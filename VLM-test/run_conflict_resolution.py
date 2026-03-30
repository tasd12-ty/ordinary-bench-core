#!/usr/bin/env python3
"""
迭代冲突消解入口脚本
====================

从已有的 VLM 评测结果出发，提取约束冲突，调用 VLM 重问冲突题目，
迭代直到收敛，最终输出噪声/系统性错误的分解诊断。

用法：
    # 仅检测冲突（不调用 API，适合预览）
    python run_conflict_resolution.py --job API-test/jobs/conflict_resolution_gemini.toml --dry-run

    # 对单个场景调试
    python run_conflict_resolution.py --job API-test/jobs/conflict_resolution_gemini.toml --scene n06_000080

    # 指定 split 和数量
    python run_conflict_resolution.py --job API-test/jobs/conflict_resolution_gemini.toml --split n06 --max-scenes 5

    # 完整执行（调用 VLM API 进行迭代消解）
    python run_conflict_resolution.py --job API-test/jobs/conflict_resolution_gemini.toml
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "API-test"))
sys.path.insert(0, str(Path(__file__).parent))

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

from reconstruct.preparation import load_questions_auto

from conflict_resolution.conflict_detector import detect_conflicts
from conflict_resolution.resolver import resolve_scene
from conflict_resolution.report import save_scene_result, generate_summary


def _load_config(path: str) -> dict:
    """加载 TOML 配置文件。"""
    with open(path, "rb") as f:
        return tomllib.load(f)


def _expand_env(val: str) -> str:
    """展开 env:VAR_NAME 格式的环境变量引用。"""
    import os
    if isinstance(val, str) and val.startswith("env:"):
        return os.environ.get(val[4:], "")
    return val


class _SimpleNS:
    """轻量命名空间，替代 dataclass 以兼容 Python 3.9（无 slots 支持）。"""
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _load_scene_objects(questions_dir: str, scene_id: str) -> list:
    """从问题文件中加载场景物体描述列表。

    尝试依次读取 qrr/trr/fdr 目录下的场景文件，取第一个有 objects 的。
    """
    for qtype in ("qrr", "trr", "fdr"):
        qpath = Path(questions_dir) / qtype / f"{scene_id}.json"
        if qpath.exists():
            data = json.load(open(qpath))
            objects = data.get("objects", [])
            if objects:
                return objects
    return []


def main():
    parser = argparse.ArgumentParser(
        description="迭代冲突消解：区分 VLM 的随机噪声与系统性误解"
    )
    parser.add_argument("--job", required=True, help="TOML 配置文件路径")
    parser.add_argument("--scene", default=None, help="单场景 ID（调试用）")
    parser.add_argument("--split", default=None, help="覆盖 TOML 中的 split 过滤")
    parser.add_argument("--max-scenes", type=int, default=None, help="每个 split 最大场景数")
    parser.add_argument("--dry-run", action="store_true", help="仅检测冲突，不调 VLM API")
    args = parser.parse_args()

    cfg = _load_config(args.job)

    # ── 解析配置 ──

    # VLM 服务配置
    prov_cfg = cfg["provider"]
    provider_spec = _SimpleNS(
        adapter=prov_cfg["adapter"],
        model=prov_cfg["model"],
        base_url=_expand_env(prov_cfg.get("base_url", "")),
        api_key=_expand_env(prov_cfg.get("api_key", "")),
        options=prov_cfg.get("options", {}),
    )

    # 图片配置
    img_cfg = cfg.get("images", {})
    image_spec = _SimpleNS(
        mode=img_cfg.get("mode", "single"),
        single_view_root=img_cfg.get("single_view_root", ""),
        multi_view_root=img_cfg.get("multi_view_root", ""),
        n_views=img_cfg.get("n_views", 1),
        wrong_image_seed=42,
    )

    # 输入路径
    inp_cfg = cfg["input"]
    source_run = inp_cfg["source_run"]
    results_dir = Path(inp_cfg["results_dir"]) / source_run / "scenes"
    questions_dir = inp_cfg["questions_dir"]

    # 消解参数
    res_cfg = cfg.get("resolution", {})
    max_rounds = res_cfg.get("max_rounds", 10)
    patience = res_cfg.get("patience", 2)
    splits = args.split.split(",") if args.split else res_cfg.get("splits", [])
    max_scenes = args.max_scenes or res_cfg.get("max_scenes_per_split")

    # 输出路径
    out_cfg = cfg.get("output", {})
    output_dir = Path(out_cfg.get("output_dir", "output/conflict_resolution"))
    run_name = out_cfg.get("run_name", "default")
    run_dir = output_dir / run_name

    # ── 发现场景 ──

    scene_files = sorted(results_dir.glob("*.json"))

    # 应用过滤条件
    if args.scene:
        scene_files = [f for f in scene_files if f.stem == args.scene]
    elif splits:
        scene_files = [f for f in scene_files if any(f.stem.startswith(sp) for sp in splits)]

    # 按 split 分别限制数量
    if max_scenes:
        by_split_files = defaultdict(list)
        for f in scene_files:
            sp = f.stem.rsplit("_", 1)[0]
            by_split_files[sp].append(f)
        scene_files = []
        for sp in sorted(by_split_files):
            scene_files.extend(by_split_files[sp][:max_scenes])

    print(f"配置: model={provider_spec.model} 基底={source_run}")
    print(f"场景: {len(scene_files)} 个 | max_rounds={max_rounds} patience={patience}")
    print(f"输出: {run_dir}")

    if not scene_files:
        print("无可处理的场景。")
        return

    # ── 逐场景处理 ──

    all_results = {}

    for i, sf in enumerate(scene_files):
        sr = json.load(open(sf))
        scene_id = sr["scene_id"]

        # 加载问题和元数据
        questions, meta = load_questions_auto(questions_dir, scene_id)

        # 加载物体描述（构建 VLM prompt 用）
        scene_objects = _load_scene_objects(questions_dir, scene_id)

        # ── dry-run 模式：仅输出冲突统计 ──
        if args.dry_run:
            report = detect_conflicts(sr, questions, meta)
            fas_size = len(report.fas_result.edges_removed) if report.fas_result else 0
            print(f"[{i+1}/{len(scene_files)}] {scene_id}: "
                  f"FAS={fas_size} 冲突题={len(report.conflict_qids)} "
                  f"总题数={report.n_total_questions}")
            continue

        # ── 解析图片 ──
        from image_resolver import resolve_scene_images
        image_inputs = resolve_scene_images(scene_id, image_spec)

        # ── 执行迭代消解 ──
        print(f"[{i+1}/{len(scene_files)}] {scene_id}:")
        result = resolve_scene(
            scene_id=scene_id,
            scene_result=sr,
            questions=questions,
            scene_objects=scene_objects,
            image_inputs=image_inputs,
            provider_spec=provider_spec,
            metadata=meta,
            max_rounds=max_rounds,
            patience=patience,
        )

        d = result.diagnosis
        print(f"  → 收敛={result.converged} 轮数={result.n_rounds} "
              f"FAS: {result.initial_fas_size}→{result.final_fas_size} "
              f"噪声翻转={d.noise_flips} 系统性冲突={d.systematic_conflicts}")

        # 保存结果
        save_scene_result(result, run_dir)
        all_results[scene_id] = result

    # ── 生成汇总报告 ──
    if all_results:
        generate_summary(all_results, run_dir)


if __name__ == "__main__":
    main()
