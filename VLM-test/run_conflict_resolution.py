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
    parser.add_argument("--voting", action="store_true", help="使用投票式消解（替代迭代覆盖）")
    parser.add_argument("--reask-rounds", type=int, default=None, help="投票模式的重问轮数 K（默认 4）")
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

        if args.voting:
            # ── 投票式消解 ──
            from conflict_resolution.voting_resolver import (
                voting_resolve_scene, voting_result_to_dict,
            )
            reask_rounds = args.reask_rounds or res_cfg.get("reask_rounds", 4)
            print(f"[{i+1}/{len(scene_files)}] {scene_id} (投票模式, K={reask_rounds}):")
            vresult = voting_resolve_scene(
                scene_id=scene_id,
                scene_result=sr,
                questions=questions,
                scene_objects=scene_objects,
                image_inputs=image_inputs,
                provider_spec=provider_spec,
                metadata=meta,
                reask_rounds=reask_rounds,
            )

            d = vresult.diagnosis
            print(f"  → FAS: {d.initial_fas_size}→{d.final_fas_size} "
                  f"纠正={d.noise_corrected} 系统性={d.systematic_wrong} "
                  f"不确定={d.uncertain} 确认正确={d.confirmed_correct}")

            # 保存投票结果
            scenes_dir = run_dir / "scenes"
            scenes_dir.mkdir(parents=True, exist_ok=True)
            with open(scenes_dir / f"{scene_id}.json", "w") as f:
                json.dump(voting_result_to_dict(vresult), f, indent=2, ensure_ascii=False)
            if vresult.final_scene_result:
                resolved_dir = run_dir / "resolved_scenes"
                resolved_dir.mkdir(parents=True, exist_ok=True)
                with open(resolved_dir / f"{scene_id}.json", "w") as f:
                    json.dump(vresult.final_scene_result, f, indent=2, ensure_ascii=False)

            all_results[scene_id] = vresult
            continue

        # ── 迭代式消解（默认）──
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

        save_scene_result(result, run_dir)
        all_results[scene_id] = result

    # ── 生成汇总报告 ──
    if all_results:
        if args.voting:
            _generate_voting_summary(all_results, run_dir)
        else:
            generate_summary(all_results, run_dir)


def _generate_voting_summary(results: dict, output_dir: Path) -> None:
    """生成投票式消解的汇总报告。"""
    from collections import defaultdict

    output_dir.mkdir(parents=True, exist_ok=True)
    by_split = defaultdict(list)
    for sid, r in results.items():
        sp = sid.rsplit("_", 1)[0]
        by_split[sp].append(r)

    summary = {"by_split": {}, "overall": {}}
    total_scenes = 0
    total_initial = 0
    total_final = 0
    total_corrected = 0
    total_systematic = 0
    total_uncertain = 0
    total_confirmed = 0

    print(f"\n{'Split':>5} | {'场景数':>6} | {'初始FAS':>8} | {'最终FAS':>8} | "
          f"{'纠正':>6} | {'系统性':>6} | {'不确定':>6} | {'确认正确':>8}")
    print("-" * 80)

    for sp in sorted(by_split):
        rs = by_split[sp]
        n = len(rs)
        avg_init = sum(r.diagnosis.initial_fas_size for r in rs) / n
        avg_final = sum(r.diagnosis.final_fas_size for r in rs) / n
        sum_corrected = sum(r.diagnosis.noise_corrected for r in rs)
        sum_systematic = sum(r.diagnosis.systematic_wrong for r in rs)
        sum_uncertain = sum(r.diagnosis.uncertain for r in rs)
        sum_confirmed = sum(r.diagnosis.confirmed_correct for r in rs)

        print(f"{sp:>5} | {n:>6} | {avg_init:>8.1f} | {avg_final:>8.1f} | "
              f"{sum_corrected:>6} | {sum_systematic:>6} | "
              f"{sum_uncertain:>6} | {sum_confirmed:>8}")

        summary["by_split"][sp] = {
            "n_scenes": n,
            "avg_initial_fas": round(avg_init, 2),
            "avg_final_fas": round(avg_final, 2),
            "noise_corrected": sum_corrected,
            "systematic_wrong": sum_systematic,
            "uncertain": sum_uncertain,
            "confirmed_correct": sum_confirmed,
        }

        total_scenes += n
        total_initial += sum(r.diagnosis.initial_fas_size for r in rs)
        total_final += sum(r.diagnosis.final_fas_size for r in rs)
        total_corrected += sum_corrected
        total_systematic += sum_systematic
        total_uncertain += sum_uncertain
        total_confirmed += sum_confirmed

    if total_scenes:
        summary["overall"] = {
            "n_scenes": total_scenes,
            "avg_initial_fas": round(total_initial / total_scenes, 2),
            "avg_final_fas": round(total_final / total_scenes, 2),
            "noise_corrected": total_corrected,
            "systematic_wrong": total_systematic,
            "uncertain": total_uncertain,
            "confirmed_correct": total_confirmed,
        }
        print("-" * 80)
        o = summary["overall"]
        print(f"{'合计':>5} | {total_scenes:>6} | {o['avg_initial_fas']:>8.1f} | "
              f"{o['avg_final_fas']:>8.1f} | {total_corrected:>6} | "
              f"{total_systematic:>6} | {total_uncertain:>6} | {total_confirmed:>8}")

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n结果已保存至: {output_dir}")


if __name__ == "__main__":
    main()
