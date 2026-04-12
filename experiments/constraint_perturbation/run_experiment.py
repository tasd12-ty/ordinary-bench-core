#!/usr/bin/env python3
"""约束扰动实验：Null Model for Feasibility Comparison.

对 140 个测试场景，sweep 扰动比例 p，用 consistent_flip 翻转 QRR 约束，
跑完整重建，记录 feasibility 和各项指标。

用法:
    python run_experiment.py                         # 全量运行（~6h with 8 cores）
    python run_experiment.py --max-scenes 2 --repeats 1  # 冒烟测试
    python run_experiment.py --workers 4             # 控制并行数
"""

import argparse
import json
import logging
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "VLM-test"))

from dsl.predicates import MetricType, extract_all_qrr, extract_all_qrr_shared_anchor, extract_all_trr
from extraction import parse_objects, load_scene
from reconstruct import reconstruct, SolverConfig
from validate_reconstruction import qrr_to_dict, trr_to_dict, extract_gt_positions

from perturbation import consistent_flip_qrr, compute_gt_satisfaction

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SCENE_DIR = Path(__file__).resolve().parents[2] / "datasets" / "test-data" / "scenes"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_FILE = RESULTS_DIR / "perturbation_results.jsonl"

FRACTIONS = [0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
N_RESTARTS = 5
TAU = 0.10


def _load_done_keys(results_file: Path) -> set:
    """扫描已完成的 trial，用于中断恢复。"""
    done = set()
    if results_file.exists():
        with open(results_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    done.add((r["scene_id"], r["fraction"], r["repeat"]))
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


def run_single_trial(
    scene_path: str,
    fraction: float,
    repeat: int,
    n_restarts: int = N_RESTARTS,
    tau: float = TAU,
) -> dict:
    """单次 trial：加载场景 → 扰动 → 重建 → 记录指标。"""
    scene = load_scene(scene_path)
    scene_id = scene["scene_id"]
    n_objects = int(scene_id.split("_")[0][1:])
    objects = parse_objects(scene)

    # 提取 GT 约束
    qrr_constraints = extract_all_qrr(objects, MetricType.DIST_3D, tau=tau, disjoint_only=True)
    qrr_constraints += extract_all_qrr_shared_anchor(objects, MetricType.DIST_3D, tau=tau)
    trr_constraints = extract_all_trr(objects, use_3d=True)

    qrr_dicts = [qrr_to_dict(c) for c in qrr_constraints]
    trr_dicts = [trr_to_dict(c) for c in trr_constraints]
    object_ids = sorted(objects.keys())
    gt_positions = extract_gt_positions(scene)

    n_total_qrr = len(qrr_dicts)
    n_strict = sum(1 for c in qrr_dicts if c["comparator"] != "~=")

    # 扰动
    seed = hash((scene_id, fraction, repeat)) % (2**31)
    rng = random.Random(seed)

    if fraction == 0:
        perturbed_qrr = qrr_dicts
        n_flipped = 0
    else:
        perturbed_qrr, n_flipped = consistent_flip_qrr(qrr_dicts, fraction, rng)

    n_target = int(n_strict * fraction)
    saturation = n_flipped / n_target if n_target > 0 else 1.0
    gt_sat = compute_gt_satisfaction(perturbed_qrr, objects, tau)

    # 重建
    t0 = time.time()
    try:
        result = reconstruct(
            qrr_constraints=perturbed_qrr,
            trr_constraints=trr_dicts,
            object_ids=object_ids,
            gt_positions=gt_positions,
            config=SolverConfig(n_restarts=n_restarts, bt_ratio_alpha=0.0),
        )
        rd = result.to_dict()
        elapsed = time.time() - t0

        return {
            "scene_id": scene_id,
            "n_objects": n_objects,
            "fraction": fraction,
            "repeat": repeat,
            "seed": seed,
            "n_total_qrr": n_total_qrr,
            "n_strict": n_strict,
            "n_target": n_target,
            "n_flipped": n_flipped,
            "saturation": round(saturation, 4),
            "gt_satisfaction": round(gt_sat, 4),
            "feasible": rd["feasible"],
            "status": rd["status"],
            "csr_qrr": rd["metrics"].get("csr_qrr"),
            "csr_trr": rd["metrics"].get("csr_trr"),
            "nrms": rd["metrics"].get("nrms"),
            "kendall_tau": rd["metrics"].get("kendall_tau"),
            "best_loss": rd["metrics"].get("best_loss"),
            "K_geom": rd["metrics"].get("K_geom"),
            "has_cycle": rd["feasibility_checks"].get("qrr_has_cycle", False),
            "elapsed_s": round(elapsed, 2),
        }
    except Exception as e:
        elapsed = time.time() - t0
        return {
            "scene_id": scene_id,
            "n_objects": n_objects,
            "fraction": fraction,
            "repeat": repeat,
            "seed": seed,
            "n_total_qrr": n_total_qrr,
            "n_strict": n_strict,
            "n_target": n_target,
            "n_flipped": n_flipped,
            "saturation": round(saturation, 4),
            "gt_satisfaction": round(gt_sat, 4),
            "feasible": False,
            "status": "error",
            "error": str(e),
            "elapsed_s": round(elapsed, 2),
        }


def main():
    parser = argparse.ArgumentParser(description="Run constraint perturbation experiment")
    parser.add_argument("--scenes-dir", default=str(SCENE_DIR), help="Scene directory")
    parser.add_argument("--output", default=str(RESULTS_FILE), help="Output JSONL file")
    parser.add_argument("--split", default=None, help="Filter by split prefix")
    parser.add_argument("--max-scenes", type=int, default=None, help="Max scenes to process")
    parser.add_argument("--repeats", "-R", type=int, default=20, help="Repeats per (scene, p) for p>0")
    parser.add_argument("--restarts", type=int, default=N_RESTARTS, help="Solver restarts")
    parser.add_argument("--workers", "-w", type=int, default=8, help="Parallel workers")
    parser.add_argument("--no-resume", action="store_true", help="Ignore existing results")
    args = parser.parse_args()

    scenes_dir = Path(args.scenes_dir)
    scene_files = sorted(scenes_dir.glob("*.json"))
    if args.split:
        scene_files = [f for f in scene_files if f.stem.startswith(args.split)]
    if args.max_scenes:
        scene_files = scene_files[: args.max_scenes]

    if not scene_files:
        logger.error("No scene files found in %s", scenes_dir)
        sys.exit(1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 中断恢复
    done_keys = set() if args.no_resume else _load_done_keys(output_path)
    if done_keys:
        logger.info("Resuming: %d trials already completed", len(done_keys))

    # 构建任务列表
    tasks = []
    for scene_path in scene_files:
        scene_id = scene_path.stem
        for fraction in FRACTIONS:
            n_repeats = 1 if fraction == 0 else args.repeats
            for repeat in range(n_repeats):
                key = (scene_id, fraction, repeat)
                if key in done_keys:
                    continue
                tasks.append((str(scene_path), fraction, repeat, args.restarts))

    n_total = len(tasks) + len(done_keys)
    logger.info(
        "%d scenes, %d total trials, %d remaining (workers=%d, restarts=%d)",
        len(scene_files), n_total, len(tasks), args.workers, args.restarts,
    )

    if not tasks:
        logger.info("All trials already completed.")
        return

    # 执行
    n_done = len(done_keys)
    t_start = time.time()

    with open(output_path, "a") as out_f:
        with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = {
                pool.submit(run_single_trial, *t): t for t in tasks
            }
            for future in as_completed(futures):
                scene_path, fraction, repeat, _ = futures[future]
                scene_id = Path(scene_path).stem
                try:
                    result = future.result()
                    out_f.write(json.dumps(result, default=str) + "\n")
                    out_f.flush()
                    n_done += 1

                    elapsed_total = time.time() - t_start
                    rate = n_done / max(elapsed_total, 1)
                    eta = (n_total - n_done) / rate if rate > 0 else 0

                    logger.info(
                        "[%d/%d] %s p=%.2f r=%d | feas=%s sat=%.2f csr=%.3f t=%.1fs | ETA %.0fm",
                        n_done, n_total,
                        result["scene_id"], fraction, repeat,
                        result["feasible"], result.get("saturation", 0),
                        result.get("csr_qrr", 0) or 0,
                        result.get("elapsed_s", 0),
                        eta / 60,
                    )
                except Exception as e:
                    logger.error("[%d/%d] %s p=%.2f r=%d FAILED: %s",
                                n_done, n_total, scene_id, fraction, repeat, e)

    total_time = time.time() - t_start
    logger.info("Done. %d trials in %.1f min. Results: %s", len(tasks), total_time / 60, output_path)


if __name__ == "__main__":
    main()
