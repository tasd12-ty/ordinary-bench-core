"""
批量 belief 重建 + SVG 渲染。

使用 v2 QRR + FDR 问题，跳过环检查，让 solver 尽力求解。
即使约束有环 (infeasible)，solver 仍能找到最优妥协解。

用法:
    cd VLM-test
    uv run python -m analysis.batch_recon_svg --model claude_opus_single_v2
    uv run python -m analysis.batch_recon_svg  # 所有模型
"""

import argparse
import json
import os
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "API-test"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from reconstruct.solver import SolverConfig
from analysis.visualize_svg import render_scene_comparison_svg


def load_v2_questions(questions_dir: Path, scene_id: str) -> list:
    """加载 v2 格式的 QRR + FDR 问题。"""
    qs = []
    for qtype in ["qrr", "fdr"]:
        p = questions_dir / qtype / f"{scene_id}.json"
        if p.exists():
            d = json.load(open(p))
            for b in d["batches"]:
                qs.extend(b["questions"])
    return qs


def reconstruct_belief_acyclic(
    scoring_result: dict,
    questions: list,
    gt_positions: dict,
    config: SolverConfig,
) -> dict:
    """Belief 重建，只处理无环场景。有环的直接跳过。"""
    from reconstruct.pipeline import reconstruct_from_scoring

    recon = reconstruct_from_scoring(
        scoring_result, questions,
        gt_positions=gt_positions,
        config=config,
        use_correct_only=False,
    )

    if not recon.positions:
        return {"status": recon.status, "positions": None}

    # 检查堆叠
    objs = list(recon.positions.keys())
    min_dist = float("inf")
    for i in range(len(objs)):
        for j in range(i + 1, len(objs)):
            d = np.linalg.norm(recon.positions[objs[i]] - recon.positions[objs[j]])
            min_dist = min(min_dist, d)

    return {
        "status": recon.status,
        "positions": recon.positions,
        "metrics": recon.metrics,
        "min_dist": min_dist,
    }


def main():
    parser = argparse.ArgumentParser(description="批量 belief 重建 + SVG")
    parser.add_argument("--model", default=None, help="模型名 (默认全部)")
    parser.add_argument("--scenes-dir", default="../datasets/test-data/scenes")
    parser.add_argument("--questions-dir", default="output/questions")
    parser.add_argument("--results-dir", default="output/results")
    parser.add_argument("--output-dir", default="output/analysis/svg_belief")
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--n-restarts", type=int, default=15)
    args = parser.parse_args()

    scenes_dir = Path(args.scenes_dir)
    questions_dir = Path(args.questions_dir)
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)

    # 发现模型
    if args.model:
        models = [args.model]
    else:
        models = sorted([
            d.name for d in results_dir.iterdir()
            if d.is_dir() and (d / "scenes").is_dir()
        ])

    # 发现场景
    scene_ids = sorted([f.stem for f in scenes_dir.glob("*.json")])

    config = SolverConfig(n_restarts=args.n_restarts)
    total_ok = 0
    total_skip = 0

    for model in models:
        model_dir = output_dir / model
        model_dir.mkdir(parents=True, exist_ok=True)

        scene_files = sorted((results_dir / model / "scenes").glob("*.json"))
        if args.max_scenes:
            scene_files = scene_files[:args.max_scenes]

        print(f"\n=== {model} ({len(scene_files)} scenes) ===")

        for sf in scene_files:
            scene_id = sf.stem
            gt_path = scenes_dir / f"{scene_id}.json"
            if not gt_path.exists():
                continue

            gt = json.load(open(gt_path))
            gt_positions = {o["id"]: o["3d_coords"][:2] for o in gt["objects"]}
            object_info = {
                o["id"]: type("OI", (), {
                    "shape": o["shape"], "color": o["color"],
                    "size": o["size"], "material": o["material"],
                })()
                for o in gt["objects"]
            }

            sr = json.load(open(sf))
            questions = load_v2_questions(questions_dir, scene_id)
            if not questions:
                continue

            result = reconstruct_belief_acyclic(
                sr["scores"], questions, gt_positions, config
            )

            if result["positions"] is None:
                total_skip += 1
                continue

            m = result["metrics"]
            print(f"  {scene_id} (n={len(gt['objects'])}): "
                  f"CSR={m.csr_qrr:.3f} tau={m.kendall_tau:.3f} "
                  f"NRMS={m.nrms:.3f} min_d={result['min_dist']:.3f}")

            svg = render_scene_comparison_svg(
                gt_positions=gt_positions,
                recon_positions=result["positions"],
                object_info=object_info,
                scene_id=scene_id,
                metrics={
                    "csr_qrr": m.csr_qrr,
                    "kendall_tau": m.kendall_tau,
                    "nrms": m.nrms,
                    "K_geom": m.K_geom,
                },
                panel_size=300,
            )
            with open(model_dir / f"{scene_id}.svg", "w") as f:
                f.write(svg)
            total_ok += 1

    print(f"\nDone: {total_ok} SVGs rendered, {total_skip} skipped")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
