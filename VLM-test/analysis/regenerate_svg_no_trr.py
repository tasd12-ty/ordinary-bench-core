"""
对所有已有 SVG 的重建场景，使用仅 QRR（不含 TRR）模式重新重建并生成对比 SVG。

输出文件名：在原 SVG 同目录下，加 _no_trr 后缀。
例: n04_000083.svg → n04_000083_no_trr.svg
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from reconstruct.preparation import (
    load_questions_auto,
    prepare_reconstruction_input_from_scoring,
    load_scene_gt_positions,
)
from reconstruct.pipeline import reconstruct_from_prepared
from analysis.visualize_svg import (
    render_scene_comparison_svg,
    load_object_info,
    ObjectInfo,
)


def regenerate_all(
    recon_base: str = "output/analysis/belief_recon",
    results_base: str = "output/results",
    questions_dir: str = "output/questions",
    scenes_dir: str = "../data-gen/output/scenes",
    images_dir: str = "../data-gen/output/images/single_view",
    n_restarts: int = 10,
):
    recon_root = Path(recon_base)
    results_root = Path(results_base)
    scenes_path = Path(scenes_dir)
    images_path = Path(images_dir)

    total = 0
    done = 0

    for model_dir in sorted(recon_root.iterdir()):
        if not model_dir.is_dir():
            continue
        model = model_dir.name
        results_dir = results_root / model / "scenes"
        if not results_dir.exists():
            continue

        svg_files = sorted(model_dir.glob("*.svg"))
        # 排除已有的 _no_trr svg
        svg_files = [f for f in svg_files if not f.stem.endswith("_no_trr")]
        if not svg_files:
            continue

        print(f"\n=== {model} ({len(svg_files)} scenes) ===")

        for svg_file in svg_files:
            scene_id = svg_file.stem
            total += 1

            result_file = results_dir / f"{scene_id}.json"
            if not result_file.exists():
                print(f"  {scene_id}: no scoring result, skip")
                continue

            gt_scene_path = scenes_path / f"{scene_id}.json"
            if not gt_scene_path.exists():
                print(f"  {scene_id}: no GT scene, skip")
                continue

            # 加载评分结果
            with open(result_file) as f:
                result = json.load(f)

            # 加载问题
            questions, _ = load_questions_auto(questions_dir, scene_id)
            if not questions:
                print(f"  {scene_id}: no questions, skip")
                continue

            scoring = result.get("scores", result)

            # 准备约束（belief 模式）
            prepared = prepare_reconstruction_input_from_scoring(
                scoring_result=scoring,
                questions=questions,
                use_correct_only=False,
            )

            # 加载 GT 位置
            gt_pos_raw = load_scene_gt_positions(str(gt_scene_path))
            if gt_pos_raw:
                for oid, pos in gt_pos_raw.items():
                    prepared.gt_positions[oid] = pos

            # 重建：仅 QRR，不含 TRR
            try:
                recon = reconstruct_from_prepared(
                    prepared_input=prepared,
                    n_restarts=n_restarts,
                    constraint_mode="fdr_qrr",
                )
            except Exception as e:
                print(f"  {scene_id}: reconstruct error: {e}")
                continue

            if not recon.positions:
                print(f"  {scene_id}: no positions, skip")
                continue

            # 加载 GT 位置和物体信息
            gt_positions = {}
            with open(gt_scene_path) as f:
                scene_data = json.load(f)
            for obj in scene_data["objects"]:
                gt_positions[obj["id"]] = np.array(obj["3d_coords"][:2], dtype=float)

            object_info = load_object_info(str(gt_scene_path))

            # 重建位置
            recon_positions = {
                oid: np.array(pos, dtype=float)
                for oid, pos in recon.positions.items()
            }

            metrics = {
                "csr_qrr": recon.metrics.csr_qrr,
                "csr_trr": None,  # 无 TRR
                "nrms": recon.metrics.nrms,
                "kendall_tau": recon.metrics.kendall_tau,
                "K_geom": recon.metrics.K_geom,
            }

            # Blender 渲染图
            blender_img = images_path / f"{scene_id}.png"
            blender_path = str(blender_img) if blender_img.exists() else None

            # 渲染 SVG
            svg = render_scene_comparison_svg(
                gt_positions=gt_positions,
                recon_positions=recon_positions,
                object_info=object_info,
                scene_id=f"{scene_id} (no TRR)",
                metrics=metrics,
                blender_image_path=blender_path,
                panel_size=300,
            )

            out_path = model_dir / f"{scene_id}_no_trr.svg"
            with open(out_path, "w") as f:
                f.write(svg)

            m = recon.metrics
            nrms_str = f"{m.nrms:.4f}" if m.nrms is not None else "N/A"
            kt_str = f"{m.kendall_tau:.3f}" if m.kendall_tau is not None else "N/A"
            print(f"  {scene_id}: CSR_QRR={m.csr_qrr:.3f}  NRMS={nrms_str}  tau={kt_str}  K={m.K_geom}  -> {out_path.name}")
            done += 1

    print(f"\n=== Done: {done}/{total} SVGs generated ===")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Regenerate SVGs with QRR-only (no TRR) reconstruction")
    parser.add_argument("--recon-base", default="output/analysis/belief_recon")
    parser.add_argument("--results-base", default="output/results")
    parser.add_argument("--questions-dir", default="output/questions")
    parser.add_argument("--scenes-dir", default="../data-gen/output/scenes")
    parser.add_argument("--images-dir", default="../data-gen/output/images/single_view")
    parser.add_argument("--restarts", type=int, default=10)

    args = parser.parse_args()
    regenerate_all(
        recon_base=args.recon_base,
        results_base=args.results_base,
        questions_dir=args.questions_dir,
        scenes_dir=args.scenes_dir,
        images_dir=args.images_dir,
        n_restarts=args.restarts,
    )
