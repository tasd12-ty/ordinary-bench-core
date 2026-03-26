"""
生成场景信念重建对比 SVG 示例图。

用法：
    python -m analysis.generate_svg_examples [--max-scenes 5] [--model MODEL_DIR]
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "API-test"))

from reconstruct import reconstruct_from_scoring, SolverConfig
from analysis.visualize_svg import (
    render_scene_comparison_svg,
    load_object_info,
    ObjectInfo,
)


def find_model_results(base_dir: str = "output/results") -> list:
    """查找所有模型结果目录。"""
    base = Path(base_dir)
    if not base.exists():
        return []
    return sorted([d for d in base.iterdir() if d.is_dir() and (d / "scenes").exists()])


def load_scene_result(model_dir: Path, scene_id: str) -> dict:
    """加载场景评分结果 JSON 文件。"""
    path = model_dir / "scenes" / f"{scene_id}.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def load_gt_positions(scene_json_path: str) -> dict:
    """从场景 JSON 文件加载真值坐标。"""
    with open(scene_json_path) as f:
        scene = json.load(f)
    return {
        obj["id"]: np.array(obj["3d_coords"][:2], dtype=float)
        for obj in scene["objects"]
    }


def generate_examples(
    max_scenes: int = 5,
    model_dir: str = None,
    scenes_dir: str = "../data-gen/output/scenes",
    images_dir: str = "../data-gen/output/images/single_view",
    output_dir: str = "output/analysis/svg",
    n_restarts: int = 10,
):
    """生成场景重建对比 SVG 示例图。"""
    os.makedirs(output_dir, exist_ok=True)

    # 查找模型结果目录
    if model_dir:
        model_dirs = [Path(model_dir)]
    else:
        model_dirs = find_model_results()

    if not model_dirs:
        print("No model results found in output/results/")
        return

    scenes_path = Path(scenes_dir)
    images_path = Path(images_dir)

    for mdir in model_dirs:
        model_name = mdir.name
        print(f"\n=== Model: {model_name} ===")

        # 获取可用的场景结果文件
        scene_files = sorted((mdir / "scenes").glob("*.json"))[:max_scenes]

        for sf in scene_files:
            scene_id = sf.stem
            print(f"\n  Processing {scene_id}...")

            # 加载 VLM 评分结果
            scene_result = load_scene_result(mdir, scene_id)
            if not scene_result:
                print(f"    Skipping: no result")
                continue

            # 加载真值场景
            gt_scene_path = scenes_path / f"{scene_id}.json"
            if not gt_scene_path.exists():
                print(f"    Skipping: no GT scene at {gt_scene_path}")
                continue

            gt_positions = load_gt_positions(str(gt_scene_path))
            object_info = load_object_info(str(gt_scene_path))

            # 加载问题文件
            questions_path = Path("output/questions") / f"{scene_id}.json"
            if not questions_path.exists():
                print(f"    Skipping: no questions at {questions_path}")
                continue

            with open(questions_path) as f:
                q_data = json.load(f)
            all_questions = []
            for batch in q_data["batches"]:
                all_questions.extend(batch["questions"])

            # 从 VLM 预测结果重建场景布局
            scoring = scene_result.get("scores", {})
            config = SolverConfig(n_restarts=n_restarts)
            recon = reconstruct_from_scoring(
                scoring, all_questions,
                gt_positions=gt_positions,
                config=config,
            )

            if recon.status == "no_constraints":
                print(f"    Skipping: no constraints extracted")
                continue

            # 获取重建坐标
            recon_positions = recon.positions

            if not recon_positions:
                print(f"    Skipping: no reconstruction result")
                continue

            # Blender 渲染图像路径
            blender_img = images_path / f"{scene_id}.png"
            blender_path = str(blender_img) if blender_img.exists() else None

            # 重建评估指标
            metrics = None
            if recon.metrics:
                metrics = {
                    "csr_qrr": recon.metrics.csr_qrr,
                    "csr_trr": recon.metrics.csr_trr,
                    "nrms": recon.metrics.nrms,
                    "kendall_tau": recon.metrics.kendall_tau,
                    "K_geom": recon.metrics.K_geom,
                }

            # 渲染 SVG
            svg = render_scene_comparison_svg(
                gt_positions=gt_positions,
                recon_positions=recon_positions,
                object_info=object_info,
                scene_id=scene_id,
                metrics=metrics,
                blender_image_path=blender_path,
                panel_size=300,
            )

            out_path = os.path.join(output_dir, f"{model_name}_{scene_id}.svg")
            with open(out_path, "w") as f:
                f.write(svg)

            m = recon.metrics
            print(f"    NRMS={m.nrms:.4f}  CSR_QRR={m.csr_qrr:.3f}  "
                  f"CSR_TRR={m.csr_trr:.3f}  K={m.K_geom}")
            print(f"    Saved: {out_path}")

    print(f"\n  All SVGs saved to {output_dir}/")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate SVG scene comparisons")
    parser.add_argument("--max-scenes", type=int, default=5)
    parser.add_argument("--model-dir", type=str, default=None)
    parser.add_argument("--scenes-dir", default="../data-gen/output/scenes")
    parser.add_argument("--images-dir", default="../data-gen/output/images/single_view")
    parser.add_argument("--output-dir", default="output/analysis/svg")
    parser.add_argument("--restarts", type=int, default=10)
    args = parser.parse_args()

    generate_examples(
        max_scenes=args.max_scenes,
        model_dir=args.model_dir,
        scenes_dir=args.scenes_dir,
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        n_restarts=args.restarts,
    )
