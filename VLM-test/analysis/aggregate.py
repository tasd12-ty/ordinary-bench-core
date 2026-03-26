"""
跨场景和模型聚合评估结果。

生成论文用的汇总表格（表1等）。
"""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from reconstruct.preparation import load_questions_auto


def load_scene_results(results_dir: str) -> List[dict]:
    """从模型的 scenes/ 目录加载所有场景结果文件。"""
    scenes_dir = Path(results_dir) / "scenes"
    if not scenes_dir.exists():
        return []

    results = []
    for f in sorted(scenes_dir.glob("*.json")):
        with open(f) as fp:
            results.append(json.load(fp))
    return results


def load_questions(questions_dir: str, scene_id: str) -> List[dict]:
    """从平铺或分目录布局加载场景的所有问题。"""
    questions, _meta = load_questions_auto(questions_dir, scene_id)
    return questions


def compute_accuracy_table(
    results_dirs: Dict[str, str],
    by_split: bool = True,
) -> dict:
    """按模型和 split 计算准确率明细。

    Args:
        results_dirs: {model_name: path_to_results_dir}
        by_split: 若为 True，则按物体数量 split 分组统计

    Returns:
        {
            "models": [model_name, ...],
            "overall": {model_name: {qrr_acc, trr_hour_acc, ...}},
            "by_split": {split: {model_name: {...}}},
        }
    """
    output = {
        "models": list(results_dirs.keys()),
        "overall": {},
        "by_split": defaultdict(dict),
    }

    for model_name, results_dir in results_dirs.items():
        scenes = load_scene_results(results_dir)
        if not scenes:
            continue

        # 汇总统计
        total = defaultdict(int)
        splits = defaultdict(lambda: defaultdict(int))

        for scene in scenes:
            s = scene["scores"]
            split = scene["scene_id"].rsplit("_", 1)[0]

            for key in ["qrr_correct", "qrr_total", "trr_hour_correct",
                        "trr_quadrant_correct", "trr_adjacent_correct",
                        "trr_total", "missing"]:
                total[key] += s[key]
                splits[split][key] += s[key]

        def _acc(c, t):
            return round(c / t, 4) if t > 0 else 0.0

        output["overall"][model_name] = {
            "qrr_accuracy": _acc(total["qrr_correct"], total["qrr_total"]),
            "trr_hour_accuracy": _acc(total["trr_hour_correct"], total["trr_total"]),
            "trr_quadrant_accuracy": _acc(total["trr_quadrant_correct"], total["trr_total"]),
            "trr_adjacent_accuracy": _acc(total["trr_adjacent_correct"], total["trr_total"]),
            "missing_rate": _acc(total["missing"],
                                 total["qrr_total"] + total["trr_total"]),
            "n_scenes": len(scenes),
            "n_questions": total["qrr_total"] + total["trr_total"],
        }

        if by_split:
            for split, counts in sorted(splits.items()):
                output["by_split"][split][model_name] = {
                    "qrr_accuracy": _acc(counts["qrr_correct"], counts["qrr_total"]),
                    "trr_hour_accuracy": _acc(counts["trr_hour_correct"], counts["trr_total"]),
                    "trr_quadrant_accuracy": _acc(counts["trr_quadrant_correct"], counts["trr_total"]),
                    "trr_adjacent_accuracy": _acc(counts["trr_adjacent_correct"], counts["trr_total"]),
                    "n_scenes": sum(1 for sc in scenes if sc["scene_id"].startswith(split)),
                    "n_qrr": counts["qrr_total"],
                    "n_trr": counts["trr_total"],
                }

    return output


def format_accuracy_table_markdown(table: dict) -> str:
    """将准确率表格格式化为论文用的 Markdown 格式。"""
    models = table["models"]
    lines = []

    # 总体汇总表
    lines.append("## Overall Accuracy")
    lines.append("")
    header = "| Model | QRR Acc | TRR Hour | TRR Quad | TRR Adj | Missing | N Scenes |"
    lines.append(header)
    lines.append("|" + "---|" * 7)

    for model in models:
        if model not in table["overall"]:
            continue
        o = table["overall"][model]
        lines.append(
            f"| {model} | {o['qrr_accuracy']:.2%} | {o['trr_hour_accuracy']:.2%} "
            f"| {o['trr_quadrant_accuracy']:.2%} | {o['trr_adjacent_accuracy']:.2%} "
            f"| {o['missing_rate']:.2%} | {o['n_scenes']} |"
        )

    # 按 split 分组表
    if table["by_split"]:
        lines.append("")
        lines.append("## Accuracy by Split")
        lines.append("")
        header = "| Split | Model | QRR Acc | TRR Hour | TRR Quad | N Scenes |"
        lines.append(header)
        lines.append("|" + "---|" * 6)

        for split in sorted(table["by_split"]):
            for model in models:
                if model not in table["by_split"][split]:
                    continue
                s = table["by_split"][split][model]
                lines.append(
                    f"| {split} | {model} | {s['qrr_accuracy']:.2%} "
                    f"| {s['trr_hour_accuracy']:.2%} "
                    f"| {s['trr_quadrant_accuracy']:.2%} | {s['n_scenes']} |"
                )

    return "\n".join(lines)


def per_scene_accuracy(results_dir: str) -> List[dict]:
    """获取逐场景准确率，用于绘图。"""
    scenes = load_scene_results(results_dir)
    output = []
    for scene in scenes:
        s = scene["scores"]
        n = scene.get("n_objects", int(scene["scene_id"][1:3]))
        qrr_acc = s["qrr_correct"] / s["qrr_total"] if s["qrr_total"] > 0 else 0
        trr_h = s["trr_hour_correct"] / s["trr_total"] if s["trr_total"] > 0 else 0
        trr_q = s["trr_quadrant_correct"] / s["trr_total"] if s["trr_total"] > 0 else 0
        output.append({
            "scene_id": scene["scene_id"],
            "split": scene["scene_id"].rsplit("_", 1)[0],
            "n_objects": n,
            "qrr_accuracy": qrr_acc,
            "trr_hour_accuracy": trr_h,
            "trr_quadrant_accuracy": trr_q,
            "n_qrr": s["qrr_total"],
            "n_trr": s["trr_total"],
            "missing": s["missing"],
        })
    return output


if __name__ == "__main__":
    import sys

    base = Path(__file__).parent.parent / "output" / "results"
    models = {}
    for d in sorted(base.iterdir()):
        if d.is_dir():
            short_name = d.name.replace("qwen--", "").replace("-thinking", "")
            models[short_name] = str(d)

    if not models:
        print("No result directories found.")
        sys.exit(1)

    table = compute_accuracy_table(models)
    print(format_accuracy_table_markdown(table))
