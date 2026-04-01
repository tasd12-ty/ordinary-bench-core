"""
Step 5: 对比分析 full-image vs subset-image 的 QRR 准确率。

用法:
    python analyze_results.py \
        --full-results output/results/full \
        --subset-results output/results/subset \
        --mapping output/question_mapping.json \
        --output output/analysis
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def load_scene_results(results_dir: Path) -> dict:
    """加载逐场景评分结果。返回 {scene_id: {per_question: [...]}}"""
    scenes_dir = results_dir / "scenes"
    if not scenes_dir.exists():
        return {}
    results = {}
    for f in sorted(scenes_dir.glob("*.json")):
        with open(f) as fh:
            data = json.load(fh)
        sid = data.get("scene_id", f.stem)
        results[sid] = data
    return results


def extract_predictions(scene_result: dict) -> dict:
    """从场景结果中提取 {qid: {predicted, correct}}。"""
    preds = {}
    per_q = scene_result.get("scores", scene_result).get("per_question", [])
    for q in per_q:
        preds[q["qid"]] = {
            "predicted": q.get("predicted"),
            "correct": q.get("correct", False),
            "variant": q.get("variant", ""),
        }
    return preds


def majority_vote(answers: list) -> str:
    """多数投票。"""
    counter = Counter(a for a in answers if a is not None)
    if not counter:
        return None
    return counter.most_common(1)[0][0]


def main():
    parser = argparse.ArgumentParser(description="对比 full-image vs subset-image QRR 准确率")
    parser.add_argument("--full-results", required=True, help="全图结果目录")
    parser.add_argument("--subset-results", required=True, help="子集图结果目录")
    parser.add_argument("--mapping", required=True, help="question_mapping.json 路径")
    parser.add_argument("--output", default="output/analysis", help="分析输出目录")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载数据
    full_results = load_scene_results(Path(args.full_results))
    subset_results = load_scene_results(Path(args.subset_results))

    with open(args.mapping) as f:
        question_mapping = json.load(f)

    # 对比分析
    comparison_rows = []
    per_split = defaultdict(lambda: {
        "full_correct": 0, "full_total": 0,
        "subset_correct": 0, "subset_total": 0,
    })
    per_variant = defaultdict(lambda: {
        "full_correct": 0, "full_total": 0,
        "subset_correct": 0, "subset_total": 0,
    })
    confusion = {"both_correct": 0, "full_only": 0, "subset_only": 0, "both_wrong": 0}

    # 自一致性分析（shared_anchor 跨子集）
    self_agreement_data = []

    for parent_id, qmap in question_mapping.items():
        split = parent_id.rsplit("_", 1)[0]  # e.g. "n10"

        # 获取 full-image 预测
        full_preds = {}
        if parent_id in full_results:
            full_preds = extract_predictions(full_results[parent_id])

        for qkey, qinfo in qmap.items():
            gt = qinfo["gt_comparator"]
            variant = qinfo["variant"]
            subset_ids = qinfo["subset_ids"]

            # 收集子集预测
            subset_answers = []
            for subset_id in subset_ids:
                if subset_id not in subset_results:
                    continue
                sub_preds = extract_predictions(subset_results[subset_id])
                sub_qid = qinfo["subset_qids"].get(subset_id)
                if sub_qid and sub_qid in sub_preds:
                    subset_answers.append(sub_preds[sub_qid]["predicted"])

            # 子集答案合并（多数投票）
            subset_answer = majority_vote(subset_answers)
            subset_correct = (subset_answer == gt) if subset_answer else False

            # 全图答案（需要找到对应的 full qid）
            # full 结果中的 qid 编号可能不同，用 pair 匹配
            full_correct = None
            full_answer = None
            for fqid, fpred in full_preds.items():
                # 简单匹配：variant + 相同 pair
                # 这里先用所有 full 预测，后续可精确匹配
                pass

            # 如果没有精确匹配，跳过 full 对比
            # (full 结果的 question 格式可能需要单独处理)

            # 自一致性（仅 shared_anchor，因为 disjoint 在 k=4 时只出现在 1 个子集）
            if len(subset_answers) > 1 and variant == "shared_anchor":
                most_common = Counter(subset_answers).most_common(1)[0]
                agreement = most_common[1] / len(subset_answers)
                self_agreement_data.append({
                    "parent_id": parent_id,
                    "qkey": qkey,
                    "variant": variant,
                    "n_subsets": len(subset_answers),
                    "agreement": agreement,
                    "answers": subset_answers,
                    "gt": gt,
                    "majority_correct": subset_correct,
                })

            row = {
                "parent_id": parent_id,
                "split": split,
                "variant": variant,
                "gt": gt,
                "subset_answer": subset_answer,
                "subset_correct": subset_correct,
                "n_subset_answers": len(subset_answers),
            }
            comparison_rows.append(row)

            # 聚合
            per_split[split]["subset_total"] += 1
            per_variant[variant]["subset_total"] += 1
            if subset_correct:
                per_split[split]["subset_correct"] += 1
                per_variant[variant]["subset_correct"] += 1

    # 汇总
    summary = {
        "per_split": {},
        "per_variant": {},
        "self_agreement": {},
        "total_unique_questions": len(comparison_rows),
    }

    for split, data in sorted(per_split.items()):
        total = data["subset_total"]
        correct = data["subset_correct"]
        summary["per_split"][split] = {
            "subset_acc": round(correct / total, 4) if total else 0,
            "subset_correct": correct,
            "subset_total": total,
        }

    for variant, data in sorted(per_variant.items()):
        total = data["subset_total"]
        correct = data["subset_correct"]
        summary["per_variant"][variant] = {
            "subset_acc": round(correct / total, 4) if total else 0,
            "subset_correct": correct,
            "subset_total": total,
        }

    # 自一致性汇总
    if self_agreement_data:
        agreements = [d["agreement"] for d in self_agreement_data]
        summary["self_agreement"] = {
            "mean": round(sum(agreements) / len(agreements), 4),
            "n_questions": len(agreements),
            "mean_correct_rate": round(
                sum(1 for d in self_agreement_data if d["majority_correct"])
                / len(self_agreement_data), 4
            ),
        }

    # 输出
    with open(output_dir / "comparison.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(output_dir / "self_agreement.json", "w") as f:
        json.dump(self_agreement_data, f, indent=2)

    # 打印摘要
    print("\n=== Subset-Image QRR Accuracy ===\n")
    print(f"{'Split':<8} {'Acc':>8} {'Correct':>8} {'Total':>8}")
    print("-" * 36)
    for split, data in sorted(summary["per_split"].items()):
        print(f"{split:<8} {data['subset_acc']:>8.4f} {data['subset_correct']:>8} {data['subset_total']:>8}")

    print(f"\n{'Variant':<16} {'Acc':>8} {'Correct':>8} {'Total':>8}")
    print("-" * 44)
    for variant, data in sorted(summary["per_variant"].items()):
        print(f"{variant:<16} {data['subset_acc']:>8.4f} {data['subset_correct']:>8} {data['subset_total']:>8}")

    if summary["self_agreement"]:
        sa = summary["self_agreement"]
        print(f"\nSelf-agreement (shared_anchor across subsets):")
        print(f"  Mean agreement: {sa['mean']:.4f} ({sa['n_questions']} questions)")
        print(f"  Majority-vote correct rate: {sa['mean_correct_rate']:.4f}")

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
