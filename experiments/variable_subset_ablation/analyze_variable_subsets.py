"""
Variable-Size Subset Ablation: 按子集大小分组分析 VLM 评测结果。

读取 manifest + VLM 评测结果，按 subset_size 分组计算:
  1. QRR 精度 (overall / disjoint / shared_anchor)
  2. 拒答检测率 + hallucination rate
  3. 跨尺寸答案一致性 (同一题在不同大小子场景中是否一致)

输入:
  - manifest.json          — enumerate_variable_subsets.py 的输出
  - results/{run}/scenes/  — run_subset_eval.py 的输出（每个子集一个 JSON）

输出:
  - analysis/accuracy_by_size.json     — 每种大小的精度汇总
  - analysis/refusal_by_size.json      — 每种大小的拒答率
  - analysis/consistency.json          — 跨尺寸一致性分析
  - analysis/raw_predictions.json      — 全部预测明细（供可视化使用）

用法:
    cd experiments/variable_subset_ablation

    python analyze_variable_subsets.py \
        --manifest output/manifest.json \
        --results-dir output/results/gpt4o/scenes \
        --output-dir output/analysis

依赖:
    仅标准库，无额外依赖。
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_results(results_dir: Path) -> dict:
    """加载所有子集评测结果。返回 {subset_id: result_dict}。"""
    results = {}
    for f in sorted(results_dir.glob("*.json")):
        with open(f) as fh:
            data = json.load(fh)
        sid = data.get("scene_id", f.stem)
        results[sid] = data
    return results


def analyze_accuracy_by_size(manifest: dict, results: dict) -> dict:
    """
    按子集大小分组计算 QRR 精度。

    返回:
        {
          size: {
            "overall": {"correct": N, "total": N, "acc": float},
            "disjoint": {...},
            "shared_anchor": {...},
            "n_scenes": int,
            "per_scene_acc": [float, ...]  # 用于计算置信区间
          }
        }
    """
    by_size = defaultdict(lambda: {
        "overall": {"correct": 0, "total": 0},
        "disjoint": {"correct": 0, "total": 0},
        "shared_anchor": {"correct": 0, "total": 0},
        "n_scenes": 0,
        "per_scene_acc": [],
    })

    for parent_id, parent_data in manifest["parent_scenes"].items():
        for subset_info in parent_data["subsets"]:
            subset_id = subset_info["subset_id"]
            size = subset_info["subset_size"]

            if subset_id not in results:
                continue

            result = results[subset_id]
            per_q = result.get("per_question", [])
            if not per_q:
                scores = result.get("scores", {})
                per_q = scores.get("per_question", [])

            scene_correct = 0
            scene_total = 0

            for q in per_q:
                if not q.get("answerable", True):
                    continue  # 只统计可答题

                correct = q.get("correct", False)
                variant = q.get("variant", "disjoint")

                by_size[size]["overall"]["total"] += 1
                by_size[size][variant]["total"] += 1
                if correct:
                    by_size[size]["overall"]["correct"] += 1
                    by_size[size][variant]["correct"] += 1
                    scene_correct += 1
                scene_total += 1

            by_size[size]["n_scenes"] += 1
            if scene_total > 0:
                by_size[size]["per_scene_acc"].append(scene_correct / scene_total)

    # 计算 acc
    output = {}
    for size in sorted(by_size.keys()):
        entry = by_size[size]
        for key in ["overall", "disjoint", "shared_anchor"]:
            t = entry[key]["total"]
            c = entry[key]["correct"]
            entry[key]["acc"] = round(c / t, 4) if t > 0 else 0.0
        output[size] = entry

    return output


def analyze_refusal_by_size(manifest: dict, results: dict) -> dict:
    """
    按子集大小分组计算拒答检测率和 hallucination rate。

    返回:
        {
          size: {
            "refusal_total": N,
            "refusal_correct": N,    # VLM 正确回答 N/A
            "refusal_rate": float,
            "hallucinated": N,       # VLM 对不可答题给出实际答案
            "hallucination_rate": float,
            "answerable_ratio": float  # 可答题占总题的比例
          }
        }
    """
    by_size = defaultdict(lambda: {
        "refusal_total": 0,
        "refusal_correct": 0,
        "hallucinated": 0,
        "answerable_total": 0,
        "all_total": 0,
    })

    for parent_id, parent_data in manifest["parent_scenes"].items():
        for subset_info in parent_data["subsets"]:
            subset_id = subset_info["subset_id"]
            size = subset_info["subset_size"]

            if subset_id not in results:
                continue

            result = results[subset_id]
            by_size[size]["refusal_total"] += result.get("refusal_total", 0)
            by_size[size]["refusal_correct"] += result.get("refusal_correct", 0)
            by_size[size]["hallucinated"] += result.get("refusal_hallucinated", 0)
            by_size[size]["answerable_total"] += result.get("answerable_total", 0)
            by_size[size]["all_total"] += result.get("total_questions", 0)

    output = {}
    for size in sorted(by_size.keys()):
        d = by_size[size]
        ref_t = d["refusal_total"]
        output[size] = {
            "refusal_total": ref_t,
            "refusal_correct": d["refusal_correct"],
            "refusal_rate": round(d["refusal_correct"] / ref_t, 4) if ref_t > 0 else 0.0,
            "hallucinated": d["hallucinated"],
            "hallucination_rate": round(d["hallucinated"] / ref_t, 4) if ref_t > 0 else 0.0,
            "answerable_total": d["answerable_total"],
            "all_total": d["all_total"],
            "answerable_ratio": round(
                d["answerable_total"] / d["all_total"], 4
            ) if d["all_total"] > 0 else 0.0,
        }

    return output


def analyze_consistency(manifest: dict, results: dict) -> dict:
    """
    跨尺寸答案一致性：同一 qid 在不同大小子场景中的 VLM 答案是否一致。

    对每个 qid，收集它在所有子场景（不同 size）中的预测值，
    计算一致率 = 最频繁答案的占比。

    返回:
        {
          "mean_consistency": float,
          "n_questions_multi_size": int,  # 在 >=2 种大小中可答的题数
          "per_question": [
            {
              "qid": str,
              "answers_by_size": {size: [answer, ...]},
              "consistency": float,
              "majority_answer": str
            }
          ]
        }
    """
    # qid → {size → [predicted_answer, ...]}
    qid_answers = defaultdict(lambda: defaultdict(list))

    for parent_id, parent_data in manifest["parent_scenes"].items():
        for subset_info in parent_data["subsets"]:
            subset_id = subset_info["subset_id"]
            size = subset_info["subset_size"]

            if subset_id not in results:
                continue

            result = results[subset_id]
            per_q = result.get("per_question", [])
            if not per_q:
                scores = result.get("scores", {})
                per_q = scores.get("per_question", [])

            for q in per_q:
                if not q.get("answerable", True):
                    continue
                pred = q.get("predicted")
                if pred is None:
                    continue
                qid = q["qid"]
                qid_answers[qid][size].append(pred)

    # 只分析在 >=2 种大小中出现的题目
    per_question = []
    for qid, size_answers in sorted(qid_answers.items()):
        if len(size_answers) < 2:
            continue

        # 收集所有答案（不分 size）
        all_answers = []
        answers_by_size_str = {}
        for size in sorted(size_answers.keys()):
            answers = size_answers[size]
            all_answers.extend(answers)
            answers_by_size_str[str(size)] = answers

        # 一致性 = 最频繁答案 / 总答案数
        from collections import Counter
        counter = Counter(all_answers)
        majority = counter.most_common(1)[0]
        consistency = majority[1] / len(all_answers)

        per_question.append({
            "qid": qid,
            "answers_by_size": answers_by_size_str,
            "consistency": round(consistency, 4),
            "majority_answer": majority[0],
            "n_sizes": len(size_answers),
            "n_total_answers": len(all_answers),
        })

    mean_consistency = 0.0
    if per_question:
        mean_consistency = sum(q["consistency"] for q in per_question) / len(per_question)

    return {
        "mean_consistency": round(mean_consistency, 4),
        "n_questions_multi_size": len(per_question),
        "per_question": per_question,
    }


def print_summary(accuracy: dict, refusal: dict, consistency: dict):
    """打印汇总表到终端。"""
    print("\n" + "=" * 80)
    print("QRR Accuracy by Subset Size")
    print("=" * 80)
    print(f"{'Size':>5} {'Scenes':>7} {'Overall':>9} {'Disjoint':>10} {'SharedAnc':>10} "
          f"{'AnsRatio':>9} {'RefusalRate':>12} {'HalluRate':>10}")
    print("-" * 80)

    for size in sorted(accuracy.keys()):
        acc = accuracy[size]
        ref = refusal.get(size, {})
        print(
            f"{size:>5} "
            f"{acc['n_scenes']:>7} "
            f"{acc['overall']['acc']:>9.4f} "
            f"{acc['disjoint']['acc']:>10.4f} "
            f"{acc['shared_anchor']['acc']:>10.4f} "
            f"{ref.get('answerable_ratio', 0):>9.4f} "
            f"{ref.get('refusal_rate', 0):>12.4f} "
            f"{ref.get('hallucination_rate', 0):>10.4f}"
        )

    print("-" * 80)
    if consistency["n_questions_multi_size"] > 0:
        print(f"\nCross-size consistency: {consistency['mean_consistency']:.4f} "
              f"({consistency['n_questions_multi_size']} questions in >=2 sizes)")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Variable-Size Subset Ablation: 按子集大小分组分析"
    )
    parser.add_argument("--manifest", required=True, help="manifest.json 路径")
    parser.add_argument("--results-dir", required=True,
                        help="VLM 评测结果目录 (含 {subset_id}.json)")
    parser.add_argument("--output-dir", default="output/analysis",
                        help="分析输出目录")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载数据
    with open(args.manifest) as f:
        manifest = json.load(f)

    results = load_results(Path(args.results_dir))
    print(f"Loaded {len(results)} result files from {args.results_dir}")

    # 分析
    accuracy = analyze_accuracy_by_size(manifest, results)
    refusal = analyze_refusal_by_size(manifest, results)
    consistency = analyze_consistency(manifest, results)

    # 打印
    print_summary(accuracy, refusal, consistency)

    # 保存
    with open(output_dir / "accuracy_by_size.json", "w") as f:
        json.dump(accuracy, f, indent=2, ensure_ascii=False)

    with open(output_dir / "refusal_by_size.json", "w") as f:
        json.dump(refusal, f, indent=2, ensure_ascii=False)

    with open(output_dir / "consistency.json", "w") as f:
        json.dump(consistency, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
