#!/usr/bin/env python3
"""
评估 VLM 对 3D 网格位置问题的预测结果。

用法：
    # 单场景
    python evaluate.py --question output/questions/g04_000000.json \
                       --prediction pred.json

    # 批量（目录）
    python evaluate.py --questions-dir output/questions \
                       --predictions-dir results/ \
                       --output report.json
"""

import argparse
import json
import sys
from pathlib import Path

import scoring


def evaluate_single(question_path: Path, prediction_path: Path) -> dict:
    """评估单个场景。"""
    with open(question_path) as f:
        question = json.load(f)
    with open(prediction_path) as f:
        predictions = json.load(f)

    gt = question["ground_truth"]
    result = scoring.score_scene(predictions, gt)
    result["scene_id"] = question["scene_id"]
    return result


def print_scene_result(result: dict) -> None:
    """将单个场景的评估结果打印到终端。"""
    sid = result.get("scene_id", "?")
    n = result["n_objects"]
    ex = result["exact"]
    st = result["structural"]
    pd = result["per_dimension"]

    print(f"\n{'=' * 50}")
    print(f"Scene: {sid}  ({n} objects)")
    print(f"{'=' * 50}")
    print(f"  Exact accuracy:      {ex['correct']}/{n} = {ex['accuracy']:.1%}")
    print(f"  Structural accuracy: {st['correct']}/{n} = {st['accuracy']:.1%}")
    print(f"  Best transform:      {st['best_transform']}"
          f"{'  ✓ aligned' if st['is_aligned'] else '  (not aligned)'}")
    print(f"  Per-dimension:       row={pd['row']:.1%}  col={pd['col']:.1%}  layer={pd['layer']:.1%}")

    print(f"\n  Per-object:")
    for obj in result["per_object"]:
        pred = obj["predicted"] or "(parse error)"
        gt = obj["gt"] or "?"
        exact_mark = "✓" if obj["exact_match"] else "✗"
        struct_mark = "✓" if obj["structural_match"] else "✗"
        print(f"    {obj['object']:30s}  pred={pred:6s}  gt={gt:6s}  "
              f"exact={exact_mark}  structural={struct_mark}")

    print(f"\n  Transform scores: {st['all_transform_scores']}")


def print_aggregate(agg: dict) -> None:
    """将汇总报告打印到终端。"""
    s = agg["summary"]
    print(f"\n{'=' * 50}")
    print(f"AGGREGATE REPORT")
    print(f"{'=' * 50}")
    print(f"  Scenes:              {s['n_scenes']}")
    print(f"  Total objects:       {s['n_objects']}")
    print(f"  Exact accuracy:      {s['exact_accuracy']:.1%}")
    print(f"  Structural accuracy: {s['structural_accuracy']:.1%}")
    print(f"  Alignment rate:      {s['alignment_rate']:.1%}")
    print(f"  Row accuracy:        {s['row_accuracy']:.1%}")
    print(f"  Col accuracy:        {s['col_accuracy']:.1%}")
    print(f"  Layer accuracy:      {s['layer_accuracy']:.1%}")
    print(f"\n  Transform distribution: {agg['transform_distribution']}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate 3D grid VLM predictions")
    parser.add_argument("--question", "-q", default=None,
                        help="Single question JSON file")
    parser.add_argument("--prediction", "-p", default=None,
                        help="Single prediction JSON file")
    parser.add_argument("--questions-dir", default=None,
                        help="Directory of question JSONs (batch mode)")
    parser.add_argument("--predictions-dir", default=None,
                        help="Directory of prediction JSONs (batch mode)")
    parser.add_argument("--output", "-o", default=None,
                        help="Output report JSON path")
    args = parser.parse_args()

    # 单场景模式
    if args.question and args.prediction:
        result = evaluate_single(Path(args.question), Path(args.prediction))
        print_scene_result(result)

        if args.output:
            with open(args.output, "w") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"\nReport saved: {args.output}")
        return

    # 批量模式
    if args.questions_dir and args.predictions_dir:
        q_dir = Path(args.questions_dir)
        p_dir = Path(args.predictions_dir)

        scene_scores = []
        for q_path in sorted(q_dir.glob("*.json")):
            p_path = p_dir / q_path.name
            if not p_path.exists():
                print(f"  Skip {q_path.name}: no prediction file")
                continue
            result = evaluate_single(q_path, p_path)
            print_scene_result(result)
            scene_scores.append(result)

        if scene_scores:
            agg = scoring.aggregate(scene_scores)
            print_aggregate(agg)

            if args.output:
                report = {**agg, "scenes": scene_scores}
                with open(args.output, "w") as f:
                    json.dump(report, f, indent=2, ensure_ascii=False)
                print(f"\nReport saved: {args.output}")
        return

    parser.print_help()
    print("\nError: provide --question + --prediction, or --questions-dir + --predictions-dir")
    sys.exit(1)


if __name__ == "__main__":
    main()
