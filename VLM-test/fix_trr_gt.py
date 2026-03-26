"""修正 TRR 真值标注：将逆时针角度转换为顺时针。

原始 compute_angle_2d() 使用 atan2 的逆时针角度，
但钟面方向为顺时针。本脚本修正所有 TRR 问题文件中的
gt_angle_deg、gt_hour 和 gt_quadrant 字段。

用法：
    python fix_trr_gt.py [--questions-dir DIR] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from dsl.predicates import angle_to_hour, hour_to_quadrant


def fix_question(q: dict) -> dict:
    """修正单个 TRR 问题的真值标注。"""
    old_angle = q["gt_angle_deg"]
    new_angle = (360 - old_angle) % 360
    new_hour = angle_to_hour(new_angle)
    new_quadrant = hour_to_quadrant(new_hour)

    q["gt_angle_deg"] = round(new_angle, 6)
    q["gt_hour"] = new_hour
    q["gt_quadrant"] = new_quadrant
    return q


def main():
    parser = argparse.ArgumentParser(description="Fix TRR ground truth (CCW → CW)")
    parser.add_argument(
        "--questions-dir",
        default=str(REPO_ROOT / "output" / "questions" / "trr"),
    )
    parser.add_argument("--dry-run", action="store_true", help="Print changes without writing")
    args = parser.parse_args()

    trr_dir = Path(args.questions_dir)
    if not trr_dir.is_dir():
        print(f"Error: {trr_dir} is not a directory")
        sys.exit(1)

    files = sorted(trr_dir.glob("*.json"))
    print(f"Found {len(files)} TRR question files in {trr_dir}")

    total_fixed = 0
    sample_changes = []

    for fpath in files:
        with open(fpath) as f:
            doc = json.load(f)

        changed = 0
        for batch in doc.get("batches", []):
            for q in batch.get("questions", []):
                old_hour = q["gt_hour"]
                fix_question(q)
                new_hour = q["gt_hour"]
                if old_hour != new_hour:
                    changed += 1
                    if len(sample_changes) < 10:
                        sample_changes.append(
                            f"  {fpath.stem} {q['qid']}: hour {old_hour} → {new_hour}"
                        )

        total_fixed += changed

        if not args.dry_run:
            with open(fpath, "w") as f:
                json.dump(doc, f, indent=2, ensure_ascii=False)

    print(f"\nFixed {total_fixed} questions across {len(files)} files")
    if sample_changes:
        print("\nSample changes:")
        for line in sample_changes:
            print(line)

    if args.dry_run:
        print("\n(dry-run mode — no files were written)")


if __name__ == "__main__":
    main()
