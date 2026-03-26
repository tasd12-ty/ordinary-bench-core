"""
从模型评分输出中准备逐场景的重建输入数据。

本脚本不运行求解器，仅提取可审查的关系约束，
并将每个场景的预处理包写入文件供后续检查使用。
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.reconstruct_scenes import prepare_all_scenes


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare reconstruction inputs")
    parser.add_argument("--results-dir", "-r", required=True, help="Path to model results directory")
    parser.add_argument("--questions-dir", "-q", default="VLM-test/output/questions",
                        help="Path to questions directory (flat or split layout)")
    parser.add_argument("--scenes-dir", "-s", default="data-gen/output/scenes",
                        help="Path to scene GT directory")
    parser.add_argument("--output-dir", "-o", required=True,
                        help="Directory to store prepared per-scene JSONs")
    parser.add_argument("--belief", action="store_true",
                        help="Use all model predictions instead of correct-only constraints")
    parser.add_argument("--max-scenes", type=int, default=None)
    args = parser.parse_args()

    prepare_all_scenes(
        results_dir=args.results_dir,
        questions_dir=args.questions_dir,
        scenes_dir=args.scenes_dir,
        output_dir=args.output_dir,
        use_correct_only=not args.belief,
        max_scenes=args.max_scenes,
    )


if __name__ == "__main__":
    main()
