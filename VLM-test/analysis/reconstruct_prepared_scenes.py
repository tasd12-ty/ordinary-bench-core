"""
Run reconstruction from previously prepared per-scene bundles.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from reconstruct import PreparedSceneInput, reconstruct_from_prepared
from analysis.reconstruct_scenes import summarize_reconstructions


def _discover_prepared_files(prepared_dir: Path) -> list[Path]:
    scene_dir = prepared_dir / "scenes"
    if scene_dir.exists():
        return sorted(scene_dir.glob("*.json"))
    return sorted(prepared_dir.glob("*.json"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconstruct scenes from prepared bundles")
    parser.add_argument("--prepared-dir", "-p", required=True,
                        help="Directory produced by prepare_reconstruction_inputs.py")
    parser.add_argument("--output", "-o", default=None,
                        help="Optional output path for reconstruction results JSON")
    parser.add_argument("--restarts", type=int, default=10)
    parser.add_argument("--max-scenes", type=int, default=None)
    args = parser.parse_args()

    prepared_dir = Path(args.prepared_dir)
    files = _discover_prepared_files(prepared_dir)
    if args.max_scenes is not None:
        files = files[:args.max_scenes]

    if not files:
        raise SystemExit(f"No prepared scene files found in {prepared_dir}")

    outputs = []
    for i, path in enumerate(files):
        with open(path) as f:
            prepared = PreparedSceneInput.from_dict(json.load(f))
        result = reconstruct_from_prepared(prepared, n_restarts=args.restarts)
        out = result.to_dict()
        out["scene_id"] = prepared.scene_id
        out["model"] = prepared.model
        out["use_correct_only"] = prepared.use_correct_only
        out["prepared_summary"] = prepared.summary
        out["prepared_integrity"] = prepared.integrity
        outputs.append(out)
        print(
            f"  [{i+1}/{len(files)}] {prepared.scene_id}: "
            f"status={out['status']} csr_qrr={out['metrics']['csr_qrr']:.3f} "
            f"csr_trr={out['metrics']['csr_trr']:.3f}"
        )

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(outputs, f, indent=2, default=str)
        summary = summarize_reconstructions(outputs)
        summary_path = out_path.with_name(f"{out_path.stem}_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\nSaved {len(outputs)} reconstruction results to {out_path}")
        print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
