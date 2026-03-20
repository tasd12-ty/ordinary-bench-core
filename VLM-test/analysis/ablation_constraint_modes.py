"""
Ablation study: compare reconstruction across constraint modes.

Runs FDR-only, QRR-only, FDR+QRR, and full (FDR+QRR+TRR) reconstruction
on the same prepared bundles, then outputs a cross-mode comparison table
and FDR vs QRR conflict report.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from reconstruct import (
    PreparedSceneInput,
    reconstruct_from_prepared,
    CONSTRAINT_MODES,
)
from reconstruct.constraints import (
    QRREntry,
    detect_fdr_qrr_conflicts,
)
from analysis.reconstruct_scenes import summarize_reconstructions


ABLATION_MODES = ["fdr_only", "qrr_only", "fdr_qrr", "all"]


def _discover_prepared_files(prepared_dir: Path) -> list:
    scene_dir = prepared_dir / "scenes"
    if scene_dir.exists():
        return sorted(scene_dir.glob("*.json"))
    return sorted(prepared_dir.glob("*.json"))


def _parse_qrr_entries(constraint_dicts: list) -> list:
    return [
        QRREntry(
            pair1=tuple(c["pair1"]),
            pair2=tuple(c["pair2"]),
            comparator=c["comparator"],
            weight=c.get("weight", 1.0),
            variant=c.get("variant", "disjoint"),
            anchor=c.get("anchor"),
        )
        for c in constraint_dicts
    ]


def run_ablation(prepared_dir: Path, modes: list, n_restarts: int,
                 max_scenes: int = None) -> dict:
    files = _discover_prepared_files(prepared_dir)
    if max_scenes is not None:
        files = files[:max_scenes]

    if not files:
        raise SystemExit(f"No prepared scene files found in {prepared_dir}")

    # Load all prepared inputs once
    prepared_inputs = []
    for path in files:
        with open(path) as f:
            prepared_inputs.append(PreparedSceneInput.from_dict(json.load(f)))

    # Run conflict detection
    conflict_report = {"per_scene": [], "aggregate": {}}
    total_overlapping = 0
    total_contradictory = 0
    total_weak = 0
    total_consistent = 0

    for prepared in prepared_inputs:
        qrr_direct = _parse_qrr_entries(prepared.qrr_constraints)
        qrr_from_fdr = _parse_qrr_entries(prepared.qrr_from_fdr)
        report = detect_fdr_qrr_conflicts(qrr_direct, qrr_from_fdr)
        report["scene_id"] = prepared.scene_id
        conflict_report["per_scene"].append(report)
        total_overlapping += report["n_overlapping"]
        total_contradictory += report["n_contradictory"]
        total_weak += report["n_weak_conflict"]
        total_consistent += report["n_consistent"]

    conflict_report["aggregate"] = {
        "n_scenes": len(prepared_inputs),
        "total_overlapping": total_overlapping,
        "total_consistent": total_consistent,
        "total_contradictory": total_contradictory,
        "total_weak_conflict": total_weak,
        "consistency_rate": total_consistent / total_overlapping if total_overlapping else 1.0,
    }

    # Run reconstruction for each mode
    results_by_mode = {}
    for mode in modes:
        print(f"\n=== Mode: {mode} ===")
        mode_outputs = []
        for i, prepared in enumerate(prepared_inputs):
            result = reconstruct_from_prepared(
                prepared, n_restarts=n_restarts, constraint_mode=mode)
            out = result.to_dict()
            out["scene_id"] = prepared.scene_id
            out["model"] = prepared.model
            out["use_correct_only"] = prepared.use_correct_only
            mode_outputs.append(out)
            print(
                f"  [{i+1}/{len(prepared_inputs)}] {prepared.scene_id}: "
                f"status={out['status']} csr_qrr={out['metrics']['csr_qrr']:.3f} "
                f"kendall={out['metrics'].get('kendall_tau', 'N/A')}"
            )
        results_by_mode[mode] = {
            "results": mode_outputs,
            "summary": summarize_reconstructions(mode_outputs),
        }

    # Build comparison table
    comparison = {}
    metric_keys = ["csr_qrr_mean", "csr_trr_mean", "kendall_tau_mean",
                   "nrms_mean", "spread_mean", "feasible_rate"]
    for mode in modes:
        s = results_by_mode[mode]["summary"]
        comparison[mode] = {k: s.get(k) for k in metric_keys}
        comparison[mode]["n_scenes"] = s.get("n_scenes", 0)
        comparison[mode]["status_counts"] = s.get("status_counts", {})

    return {
        "modes": modes,
        "comparison": comparison,
        "conflict_report": conflict_report,
        "per_mode_details": {
            mode: results_by_mode[mode]["summary"]
            for mode in modes
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ablation: compare reconstruction across constraint modes")
    parser.add_argument("--prepared-dir", "-p", required=True,
                        help="Directory with prepared scene bundles")
    parser.add_argument("--output", "-o", required=True,
                        help="Output path for ablation results JSON")
    parser.add_argument("--restarts", type=int, default=10)
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--modes", nargs="+", default=ABLATION_MODES,
                        choices=CONSTRAINT_MODES,
                        help=f"Modes to compare (default: {ABLATION_MODES})")
    args = parser.parse_args()

    result = run_ablation(
        prepared_dir=Path(args.prepared_dir),
        modes=args.modes,
        n_restarts=args.restarts,
        max_scenes=args.max_scenes,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    # Print comparison table
    print("\n" + "=" * 70)
    print("CROSS-MODE COMPARISON")
    print("=" * 70)
    header = f"{'Mode':<12} {'CSR_QRR':>8} {'CSR_TRR':>8} {'Kendall':>8} {'NRMS':>8} {'Feasible':>8}"
    print(header)
    print("-" * len(header))
    for mode in args.modes:
        c = result["comparison"][mode]
        print(f"{mode:<12} "
              f"{c.get('csr_qrr_mean', 0):.4f}   "
              f"{c.get('csr_trr_mean', 0):.4f}   "
              f"{c.get('kendall_tau_mean', 0) or 0:.4f}   "
              f"{c.get('nrms_mean', 0) or 0:.4f}   "
              f"{c.get('feasible_rate', 0):.4f}")

    # Print conflict summary
    agg = result["conflict_report"]["aggregate"]
    print(f"\nFDR vs QRR CONFLICT REPORT")
    print(f"  Overlapping constraints: {agg['total_overlapping']}")
    print(f"  Consistent:              {agg['total_consistent']}")
    print(f"  Contradictory (< vs >):  {agg['total_contradictory']}")
    print(f"  Weak conflicts:          {agg['total_weak_conflict']}")
    print(f"  Consistency rate:        {agg['consistency_rate']:.4f}")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
