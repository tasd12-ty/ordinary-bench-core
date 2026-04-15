"""
将 quick-sort / adaptive-sort 场景结果转换为全量 QRR 约束，并运行重建。

核心思路：
1. 每个场景的 `vlm_result.ranking` 给出所有距离对的全序（或带 tie 的弱序）
2. 若共有 d = C(n, 2) 个距离对，则该弱序蕴含 C(d, 2) 个 QRR 比较关系
3. 对同一 tie group 内的距离对，生成 `~=`
4. 对更早 group vs 更晚 group，生成 `<`
5. 使用现有 reconstruct pipeline 的 `qrr_only` 模式做重建测试

本脚本故意保持独立：不修改现有 reconstruction / solver 代码。
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable, Optional, Sequence


_VLM_TEST_ROOT = Path(__file__).resolve().parent.parent
if str(_VLM_TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(_VLM_TEST_ROOT))

from reconstruct import (
    PreparedSceneInput,
    load_scene_gt_positions,
    reconstruct_from_prepared,
    SolverConfig,
)


VALID_COMPARATORS = {"<", "~=", ">"}


class QuickSortSceneError(ValueError):
    """输入 quick-sort scene 结果不满足转换要求。"""


@dataclass
class NormalizedOrder:
    ranking: list[tuple[str, str]]
    tie_groups: list[list[tuple[str, str]]]


def pair_key(pair: tuple[str, str]) -> str:
    return f"{pair[0]}_{pair[1]}"


def canonicalize_distance_pair(pair: Sequence[str]) -> tuple[str, str]:
    if not isinstance(pair, Sequence) or len(pair) != 2:
        raise QuickSortSceneError(f"Invalid distance pair: {pair!r}")
    a, b = pair
    if not isinstance(a, str) or not isinstance(b, str):
        raise QuickSortSceneError(f"Distance pair must contain strings: {pair!r}")
    if a == b:
        raise QuickSortSceneError(
            f"Distance pair must contain two distinct objects: {pair!r}"
        )
    return tuple(sorted((a, b)))


def _get_result_payload(scene_doc: dict) -> tuple[str, dict]:
    if isinstance(scene_doc.get("vlm_result"), dict):
        return "vlm_result", scene_doc["vlm_result"]
    if isinstance(scene_doc.get("quicksort_result"), dict):
        return "quicksort_result", scene_doc["quicksort_result"]
    raise QuickSortSceneError(
        "Scene document has neither 'vlm_result' nor 'quicksort_result'"
    )


def _ensure_unique_pairs(pairs: Iterable[tuple[str, str]], context: str) -> None:
    seen: set[str] = set()
    duplicates: list[str] = []
    for pair in pairs:
        key = pair_key(pair)
        if key in seen:
            duplicates.append(key)
        seen.add(key)
    if duplicates:
        raise QuickSortSceneError(
            f"Duplicate distance pairs in {context}: {duplicates[:10]}"
        )


def normalize_total_order(
    ranking_raw: object, tie_groups_raw: object
) -> NormalizedOrder:
    if not isinstance(ranking_raw, list) or not ranking_raw:
        raise QuickSortSceneError("Missing or empty ranking")

    ranking = [canonicalize_distance_pair(pair) for pair in ranking_raw]
    _ensure_unique_pairs(ranking, "ranking")
    ranking_index = {pair_key(pair): idx for idx, pair in enumerate(ranking)}

    if not tie_groups_raw:
        return NormalizedOrder(ranking=ranking, tie_groups=[[pair] for pair in ranking])

    if not isinstance(tie_groups_raw, list):
        raise QuickSortSceneError("tie_groups must be a list")

    groups_with_indices: list[tuple[int, list[int]]] = []
    flat_pairs: list[tuple[str, str]] = []
    for group_idx, group in enumerate(tie_groups_raw):
        if not isinstance(group, list) or not group:
            raise QuickSortSceneError(
                f"tie_groups[{group_idx}] must be a non-empty list"
            )
        normalized_group = [canonicalize_distance_pair(pair) for pair in group]
        _ensure_unique_pairs(normalized_group, f"tie_groups[{group_idx}]")
        indices = sorted(
            ranking_index.get(pair_key(pair), -1) for pair in normalized_group
        )
        if any(index < 0 for index in indices):
            raise QuickSortSceneError(
                f"tie_groups[{group_idx}] contains pair not present in ranking"
            )
        if indices != list(range(indices[0], indices[0] + len(indices))):
            raise QuickSortSceneError(
                f"tie_groups[{group_idx}] is not contiguous in ranking: indices={indices}"
            )
        groups_with_indices.append((indices[0], indices))
        flat_pairs.extend(normalized_group)

    _ensure_unique_pairs(flat_pairs, "tie_groups(flattened)")

    if len(flat_pairs) != len(ranking):
        raise QuickSortSceneError(
            f"tie_groups flatten size mismatch: tie_groups={len(flat_pairs)} ranking={len(ranking)}"
        )
    if {pair_key(pair) for pair in flat_pairs} != set(ranking_index):
        raise QuickSortSceneError(
            "tie_groups and ranking contain different distance-pair sets"
        )

    groups_with_indices.sort(key=lambda item: item[0])
    ordered_tie_groups = [
        [ranking[i] for i in indices] for _, indices in groups_with_indices
    ]
    flattened = [pair for group in ordered_tie_groups for pair in group]
    if flattened != ranking:
        raise QuickSortSceneError("tie_groups do not reproduce the ranking order")

    return NormalizedOrder(ranking=ranking, tie_groups=ordered_tie_groups)


def infer_object_ids(
    normalized: NormalizedOrder,
    declared_n_objects: Optional[int] = None,
    scene_gt_path: Optional[Path] = None,
) -> tuple[list[str], str, dict[str, list[float]]]:
    gt_positions = {}
    if scene_gt_path is not None and scene_gt_path.exists():
        loaded = load_scene_gt_positions(str(scene_gt_path))
        if loaded:
            object_ids = sorted(loaded)
            gt_positions = dict(loaded)
            if declared_n_objects is not None and len(object_ids) != declared_n_objects:
                raise QuickSortSceneError(
                    f"GT object count mismatch: gt={len(object_ids)} declared={declared_n_objects}"
                )
            return object_ids, "scene_gt", gt_positions

    obj_ids = sorted({obj for pair in normalized.ranking for obj in pair})
    if declared_n_objects is not None and len(obj_ids) != declared_n_objects:
        raise QuickSortSceneError(
            f"Recovered object count mismatch: recovered={len(obj_ids)} declared={declared_n_objects}"
        )
    return obj_ids, "ranking_union", gt_positions


def _classify_qrr_variant(
    pair1: tuple[str, str],
    pair2: tuple[str, str],
) -> tuple[str, Optional[str]]:
    overlap = sorted(set(pair1) & set(pair2))
    if len(overlap) == 1:
        return "shared_anchor", overlap[0]
    if len(overlap) == 0:
        return "disjoint", None
    raise QuickSortSceneError(
        f"Invalid pair comparison; expected distinct distance pairs: {pair1} vs {pair2}"
    )


def build_exhaustive_qrr_constraints(normalized: NormalizedOrder) -> list[dict]:
    constraints: list[dict] = []
    counter = 0

    for group in normalized.tie_groups:
        for left, right in combinations(group, 2):
            variant, anchor = _classify_qrr_variant(left, right)
            counter += 1
            constraints.append(
                {
                    "qid": f"qsort_qrr_{counter:06d}",
                    "source_type": "quicksort_total_order",
                    "pair1": list(left),
                    "pair2": list(right),
                    "comparator": "~=",
                    "weight": 1.0,
                    "variant": variant,
                    "anchor": anchor,
                }
            )

    for group_idx, group in enumerate(normalized.tie_groups):
        for later_group in normalized.tie_groups[group_idx + 1 :]:
            for left in group:
                for right in later_group:
                    variant, anchor = _classify_qrr_variant(left, right)
                    counter += 1
                    constraints.append(
                        {
                            "qid": f"qsort_qrr_{counter:06d}",
                            "source_type": "quicksort_total_order",
                            "pair1": list(left),
                            "pair2": list(right),
                            "comparator": "<",
                            "weight": 1.0,
                            "variant": variant,
                            "anchor": anchor,
                        }
                    )

    expected = math.comb(len(normalized.ranking), 2)
    if len(constraints) != expected:
        raise QuickSortSceneError(
            f"Exhaustive QRR count mismatch: built={len(constraints)} expected={expected}"
        )
    return constraints


def build_prepared_scene_input(
    scene_doc: dict,
    source_path: Path,
    scene_gt_path: Optional[Path] = None,
) -> PreparedSceneInput:
    payload_name, payload = _get_result_payload(scene_doc)
    if payload.get("failed"):
        raise QuickSortSceneError(
            f"Source quick-sort run failed: {payload.get('fail_reason', 'unknown')}"
        )

    normalized = normalize_total_order(
        ranking_raw=payload.get("ranking"),
        tie_groups_raw=payload.get("tie_groups"),
    )

    declared_n_objects = scene_doc.get("n_objects")
    object_ids, object_source, gt_positions = infer_object_ids(
        normalized=normalized,
        declared_n_objects=declared_n_objects,
        scene_gt_path=scene_gt_path,
    )

    expected_pairs = math.comb(len(object_ids), 2)
    if len(normalized.ranking) != expected_pairs:
        raise QuickSortSceneError(
            f"Ranking does not cover all distance pairs: actual={len(normalized.ranking)} expected={expected_pairs}"
        )

    qrr_constraints = build_exhaustive_qrr_constraints(normalized)
    n_shared = sum(1 for row in qrr_constraints if row["variant"] == "shared_anchor")
    n_disjoint = sum(1 for row in qrr_constraints if row["variant"] == "disjoint")
    n_approx = sum(1 for row in qrr_constraints if row["comparator"] == "~=")
    n_lt = sum(1 for row in qrr_constraints if row["comparator"] == "<")

    metadata = {
        "source_path": str(source_path),
        "source_result_field": payload_name,
        "source_model": scene_doc.get("model"),
        "n_objects_declared": declared_n_objects,
        "object_id_source": object_source,
        "source_total_comparisons": payload.get("total_comparisons"),
        "source_total_api_calls": payload.get("total_api_calls"),
        "source_failed": payload.get("failed", False),
        "source_fail_reason": payload.get("fail_reason"),
    }
    integrity = {
        "n_ranking_pairs": len(normalized.ranking),
        "n_tie_groups": len(normalized.tie_groups),
        "ranking_complete": True,
        "object_count_matches_declared": declared_n_objects is None
        or len(object_ids) == declared_n_objects,
        "expected_n_pairs_from_objects": expected_pairs,
        "source_payload": payload_name,
        "scene_gt_path": str(scene_gt_path) if scene_gt_path is not None else None,
    }
    summary = {
        "n_objects": len(object_ids),
        "n_distance_pairs": len(normalized.ranking),
        "n_tie_groups": len(normalized.tie_groups),
        "n_qrr_direct": len(qrr_constraints),
        "n_qrr_direct_disjoint": n_disjoint,
        "n_qrr_direct_shared_anchor": n_shared,
        "n_qrr_direct_approx": n_approx,
        "n_qrr_direct_lt": n_lt,
        "n_trr": 0,
        "n_fdr": 0,
        "n_qrr_from_fdr": 0,
        "n_qrr_total": len(qrr_constraints),
        "n_selected_questions": len(qrr_constraints),
        "n_skipped_questions": 0,
    }

    return PreparedSceneInput(
        scene_id=str(scene_doc.get("scene_id") or source_path.stem),
        model=scene_doc.get("model"),
        use_correct_only=False,
        metadata=metadata,
        object_ids=object_ids,
        gt_positions=gt_positions,
        per_question_audit=[],
        qrr_constraints=qrr_constraints,
        trr_constraints=[],
        fdr_constraints=[],
        qrr_from_fdr=[],
        qrr_all=list(qrr_constraints),
        summary=summary,
        integrity=integrity,
    )


def summarize_reconstructions(results: list[dict]) -> dict:
    if not results:
        return {"n_scenes": 0, "status_counts": {}, "feasible_rate": 0.0}

    summary: dict[str, object] = {
        "n_scenes": len(results),
        "status_counts": {},
        "feasible_rate": sum(1 for row in results if row.get("feasible"))
        / len(results),
    }

    for row in results:
        status = row.get("status", "unknown")
        summary["status_counts"][status] = summary["status_counts"].get(status, 0) + 1

    for key in [
        "csr_qrr",
        "csr_trr",
        "csr_qrr_aligned",
        "kendall_tau",
        "nrms",
        "spread",
        "best_loss",
    ]:
        values = [
            row["metrics"].get(key)
            for row in results
            if row.get("metrics", {}).get(key) is not None
        ]
        if values:
            summary[f"{key}_mean"] = statistics.fmean(values)
            summary[f"{key}_median"] = statistics.median(values)
            summary[f"{key}_min"] = min(values)
            summary[f"{key}_max"] = max(values)
            summary[f"{key}_n"] = len(values)

    return summary


def discover_run_dirs(results_root: Path) -> list[Path]:
    if (results_root / "summary.json").is_file() and (results_root / "scenes").is_dir():
        return [results_root]

    run_dirs = sorted(
        {
            summary_path.parent
            for summary_path in results_root.rglob("summary.json")
            if (summary_path.parent / "scenes").is_dir()
        }
    )
    if not run_dirs:
        raise FileNotFoundError(
            f"No run directories found under {results_root} (expected summary.json + scenes/)"
        )
    return run_dirs


def _relative_run_dir(results_root: Path, run_dir: Path) -> Path:
    try:
        rel = run_dir.relative_to(results_root)
    except ValueError:
        return Path(run_dir.name)
    return rel if str(rel) != "." else Path(run_dir.name)


def reconstruct_run(
    run_dir: Path,
    results_root: Path,
    scenes_dir: Optional[Path],
    output_root: Path,
    n_restarts: int,
    bt_ratio_alpha: float,
    max_scenes: Optional[int],
) -> dict:
    rel_run = _relative_run_dir(results_root, run_dir)
    run_output_dir = output_root / rel_run
    prepared_scene_dir = run_output_dir / "prepared" / "scenes"
    recon_scene_dir = run_output_dir / "recon" / "scenes"
    prepared_scene_dir.mkdir(parents=True, exist_ok=True)
    recon_scene_dir.mkdir(parents=True, exist_ok=True)

    scene_files = sorted((run_dir / "scenes").glob("*.json"))
    if max_scenes is not None:
        scene_files = scene_files[:max_scenes]

    prepared_rows: list[dict] = []
    recon_rows: list[dict] = []
    skipped_rows: list[dict] = []
    solver_config = SolverConfig(n_restarts=n_restarts, bt_ratio_alpha=bt_ratio_alpha)

    for scene_path in scene_files:
        with open(scene_path) as f:
            scene_doc = json.load(f)
        scene_id = scene_doc.get("scene_id", scene_path.stem)
        gt_scene_path = (
            scenes_dir / f"{scene_id}.json" if scenes_dir is not None else None
        )

        try:
            prepared = build_prepared_scene_input(
                scene_doc=scene_doc,
                source_path=scene_path,
                scene_gt_path=gt_scene_path,
            )
            prepared_dict = prepared.to_dict()
            prepared_rows.append(prepared_dict)
            with open(prepared_scene_dir / f"{scene_id}.json", "w") as f:
                json.dump(prepared_dict, f, indent=2, ensure_ascii=False)

            result = reconstruct_from_prepared(
                prepared_input=prepared,
                n_restarts=n_restarts,
                config=solver_config,
                constraint_mode="qrr_only",
            )
            recon_dict = result.to_dict()
            recon_dict.update(
                {
                    "scene_id": scene_id,
                    "model": scene_doc.get("model"),
                    "n_objects": scene_doc.get("n_objects"),
                    "prepared_summary": prepared.summary,
                    "prepared_integrity": prepared.integrity,
                    "source_scene_result": str(scene_path),
                    "constraint_source": "quicksort_total_order_to_qrr",
                }
            )
            recon_rows.append(recon_dict)
            with open(recon_scene_dir / f"{scene_id}.json", "w") as f:
                json.dump(recon_dict, f, indent=2, ensure_ascii=False)
        except Exception as exc:  # preserve per-scene progress
            skipped_rows.append(
                {
                    "scene_id": scene_id,
                    "source_scene_result": str(scene_path),
                    "reason": str(exc),
                }
            )

    prepared_summary = {
        "run_dir": str(run_dir),
        "n_scene_files": len(scene_files),
        "n_prepared": len(prepared_rows),
        "n_skipped": len(skipped_rows),
        "bt_ratio_alpha": bt_ratio_alpha,
        "scenes": [
            {
                "scene_id": row["scene_id"],
                "n_objects": row["summary"]["n_objects"],
                "n_qrr_total": row["summary"]["n_qrr_total"],
                "n_qrr_direct_shared_anchor": row["summary"][
                    "n_qrr_direct_shared_anchor"
                ],
                "n_qrr_direct_disjoint": row["summary"]["n_qrr_direct_disjoint"],
                "n_qrr_direct_approx": row["summary"].get("n_qrr_direct_approx", 0),
            }
            for row in prepared_rows
        ],
        "skipped": skipped_rows,
    }
    with open(run_output_dir / "prepared" / "summary.json", "w") as f:
        json.dump(prepared_summary, f, indent=2, ensure_ascii=False)

    recon_summary = summarize_reconstructions(recon_rows)
    recon_summary.update(
        {
            "run_dir": str(run_dir),
            "n_scene_files": len(scene_files),
            "n_reconstructed": len(recon_rows),
            "n_skipped": len(skipped_rows),
            "skipped": skipped_rows,
            "constraint_mode": "qrr_only",
            "n_restarts": n_restarts,
            "bt_ratio_alpha": bt_ratio_alpha,
        }
    )
    with open(run_output_dir / "recon" / "summary.json", "w") as f:
        json.dump(recon_summary, f, indent=2, ensure_ascii=False)

    return {
        "run_dir": str(run_dir),
        "relative_run_dir": str(rel_run),
        "output_dir": str(run_output_dir),
        "prepared_summary": prepared_summary,
        "recon_summary": recon_summary,
    }


def summarize_all_runs(run_summaries: list[dict]) -> dict:
    overall = {
        "n_runs": len(run_summaries),
        "n_scene_files": sum(
            row["prepared_summary"]["n_scene_files"] for row in run_summaries
        ),
        "n_prepared": sum(
            row["prepared_summary"]["n_prepared"] for row in run_summaries
        ),
        "n_reconstructed": sum(
            row["recon_summary"].get("n_reconstructed", 0) for row in run_summaries
        ),
        "n_skipped": sum(row["prepared_summary"]["n_skipped"] for row in run_summaries),
        "bt_ratio_alpha": run_summaries[0]["recon_summary"].get("bt_ratio_alpha")
        if run_summaries
        else None,
        "runs": [],
    }
    for row in run_summaries:
        recon_summary = row["recon_summary"]
        overall["runs"].append(
            {
                "run_dir": row["run_dir"],
                "relative_run_dir": row["relative_run_dir"],
                "output_dir": row["output_dir"],
                "n_scene_files": row["prepared_summary"]["n_scene_files"],
                "n_prepared": row["prepared_summary"]["n_prepared"],
                "n_reconstructed": recon_summary.get("n_reconstructed", 0),
                "n_skipped": row["prepared_summary"]["n_skipped"],
                "bt_ratio_alpha": recon_summary.get("bt_ratio_alpha"),
                "feasible_rate": recon_summary.get("feasible_rate"),
                "status_counts": recon_summary.get("status_counts", {}),
                "csr_qrr_mean": recon_summary.get("csr_qrr_mean"),
                "kendall_tau_mean": recon_summary.get("kendall_tau_mean"),
                "nrms_mean": recon_summary.get("nrms_mean"),
            }
        )
    return overall


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert quick-sort total orders into exhaustive QRR and run qrr_only reconstruction"
    )
    parser.add_argument(
        "--results-dir",
        required=True,
        help="Run dir or parent dir containing summary.json + scenes/",
    )
    parser.add_argument(
        "--scenes-dir",
        default=None,
        help="Optional GT scene directory; if provided, GT-backed reconstruction metrics will be available",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for prepared/reconstruction artifacts",
    )
    parser.add_argument(
        "--restarts", type=int, default=10, help="Number of solver restarts"
    )
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=None,
        help="Optional maximum number of scenes to process per run",
    )
    parser.add_argument(
        "--bt-ratio-alpha",
        type=float,
        default=1.0,
        help="Override SolverConfig.bt_ratio_alpha (set 0 to disable BT ratio loss)",
    )
    args = parser.parse_args(argv)

    results_root = Path(args.results_dir).resolve()
    output_root = Path(args.output_dir).resolve()
    scenes_dir = Path(args.scenes_dir).resolve() if args.scenes_dir else None

    run_dirs = discover_run_dirs(results_root)
    run_summaries = []
    for run_dir in run_dirs:
        print(f"[reconstruct-quicksort] processing run: {run_dir}", flush=True)
        run_summary = reconstruct_run(
            run_dir=run_dir,
            results_root=results_root,
            scenes_dir=scenes_dir,
            output_root=output_root,
            n_restarts=args.restarts,
            bt_ratio_alpha=args.bt_ratio_alpha,
            max_scenes=args.max_scenes,
        )
        run_summaries.append(run_summary)
        recon_summary = run_summary["recon_summary"]
        print(
            f"  reconstructed={recon_summary.get('n_reconstructed', 0)} "
            f"skipped={recon_summary.get('n_skipped', 0)} "
            f"feasible_rate={recon_summary.get('feasible_rate', 0.0):.1%}",
            flush=True,
        )

    overall = summarize_all_runs(run_summaries)
    output_root.mkdir(parents=True, exist_ok=True)
    with open(output_root / "summary.json", "w") as f:
        json.dump(overall, f, indent=2, ensure_ascii=False)

    print("[reconstruct-quicksort] done", flush=True)
    print(json.dumps(overall, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
