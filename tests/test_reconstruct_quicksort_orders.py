from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys
import unittest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "VLM-test"
    / "analysis"
    / "reconstruct_quicksort_orders.py"
)
SPEC = importlib.util.spec_from_file_location(
    "reconstruct_quicksort_orders", MODULE_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class QuickSortReconstructionTests(unittest.TestCase):
    def test_strict_total_order_expands_to_all_qrr(self) -> None:
        ranking = [
            ["obj_1", "obj_2"],
            ["obj_1", "obj_3"],
            ["obj_1", "obj_4"],
            ["obj_2", "obj_3"],
            ["obj_2", "obj_4"],
            ["obj_3", "obj_4"],
        ]
        normalized = MODULE.normalize_total_order(ranking, None)
        constraints = MODULE.build_exhaustive_qrr_constraints(normalized)
        self.assertEqual(len(constraints), math.comb(len(ranking), 2))
        self.assertTrue(all(row["comparator"] == "<" for row in constraints))

    def test_tie_group_emits_approx_and_lt_constraints(self) -> None:
        ranking = [
            ["obj_1", "obj_2"],
            ["obj_1", "obj_3"],
            ["obj_2", "obj_3"],
        ]
        tie_groups = [
            [["obj_1", "obj_2"], ["obj_1", "obj_3"]],
            [["obj_2", "obj_3"]],
        ]
        normalized = MODULE.normalize_total_order(ranking, tie_groups)
        constraints = MODULE.build_exhaustive_qrr_constraints(normalized)
        comparators = [row["comparator"] for row in constraints]
        self.assertEqual(comparators.count("~="), 1)
        self.assertEqual(comparators.count("<"), 2)
        self.assertEqual(len(constraints), 3)

    def test_non_contiguous_tie_group_is_rejected(self) -> None:
        ranking = [
            ["obj_1", "obj_2"],
            ["obj_1", "obj_3"],
            ["obj_2", "obj_3"],
        ]
        tie_groups = [
            [["obj_1", "obj_2"], ["obj_2", "obj_3"]],
            [["obj_1", "obj_3"]],
        ]
        with self.assertRaises(MODULE.QuickSortSceneError):
            MODULE.normalize_total_order(ranking, tie_groups)

    def test_build_prepared_scene_input_generates_complete_counts(self) -> None:
        scene_doc = {
            "scene_id": "scene_a",
            "model": "demo",
            "n_objects": 4,
            "vlm_result": {
                "ranking": [
                    ["obj_1", "obj_2"],
                    ["obj_1", "obj_3"],
                    ["obj_1", "obj_4"],
                    ["obj_2", "obj_3"],
                    ["obj_2", "obj_4"],
                    ["obj_3", "obj_4"],
                ],
                "tie_groups": [],
                "failed": False,
            },
        }
        prepared = MODULE.build_prepared_scene_input(
            scene_doc, Path("scene_a.json"), None
        )
        self.assertEqual(prepared.summary["n_objects"], 4)
        self.assertEqual(prepared.summary["n_qrr_total"], 15)
        self.assertEqual(prepared.summary["n_qrr_direct_shared_anchor"], 12)
        self.assertEqual(prepared.summary["n_qrr_direct_disjoint"], 3)
        self.assertEqual(len(prepared.object_ids), 4)

    def test_solver_config_override_helper_path(self) -> None:
        cfg = MODULE.SolverConfig(n_restarts=7, bt_ratio_alpha=0.0)
        self.assertEqual(cfg.n_restarts, 7)
        self.assertEqual(cfg.bt_ratio_alpha, 0.0)


if __name__ == "__main__":
    unittest.main()
