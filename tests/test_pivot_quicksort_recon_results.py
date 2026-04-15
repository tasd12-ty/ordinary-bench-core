from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "VLM-test"
    / "analysis"
    / "pivot_quicksort_recon_results.py"
)
SPEC = importlib.util.spec_from_file_location(
    "pivot_quicksort_recon_results", MODULE_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class QuickSortReconPivotTests(unittest.TestCase):
    def test_build_scene_rows_and_aggregate_handle_anomaly_and_skips(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_root = root / "source"
            recon_root = root / "recon"

            source_run = source_root / "results-397-m" / "27-m"
            (source_run / "scenes").mkdir(parents=True)
            with open(source_run / "summary.json", "w") as f:
                json.dump({"model": "qwen3p5-27", "n_scenes": 2}, f)
            with open(
                source_run / "scenes" / "n20_000000__sz10_s0001.json", "w"
            ) as f:
                json.dump(
                    {
                        "scene_id": "n20_000000__sz10_s0001",
                        "model": "qwen3p5-27",
                        "n_objects": 10,
                    },
                    f,
                )
            with open(
                source_run / "scenes" / "n20_000000__sz11_s0002.json", "w"
            ) as f:
                json.dump(
                    {
                        "scene_id": "n20_000000__sz11_s0002",
                        "model": "qwen3p5-27",
                    },
                    f,
                )

            recon_run = recon_root / "results-397-m__27-m" / "27-m" / "recon"
            (recon_run / "scenes").mkdir(parents=True)
            with open(recon_run / "summary.json", "w") as f:
                json.dump(
                    {
                        "n_reconstructed": 1,
                        "n_skipped": 1,
                        "skipped": [
                            {
                                "scene_id": "n20_000000__sz11_s0002",
                                "source_scene_result": str(
                                    source_run
                                    / "scenes"
                                    / "n20_000000__sz11_s0002.json"
                                ),
                                "reason": "Source quick-sort run failed",
                            }
                        ],
                    },
                    f,
                )
            with open(recon_run / "scenes" / "n20_000000__sz10_s0001.json", "w") as f:
                json.dump(
                    {
                        "scene_id": "n20_000000__sz10_s0001",
                        "model": "qwen3p5-27",
                        "n_objects": 10,
                        "status": "multimodal",
                        "feasible": True,
                        "metrics": {
                            "csr_qrr": 0.9,
                            "csr_qrr_aligned": 0.95,
                            "kendall_tau": 0.5,
                            "nrms": 0.1,
                            "best_loss": 0.01,
                            "n_solutions": 2,
                        },
                    },
                    f,
                )

            rows = MODULE.build_scene_long_rows(recon_root, source_root)
            self.assertEqual(len(rows), 2)

            reconstructed = next(row for row in rows if not row["is_skipped"])
            self.assertEqual(reconstructed["model_family"], "qwen27")
            self.assertEqual(reconstructed["run_label"], "results-397-m/27-m")
            self.assertEqual(reconstructed["status"], "multimodal")
            self.assertTrue(reconstructed["feasible"])
            self.assertEqual(reconstructed["n_objects"], 10)

            skipped = next(row for row in rows if row["is_skipped"])
            self.assertEqual(skipped["model_family"], "qwen27")
            self.assertEqual(skipped["run_label"], "results-397-m/27-m")
            self.assertEqual(skipped["status"], "skipped")
            self.assertEqual(skipped["n_objects"], 11)
            self.assertEqual(skipped["skip_reason"], "Source quick-sort run failed")

            by_run = MODULE.aggregate_scene_rows(
                rows, ("model_family", "run_label"), include_status_counts=True
            )
            self.assertEqual(len(by_run), 1)
            run_row = by_run[0]
            self.assertEqual(run_row["source_scenes"], 2)
            self.assertEqual(run_row["reconstructed"], 1)
            self.assertEqual(run_row["skipped"], 1)
            self.assertEqual(run_row["feasible"], 1)
            self.assertEqual(run_row["multimodal"], 1)
            self.assertEqual(run_row["infeasible"], 0)
            self.assertEqual(run_row["single_mode"], 0)
            self.assertEqual(run_row["other_status"], 0)
            self.assertEqual(run_row["feasible_reconstructed"], 1.0)
            self.assertEqual(run_row["feasible_total"], 0.5)
            self.assertEqual(run_row["status_counts"], '{"multimodal": 1}')

    def test_infer_n_objects_prefers_subset_size(self) -> None:
        self.assertEqual(MODULE.infer_n_objects("n20_000000__sz13_s0080"), 13)
        self.assertEqual(MODULE.infer_n_objects("n04_000088"), 4)
        self.assertEqual(MODULE.infer_n_objects("anything", 7), 7)


if __name__ == "__main__":
    unittest.main()
