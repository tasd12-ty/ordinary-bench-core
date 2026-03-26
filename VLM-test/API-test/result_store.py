"""Result persistence for the unified evaluation runner."""

from __future__ import annotations

import json
from pathlib import Path


class ResultStore:
    def __init__(self, job):
        self.job = job
        self.run_dir = Path(job.output.results_dir) / job.run_name
        self.raw_dir = self.run_dir / "raw"
        self.scenes_dir = self.run_dir / "scenes"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.scenes_dir.mkdir(parents=True, exist_ok=True)

    def save_raw(self, *, scene_id: str, batch_label: str, record: dict) -> None:
        filename = f"{scene_id}_{batch_label}.json"
        with open(self.raw_dir / filename, "w") as handle:
            json.dump(record, handle, indent=2, ensure_ascii=False)

    def save_scene_result(self, result: dict) -> None:
        with open(self.scenes_dir / f"{result['scene_id']}.json", "w") as handle:
            json.dump(result, handle, indent=2, ensure_ascii=False)

    def save_summary(self, summary: dict) -> None:
        with open(self.run_dir / "summary.json", "w") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)
