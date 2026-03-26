#!/usr/bin/env python3
"""对 Infinigen 适配器进行冒烟测试，使用模拟帧包验证基本功能。"""

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "VLM-test"))

from extraction import extract_gt  # type: ignore
from adapter import adapt_scene


def main() -> None:
    fixture_root = Path(__file__).resolve().parent / "fixtures" / "mock_frame"
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        scene = adapt_scene(
            source_root=fixture_root,
            scene_id="ifg_mock_000000",
            split="ifg",
            max_objects=4,
            query_terms=[],
            min_depth=0.2,
            min_screen_margin=0,
            image_dst=tmp / "single_view" / "ifg_mock_000000.png",
            multi_view_dir=tmp / "multi_view" / "ifg_mock_000000",
        )
    gt = extract_gt(scene, tau=0.10)
    summary = {
        "scene_id": scene["scene_id"],
        "n_objects": scene["n_objects"],
        "object_ids": [o["id"] for o in scene["objects"]],
        "n_views": len(scene.get("views", [])),
        "view_ids": [v["view_id"] for v in scene.get("views", [])],
        "view_object_counts": [len(v.get("objects", [])) for v in scene.get("views", [])],
        "n_qrr": len(gt["qrr"]),
        "n_trr": len(gt["trr"]),
        "n_fdr": len(gt["fdr"]),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
