"""Parse VLM response to extract {obj_id: [x, y]} coordinate predictions."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

_API_DIR = Path(__file__).resolve().parents[2] / "VLM-test" / "API-test"
if str(_API_DIR) not in sys.path:
    sys.path.insert(0, str(_API_DIR))

from response_parser import extract_json


def _normalize_coords(value) -> Optional[List[float]]:
    """Normalize various coordinate formats to [x, y]."""
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            return [float(value[0]), float(value[1])]
        except (ValueError, TypeError):
            return None
    if isinstance(value, dict):
        # {"x": 1.0, "y": 2.0}
        x = value.get("x", value.get("X"))
        y = value.get("y", value.get("Y"))
        if x is not None and y is not None:
            try:
                return [float(x), float(y)]
            except (ValueError, TypeError):
                return None
    return None


def parse_coordinate_response(
    raw: str,
    expected_ids: List[str],
) -> Dict[str, Optional[List[float]]]:
    """Extract {obj_id: [x, y]} from VLM output.

    Returns dict with None for missing/invalid objects.
    """
    result = {oid: None for oid in expected_ids}

    parsed = extract_json(raw)
    if parsed is None:
        return result

    # Format 1: {"obj_0": [x, y], ...}  or  {"obj_0": {"x": ..., "y": ...}}
    if isinstance(parsed, dict):
        for oid in expected_ids:
            if oid in parsed:
                coords = _normalize_coords(parsed[oid])
                if coords is not None:
                    result[oid] = coords
        return result

    # Format 2: [{"id": "obj_0", "x": ..., "y": ...}, ...]
    if isinstance(parsed, list):
        for item in parsed:
            if not isinstance(item, dict):
                continue
            oid = item.get("id", item.get("obj_id", item.get("object_id")))
            if oid not in result:
                continue
            # Try [x, y] in "position" or "coords" field
            for key in ("position", "coords", "coordinates"):
                if key in item:
                    coords = _normalize_coords(item[key])
                    if coords is not None:
                        result[oid] = coords
                        break
            else:
                # Try x, y directly in item
                coords = _normalize_coords(item)
                if coords is not None:
                    result[oid] = coords

    return result
