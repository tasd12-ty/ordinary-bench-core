"""
Scoring for 3D grid position questions.

Handles D4 symmetry: if a VLM's answers are a consistent rotation/reflection
of the ground truth in the Row-Col plane, that counts as structurally correct.
"""

import re
from typing import Optional

ROW_LABELS = ['A', 'B', 'C', 'D']
ROW_TO_IDX = {ch: i for i, ch in enumerate(ROW_LABELS)}

# D4 symmetry group: 8 transformations of the Row-Col plane (index 0-3)
D4_TRANSFORMS = {
    "identity":      lambda r, c: (r, c),
    "rot90_cw":      lambda r, c: (c, 3 - r),
    "rot180":        lambda r, c: (3 - r, 3 - c),
    "rot90_ccw":     lambda r, c: (3 - c, r),
    "flip_h":        lambda r, c: (r, 3 - c),
    "flip_v":        lambda r, c: (3 - r, c),
    "flip_diag":     lambda r, c: (c, r),
    "flip_antidiag": lambda r, c: (3 - c, 3 - r),
}


def parse_cell_label(label: str) -> Optional[tuple]:
    """Parse 'B3-4' → (row=1, col=2, layer=3). Returns None on failure."""
    if not label or not isinstance(label, str):
        return None
    m = re.match(r'^([A-Da-d])(\d)-(\d)$', label.strip())
    if not m:
        return None
    row_ch, col_str, layer_str = m.group(1), m.group(2), m.group(3)
    row = ROW_TO_IDX.get(row_ch.upper())
    if row is None:
        return None
    col = int(col_str) - 1   # 1-based → 0-based
    layer = int(layer_str) - 1
    if not (0 <= col <= 3 and 0 <= layer <= 3):
        return None
    return (row, col, layer)


def format_cell_label(row: int, col: int, layer: int) -> str:
    """(1, 2, 3) → 'B3-4'"""
    return f"{ROW_LABELS[row]}{col + 1}-{layer + 1}"


def match_objects(predictions: list, ground_truth: list) -> list:
    """
    Match predictions to ground truth by object description.
    Returns list of (pred_parsed, gt_parsed, object_desc) tuples.
    Unmatched objects get pred_parsed=None.
    """
    # Build lookup from description → gt
    gt_by_desc = {}
    for gt in ground_truth:
        desc = gt["object"]
        gt_by_desc[desc] = parse_cell_label(gt["cell"])

    # Build lookup from description → pred
    pred_by_desc = {}
    for pred in predictions:
        desc = pred.get("object", "")
        cell = pred.get("cell", "")
        pred_by_desc[desc] = parse_cell_label(cell)

    pairs = []
    for desc, gt_parsed in gt_by_desc.items():
        pred_parsed = pred_by_desc.get(desc)
        pairs.append((pred_parsed, gt_parsed, desc))

    return pairs


def find_best_transform(pairs: list) -> dict:
    """
    Try all 8 D4 transforms. For each, check if applying the inverse
    transform to predictions yields the ground truth positions.

    We apply the transform to GT and check if it matches predictions.
    """
    valid_pairs = [(p, g) for p, g, _ in pairs if p is not None and g is not None]
    total = len(valid_pairs)

    if total == 0:
        return {
            "best_transform": "identity",
            "match_count": 0,
            "total": 0,
            "all_transform_scores": {name: 0 for name in D4_TRANSFORMS},
        }

    scores = {}
    for name, transform in D4_TRANSFORMS.items():
        count = 0
        for pred, gt in valid_pairs:
            pr, pc, pl = pred
            gr, gc, gl = gt
            # Transform GT row,col and compare with prediction
            tr, tc = transform(gr, gc)
            if tr == pr and tc == pc and gl == pl:
                count += 1
        scores[name] = count

    best_name = max(scores, key=scores.get)
    return {
        "best_transform": best_name,
        "match_count": scores[best_name],
        "total": total,
        "all_transform_scores": scores,
    }


def score_scene(predictions: list, ground_truth: list) -> dict:
    """
    Score one scene's predictions against ground truth.

    Returns:
        dict with exact, structural, per_dimension, and per_object scores.
    """
    pairs = match_objects(predictions, ground_truth)
    n_objects = len(pairs)

    # --- Exact match ---
    exact_correct = 0
    for pred, gt, desc in pairs:
        if pred is not None and gt is not None and pred == gt:
            exact_correct += 1

    # --- D4 symmetry-aware match ---
    transform_result = find_best_transform(pairs)
    best_transform_name = transform_result["best_transform"]
    best_transform = D4_TRANSFORMS[best_transform_name]
    structural_correct = transform_result["match_count"]

    # --- Per-dimension accuracy under best transform ---
    row_correct = 0
    col_correct = 0
    layer_correct = 0
    valid_count = 0

    per_object = []
    for pred, gt, desc in pairs:
        obj_result = {
            "object": desc,
            "predicted": format_cell_label(*pred) if pred else None,
            "gt": format_cell_label(*gt) if gt else None,
            "exact_match": False,
            "structural_match": False,
            "parse_error": pred is None,
        }

        if pred is not None and gt is not None:
            valid_count += 1
            pr, pc, pl = pred
            gr, gc, gl = gt

            # Exact
            obj_result["exact_match"] = (pred == gt)

            # Structural (under best transform)
            tr, tc = best_transform(gr, gc)
            obj_result["structural_match"] = (tr == pr and tc == pc and gl == pl)

            # Per-dimension (under best transform)
            if tr == pr:
                row_correct += 1
            if tc == pc:
                col_correct += 1
            if gl == pl:
                layer_correct += 1

        per_object.append(obj_result)

    safe_div = lambda a, b: a / b if b > 0 else 0.0

    return {
        "n_objects": n_objects,
        "n_valid": valid_count,
        "exact": {
            "accuracy": safe_div(exact_correct, n_objects),
            "correct": exact_correct,
        },
        "structural": {
            "accuracy": safe_div(structural_correct, n_objects),
            "correct": structural_correct,
            "best_transform": best_transform_name,
            "is_aligned": best_transform_name == "identity",
            "all_transform_scores": transform_result["all_transform_scores"],
        },
        "per_dimension": {
            "row": safe_div(row_correct, valid_count),
            "col": safe_div(col_correct, valid_count),
            "layer": safe_div(layer_correct, valid_count),
            "note": f"under best transform ({best_transform_name})",
        },
        "per_object": per_object,
    }


def aggregate(scene_scores: list) -> dict:
    """Aggregate scores across multiple scenes."""
    total_objects = 0
    total_exact = 0
    total_structural = 0
    total_aligned = 0
    total_row = 0
    total_col = 0
    total_layer = 0
    total_valid = 0
    transform_counts = {name: 0 for name in D4_TRANSFORMS}

    for s in scene_scores:
        n = s["n_objects"]
        total_objects += n
        total_exact += s["exact"]["correct"]
        total_structural += s["structural"]["correct"]
        total_valid += s["n_valid"]

        if s["structural"]["is_aligned"]:
            total_aligned += 1

        bt = s["structural"]["best_transform"]
        transform_counts[bt] += 1

        v = s["n_valid"]
        total_row += int(s["per_dimension"]["row"] * v)
        total_col += int(s["per_dimension"]["col"] * v)
        total_layer += int(s["per_dimension"]["layer"] * v)

    n_scenes = len(scene_scores)
    safe_div = lambda a, b: round(a / b, 4) if b > 0 else 0.0

    return {
        "summary": {
            "n_scenes": n_scenes,
            "n_objects": total_objects,
            "exact_accuracy": safe_div(total_exact, total_objects),
            "structural_accuracy": safe_div(total_structural, total_objects),
            "alignment_rate": safe_div(total_aligned, n_scenes),
            "row_accuracy": safe_div(total_row, total_valid),
            "col_accuracy": safe_div(total_col, total_valid),
            "layer_accuracy": safe_div(total_layer, total_valid),
        },
        "transform_distribution": transform_counts,
    }
