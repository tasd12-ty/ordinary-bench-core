#!/usr/bin/env python3
"""
Experimental scene storyboard renderer for reconstruction analysis.

This script is self-contained on purpose. It does not modify or depend on the
main reconstruction pipeline implementation details.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


COLOR_MAP = {
    "red": ("#e74c3c", "#b03a2e"),
    "blue": ("#3498db", "#21618c"),
    "green": ("#2ecc71", "#1e8449"),
    "yellow": ("#f1c40f", "#b7950b"),
    "brown": ("#a0785a", "#7e5a3a"),
    "purple": ("#8e44ad", "#6c3483"),
    "cyan": ("#1abc9c", "#117a65"),
    "gray": ("#95a5a6", "#566573"),
    "orange": ("#e67e22", "#af601a"),
    "white": ("#ecf0f1", "#aeb6bf"),
    "black": ("#2c3e50", "#17202a"),
}
DEFAULT_COLOR = ("#95a5a6", "#566573")

SIZE_RADIUS = {"small": 13, "large": 19}
MATERIAL_OPACITY = {"metal": 1.0, "rubber": 0.86}


@dataclass
class ObjectInfo:
    obj_id: str
    shape: str
    color: str
    material: str
    size: str
    gt_xy: np.ndarray
    pixel_xy: Tuple[float, float]


def _svg_header(width: int, height: int) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'style="font-family: \'IBM Plex Sans\', \'Avenir Next\', sans-serif;">\n'
        f'<rect width="{width}" height="{height}" fill="#f7f4ed"/>\n'
    )


def _svg_footer() -> str:
    return "</svg>\n"


def _svg_rect(x: float, y: float, w: float, h: float, fill: str, stroke: str = "#d6d0c4", rx: int = 18) -> str:
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
        f'rx="{rx}" ry="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="1.2"/>'
    )


def _svg_text(x: float, y: float, text: str, size: int = 14, fill: str = "#1f2933",
              weight: str = "400", anchor: str = "start") -> str:
    text = (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
        f'font-size="{size}" font-weight="{weight}" fill="{fill}">{text}</text>'
    )


def _wrap_text(text: str, width: int) -> List[str]:
    words = text.split()
    if not words:
        return [""]
    lines: List[str] = []
    line = words[0]
    for word in words[1:]:
        candidate = f"{line} {word}"
        if len(candidate) <= width:
            line = candidate
        else:
            lines.append(line)
            line = word
    lines.append(line)
    return lines


def _svg_multiline(x: float, y: float, text: str, size: int = 12, fill: str = "#52606d",
                   weight: str = "400", line_gap: int = 15, width: int = 44) -> str:
    lines = _wrap_text(text, width)
    parts = []
    for i, line in enumerate(lines):
        parts.append(_svg_text(x, y + i * line_gap, line, size=size, fill=fill, weight=weight))
    return "\n".join(parts)


def _svg_badge(x: float, y: float, label: str, value: str, fill: str = "#ffffff") -> str:
    width = max(88, 10 * max(len(label), len(value)) + 18)
    return "\n".join([
        _svg_rect(x, y, width, 46, fill=fill, stroke="#d9d2c6", rx=14),
        _svg_text(x + 12, y + 18, label, size=10, fill="#7b8794", weight="600"),
        _svg_text(x + 12, y + 35, value, size=16, fill="#102a43", weight="700"),
    ])


def _svg_grid(x: float, y: float, w: float, h: float, n: int = 5) -> str:
    parts = []
    for i in range(1, n):
        gx = x + i * w / n
        gy = y + i * h / n
        parts.append(f'<line x1="{gx:.1f}" y1="{y:.1f}" x2="{gx:.1f}" y2="{y + h:.1f}" stroke="#ebe6db" stroke-width="1"/>')
        parts.append(f'<line x1="{x:.1f}" y1="{gy:.1f}" x2="{x + w:.1f}" y2="{gy:.1f}" stroke="#ebe6db" stroke-width="1"/>')
    parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" fill="none" stroke="#d9d2c6" stroke-width="1.2"/>')
    return "\n".join(parts)


def _shape_svg(cx: float, cy: float, info: ObjectInfo, ghost: bool = False) -> str:
    fill, stroke = COLOR_MAP.get(info.color, DEFAULT_COLOR)
    opacity = MATERIAL_OPACITY.get(info.material, 0.9)
    if ghost:
        opacity *= 0.38
    r = SIZE_RADIUS.get(info.size, 15)
    stroke_width = 2 if not ghost else 1.2
    dash = "" if not ghost else ' stroke-dasharray="5,3"'
    common = f'fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}" opacity="{opacity}"{dash}'
    shape = info.shape
    if shape == "sphere":
        base = f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" {common}/>'
    elif shape == "cube":
        half = r * 0.85
        base = f'<rect x="{cx-half:.1f}" y="{cy-half:.1f}" width="{2*half:.1f}" height="{2*half:.1f}" rx="4" ry="4" {common}/>'
    else:
        w = r * 0.78
        h = r * 1.1
        base = f'<rect x="{cx-w:.1f}" y="{cy-h:.1f}" width="{2*w:.1f}" height="{2*h:.1f}" rx="{w:.1f}" ry="{w*0.35:.1f}" {common}/>'
    label = _svg_text(cx, cy - r - 7, info.obj_id, size=11 if not ghost else 10, fill="#243b53", weight="600", anchor="middle")
    return f"{base}\n{label}"


def _svg_arrow(x1: float, y1: float, x2: float, y2: float, color: str = "#d64545") -> str:
    dx = x2 - x1
    dy = y2 - y1
    dist = math.hypot(dx, dy)
    if dist < 3:
        return ""
    head = min(8, dist * 0.25)
    ang = math.atan2(dy, dx)
    a1 = ang + math.radians(150)
    a2 = ang - math.radians(150)
    hx1, hy1 = x2 + head * math.cos(a1), y2 + head * math.sin(a1)
    hx2, hy2 = x2 + head * math.cos(a2), y2 + head * math.sin(a2)
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="1.9" opacity="0.8"/>'
        f'<polygon points="{x2:.1f},{y2:.1f} {hx1:.1f},{hy1:.1f} {hx2:.1f},{hy2:.1f}" fill="{color}" opacity="0.8"/>'
    )


def _world_to_panel(points: Dict[str, np.ndarray], x: float, y: float, w: float, h: float, pad: float = 30.0) -> Dict[str, Tuple[float, float]]:
    ids = list(points.keys())
    arr = np.array([points[k][:2] for k in ids], dtype=float)
    mins = arr.min(axis=0)
    maxs = arr.max(axis=0)
    span = np.maximum(maxs - mins, 1e-6)
    scale = min((w - 2 * pad) / span[0], (h - 2 * pad) / span[1])
    center = (mins + maxs) / 2.0
    out: Dict[str, Tuple[float, float]] = {}
    for oid, p in zip(ids, arr):
        px = x + w / 2 + (p[0] - center[0]) * scale
        py = y + h / 2 - (p[1] - center[1]) * scale
        out[oid] = (float(px), float(py))
    return out


def procrustes_align(source: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, float]:
    """Similarity align source to target."""
    src = np.asarray(source, dtype=float)
    tgt = np.asarray(target, dtype=float)
    src_mean = src.mean(axis=0)
    tgt_mean = tgt.mean(axis=0)
    src0 = src - src_mean
    tgt0 = tgt - tgt_mean
    src_norm = np.linalg.norm(src0)
    tgt_norm = np.linalg.norm(tgt0)
    if src_norm < 1e-9 or tgt_norm < 1e-9:
        aligned = src - src_mean + tgt_mean
        rms = float(np.sqrt(np.mean(np.sum((aligned - tgt) ** 2, axis=1))))
        return aligned, rms
    src0 /= src_norm
    tgt0 /= tgt_norm
    u, _s, vt = np.linalg.svd(src0.T @ tgt0)
    r = u @ vt
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1
        r = u @ vt
    scale = tgt_norm / src_norm
    aligned = (source - src_mean) @ r * scale + tgt_mean
    rms = float(np.sqrt(np.mean(np.sum((aligned - target) ** 2, axis=1))))
    return aligned, rms


def load_scene(scene_json_path: str) -> Tuple[str, Dict[str, ObjectInfo]]:
    with open(scene_json_path) as f:
        scene = json.load(f)
    object_map: Dict[str, ObjectInfo] = {}
    for obj in scene["objects"]:
        object_map[obj["id"]] = ObjectInfo(
            obj_id=obj["id"],
            shape=obj.get("shape", "sphere"),
            color=obj.get("color", "gray"),
            material=obj.get("material", "rubber"),
            size=obj.get("size", "large"),
            gt_xy=np.array(obj.get("3d_coords", [0, 0])[:2], dtype=float),
            pixel_xy=(float(obj.get("pixel_coords", [0, 0])[0]), float(obj.get("pixel_coords", [0, 0])[1])),
        )
    return scene.get("scene_id", Path(scene_json_path).stem), object_map


def load_questions_auto(questions_dir: str, scene_id: str) -> List[dict]:
    base = Path(questions_dir)
    split_paths = [base / qtype / f"{scene_id}.json" for qtype in ("qrr", "trr", "fdr")]
    found = [p for p in split_paths if p.exists()]
    if found:
        questions: List[dict] = []
        for path in found:
            with open(path) as f:
                doc = json.load(f)
            for batch in doc.get("batches", []):
                questions.extend(batch.get("questions", []))
        return questions
    flat_path = base / f"{scene_id}.json"
    if flat_path.exists():
        with open(flat_path) as f:
            doc = json.load(f)
        questions = []
        for batch in doc.get("batches", []):
            questions.extend(batch.get("questions", []))
        return questions
    return []


def load_recon_entry(recon_json_path: str, scene_id: str) -> dict:
    with open(recon_json_path) as f:
        data = json.load(f)
    if isinstance(data, list):
        for row in data:
            if row.get("scene_id") == scene_id:
                return row
        raise KeyError(f"scene_id {scene_id} not found in {recon_json_path}")
    if data.get("scene_id") == scene_id or "positions" in data:
        return data
    raise ValueError(f"Unsupported recon json format: {recon_json_path}")


def load_scene_result(scene_result_json: str) -> dict:
    with open(scene_result_json) as f:
        return json.load(f)


def _qid_lookup(items: Iterable[dict]) -> Dict[str, dict]:
    return {row["qid"]: row for row in items if "qid" in row}


def object_error_tally(questions: List[dict], per_question: List[dict]) -> Dict[str, dict]:
    q_lookup = _qid_lookup(questions)
    stats: Dict[str, dict] = {}

    def touch(obj_id: str, qtype: str, weight: float = 1.0) -> None:
        row = stats.setdefault(obj_id, {"total": 0.0, "qrr": 0.0, "trr": 0.0, "fdr": 0.0})
        row["total"] += weight
        row[qtype] += weight

    for score_row in per_question:
        qid = score_row.get("qid")
        question = q_lookup.get(qid)
        if question is None:
            continue
        qtype = question["type"]
        is_error = False
        if qtype == "qrr":
            is_error = not bool(score_row.get("correct", False))
            objs = list(question.get("pair1", [])) + list(question.get("pair2", []))
        elif qtype == "trr":
            is_error = not bool(score_row.get("hour_correct", False))
            objs = [question["target"], question["ref1"], question["ref2"]]
        else:
            is_error = not bool(score_row.get("exact_correct", False))
            objs = [question["anchor"], *question.get("gt_ranking", [])]
        if not is_error:
            continue
        for oid in objs:
            touch(oid, qtype)
    return stats


def top_relation_mismatches(questions: List[dict], per_question: List[dict], limit: int = 6) -> List[str]:
    q_lookup = _qid_lookup(questions)
    rows: List[str] = []
    for score_row in per_question:
        qid = score_row.get("qid")
        question = q_lookup.get(qid)
        if question is None:
            continue
        qtype = question["type"]
        if qtype == "qrr" and not score_row.get("correct", False):
            rows.append(
                f"QRR {question['pair1']} ? {question['pair2']} :: pred {score_row.get('predicted')} vs gt {question['gt_comparator']}"
            )
        elif qtype == "trr" and not score_row.get("hour_correct", False):
            rows.append(
                f"TRR {question['target']} wrt {question['ref1']},{question['ref2']} :: pred {score_row.get('predicted')} vs gt {question['gt_hour']}"
            )
        elif qtype == "fdr" and not score_row.get("exact_correct", False):
            pred = score_row.get("predicted")
            pred_text = pred if isinstance(pred, list) else "None"
            rows.append(
                f"FDR {question['anchor']} :: pred {pred_text} vs gt {question['gt_ranking']}"
            )
        if len(rows) >= limit:
            break
    return rows


def _panel_title_block(parts: List[str], x: float, y: float, w: float, title: str, subtitle: Optional[str] = None) -> None:
    parts.append(_svg_text(x + 18, y + 26, title, size=18, weight="700"))
    if subtitle:
        parts.append(_svg_text(x + 18, y + 46, subtitle, size=11, fill="#7b8794", weight="500"))
    parts.append(f'<line x1="{x+16:.1f}" y1="{y+58:.1f}" x2="{x+w-16:.1f}" y2="{y+58:.1f}" stroke="#e6e1d8" stroke-width="1"/>')


def _render_image_panel(parts: List[str], panel_x: float, panel_y: float, panel_w: float, panel_h: float,
                        image_path: Optional[str], object_map: Dict[str, ObjectInfo]) -> None:
    _panel_title_block(parts, panel_x, panel_y, panel_w, "What The VLM Sees", "Original RGB input with object anchors")
    img_x, img_y = panel_x + 16, panel_y + 72
    img_w, img_h = panel_w - 32, panel_h - 92
    parts.append(_svg_rect(img_x, img_y, img_w, img_h, fill="#fbfaf7", stroke="#d9d2c6", rx=12))
    if image_path and os.path.exists(image_path):
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        parts.append(
            f'<image x="{img_x+4:.1f}" y="{img_y+4:.1f}" width="{img_w-8:.1f}" height="{img_h-8:.1f}" '
            f'xlink:href="data:image/png;base64,{b64}" preserveAspectRatio="xMidYMid meet"/>'
        )
        # Current benchmark images are 480x320.
        scale_x = (img_w - 8) / 480.0
        scale_y = (img_h - 8) / 320.0
        for info in object_map.values():
            px = img_x + 4 + info.pixel_xy[0] * scale_x
            py = img_y + 4 + info.pixel_xy[1] * scale_y
            fill, stroke = COLOR_MAP.get(info.color, DEFAULT_COLOR)
            parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="10" fill="#fffdf8" stroke="{stroke}" stroke-width="2.2" opacity="0.95"/>')
            parts.append(_svg_text(px, py + 4, info.obj_id.replace("obj_", ""), size=11, fill=stroke, weight="700", anchor="middle"))
    else:
        parts.append(_svg_text(img_x + img_w / 2, img_y + img_h / 2, "RGB missing", size=16, fill="#9aa5b1", anchor="middle"))


def _render_map_panel(parts: List[str], panel_x: float, panel_y: float, panel_w: float, panel_h: float,
                      title: str, subtitle: str, points: Dict[str, np.ndarray], object_map: Dict[str, ObjectInfo],
                      ghost: bool = False, arrows_from: Optional[Dict[str, Tuple[float, float]]] = None,
                      arrows_to: Optional[Dict[str, Tuple[float, float]]] = None) -> Dict[str, Tuple[float, float]]:
    _panel_title_block(parts, panel_x, panel_y, panel_w, title, subtitle)
    map_x, map_y = panel_x + 18, panel_y + 78
    map_w, map_h = panel_w - 36, panel_h - 102
    parts.append(_svg_grid(map_x, map_y, map_w, map_h, n=5))
    panel_pts = _world_to_panel(points, map_x, map_y, map_w, map_h)
    if arrows_from is not None and arrows_to is not None:
        for oid in sorted(set(arrows_from) & set(arrows_to)):
            ax, ay = arrows_from[oid]
            bx, by = arrows_to[oid]
            parts.append(_svg_arrow(ax, ay, bx, by))
    for oid, (px, py) in panel_pts.items():
        parts.append(_shape_svg(px, py, object_map[oid], ghost=ghost))
    return panel_pts


def render_storyboard(scene_id: str, object_map: Dict[str, ObjectInfo], image_path: Optional[str], model_name: str,
                      recon_entry: dict, scene_result: dict, questions: List[dict]) -> str:
    gt_points = {oid: info.gt_xy for oid, info in object_map.items()}
    recon_positions = {
        oid: np.array(coords[:2], dtype=float)
        for oid, coords in recon_entry.get("positions", {}).items()
        if oid in object_map
    }
    common_ids = sorted(set(gt_points) & set(recon_positions))
    if len(common_ids) < 2:
        raise ValueError(f"Need at least 2 common objects, got {len(common_ids)}")

    gt_mat = np.array([gt_points[oid] for oid in common_ids], dtype=float)
    recon_mat = np.array([recon_positions[oid] for oid in common_ids], dtype=float)
    recon_aligned, rms = procrustes_align(recon_mat, gt_mat)
    recon_aligned_map = {oid: recon_aligned[i] for i, oid in enumerate(common_ids)}

    extent = float(np.max(np.ptp(gt_mat, axis=0)))
    nrms_overlay = rms / extent if extent > 1e-6 else rms

    error_stats = object_error_tally(questions, scene_result["scores"].get("per_question", []))
    top_objects = sorted(error_stats.items(), key=lambda kv: (-kv[1]["total"], kv[0]))[:4]
    top_mismatches = top_relation_mismatches(questions, scene_result["scores"].get("per_question", []), limit=6)

    disp_rows = []
    for oid in common_ids:
        err = float(np.linalg.norm(recon_aligned_map[oid] - gt_points[oid]))
        disp_rows.append((err, oid))
    disp_rows.sort(reverse=True)

    width = 1760
    height = 940
    margin = 28
    col_gap = 18
    top_y = 130
    panel_h = 520
    panel_w = 330
    sidebar_w = width - 2 * margin - 4 * panel_w - 3 * col_gap
    xs = [
        margin,
        margin + panel_w + col_gap,
        margin + 2 * (panel_w + col_gap),
        margin + 3 * (panel_w + col_gap),
        margin + 4 * panel_w + 3 * col_gap,
    ]

    parts = [_svg_header(width, height)]
    parts.append(_svg_text(margin, 46, "Scene Belief Storyboard", size=28, weight="800"))
    parts.append(_svg_text(margin, 72, f"{scene_id}  |  {model_name}", size=15, fill="#52606d", weight="600"))
    parts.append(_svg_text(margin, 95, "Comparing image input, geometric truth, and the world implied by VLM answers", size=13, fill="#7b8794"))

    metrics = recon_entry.get("metrics", {})
    badge_y = 28
    badge_x = width - margin - 5 * 104
    parts.append(_svg_badge(badge_x + 0, badge_y, "Status", str(recon_entry.get("status", "N/A")).upper()))
    parts.append(_svg_badge(badge_x + 112, badge_y, "CSR QRR", f"{metrics.get('csr_qrr', 0):.3f}"))
    parts.append(_svg_badge(badge_x + 224, badge_y, "CSR TRR", f"{metrics.get('csr_trr', 0):.3f}"))
    parts.append(_svg_badge(badge_x + 336, badge_y, "NRMS", f"{metrics.get('nrms', 0):.3f}"))
    parts.append(_svg_badge(badge_x + 448, badge_y, "K geom", f"{metrics.get('K_geom', 'NA')}"))

    # Main panels.
    card_fill = "#fffdf8"
    for i in range(4):
        parts.append(_svg_rect(xs[i], top_y, panel_w, panel_h, fill=card_fill))
    parts.append(_svg_rect(xs[4], top_y, sidebar_w, height - top_y - margin, fill=card_fill))

    _render_image_panel(parts, xs[0], top_y, panel_w, panel_h, image_path, object_map)
    gt_panel = _render_map_panel(parts, xs[1], top_y, panel_w, panel_h, "Ground Truth", "Top-down scene geometry from metadata", gt_points, object_map)
    belief_panel = _render_map_panel(parts, xs[2], top_y, panel_w, panel_h, "VLM Belief World", "Reconstruction aligned to GT for comparison", recon_aligned_map, object_map)
    _render_map_panel(parts, xs[3], top_y, panel_w, panel_h, "Overlay / Distortion", f"Arrow field after Procrustes alignment (NRMS={nrms_overlay:.3f})", gt_points, object_map, arrows_from=gt_panel, arrows_to=belief_panel)
    # Add ghost belief layer on overlay.
    overlay_area = _world_to_panel(recon_aligned_map, xs[3] + 18, top_y + 78, panel_w - 36, panel_h - 102)
    for oid, (px, py) in overlay_area.items():
        parts.append(_shape_svg(px, py, object_map[oid], ghost=True))

    # Sidebar.
    side_x = xs[4] + 18
    side_y = top_y + 26
    parts.append(_svg_text(side_x, side_y, "Audit Sidebar", size=20, weight="800"))
    parts.append(_svg_text(side_x, side_y + 24, "Where the model's externalized world departs from the scene", size=11, fill="#7b8794"))

    block_y = side_y + 62
    parts.append(_svg_text(side_x, block_y, "Reading Guide", size=14, weight="700"))
    parts.append(_svg_multiline(side_x, block_y + 18, "The belief map is aligned to GT because reconstruction is only identifiable up to translation, rotation, and scale.", size=12))

    block_y += 74
    parts.append(_svg_text(side_x, block_y, "Largest Object Displacements", size=14, weight="700"))
    y = block_y + 22
    for err, oid in disp_rows[:4]:
        parts.append(_svg_text(side_x, y, f"{oid}", size=12, weight="700"))
        parts.append(_svg_text(side_x + 58, y, f"{err:.3f} world units", size=12, fill="#52606d"))
        y += 18

    block_y = y + 16
    parts.append(_svg_text(side_x, block_y, "Objects Most Often In Wrong Answers", size=14, weight="700"))
    y = block_y + 22
    if top_objects:
        for oid, row in top_objects:
            parts.append(_svg_text(side_x, y, f"{oid}", size=12, weight="700"))
            parts.append(_svg_text(side_x + 58, y, f"total {row['total']:.0f}  qrr {row['qrr']:.0f}  trr {row['trr']:.0f}  fdr {row['fdr']:.0f}", size=12, fill="#52606d"))
            y += 18
    else:
        parts.append(_svg_text(side_x, y, "No scored mistakes recorded for this scene.", size=12, fill="#52606d"))
        y += 18

    block_y = y + 18
    parts.append(_svg_text(side_x, block_y, "Example Relation Mismatches", size=14, weight="700"))
    y = block_y + 22
    if top_mismatches:
        for row in top_mismatches:
            parts.append(_svg_multiline(side_x, y, row, size=11, width=50))
            y += 34
    else:
        parts.append(_svg_text(side_x, y, "No explicit mismatch rows available.", size=12, fill="#52606d"))
        y += 18

    block_y = max(y + 14, top_y + panel_h + 22)
    parts.append(_svg_text(margin, block_y, "Interpretation", size=18, weight="800"))
    interpretation = (
        f"This scene is marked as {recon_entry.get('status', 'unknown')}. "
        f"The reconstruction satisfies QRR/TRR constraints at "
        f"{metrics.get('csr_qrr', 0):.3f}/{metrics.get('csr_trr', 0):.3f}, "
        f"but its aligned geometry still differs from GT with NRMS {metrics.get('nrms', 0):.3f}. "
        f"K_geom={metrics.get('K_geom', 'NA')} and spread={metrics.get('spread', 0):.3f} "
        f"indicate how ambiguous the model-implied world remains."
    )
    parts.append(_svg_multiline(margin, block_y + 22, interpretation, size=13, width=160, line_gap=18))

    parts.append(_svg_footer())
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render experimental reconstruction storyboard SVG")
    parser.add_argument("--scene-json", required=True)
    parser.add_argument("--scene-result-json", required=True)
    parser.add_argument("--recon-json", required=True, help="Aggregated or per-scene reconstruction json")
    parser.add_argument("--questions-dir", required=True)
    parser.add_argument("--scene-id", default=None)
    parser.add_argument("--image-path", default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    scene_id, object_map = load_scene(args.scene_json)
    if args.scene_id:
        scene_id = args.scene_id
    scene_result = load_scene_result(args.scene_result_json)
    recon_entry = load_recon_entry(args.recon_json, scene_id)
    questions = load_questions_auto(args.questions_dir, scene_id)
    if not questions:
        raise SystemExit(f"No questions found for {scene_id} under {args.questions_dir}")
    image_path = args.image_path or str(Path(args.scene_json).resolve().parents[1] / "images" / "single_view" / f"{scene_id}.png")
    model_name = scene_result.get("model", recon_entry.get("model", "unknown-model"))
    svg = render_storyboard(scene_id, object_map, image_path, model_name, recon_entry, scene_result, questions)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(svg)
    print(f"Saved storyboard to {output_path}")


if __name__ == "__main__":
    main()
