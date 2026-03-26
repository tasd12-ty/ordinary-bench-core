"""
SVG vector renderer for scene belief reconstruction comparison.

Renders GT vs reconstructed scene layouts as publication-quality SVG.
- Real object shapes: sphere→circle, cube→rect, cylinder→rounded-rect
- Real colors from scene JSON
- GT / Recon / Overlay panels with displacement arrows
- Optional embedded Blender rendered image
- No matplotlib dependency — pure Python SVG generation
"""

import base64
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from reconstruct.utils import procrustes_align


# ── Color Mapping ──

# Blender-style colors (name → hex fill, darker stroke)
COLOR_MAP = {
    "red":    ("#e74c3c", "#c0392b"),
    "blue":   ("#3498db", "#2980b9"),
    "green":  ("#2ecc71", "#27ae60"),
    "yellow": ("#f1c40f", "#d4ac0f"),
    "brown":  ("#a0785a", "#8b6545"),
    "purple": ("#9b59b6", "#8e44ad"),
    "cyan":   ("#1abc9c", "#16a085"),
    "gray":   ("#95a5a6", "#7f8c8d"),
    "orange": ("#e67e22", "#d35400"),
    "white":  ("#ecf0f1", "#bdc3c7"),
    "black":  ("#2c3e50", "#1a252f"),
    "pink":   ("#e91e8c", "#c0187a"),
}

DEFAULT_COLOR = ("#95a5a6", "#7f8c8d")

# Material → opacity
MATERIAL_OPACITY = {
    "metal": 1.0,
    "rubber": 0.85,
}

# Size → SVG shape radius
SIZE_RADIUS = {
    "large": 18,
    "small": 12,
}
DEFAULT_RADIUS = 15


# ── SVG Primitives ──

def _svg_header(width: int, height: int, bg: str = "#ffffff") -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" '
        f'style="font-family: \'Helvetica Neue\', Arial, sans-serif;">\n'
        f'<rect width="{width}" height="{height}" fill="{bg}"/>\n'
    )


def _svg_footer() -> str:
    return '</svg>\n'


def _svg_shape(
    cx: float, cy: float,
    shape: str = "sphere",
    color: str = "gray",
    material: str = "rubber",
    size: str = "large",
    obj_id: str = "",
    ghost: bool = False,
) -> str:
    """Render a single object as SVG shape at (cx, cy)."""
    fill, stroke = COLOR_MAP.get(color, DEFAULT_COLOR)
    r = SIZE_RADIUS.get(size, DEFAULT_RADIUS)
    opacity = MATERIAL_OPACITY.get(material, 0.9)
    if ghost:
        opacity *= 0.35
    stroke_w = 2.0 if not ghost else 1.0
    stroke_dash = '' if not ghost else ' stroke-dasharray="4,3"'

    # Metal → add a subtle gradient shimmer
    gradient_id = f"grad_{obj_id}" if material == "metal" and not ghost else ""
    gradient_def = ""
    if gradient_id:
        gradient_def = (
            f'<defs><radialGradient id="{gradient_id}" cx="35%" cy="35%">'
            f'<stop offset="0%" stop-color="white" stop-opacity="0.4"/>'
            f'<stop offset="100%" stop-color="{fill}" stop-opacity="0"/>'
            f'</radialGradient></defs>\n'
        )
        fill_attr = f'fill="{fill}" '
    else:
        fill_attr = f'fill="{fill}" '

    common = (
        f'{fill_attr}stroke="{stroke}" stroke-width="{stroke_w}"{stroke_dash} '
        f'opacity="{opacity}"'
    )

    parts = [gradient_def]

    if shape in ("sphere",):
        parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" {common}/>')
        if gradient_id:
            parts.append(
                f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" '
                f'fill="url(#{gradient_id})" opacity="{opacity}"/>'
            )
    elif shape in ("cube",):
        half = r * 0.85
        parts.append(
            f'<rect x="{cx - half:.1f}" y="{cy - half:.1f}" '
            f'width="{2 * half:.1f}" height="{2 * half:.1f}" '
            f'rx="3" ry="3" {common}/>'
        )
    elif shape in ("cylinder",):
        w = r * 0.75
        h = r * 1.1
        parts.append(
            f'<rect x="{cx - w:.1f}" y="{cy - h:.1f}" '
            f'width="{2 * w:.1f}" height="{2 * h:.1f}" '
            f'rx="{w:.1f}" ry="{w * 0.35:.1f}" {common}/>'
        )
    else:
        # Fallback: circle
        parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" {common}/>')

    # Label
    label_y = cy - r - 5
    font_size = 10 if not ghost else 9
    parts.append(
        f'<text x="{cx:.1f}" y="{label_y:.1f}" text-anchor="middle" '
        f'font-size="{font_size}" fill="#333" opacity="{min(1.0, opacity + 0.3)}">'
        f'{obj_id}</text>'
    )

    return '\n'.join(parts)


def _svg_arrow(
    x1: float, y1: float, x2: float, y2: float,
    color: str = "#e74c3c", width: float = 1.5, opacity: float = 0.7,
) -> str:
    """SVG arrow from (x1,y1) to (x2,y2)."""
    dx, dy = x2 - x1, y2 - y1
    length = math.sqrt(dx * dx + dy * dy)
    if length < 2:
        return ''

    # Arrowhead
    head_len = min(8, length * 0.3)
    angle = math.atan2(dy, dx)
    ha1 = angle + math.radians(150)
    ha2 = angle - math.radians(150)

    hx1 = x2 + head_len * math.cos(ha1)
    hy1 = y2 + head_len * math.sin(ha1)
    hx2 = x2 + head_len * math.cos(ha2)
    hy2 = y2 + head_len * math.sin(ha2)

    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{color}" stroke-width="{width}" opacity="{opacity}"/>\n'
        f'<polygon points="{x2:.1f},{y2:.1f} {hx1:.1f},{hy1:.1f} {hx2:.1f},{hy2:.1f}" '
        f'fill="{color}" opacity="{opacity}"/>'
    )


def _svg_title(x: float, y: float, text: str, font_size: int = 14) -> str:
    return (
        f'<text x="{x:.0f}" y="{y:.0f}" text-anchor="middle" '
        f'font-size="{font_size}" font-weight="bold" fill="#2c3e50">'
        f'{text}</text>'
    )


def _svg_subtitle(x: float, y: float, text: str, font_size: int = 10) -> str:
    return (
        f'<text x="{x:.0f}" y="{y:.0f}" text-anchor="middle" '
        f'font-size="{font_size}" fill="#7f8c8d">'
        f'{text}</text>'
    )


def _svg_grid(
    ox: float, oy: float, w: float, h: float,
    n_lines: int = 5, color: str = "#ecf0f1",
) -> str:
    """Light grid background."""
    lines = []
    for i in range(n_lines + 1):
        x = ox + i * w / n_lines
        y = oy + i * h / n_lines
        lines.append(
            f'<line x1="{x:.1f}" y1="{oy:.1f}" x2="{x:.1f}" y2="{oy + h:.1f}" '
            f'stroke="{color}" stroke-width="0.5"/>'
        )
        lines.append(
            f'<line x1="{ox:.1f}" y1="{y:.1f}" x2="{ox + w:.1f}" y2="{y:.1f}" '
            f'stroke="{color}" stroke-width="0.5"/>'
        )
    # Border
    lines.append(
        f'<rect x="{ox:.1f}" y="{oy:.1f}" width="{w:.1f}" height="{h:.1f}" '
        f'fill="none" stroke="#ddd" stroke-width="1"/>'
    )
    return '\n'.join(lines)


def _svg_legend(x: float, y: float) -> str:
    """Small legend for overlay panel."""
    return (
        f'<circle cx="{x}" cy="{y}" r="5" fill="#95a5a6" stroke="#333" stroke-width="1"/>'
        f'<text x="{x + 10}" y="{y + 4}" font-size="9" fill="#555">GT</text>'
        f'<circle cx="{x}" cy="{y + 16}" r="5" fill="#95a5a6" stroke="#333" '
        f'stroke-width="1" stroke-dasharray="3,2" opacity="0.5"/>'
        f'<text x="{x + 10}" y="{y + 20}" font-size="9" fill="#555">Recon</text>'
        f'<line x1="{x - 6}" y1="{y + 32}" x2="{x + 6}" y2="{y + 32}" '
        f'stroke="#e74c3c" stroke-width="1.5"/>'
        f'<text x="{x + 10}" y="{y + 36}" font-size="9" fill="#555">Distortion</text>'
    )


# ── Coordinate Transform ──

def _world_to_svg(
    positions: Dict[str, np.ndarray],
    panel_ox: float, panel_oy: float,
    panel_w: float, panel_h: float,
    margin: float = 35,
) -> Dict[str, Tuple[float, float]]:
    """Map world coordinates to SVG pixel coordinates within a panel."""
    if not positions:
        return {}

    coords = np.array([positions[k][:2] for k in sorted(positions.keys())])
    xmin, ymin = coords.min(axis=0)
    xmax, ymax = coords.max(axis=0)

    # Add padding
    dx = xmax - xmin if xmax - xmin > 1e-6 else 1.0
    dy = ymax - ymin if ymax - ymin > 1e-6 else 1.0

    usable_w = panel_w - 2 * margin
    usable_h = panel_h - 2 * margin
    scale = min(usable_w / dx, usable_h / dy)

    # Center
    cx_world = (xmin + xmax) / 2
    cy_world = (ymin + ymax) / 2
    cx_svg = panel_ox + panel_w / 2
    cy_svg = panel_oy + panel_h / 2

    result = {}
    for k in sorted(positions.keys()):
        wx, wy = positions[k][0], positions[k][1]
        sx = cx_svg + (wx - cx_world) * scale
        sy = cy_svg - (wy - cy_world) * scale  # flip Y
        result[k] = (sx, sy)

    return result


# ── Public API ──

@dataclass
class ObjectInfo:
    """Visual attributes of a scene object."""
    obj_id: str
    shape: str = "sphere"
    color: str = "gray"
    material: str = "rubber"
    size: str = "large"


def load_object_info(scene_json_path: str) -> Dict[str, ObjectInfo]:
    """Load object visual attributes from scene JSON."""
    with open(scene_json_path) as f:
        scene = json.load(f)

    info = {}
    for obj in scene["objects"]:
        info[obj["id"]] = ObjectInfo(
            obj_id=obj["id"],
            shape=obj.get("shape", "sphere"),
            color=obj.get("color", "gray"),
            material=obj.get("material", "rubber"),
            size=obj.get("size", "large"),
        )
    return info


def render_scene_comparison_svg(
    gt_positions: Dict[str, np.ndarray],
    recon_positions: Dict[str, np.ndarray],
    object_info: Optional[Dict[str, ObjectInfo]] = None,
    scene_id: str = "",
    metrics: Optional[dict] = None,
    blender_image_path: Optional[str] = None,
    panel_size: int = 300,
    show_overlay: bool = True,
) -> str:
    """Render GT vs Recon comparison as SVG string.

    Args:
        gt_positions: {obj_id: [x, y, ...]} ground truth
        recon_positions: {obj_id: [x, y, ...]} reconstructed
        object_info: {obj_id: ObjectInfo} visual attributes
        scene_id: scene identifier
        metrics: optional {csr_qrr, csr_trr, nrms, kendall_tau, K_geom}
        blender_image_path: optional path to Blender rendered PNG
        panel_size: pixel size of each panel
        show_overlay: whether to include overlay panel

    Returns:
        SVG string
    """
    obj_ids = sorted(set(gt_positions.keys()) & set(recon_positions.keys()))
    if len(obj_ids) < 2:
        return '<svg xmlns="http://www.w3.org/2000/svg" width="100" height="50">' \
               '<text x="10" y="30" font-size="12">Too few objects</text></svg>'

    # Procrustes-align recon to GT
    gt_mat = np.array([gt_positions[oid][:2] for oid in obj_ids])
    recon_mat = np.array([recon_positions[oid][:2] for oid in obj_ids])
    recon_aligned, rms = procrustes_align(recon_mat, gt_mat, allow_reflection=True)

    gt_dict = {oid: gt_mat[i] for i, oid in enumerate(obj_ids)}
    recon_dict = {oid: recon_aligned[i] for i, oid in enumerate(obj_ids)}

    # Panel layout
    has_blender = bool(blender_image_path and os.path.exists(blender_image_path))
    n_panels = 2 + int(show_overlay) + int(has_blender)

    header_h = 65
    pw = panel_size
    ph = panel_size
    gap = 15
    total_w = n_panels * pw + (n_panels - 1) * gap + 40
    total_h = ph + header_h + 30

    parts = [_svg_header(total_w, total_h, "#fafafa")]

    # Title
    parts.append(_svg_title(total_w / 2, 22, f"Scene: {scene_id}"))

    # Metrics subtitle
    if metrics:
        metric_parts = []
        if isinstance(metrics.get("csr_qrr"), (int, float)):
            metric_parts.append(f"CSR_QRR={metrics['csr_qrr']:.3f}")
        if isinstance(metrics.get("csr_trr"), (int, float)):
            metric_parts.append(f"CSR_TRR={metrics['csr_trr']:.3f}")
        if isinstance(metrics.get("nrms"), (int, float)):
            metric_parts.append(f"NRMS={metrics['nrms']:.4f}")
        if isinstance(metrics.get("kendall_tau"), (int, float)):
            metric_parts.append(f"\u03C4={metrics['kendall_tau']:.3f}")
        if metrics.get("K_geom") is not None:
            metric_parts.append(f"K={metrics['K_geom']}")
        parts.append(_svg_subtitle(total_w / 2, 42, "  |  ".join(metric_parts)))

    panel_y = header_h
    panel_idx = 0

    def panel_x(idx):
        return 20 + idx * (pw + gap)

    # Helper: default object info
    def get_info(oid):
        if object_info and oid in object_info:
            return object_info[oid]
        return ObjectInfo(obj_id=oid)

    # ── Panel: Blender Image ──
    if has_blender:
        px = panel_x(panel_idx)
        parts.append(_svg_grid(px, panel_y, pw, ph))
        parts.append(_svg_title(px + pw / 2, panel_y - 5, "Blender Render", 12))

        with open(blender_image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()
        # Fit image into panel with padding
        img_margin = 5
        parts.append(
            f'<image x="{px + img_margin}" y="{panel_y + img_margin}" '
            f'width="{pw - 2 * img_margin}" height="{ph - 2 * img_margin}" '
            f'xlink:href="data:image/png;base64,{img_b64}" '
            f'preserveAspectRatio="xMidYMid meet"/>'
        )
        panel_idx += 1

    # ── Panel: Ground Truth ──
    px = panel_x(panel_idx)
    parts.append(_svg_grid(px, panel_y, pw, ph))
    parts.append(_svg_title(px + pw / 2, panel_y - 5, "Ground Truth", 12))

    gt_svg = _world_to_svg(gt_dict, px, panel_y, pw, ph)
    for oid in obj_ids:
        info = get_info(oid)
        sx, sy = gt_svg[oid]
        parts.append(_svg_shape(
            sx, sy, shape=info.shape, color=info.color,
            material=info.material, size=info.size, obj_id=oid,
        ))
    panel_idx += 1

    # ── Panel: Reconstructed ──
    px = panel_x(panel_idx)
    parts.append(_svg_grid(px, panel_y, pw, ph))
    parts.append(_svg_title(px + pw / 2, panel_y - 5, "Reconstructed", 12))

    recon_svg = _world_to_svg(recon_dict, px, panel_y, pw, ph)
    for oid in obj_ids:
        info = get_info(oid)
        sx, sy = recon_svg[oid]
        parts.append(_svg_shape(
            sx, sy, shape=info.shape, color=info.color,
            material=info.material, size=info.size, obj_id=oid,
        ))
    panel_idx += 1

    # ── Panel: Overlay ──
    if show_overlay:
        px = panel_x(panel_idx)
        parts.append(_svg_grid(px, panel_y, pw, ph))

        extent = np.max(np.ptp(gt_mat, axis=0))
        nrms_val = rms / extent if extent > 1e-6 else rms
        parts.append(_svg_title(
            px + pw / 2, panel_y - 5,
            f"Overlay (NRMS={nrms_val:.3f})", 12,
        ))

        # Use GT coordinate space for both
        overlay_gt = _world_to_svg(gt_dict, px, panel_y, pw, ph)
        overlay_recon = _world_to_svg(recon_dict, px, panel_y, pw, ph)

        # Draw arrows first (behind shapes)
        for oid in obj_ids:
            gx, gy = overlay_gt[oid]
            rx, ry = overlay_recon[oid]
            info = get_info(oid)
            arrow_color, _ = COLOR_MAP.get(info.color, DEFAULT_COLOR)
            parts.append(_svg_arrow(gx, gy, rx, ry, color=arrow_color, width=1.8))

        # GT shapes (solid)
        for oid in obj_ids:
            info = get_info(oid)
            sx, sy = overlay_gt[oid]
            parts.append(_svg_shape(
                sx, sy, shape=info.shape, color=info.color,
                material=info.material, size=info.size, obj_id=oid,
            ))

        # Recon shapes (ghost)
        for oid in obj_ids:
            info = get_info(oid)
            sx, sy = overlay_recon[oid]
            parts.append(_svg_shape(
                sx, sy, shape=info.shape, color=info.color,
                material=info.material, size=info.size,
                obj_id="", ghost=True,
            ))

        # Legend
        parts.append(_svg_legend(px + pw - 70, panel_y + 10))

    parts.append(_svg_footer())
    return '\n'.join(parts)


def render_three_condition_svg(
    gt_positions: Dict[str, np.ndarray],
    recon_a: Optional[Dict[str, np.ndarray]],
    recon_b: Optional[Dict[str, np.ndarray]],
    recon_c: Optional[Dict[str, np.ndarray]],
    object_info: Optional[Dict[str, ObjectInfo]] = None,
    scene_id: str = "",
    blender_image_path: Optional[str] = None,
    panel_size: int = 260,
) -> str:
    """Render three-condition comparison (A/B/C) as SVG.

    GT | Blender | Cond A | Cond B | Cond C
    """
    obj_ids = sorted(gt_positions.keys())
    gt_mat = np.array([gt_positions[oid][:2] for oid in obj_ids])
    gt_dict = {oid: gt_mat[i] for i, oid in enumerate(obj_ids)}

    has_blender = blender_image_path and os.path.exists(blender_image_path)
    conditions = [
        ("A: Correct Image", recon_a),
        ("B: Wrong Image", recon_b),
        ("C: No Image", recon_c),
    ]
    active_conditions = [(name, pos) for name, pos in conditions if pos is not None]

    n_panels = 1 + int(has_blender) + len(active_conditions)
    pw, ph = panel_size, panel_size
    gap = 12
    header_h = 50
    total_w = n_panels * pw + (n_panels - 1) * gap + 40
    total_h = ph + header_h + 25

    parts = [_svg_header(total_w, total_h, "#fafafa")]
    parts.append(_svg_title(total_w / 2, 25, f"Three-Condition Comparison: {scene_id}"))

    panel_y = header_h
    panel_idx = 0

    def panel_x(idx):
        return 20 + idx * (pw + gap)

    def get_info(oid):
        if object_info and oid in object_info:
            return object_info[oid]
        return ObjectInfo(obj_id=oid)

    # GT panel
    px = panel_x(panel_idx)
    parts.append(_svg_grid(px, panel_y, pw, ph))
    parts.append(_svg_title(px + pw / 2, panel_y - 5, "Ground Truth", 11))
    gt_svg = _world_to_svg(gt_dict, px, panel_y, pw, ph)
    for oid in obj_ids:
        info = get_info(oid)
        sx, sy = gt_svg[oid]
        parts.append(_svg_shape(sx, sy, info.shape, info.color,
                                info.material, info.size, oid))
    panel_idx += 1

    # Blender panel
    if has_blender:
        px = panel_x(panel_idx)
        parts.append(_svg_grid(px, panel_y, pw, ph))
        parts.append(_svg_title(px + pw / 2, panel_y - 5, "Blender", 11))
        with open(blender_image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()
        parts.append(
            f'<image x="{px + 3}" y="{panel_y + 3}" '
            f'width="{pw - 6}" height="{ph - 6}" '
            f'xlink:href="data:image/png;base64,{img_b64}" '
            f'preserveAspectRatio="xMidYMid meet"/>'
        )
        panel_idx += 1

    # Condition panels
    for cond_name, cond_pos in active_conditions:
        px = panel_x(panel_idx)
        parts.append(_svg_grid(px, panel_y, pw, ph))

        # Align to GT
        common = sorted(set(obj_ids) & set(cond_pos.keys()))
        if len(common) >= 3:
            c_mat = np.array([cond_pos[oid][:2] for oid in common])
            g_mat = np.array([gt_positions[oid][:2] for oid in common])
            aligned, rms = procrustes_align(c_mat, g_mat, allow_reflection=True)
            extent = np.max(np.ptp(g_mat, axis=0))
            nrms = rms / extent if extent > 1e-6 else rms
            aligned_dict = {oid: aligned[i] for i, oid in enumerate(common)}

            parts.append(_svg_title(
                px + pw / 2, panel_y - 5,
                f"{cond_name} (NRMS={nrms:.3f})", 11,
            ))

            cond_svg = _world_to_svg(aligned_dict, px, panel_y, pw, ph)
            for oid in common:
                info = get_info(oid)
                sx, sy = cond_svg[oid]
                parts.append(_svg_shape(sx, sy, info.shape, info.color,
                                        info.material, info.size, oid))
        else:
            parts.append(_svg_title(px + pw / 2, panel_y - 5, cond_name, 11))
            parts.append(
                f'<text x="{px + pw / 2}" y="{panel_y + ph / 2}" '
                f'text-anchor="middle" font-size="11" fill="#999">N/A</text>'
            )
        panel_idx += 1

    parts.append(_svg_footer())
    return '\n'.join(parts)


# ── CLI Entry Point ──

def render_scene_from_files(
    scene_json_path: str,
    recon_result: dict,
    output_path: str,
    blender_image_path: Optional[str] = None,
    panel_size: int = 300,
):
    """Convenience: render from file paths and save SVG.

    Args:
        scene_json_path: path to scene JSON (for GT positions + object info)
        recon_result: reconstruction result dict (from pipeline.ReconstructResult.to_dict())
        output_path: where to save the SVG
        blender_image_path: optional Blender rendered image
        panel_size: panel size in px
    """
    # Load GT
    with open(scene_json_path) as f:
        scene = json.load(f)

    gt_positions = {}
    for obj in scene["objects"]:
        gt_positions[obj["id"]] = np.array(obj["3d_coords"][:2], dtype=float)

    # Load object info
    object_info = load_object_info(scene_json_path)

    # Reconstructed positions
    recon_positions = {}
    for oid, coords in recon_result["positions"].items():
        recon_positions[oid] = np.array(coords, dtype=float)

    metrics = recon_result.get("metrics", {})

    svg = render_scene_comparison_svg(
        gt_positions=gt_positions,
        recon_positions=recon_positions,
        object_info=object_info,
        scene_id=scene.get("scene_id", ""),
        metrics=metrics,
        blender_image_path=blender_image_path,
        panel_size=panel_size,
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        f.write(svg)

    return output_path
