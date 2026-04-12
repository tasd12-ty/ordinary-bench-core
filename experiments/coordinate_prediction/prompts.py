"""Coordinate prediction prompt templates."""

from typing import Dict, List


COORD_SYSTEM_PROMPT = """\
You are a spatial reasoning assistant. You will be shown one or more images \
of a 3D scene containing several objects on a flat surface.

Your task: estimate the 2D positions of all listed objects as they would \
appear on the ground plane viewed from directly above (bird's-eye view).

Output ONLY a JSON object mapping each object ID to an [x, y] coordinate pair. \
Use any coordinate system and scale you find natural — only the relative \
positions matter.

Example output format:
{"obj_0": [0.0, 1.5], "obj_1": [-2.0, 0.3], "obj_2": [1.0, -1.0]}
"""

_MODE_INTRO = {
    "single": (
        "The image below shows the scene photographed from an elevated side angle. "
        "Mentally project the objects onto the ground plane as seen from above."
    ),
    "multi_view": (
        "The images below show the same scene from {n_views} different side angles. "
        "Use all views to determine where each object sits on the ground plane."
    ),
    "top_view": (
        "The image below is an orthographic top-down view of the scene. "
        "Estimate the 2D ground-plane position of each object directly from this view."
    ),
}


def format_coord_user_prompt(
    objects: List[Dict],
    image_mode: str = "single",
    n_views: int = 4,
) -> str:
    intro = _MODE_INTRO.get(image_mode, _MODE_INTRO["single"])
    if "{n_views}" in intro:
        intro = intro.format(n_views=n_views)

    obj_lines = []
    for obj in objects:
        obj_lines.append(f"  - {obj['id']}: {obj['desc']}")

    return (
        f"{intro}\n\n"
        f"Objects in the scene:\n"
        + "\n".join(obj_lines)
        + "\n\n"
        "Estimate the 2D ground-plane position of each object. "
        "Output a JSON object mapping each obj_id to [x, y]."
    )
