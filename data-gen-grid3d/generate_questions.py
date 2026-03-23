#!/usr/bin/env python3
"""
Generate VLM test questions for 3D grid scenes.

Reads scene JSONs and produces question JSONs with:
- System/user prompts with coordinate system explanation
- 6 orthographic view image paths with axis labels
- Ground truth cell positions

Usage:
    python generate_questions.py --data output
    python generate_questions.py --data output --split g04
    python generate_questions.py --data output --counts
"""

import argparse
import json
import sys
from pathlib import Path


SYSTEM_PROMPT = """\
You are analyzing a 3D scene where objects are placed in a 4×4×4 grid.

Coordinate system:
- Row: A, B, C, D (A=front, D=back)
- Column: 1, 2, 3, 4 (1=left, 4=right)
- Layer: 1, 2, 3, 4 (1=bottom, 4=top)
- Position format: RowCol-Layer, e.g. "B3-4" = Row B, Column 3, Layer 4

You will receive 6 orthographic views of the scene, each labeled.
Use multiple views together to determine each object's 3D grid cell.

Respond ONLY with a JSON array."""

VIEW_SPECS = [
    {
        "view": "top",
        "label": "TOP VIEW (looking down): rows=A-D top-to-bottom, columns=1-4 left-to-right",
    },
    {
        "view": "front",
        "label": "FRONT VIEW (looking from front): columns=1-4 left-to-right, layers=1-4 bottom-to-top",
    },
    {
        "view": "right",
        "label": "RIGHT VIEW (looking from right side): rows=A-D left-to-right, layers=1-4 bottom-to-top",
    },
    {
        "view": "back",
        "label": "BACK VIEW (looking from back): columns=4-1 left-to-right, layers=1-4 bottom-to-top",
    },
    {
        "view": "left",
        "label": "LEFT VIEW (looking from left side): rows=D-A left-to-right, layers=1-4 bottom-to-top",
    },
    {
        "view": "bottom",
        "label": "BOTTOM VIEW (looking up): rows=A-D top-to-bottom, columns=4-1 left-to-right",
    },
]


def object_desc(obj: dict) -> str:
    """Build human-readable object description: 'cyan rubber cylinder'."""
    return f"{obj['color']} {obj['material']} {obj['shape']}"


def build_user_prompt(objects: list[dict]) -> str:
    """Build the user prompt with view labels and object list."""
    lines = ["Here are 6 orthographic views of a 3D grid scene:\n"]

    for i, spec in enumerate(VIEW_SPECS, 1):
        lines.append(f"[Image {i} — {spec['label']}]\n")

    lines.append("Objects in this scene:")
    for j, obj in enumerate(objects, 1):
        lines.append(f"  {j}. {object_desc(obj)}")

    lines.append("")
    lines.append(
        "For each object, determine its grid position by combining information "
        "from at least two views. Answer as JSON:"
    )
    lines.append("")
    lines.append("[")
    for j, obj in enumerate(objects):
        comma = "," if j < len(objects) - 1 else ""
        lines.append(f'  {{"object": "{object_desc(obj)}", "cell": "?"}}{comma}')
    lines.append("]")

    return "\n".join(lines)


def generate_for_scene(scene_path: Path, data_dir: Path) -> dict:
    """Generate question JSON for one scene."""
    with open(scene_path) as f:
        scene = json.load(f)

    scene_id = scene["scene_id"]
    objects = scene["objects"]

    # Build image list with paths and labels
    images = []
    for spec in VIEW_SPECS:
        img_path = f"images/{spec['view']}/{scene_id}.png"
        full_path = data_dir / img_path
        images.append({
            "view": spec["view"],
            "label": spec["label"],
            "path": img_path,
            "exists": full_path.exists(),
        })

    # Build object list and ground truth
    obj_list = []
    ground_truth = []
    for obj in objects:
        desc = object_desc(obj)
        obj_list.append({
            "id": obj["id"],
            "desc": desc,
        })
        ground_truth.append({
            "object": desc,
            "cell": obj["cell_label"],
            "row": obj["cell_row"],
            "col": obj["cell_col"],
            "layer": obj["cell_layer"],
        })

    return {
        "scene_id": scene_id,
        "split": scene.get("split", ""),
        "n_objects": len(objects),
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt": build_user_prompt(objects),
        "images": images,
        "objects": obj_list,
        "ground_truth": ground_truth,
    }


def find_scenes(data_dir: Path, split: str = None) -> list[Path]:
    """Find scene JSON files, optionally filtered by split."""
    scenes_dir = data_dir / "scenes"
    if not scenes_dir.exists():
        print(f"Error: scenes directory not found: {scenes_dir}")
        sys.exit(1)

    paths = sorted(scenes_dir.glob("*.json"))
    if split:
        paths = [p for p in paths if p.stem.startswith(split)]
    return paths


def main():
    parser = argparse.ArgumentParser(description="Generate 3D grid VLM questions")
    parser.add_argument("--data", "-d", default="output",
                        help="Data directory containing scenes/ and images/")
    parser.add_argument("--split", "-s", default=None,
                        help="Filter by split prefix (e.g. g04)")
    parser.add_argument("--counts", action="store_true",
                        help="Just print question counts and exit")
    args = parser.parse_args()

    data_dir = Path(args.data)
    scene_paths = find_scenes(data_dir, args.split)

    if not scene_paths:
        print("No scenes found.")
        sys.exit(1)

    if args.counts:
        total_questions = 0
        for sp in scene_paths:
            with open(sp) as f:
                scene = json.load(f)
            n = len(scene.get("objects", []))
            total_questions += n
            print(f"  {sp.stem}: {n} objects")
        print(f"\nTotal: {len(scene_paths)} scenes, {total_questions} object-position questions")
        return

    # Generate questions
    out_dir = data_dir / "questions"
    out_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for sp in scene_paths:
        question = generate_for_scene(sp, data_dir)
        out_path = out_dir / f"{question['scene_id']}.json"
        with open(out_path, "w") as f:
            json.dump(question, f, indent=2, ensure_ascii=False)
        total += 1
        print(f"  Generated: {out_path.name} ({question['n_objects']} objects)")

    print(f"\nDone: {total} question files → {out_dir}")


if __name__ == "__main__":
    main()
