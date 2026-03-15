"""
Download 3D models from Objaverse for demo scenes.

Targets: 2 humans, 1 table, 1 chair, 1 plant.
Downloads .glb files and copies them to models/ with descriptive names.

Usage:
    pip install objaverse
    python download_assets.py
"""

import os
import shutil
import json

import objaverse


# Target models: category -> (filename, search keywords)
# We'll search annotations to find suitable UIDs, then download.
TARGETS = {
    "human_1": {
        "keywords": ["human", "person", "man", "character"],
        "description": "Standing human figure",
    },
    "human_2": {
        "keywords": ["woman", "female", "person", "character"],
        "description": "Second human figure",
    },
    "table": {
        "keywords": ["table", "desk"],
        "description": "Table / desk",
    },
    "chair": {
        "keywords": ["chair", "seat"],
        "description": "Chair",
    },
    "plant": {
        "keywords": ["plant", "potted plant", "flower pot", "houseplant"],
        "description": "Potted plant",
    },
}

# Hand-picked UIDs from Objaverse that are known to be reasonable quality
# and appropriately sized .glb models. These were selected by searching
# objaverse annotations for common indoor objects.
#
# If these UIDs become unavailable, run search_and_pick() below to find
# replacements interactively.
CURATED_UIDS = {
    "human_1": "16522107ad84410dad419e2e3977e721",  # low poly man walking+phone
    "human_2": "cf9b293059da44059ead548c043af92d",  # low-poly statuette boy with girl (textured)
    "table":   "786da44988f646a99fbd61a4e26af886",  # office table
    "chair":   "0115bca2d7bf45f092e9e93064ffada8",  # moora chair (stool)
    "plant":   "4ab9c39ca9424750a56712a9f3d938ef",  # potted flowers
}

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


def search_annotations(annotations, keywords, top_k=20):
    """Search objaverse annotations for models matching keywords."""
    results = []
    for uid, meta in annotations.items():
        name = (meta.get("name") or "").lower()
        tags = [t.get("name", "").lower() for t in (meta.get("tags") or [])]
        categories = [c.get("name", "").lower() for c in (meta.get("categories") or [])]
        all_text = name + " " + " ".join(tags) + " " + " ".join(categories)

        score = sum(1 for kw in keywords if kw.lower() in all_text)
        if score > 0:
            results.append((score, uid, name))

    results.sort(key=lambda x: -x[0])
    return results[:top_k]


def search_and_pick():
    """Interactive search: find and display candidate UIDs for each target."""
    print("Loading Objaverse annotations...")
    annotations = objaverse.load_annotations()
    print(f"Loaded {len(annotations)} annotations.\n")

    for target_name, info in TARGETS.items():
        print(f"--- {target_name}: {info['description']} ---")
        candidates = search_annotations(annotations, info["keywords"])
        for score, uid, name in candidates[:10]:
            print(f"  score={score}  uid={uid}  name={name}")
        print()


def download_curated():
    """Download the curated set of models."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    uids = list(CURATED_UIDS.values())
    uid_to_name = {v: k for k, v in CURATED_UIDS.items()}

    print(f"Downloading {len(uids)} models from Objaverse...")
    downloaded = objaverse.load_objects(uids=uids)

    print(f"\nDownloaded {len(downloaded)} files. Copying to {OUTPUT_DIR}/")
    manifest = {}

    for uid, local_path in downloaded.items():
        target_name = uid_to_name.get(uid, uid)
        dest_filename = f"{target_name}.glb"
        dest_path = os.path.join(OUTPUT_DIR, dest_filename)

        shutil.copy2(local_path, dest_path)
        file_size_mb = os.path.getsize(dest_path) / (1024 * 1024)
        print(f"  {dest_filename}  ({file_size_mb:.1f} MB)")
        manifest[target_name] = {
            "uid": uid,
            "filename": dest_filename,
            "size_mb": round(file_size_mb, 2),
        }

    # Save manifest
    manifest_path = os.path.join(OUTPUT_DIR, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest saved to {manifest_path}")


if __name__ == "__main__":
    import sys
    if "--search" in sys.argv:
        search_and_pick()
    else:
        download_curated()
