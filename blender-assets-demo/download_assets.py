"""
从 Objaverse 下载演示场景所需的 3D 模型。

目标：2 个人物、1 张桌子、1 把椅子、1 株植物。
下载 .glb 文件并以具描述性的文件名复制到 models/ 目录。

用法：
    pip install objaverse
    python download_assets.py
"""

import os
import shutil
import json

import objaverse


# 目标模型：类别 -> (文件名, 搜索关键词)
# 将通过搜索注释来查找合适的 UID，然后下载。
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

# 从 Objaverse 手动筛选的 UID，质量可接受且尺寸合适的 .glb 模型。
# 通过搜索 objaverse 注释中常见室内物体关键词选出。
#
# 若这些 UID 不再可用，可运行下方的 search_and_pick() 以交互方式查找替代品。
CURATED_UIDS = {
    "human_1": "16522107ad84410dad419e2e3977e721",  # low poly man walking+phone
    "human_2": "cf9b293059da44059ead548c043af92d",  # low-poly statuette boy with girl (textured)
    "table":   "786da44988f646a99fbd61a4e26af886",  # office table
    "chair":   "0115bca2d7bf45f092e9e93064ffada8",  # moora chair (stool)
    "plant":   "4ab9c39ca9424750a56712a9f3d938ef",  # potted flowers
}

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


def search_annotations(annotations, keywords, top_k=20):
    """在 objaverse 注释中搜索匹配关键词的模型。"""
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
    """交互式搜索：为每个目标查找并展示候选 UID。"""
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
    """下载精选的模型集合。"""
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

    # 保存清单文件
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
