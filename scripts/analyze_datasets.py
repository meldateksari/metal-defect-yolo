import json
from pathlib import Path
from collections import Counter
from PIL import Image
from tqdm import tqdm
import numpy as np

# ==== PATH FIX FOR COLAB ====
if "__file__" in globals():
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
else:
    PROJECT_ROOT = Path("/content/metal-defect-yolo")

DATA_DIR = PROJECT_ROOT / "data"

# NEU (YOLO format)
NEU_IMG_DIR = DATA_DIR / "raw" / "neu_det" / "images"
NEU_ANN_DIR = DATA_DIR / "raw" / "neu_det" / "annotations"

# GC10 DET (folder format)
GC10_DIR = DATA_DIR / "raw" / "gc10_det"


# ============ NEU SCAN ===============
def scan_neu():
    if not NEU_IMG_DIR.exists():
        print("[ERROR] NEU images directory not found:", NEU_IMG_DIR)
        return {}, []

    # ALL FIXED: scan subfolders using rglob
    image_paths = [p for p in NEU_IMG_DIR.rglob("*") if p.suffix.lower() in ['.jpg', '.png', '.jpeg', '.bmp']]

    print(f"[NEU-DET] Found {len(image_paths)} images")

    records = []
    corrupt = []
    class_ids = []

    for img_path in tqdm(image_paths):
        # annotation path
        txt_path = NEU_ANN_DIR / (img_path.stem + ".txt")

        # read class id (if exists)
        class_id = None
        if txt_path.exists():
            try:
                with open(txt_path, "r") as f:
                    first_line = f.readline().strip()
                    class_id = int(first_line.split()[0])
            except:
                class_id = None

        class_ids.append(class_id)

        # read image attributes
        try:
            with Image.open(img_path) as im:
                w, h = im.size
                mode = im.mode
        except:
            corrupt.append(str(img_path))
            continue

        records.append({
            "path": str(img_path),
            "class_id": class_id,
            "width": w,
            "height": h,
            "mode": mode
        })

    # compute stats
    if len(records) == 0:
        stats = {}
    else:
        stats = {
            "num_images": len(records),
            "classes": dict(Counter(class_ids)),
            "modes": dict(Counter([r["mode"] for r in records])),
            "width_range": [min(r["width"] for r in records), max(r["width"] for r in records)],
            "height_range": [min(r["height"] for r in records), max(r["height"] for r in records)],
        }

    return stats, corrupt


# ============ GC10 SCAN ===============
def scan_gc10():
    img_paths = [p for p in GC10_DIR.rglob("*") if p.suffix.lower() in ['.jpg', '.png', '.bmp']]
    records = []
    corrupt = []

    print(f"[GC10-DET] Found {len(img_paths)} images")

    for img_path in tqdm(img_paths):
        class_name = img_path.parent.name

        try:
            with Image.open(img_path) as im:
                w, h = im.size
                mode = im.mode
        except:
            corrupt.append(str(img_path))
            continue

        records.append({
            "path": str(img_path),
            "class": class_name,
            "width": w,
            "height": h,
            "mode": mode
        })

    if len(records) == 0:
        stats = {}
    else:
        stats = {
            "num_images": len(records),
            "classes": dict(Counter([r["class"] for r in records])),
            "modes": dict(Counter([r["mode"] for r in records])),
            "width_range": [min(r["width"] for r in records), max(r["width"] for r in records)],
            "height_range": [min(r["height"] for r in records), max(r["height"] for r in records)],
        }

    return stats, corrupt


# ============ MAIN ===============
def main():
    neu_stats, neu_corrupt = scan_neu()
    gc10_stats, gc10_corrupt = scan_gc10()

    dataset_info = {
        "NEU-DET": {"stats": neu_stats, "corrupt": neu_corrupt},
        "GC10-DET": {"stats": gc10_stats, "corrupt": gc10_corrupt}
    }

    out_path = DATA_DIR / "dataset_info.json"
    with open(out_path, "w") as f:
        json.dump(dataset_info, f, indent=4)

    print(f"\n[OK] dataset_info.json saved to {out_path}")


if __name__ == "__main__":
    main()
