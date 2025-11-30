import os
import xml.etree.ElementTree as ET
from pathlib import Path
from sklearn.model_selection import train_test_split
import shutil

# ==== PATHS ====
ROOT = Path("/content/metal-defect-yolo")
RAW_DIR = ROOT / "data/raw/gc10_det"
OUT_DIR = ROOT / "data/yolo_gc10"

# Class list (GC10 has 10 defect categories)
CLASSES = [
    "punching_hole",
    "welding_line",
    "crescent_gap",
    "water_spot",
    "oil_spot",
    "silk_spot",
    "inclusion",
    "rolled_pit",
    "crease",
    "waist_folding"
]

CLASS_TO_ID = {cls: i for i, cls in enumerate(CLASSES)}

def parse_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    img_w = int(root.find("size").find("width").text)
    img_h = int(root.find("size").find("height").text)

    yolo_labels = []

    for obj in root.findall("object"):
        cls_name = obj.find("name").text
        cls_id = CLASS_TO_ID[cls_name]

        bbox = obj.find("bndbox")
        xmin = float(bbox.find("xmin").text)
        ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)

        # Convert to YOLO format
        x_center = (xmin + xmax) / 2 / img_w
        y_center = (ymin + ymax) / 2 / img_h
        width = (xmax - xmin) / img_w
        height = (ymax - ymin) / img_h

        yolo_labels.append(f"{cls_id} {x_center} {y_center} {width} {height}")

    return yolo_labels

def convert_gc10():
    # Output folders
    for split in ["train", "val", "test"]:
        (OUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Collect all images
    img_files = list(RAW_DIR.rglob("*.jpg"))
    print("Found images:", len(img_files))

    # Split dataset
    train_imgs, temp_imgs = train_test_split(img_files, test_size=0.30, random_state=42)
    val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.50, random_state=42)

    splits = {"train": train_imgs, "val": val_imgs, "test": test_imgs}

    for split, images in splits.items():
        print(f"Processing {split}: {len(images)} images")
        for img_path in images:
            xml_path = img_path.with_suffix(".xml")
            if not xml_path.exists():
                print("Missing XML:", xml_path)
                continue

            # → Parse XML
            labels = parse_xml(xml_path)

            # → Copy image
            new_img = OUT_DIR / "images" / split / img_path.name
            shutil.copy(img_path, new_img)

            # → Write YOLO label
            label_path = OUT_DIR / "labels" / split / (img_path.stem + ".txt")
            with open(label_path, "w") as f:
                f.write("\n".join(labels))

    print("\nConversion completed!")
    print("YOLO dataset saved to:", OUT_DIR)

if __name__ == "__main__":
    convert_gc10()
