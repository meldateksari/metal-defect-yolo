import os
import shutil

TRAIN_IMG_ROOT = "/content/metal-defect-yolo/data/raw/neu_det/NEU-DET/train/images"
VAL_IMG_ROOT   = "/content/metal-defect-yolo/data/raw/neu_det/NEU-DET/validation/images"

OUT = "/content/metal-defect-yolo/data/yolo_neu"

CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches"
]

def ensure():
    shutil.rmtree(OUT, ignore_errors=True)
    os.makedirs(OUT + "/images/train", exist_ok=True)
    os.makedirs(OUT + "/images/val", exist_ok=True)
    os.makedirs(OUT + "/labels/train", exist_ok=True)
    os.makedirs(OUT + "/labels/val", exist_ok=True)

def collect_images(root, subset):
    items = []
    for cid, cname in enumerate(CLASSES):
        folder = os.path.join(root, cname)
        if not os.path.isdir(folder):
            print(f"⚠ Uyarı: Sınıf klasörü bulunamadı → {folder}")
            continue

        imgs = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".jpg")]
        print(f"{subset} - {cname}: {len(imgs)} görüntü")
        items.extend([(img, cid) for img in imgs])

    return items

def write_yolo(items, subset):
    for img_path, cid in items:
        imgname = os.path.basename(img_path)
        shutil.copy(img_path, f"{OUT}/images/{subset}/{imgname}")

        with open(f"{OUT}/labels/{subset}/{imgname.replace('.jpg', '.txt')}", "w") as f:
            f.write(f"{cid} 0.5 0.5 1 1\n")

def convert_neu():
    ensure()
    print("🔍 NEU-DET dataset taranıyor...")

    train_set = collect_images(TRAIN_IMG_ROOT, "train")
    val_set   = collect_images(VAL_IMG_ROOT, "validation")

    write_yolo(train_set, "train")
    write_yolo(val_set, "val")

    print("\n✅ NEU YOLO dönüştürme tamamlandı!")
    print(f"Train toplam: {len(train_set)}")
    print(f"Validation toplam: {len(val_set)}")
    print(f"Çıktı klasörü: {OUT}")

if __name__ == "__main__":
    convert_neu()
