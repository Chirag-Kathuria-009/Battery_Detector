import random, shutil
from pathlib import Path

random.seed(42)
root = Path("BatteryDataset/E-Waste_Dataset.yolov8")
all_images = []
for split in ["train", "valid", "test"]:
    all_images += list((root / split / "images").glob("*.*"))

random.shuffle(all_images)
n = len(all_images)
train_end = int(n * 0.8)
val_end = train_end + int(n * 0.1)

splits = {
    "train": all_images[:train_end],
    "valid": all_images[train_end:val_end],
    "test": all_images[val_end:],
}

new_root = root.parent / "battery_resplit"
print(f"Creating new dataset at {new_root}")

for split, imgs in splits.items():
    (new_root / split / "images").mkdir(parents=True, exist_ok=True)
    (new_root / split / "labels").mkdir(parents=True, exist_ok=True)
    for img in imgs:
        old_split = img.parts[-3]  # original split name
        label = img.parent.parent / "labels" / (img.stem + ".txt")
        shutil.copy(img, new_root / split / "images" / img.name)
        if label.exists():
            shutil.copy(label, new_root / split / "labels" / label.name)


print({k: len(v) for k, v in splits.items()})