import os
from collections import Counter
from pathlib import Path
import zipfile

dataset_root = Path("BatteryDataset/battery_resplit")  # update this

for split in ["train", "valid", "test"]:
    img_dir = dataset_root / split / "images"
    lbl_dir = dataset_root / split / "labels"
    #print(os.getcwd())
    if not img_dir.exists():
        print(f"{split}: folder not found, skipping")
        continue

    image_count = len(list(img_dir.glob("*.*")))
    class_counter = Counter()
    annotation_type = set()

    for label_file in lbl_dir.glob("*.txt"):
        with open(label_file) as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                class_id = parts[0]
                class_counter[class_id] += 1
                num_values = len(parts) - 1
                annotation_type.add("segmentation" if num_values > 5 else "bbox")

    print(f"\n=== {split.upper()} ===")
    print(f"Images: {image_count}")
    print(f"Annotation type detected: {annotation_type}")
    print(f"Class distribution (by class_id): {dict(class_counter)}")
    
    #\BatteryDataset\E-Waste_Dataset.yolov8\test\images
    #\BatteryDataset\E-Waste_Dataset.yolov8\test\images


for split in ["train", "valid", "test"]:
    lbl_dir = dataset_root / split / "labels"
    bbox_only = 0
    segmented = 0

    for label_file in lbl_dir.glob("*.txt"):
        with open(label_file) as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                num_values = len(parts) - 1
                if num_values > 5:
                    segmented += 1
                else:
                    bbox_only += 1

    total = bbox_only + segmented
    pct_seg = (segmented / total * 100) if total else 0
    print(f"{split}: {segmented} segmented / {bbox_only} bbox-only "
          f"({pct_seg:.1f}% have masks)")