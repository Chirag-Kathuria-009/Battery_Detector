"""
Recompute the domain-gap headline metrics from the reviewed per-image results.

Method note: predictions themselves come from the original Kaggle inference run
(BatteryDataset/YOLO_Output/predictions.zip). Each annotated image was reviewed by
hand and the outcome recorded in results/domain_gap/predictions.csv. This script
does NOT re-run the model - it joins the human ground truth with those reviewed
detections and recomputes the confusion matrix + metrics, so the arithmetic in
metrics.md is auditable/reproducible.

Scoring is image level (single-class detector, no per-image GT boxes):
    predicted_positive = model drew >= 1 battery box   (predictions.csv `detected`)
    actual_positive    = isBattery == TRUE             (ground_truth.csv)

Usage:  python scripts/score_domain_gap.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GROUND_TRUTH = ROOT / "data" / "personal_test" / "ground_truth.csv"
PREDICTIONS = ROOT / "results" / "domain_gap" / "predictions.csv"
SCORES_OUT = ROOT / "results" / "domain_gap" / "scores.json"


def _as_bool(value: str) -> bool:
    return str(value).strip().upper() in {"TRUE", "1", "YES"}


def load_ground_truth() -> dict[str, bool]:
    with open(GROUND_TRUTH, newline="", encoding="utf-8-sig") as f:
        return {r["Name"].strip(): _as_bool(r["isBattery"]) for r in csv.DictReader(f)}


def load_detected() -> dict[str, bool]:
    with open(PREDICTIONS, newline="", encoding="utf-8-sig") as f:
        return {r["Name"].strip(): _as_bool(r["detected"]) for r in csv.DictReader(f)}


def main() -> None:
    gt = load_ground_truth()
    detected = load_detected()

    missing = set(gt) ^ set(detected)
    if missing:
        raise SystemExit(f"Ground truth / predictions mismatch on: {sorted(missing)}")

    tp = fp = fn = tn = 0
    for name, actual in gt.items():
        pred = detected[name]
        if actual and pred:
            tp += 1
        elif actual and not pred:
            fn += 1
        elif not actual and pred:
            fp += 1
        else:
            tn += 1

    total = tp + fp + fn + tn
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    accuracy = (tp + tn) / total if total else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    scores = {
        "n_images": total,
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "accuracy": round(accuracy, 3),
        "specificity": round(specificity, 3),
    }
    SCORES_OUT.write_text(json.dumps(scores, indent=2), encoding="utf-8")

    print(f"Images scored: {total}  ({tp + fn} battery, {fp + tn} non-battery)\n")
    print("                     detected   no-detection")
    print(f"  actually battery      TP={tp:<3}     FN={fn}")
    print(f"  not a battery         FP={fp:<3}     TN={tn}\n")
    print(f"  Precision   = {precision:.3f}")
    print(f"  Recall      = {recall:.3f}")
    print(f"  Accuracy    = {accuracy:.3f}")
    print(f"  Specificity = {specificity:.3f}")
    print(f"\nWrote {SCORES_OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
