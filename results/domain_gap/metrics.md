# Domain Gap Findings — personal phone photos

Model: `best.pt` (YOLOv8n, single class `battery`) run at **conf = 0.25** on
**20 personal phone photos** that were never part of training. Predictions come
from the original Kaggle inference run (`predictions.zip`); each annotated image
was reviewed by eye to record (a) whether the model fired and (b) whether the box
actually landed on the battery.

**Framing — image level.** Ground truth (`ground_truth.csv`) has only
`isBattery` TRUE/FALSE per image (no bounding boxes), so this is scored at the
image level, not with IoU:
- **Predicted positive** = model drew ≥ 1 battery box on the image
- **Actual positive** = human label `isBattery == TRUE`

## Confusion matrix (image level)

|                        | Detected battery | No detection |
|------------------------|------------------|--------------|
| **Actually battery (13)** | TP = 11 | FN = 2 |
| **Not a battery (7)**     | FP = 6  | TN = 1 |

## Metrics

| Metric | Value | Formula |
|--------|-------|---------|
| Precision   | **0.647** | 11 / (11 + 6) |
| Recall      | **0.846** | 11 / (11 + 2) |
| Accuracy    | **0.600** | (11 + 1) / 20 |
| Specificity | **0.143** | 1 / (1 + 6) |

Compare to the in-domain public **validation** set: Precision 0.837, Recall 0.697,
mAP50 0.73, mAP50-95 0.569. Precision fell **0.84 → 0.65** and specificity is
near zero — a clear domain gap.

## Finding 1 — over-triggers on look-alike electronics (precision collapse)
Only **1 of 7** non-batteries was correctly ignored (the laptop charger, Image1).
Every other non-battery was flagged:
- **All 4 drone controllers** (Image16–19) → false positives at 0.45–0.79. This is
  exactly the shape/material confuser called out in the project brief.
- **Scientific calculator** (Image6) flagged at **0.83** — higher than most true
  batteries.
- A **sunglasses case** was repeatedly boxed (Image16–18), and earbuds + a
  smartwatch were flagged (Image2).

The detector appears to have learned "dark, rectangular, hand-held electronic
object → battery," which generalizes to the wrong things.

## Finding 2 — image-level recall overstates real localization quality
Recall looks healthy (0.85) but of the 11 TP images only ~6 have a box actually on
the battery (Image3, 9, 12, 14, 15, 20). The others are "right image, wrong box":
- Boxed the **window frame**, missed the powerbank (Image8)
- Boxed the **controller**, missed the drone battery (Image13)
- One box covering the **whole frame** (Image5)
- Only the **partial corner** battery, missed the main hub (Image11)

Most true-positive images also carried extra junk boxes on logos, keyboards and
desks. With strict IoU scoring, precision would be worse still.

## Finding 3 — low confidence and outright misses
True detections on personal photos cluster at **0.27–0.65**, well below the
in-domain confidence. Two real powerbanks were missed entirely: backlit on a
windowsill (Image7) and lying on a booklet (Image10).

## Takeaway for the R3 disassembly use case
Image-level recall is encouraging, but precision/specificity are not yet
trustworthy: on an unseen scene the model would false-fire on controllers,
calculators and cases, and sometimes box the wrong object. Closing this gap needs
domain-specific data (real battery packs, in-situ lighting/backgrounds) and, for
localization, IoU-based evaluation against proper bounding-box labels — see the
README "Next steps".
