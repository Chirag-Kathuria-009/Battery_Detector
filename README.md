# Battery Detection — proof-of-concept

A small, honestly-scoped computer-vision service that detects batteries in images
and serves the model behind a tested, Dockerized API. Built in under a week as an
interview portfolio piece for a **Senior CV Engineer** role at **R3 Robotics**
(automated EV-battery disassembly).

> **Scope, stated up front.** This is a *detection-only* proof-of-concept, not a
> finished product. It deliberately does **not** attempt camera calibration, 6D
> pose, segmentation masks, depth, or robot integration. Those are named in
> [Out of scope](#out-of-scope) with a line on why and what would come next. The
> value here is engineering judgment under a real time constraint and an honest
> failure analysis — not the appearance of completeness.

---

## Why this problem

On a disassembly line, a robot first has to **find the battery** in a cluttered
scene of look-alike electronic parts before it can plan a grasp. This project
tackles that first perception step in miniature: given a photo, put a box around
any battery. The interesting, honest part is measuring how well a model trained on
a *public* dataset holds up on *my own* photos — the domain gap a real deployment
would have to close.

## What's in the box

| Piece | What it is |
|-------|-----------|
| `app/main.py` | FastAPI service: `GET /health`, `POST /detect` |
| `models/best.pt` | YOLOv8n detector, single `battery` class |
| `scripts/score_domain_gap.py` | Recomputes the domain-gap confusion matrix / metrics from the reviewed results |
| `results/domain_gap/` | The domain-gap tally: `metrics.md`, `predictions.csv`, annotated images |
| `tests/` | `pytest` API tests |
| `Dockerfile` | CPU-only container that serves the API |
| `Kaggle/batteryyoloexp.ipynb` | The training/inference notebook |

## Dataset & training

- **Source:** a public e-waste dataset from Roboflow Universe, filtered to a single
  `battery` class. The dataset mixes bounding-box and segmentation labels, but only
  ~46% of training instances have masks — **not enough for reliable instance
  segmentation**, so this was scoped to **detection only** (see Out of scope).
- **Re-split:** the original splits were badly proportioned (val larger than
  train), so the 432 images were re-split locally to a clean **80/10/10 =
  345 / 43 / 44**.
- **Model:** YOLOv8n (nano), 50 epochs (early-stopped at 44), trained on Kaggle
  (Tesla T4) in ~3.5 minutes.

**Results on the public validation set:**

| Precision | Recall | mAP50 | mAP50-95 |
|-----------|--------|-------|----------|
| 0.837 | 0.697 | 0.730 | 0.569 |

## Domain gap findings (the honest part)

I photographed **20** objects with my phone — real batteries (power bank, AA cells,
a DJI drone battery) and deliberate look-alike negatives (laptop charger,
scientific calculator, earbuds/smartwatch, and the **drone controller** whose
shape/material mimics the battery). None were used in training. Ground truth is a
per-image `isBattery` label; the model is single-class, so scoring is at the
**image level** (did it draw ≥1 box?), and each detection was reviewed by eye for
whether the box actually landed on the battery.

|                        | Detected battery | No detection |
|------------------------|------------------|--------------|
| **Actually battery (13)** | TP = 11 | FN = 2 |
| **Not a battery (7)**     | FP = 6  | TN = 1 |

**Precision 0.65 · Recall 0.85 · Accuracy 0.60 · Specificity 0.14**

Three findings (full write-up in [`results/domain_gap/metrics.md`](results/domain_gap/metrics.md)):

1. **It over-triggers on look-alike electronics.** Only 1 of 7 non-batteries (the
   laptop charger) was correctly ignored. **All four drone controllers** were
   flagged as battery, the **calculator at 0.83 confidence**, plus earbuds, a
   smartwatch and a sunglasses case. The model over-generalized to "dark
   rectangular gadget = battery." Precision fell from **0.84 (in-domain) → 0.65**.
2. **Image-level recall overstates real quality.** Of 11 true-positive images, only
   ~6 have the box on the actual battery. The rest boxed the *window frame*, the
   *controller*, the *whole frame*, or only a *corner* — while junk boxes littered
   logos and keyboards. Strict IoU scoring would look worse.
3. **Low confidence and outright misses.** True detections cluster at 0.27–0.65,
   and two real power banks were missed entirely (one backlit, one on a booklet).

**Takeaway for R3:** encouraging recall, but precision/specificity are not yet
trustworthy on unseen scenes — exactly the gap that domain-specific data and
proper box-level evaluation would need to close.

## Run it

### Local (Python 3.12)

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
# open http://127.0.0.1:8000/docs
```

```bash
# health
curl http://127.0.0.1:8000/health
# detect (returns JSON: class, confidence, bbox per object)
curl -F "file=@BatteryDataset/PersonalImages/Image3.jpeg" \
     "http://127.0.0.1:8000/detect?conf=0.25"
```

Example response:

```json
{
  "filename": "Image3.jpeg",
  "conf_threshold": 0.25,
  "image_size": [1197, 1600],
  "count": 2,
  "detections": [
    {"class": "battery", "confidence": 0.8783, "bbox": [432.4, 511.3, 881.3, 1442.8]},
    {"class": "battery", "confidence": 0.5624, "bbox": [252.8, 818.2, 451.9, 1173.1]}
  ]
}
```

### Docker

```bash
docker build -t battery-detect .
docker run -p 8000:8000 battery-detect
# same curl calls as above against http://127.0.0.1:8000
```

### Tests

```bash
pytest -q
```

Covers the health check, the `/detect` happy path (well-formed JSON), a blank
image (valid request, zero detections), a missing file (422), and non-image bytes
(400).

### Reproduce the domain-gap eval

Predictions come from the original inference run (`predictions.zip`); each image
was reviewed by hand into `results/domain_gap/predictions.csv`. To regenerate the
annotated images and recompute the headline metrics from those reviewed results:

```bash
# (optional) unpack the saved annotated predictions
unzip -o BatteryDataset/YOLO_Output/predictions.zip -d results/domain_gap/annotated

# recompute the confusion matrix + precision/recall/accuracy/specificity
python scripts/score_domain_gap.py   # writes results/domain_gap/scores.json
```

## Out of scope

Named deliberately, not hidden — each is a real next step, not something faked here:

- **Camera calibration** — needed to turn pixels into metric coordinates. Skipped
  because it needs the actual camera + a calibration target; *next:* intrinsic/
  extrinsic calibration with a checkerboard on the real rig.
- **6D pose estimation** — a grasp needs object orientation, not just a 2D box.
  *Next:* a pose head (e.g. PVN3D/FoundationPose) once depth/CAD models are
  available.
- **Instance segmentation (SAM)** — the dataset lacked enough masks to train seg
  reliably. *Next:* feed detector boxes to MobileSAM/FastSAM for per-instance masks
  as a second stage.
- **Depth / 3D** — no calibrated depth sensor here. *Next:* stereo or a depth
  camera; monocular depth (MiDaS) only gives *relative*, not metric, depth.
- **Robot / closed-loop integration** — no hardware in scope. *Next:* wire
  detections into a grasp planner and test on the real cell.

## Notes on honesty

Metrics in this README are the real numbers from the runs above — nothing is
inflated. The domain-gap section is intentionally unflattering: the failure
analysis is the most useful thing this project demonstrates.
