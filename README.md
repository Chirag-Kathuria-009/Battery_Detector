# Battery detection (POC)

Small computer-vision service that detects batteries in images and serves the
model over a Dockerized API. I built it in about a week as a portfolio piece for a
Senior CV Engineer interview at R3 Robotics (automated EV-battery disassembly).

It's detection-only on purpose. No camera calibration, pose, segmentation masks,
or robot integration. Those are listed under [Out of scope](#out-of-scope) with a
note on what I'd do next. The goal was to get something working end to end and
actually measure where it breaks, not to fake a finished product.

## The problem

On a disassembly line the robot has to find the battery in a pile of look-alike
parts before it can plan a grasp. This does that first step in miniature: given a
photo, put a box around any battery. What I was actually interested in was how a
model trained on a public dataset holds up on my own photos, i.e. the domain gap
you'd hit in a real deployment.

## Repo layout

| Path | What |
|------|------|
| `app/main.py` | FastAPI service: `GET /health`, `POST /detect` |
| `models/best.pt` | YOLOv8n weights, single `battery` class |
| `scripts/score_domain_gap.py` | Recomputes the domain-gap metrics from the reviewed results |
| `results/domain_gap/` | `metrics.md`, `predictions.csv`, annotated images |
| `tests/` | pytest API tests |
| `Dockerfile` | CPU-only container |
| `Kaggle/batteryyoloexp.ipynb` | Training / inference notebook |

## Dataset and training

Public e-waste dataset from Roboflow Universe, filtered down to a single `battery`
class. It mixes bbox and segmentation labels but only ~46% of training instances
have masks, which isn't enough to train segmentation reliably, so I kept it to
detection. The original splits were off (val bigger than train), so I re-split the
432 images to 80/10/10 (345 / 43 / 44).

Model is YOLOv8n, 50 epochs (early-stopped at 44), trained on Kaggle (T4) in about
3.5 minutes.

Public validation set:

| Precision | Recall | mAP50 | mAP50-95 |
|-----------|--------|-------|----------|
| 0.837 | 0.697 | 0.730 | 0.569 |

## Domain gap

I shot 20 objects on my phone: real batteries (power bank, AA cells, a DJI drone
battery) and some deliberate look-alikes (laptop charger, calculator,
earbuds/smartwatch, and the drone controller, which is close to the battery in
shape and material). None were in training. The model is single-class, so I scored
at the image level (did it draw at least one box?) and eyeballed each detection for
whether the box actually landed on the battery.

|                        | Detected | No detection |
|------------------------|----------|--------------|
| Actually battery (13)  | TP = 11  | FN = 2 |
| Not a battery (7)      | FP = 6   | TN = 1 |

Precision 0.65, Recall 0.85, Accuracy 0.60, Specificity 0.14.

What stood out (full write-up in [`results/domain_gap/metrics.md`](results/domain_gap/metrics.md)):

- It over-triggers on look-alike electronics. Only 1 of 7 non-batteries (the
  charger) was correctly ignored. All four controller shots got flagged, the
  calculator at 0.83, plus earbuds, a smartwatch and a sunglasses case. Precision
  dropped from 0.84 in-domain to 0.65 here.
- Image-level recall flatters it. Of the 11 true-positive images, maybe 6 actually
  have the box on the battery. The rest boxed the window frame, the controller, the
  whole frame, or just a corner. Strict IoU scoring would look worse.
- Confidence is low and there were flat misses. Real detections sit around
  0.27–0.65 and two power banks were missed outright (one backlit, one on a booklet).

So recall is promising but precision and specificity aren't trustworthy on unseen
scenes yet. That's the gap domain-specific data and proper box-level eval would
need to close.

## Running it

Local (Python 3.12):

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
# http://127.0.0.1:8000/docs
```

```bash
curl http://127.0.0.1:8000/health
curl -F "file=@BatteryDataset/PersonalImages/Image3.jpeg" \
     "http://127.0.0.1:8000/detect?conf=0.25"
```

Response:

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

Docker:

```bash
docker build -t battery-detect .
docker run -p 8000:8000 battery-detect
```

Tests:

```bash
pytest -q
```

They cover the health check, the `/detect` happy path, a blank image (valid, zero
detections), a missing file (422) and non-image bytes (400).

To regenerate the domain-gap outputs from the reviewed CSV:

```bash
unzip -o BatteryDataset/YOLO_Output/predictions.zip -d results/domain_gap/annotated
python scripts/score_domain_gap.py   # writes results/domain_gap/scores.json
```

## Out of scope

Left out on purpose, with what I'd do next:

- Camera calibration, to go from pixels to metric coordinates. Needs the real
  camera and a checkerboard target.
- 6D pose, since a grasp needs orientation not just a 2D box. Would add a pose head
  once depth/CAD models exist.
- Instance segmentation (SAM). Not enough masks in the data to train it; would run
  detector boxes through MobileSAM/FastSAM as a second stage.
- Depth / 3D. No calibrated depth here; stereo or a depth camera would be the move
  (monocular depth is only relative).
- Robot / closed-loop integration. No hardware in scope; detections would feed a
  grasp planner.
