"""
FastAPI wrapper around the trained YOLOv8n battery detector.

Endpoints
---------
GET  /health   -> basic liveness + whether the weights loaded
POST /detect   -> accepts an image, returns detected objects (class, confidence, bbox)

Scope note: this is a detection-only proof-of-concept (single "battery" class).
No segmentation, depth, or pose estimation - see README "Out of scope".
"""

from __future__ import annotations

import io
import os
from functools import lru_cache

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from PIL import Image, UnidentifiedImageError

# Path to the trained weights. Overridable via env var so tests and the Docker
# image can point at a different location without code changes.
MODEL_PATH = os.getenv("MODEL_PATH", "models/best.pt")

app = FastAPI(
    title="R3 Battery Detector",
    version="0.1.0",
    description="Detection-only proof-of-concept: YOLOv8n fine-tuned on a public "
    "e-waste dataset, filtered to a single 'battery' class.",
)


@lru_cache(maxsize=1)
def get_model():
    """Load the YOLO model once and cache it.

    ultralytics/torch are imported lazily inside the function so that importing
    this module (e.g. during test collection) does not pay the heavy import cost
    until a model is actually needed.
    """
    from ultralytics import YOLO

    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model weights not found at MODEL_PATH={MODEL_PATH!r}")
    return YOLO(MODEL_PATH)


@app.get("/health")
def health() -> dict:
    """Liveness probe. Reports whether the weights file is present so a failed
    model mount shows up here rather than only on the first /detect call."""
    return {
        "status": "ok",
        "model_path": MODEL_PATH,
        "model_loaded": os.path.exists(MODEL_PATH),
    }


@app.post("/detect")
async def detect(
    file: UploadFile = File(..., description="Image file (jpg/png/...)."),
    conf: float = Query(
        0.25, ge=0.0, le=1.0, description="Minimum confidence threshold."
    ),
) -> dict:
    """Run battery detection on an uploaded image.

    Returns one entry per detected box with the class name, confidence, and an
    [x1, y1, x2, y2] bounding box in pixel coordinates of the input image.
    """
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file upload.")

    # Decode + validate the bytes are actually a readable image.
    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=400,
            detail="Uploaded bytes could not be decoded as an image.",
        )

    try:
        model = get_model()
    except RuntimeError as exc:  # weights missing / unreadable
        raise HTTPException(status_code=503, detail=str(exc))

    # verbose=False keeps the server logs clean; predict returns a list (1 per image).
    results = model.predict(image, conf=conf, verbose=False)
    result = results[0]
    names = result.names  # {class_id: class_name}

    detections = []
    for box in result.boxes:
        cls_id = int(box.cls[0])
        x1, y1, x2, y2 = (round(float(v), 1) for v in box.xyxy[0].tolist())
        detections.append(
            {
                "class": names.get(cls_id, str(cls_id)),
                "confidence": round(float(box.conf[0]), 4),
                "bbox": [x1, y1, x2, y2],
            }
        )

    width, height = image.size
    return {
        "filename": file.filename,
        "conf_threshold": conf,
        "image_size": [width, height],
        "count": len(detections),
        "detections": detections,
    }
