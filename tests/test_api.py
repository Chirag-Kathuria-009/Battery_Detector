"""
Minimal API tests for the battery detector service.

Breadth over depth (per project scope): these cover the health check, the happy
path of /detect returning a well-formed JSON structure, and the two input-error
paths. They intentionally do NOT assert exact detection counts/coordinates - that
would be brittle and is what the domain-gap eval is for.

Run: pytest -q
"""

import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app

client = TestClient(app)

# A real personal photo known to contain batteries (powerbank + AA cells).
SAMPLE_IMAGE = (
    Path(__file__).resolve().parents[1]
    / "BatteryDataset" / "PersonalImages" / "Image3.jpeg"
)


def test_health_ok():
    """Health check returns 200 and reports the model is present."""
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_detect_returns_expected_json_structure():
    """A valid image yields the documented JSON shape; every detection has
    class / confidence / a 4-value bbox."""
    with open(SAMPLE_IMAGE, "rb") as f:
        resp = client.post("/detect", files={"file": ("Image3.jpeg", f, "image/jpeg")})

    assert resp.status_code == 200
    body = resp.json()
    for key in ("filename", "conf_threshold", "image_size", "count", "detections"):
        assert key in body
    assert body["count"] == len(body["detections"])
    for det in body["detections"]:
        assert set(det) == {"class", "confidence", "bbox"}
        assert 0.0 <= det["confidence"] <= 1.0
        assert len(det["bbox"]) == 4


def test_detect_synthetic_blank_image_is_valid_request():
    """A decodable image with no batteries still returns 200 with an empty
    detections list (not an error)."""
    buf = io.BytesIO()
    Image.new("RGB", (640, 480), color="white").save(buf, format="JPEG")
    buf.seek(0)
    resp = client.post("/detect", files={"file": ("blank.jpg", buf, "image/jpeg")})

    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 0
    assert body["detections"] == []


def test_detect_missing_file_is_422():
    """Omitting the required file field is a FastAPI validation error."""
    resp = client.post("/detect")
    assert resp.status_code == 422


def test_detect_non_image_bytes_is_400():
    """Uploading bytes that aren't a decodable image returns a clear 400."""
    resp = client.post(
        "/detect",
        files={"file": ("notimg.txt", io.BytesIO(b"this is not an image"), "text/plain")},
    )
    assert resp.status_code == 400
    assert "image" in resp.json()["detail"].lower()
