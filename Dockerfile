# Battery detector API - CPU-only image.
FROM python:3.12-slim

# System libs required by OpenCV (pulled in by ultralytics).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Keep the image lean: install CPU-only torch first from the PyTorch CPU index,
# then the rest of the requirements (torch is already satisfied, so pip won't
# pull the multi-GB CUDA build).
COPY requirements.txt .
RUN pip install --no-cache-dir \
        torch==2.7.1 torchvision==0.22.1 \
        --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# App code + weights.
COPY app/ ./app/
COPY models/ ./models/

# ultralytics needs a writable config dir; point it somewhere world-writable.
ENV YOLO_CONFIG_DIR=/tmp/Ultralytics \
    MODEL_PATH=models/best.pt

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
