# Docker Deployment

Deploy MetaPathPredict using Docker containers.

## Quick Start

### Pull Image

```bash
docker pull ghcr.io/your-org/metapathpredict:latest
```

### Run Prediction

```bash
docker run --gpus all -v $(pwd)/data:/data \
    ghcr.io/your-org/metapathpredict:latest \
    predict /data/input.fasta -o /data/output.csv
```

## Dockerfile

### Production Image

```dockerfile
# Multi-stage build for minimal image
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY pyproject.toml poetry.lock ./
RUN pip install poetry && \
    poetry config virtualenvs.create false && \
    poetry install --only main --no-interaction

# Copy source code
COPY src/ src/

# Production stage
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app/src /app/src

# Copy model weights
COPY data/weights /app/weights

# Set environment
ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=/app/weights/unified/model_best.pt

# Run inference
ENTRYPOINT ["python", "-m", "metapathpredict"]
CMD ["--help"]
```

### GPU Image

```dockerfile
FROM nvidia/cuda:12.1-runtime-ubuntu22.04

# Install Python
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install PyTorch with CUDA
RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install package
COPY pyproject.toml poetry.lock ./
RUN pip3 install poetry && \
    poetry config virtualenvs.create false && \
    poetry install --only main

COPY src/ src/
COPY data/weights /app/weights

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

ENTRYPOINT ["python3", "-m", "metapathpredict"]
```

### Development Image

```dockerfile
FROM python:3.11

WORKDIR /app

# Install all dependencies including dev
RUN pip install poetry
COPY pyproject.toml poetry.lock ./
RUN poetry config virtualenvs.create false && \
    poetry install --with dev,test

# Copy source
COPY . .

# Install package in editable mode
RUN pip install -e .

# Jupyter for notebooks
EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root"]
```

## Docker Compose

### Basic Setup

```yaml
# docker-compose.yml
version: '3.8'

services:
  metapathpredict:
    build: .
    volumes:
      - ./data/input:/data/input:ro
      - ./data/output:/data/output
      - ./data/weights:/app/weights:ro
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    command: predict /data/input/sequences.fasta -o /data/output/predictions.csv
```

### Full Stack

```yaml
# docker-compose.full.yml
version: '3.8'

services:
  # Main inference service
  inference:
    build:
      context: .
      dockerfile: Dockerfile.gpu
    volumes:
      - input_data:/data/input:ro
      - output_data:/data/output
      - model_weights:/app/weights:ro
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    environment:
      - BATCH_SIZE=64
      - NUM_WORKERS=4

  # API server
  api:
    build:
      context: .
      dockerfile: Dockerfile.api
    ports:
      - "8000:8000"
    volumes:
      - model_weights:/app/weights:ro
    environment:
      - MODEL_PATH=/app/weights/unified/model_best.pt
      - WORKERS=4
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # MLflow tracking
  mlflow:
    image: ghcr.io/mlflow/mlflow:v2.10.0
    ports:
      - "5000:5000"
    volumes:
      - mlflow_data:/mlflow
    command: mlflow server --host 0.0.0.0 --backend-store-uri sqlite:///mlflow/mlflow.db

  # MinIO for artifacts
  minio:
    image: minio/minio:latest
    ports:
      - "9000:9000"
      - "9001:9001"
    volumes:
      - minio_data:/data
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin
    command: server /data --console-address ":9001"

volumes:
  input_data:
  output_data:
  model_weights:
  mlflow_data:
  minio_data:
```

## Building Images

### Local Build

```bash
# Build production image
docker build -t metapathpredict:latest .

# Build GPU image
docker build -f Dockerfile.gpu -t metapathpredict:gpu .

# Build development image
docker build -f Dockerfile.dev -t metapathpredict:dev .
```

### Multi-Platform Build

```bash
# Create builder for multi-arch
docker buildx create --name multibuilder --use

# Build for multiple platforms
docker buildx build \
    --platform linux/amd64,linux/arm64 \
    -t ghcr.io/your-org/metapathpredict:latest \
    --push .
```

## Running Containers

### Inference

```bash
# CPU inference
docker run -v $(pwd)/data:/data metapathpredict:latest \
    predict /data/input.fasta -o /data/output.csv

# GPU inference
docker run --gpus all -v $(pwd)/data:/data metapathpredict:gpu \
    predict /data/input.fasta -o /data/output.csv --device cuda

# With custom config
docker run --gpus all \
    -v $(pwd)/data:/data \
    -v $(pwd)/config.yaml:/app/config.yaml:ro \
    metapathpredict:gpu \
    predict /data/input.fasta --config /app/config.yaml
```

### Training

```bash
# Train with GPU
docker run --gpus all \
    -v $(pwd)/data:/data \
    -v $(pwd)/checkpoints:/app/checkpoints \
    metapathpredict:gpu \
    train \
    --data-dir /data/datasets/unified \
    --output-dir /app/checkpoints \
    --epochs 100

# Train with MLflow tracking
docker run --gpus all \
    -v $(pwd)/data:/data \
    -e MLFLOW_TRACKING_URI=http://mlflow:5000 \
    --network metapathpredict_default \
    metapathpredict:gpu train --track-experiments
```

### Interactive Development

```bash
# Start Jupyter
docker run -it --gpus all \
    -p 8888:8888 \
    -v $(pwd):/app \
    metapathpredict:dev

# Interactive shell
docker run -it --gpus all \
    -v $(pwd):/app \
    metapathpredict:dev bash
```

## API Server

### Dockerfile.api

```dockerfile
FROM metapathpredict:gpu

# Install API dependencies
RUN pip install fastapi uvicorn python-multipart

# Copy API code
COPY api/ /app/api/

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### API Example

```python
# api/main.py
from fastapi import FastAPI, File, UploadFile
from metapathpredict import Predictor
import tempfile

app = FastAPI(title="MetaPathPredict API")
predictor = Predictor("weights/unified/model_best.pt")

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(suffix=".fasta") as tmp:
        tmp.write(await file.read())
        tmp.flush()
        results = predictor.predict_fasta(tmp.name)
    return results.to_dict(orient="records")
```

## Resource Management

### Memory Limits

```yaml
services:
  inference:
    deploy:
      resources:
        limits:
          memory: 16G
        reservations:
          memory: 8G
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### CPU Limits

```bash
docker run --cpus=4 --memory=8g metapathpredict:latest predict input.fasta
```

## Health Checks

```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from metapathpredict import Predictor; print('ok')"
```

## Logging

```bash
# View logs
docker logs -f container_name

# Log to file
docker run -v $(pwd)/logs:/logs \
    -e LOG_FILE=/logs/inference.log \
    metapathpredict:latest predict input.fasta
```

## CI/CD Integration

### GitHub Actions

```yaml
- name: Build and push Docker image
  uses: docker/build-push-action@v5
  with:
    context: .
    push: true
    tags: ghcr.io/${{ github.repository }}:${{ github.sha }}
    cache-from: type=gha
    cache-to: type=gha,mode=max
```

See [Cloud (GCP)](cloud.md) for Kubernetes deployment.
