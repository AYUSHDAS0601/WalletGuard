FROM python:3.10-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python deps (CPU-only PyTorch for Docker; override with GPU image for GPU)
COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir torch-geometric && \
    pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Create directories
RUN mkdir -p checkpoints data/raw data/cache

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
