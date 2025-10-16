# -----------------------------
# 1. Base image with Python and CUDA support (for GPU)
# -----------------------------
# Use CPU-only base if you don't need GPU:
# FROM python:3.11-slim
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# -----------------------------
# 2. Environment setup
# -----------------------------
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# -----------------------------
# 3. Install system dependencies
# -----------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    curl \
    ca-certificates \
    libffi-dev \
    libssl-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------
# 4. Set workdir
# -----------------------------
WORKDIR /app

# -----------------------------
# 5. Copy requirements and install
# -----------------------------
COPY requirements.txt .

# Install PyTorch + PyG (match CUDA version)
RUN pip install --upgrade pip setuptools wheel

# Install PyTorch + torchvision + torchaudio (CPU or GPU)
RUN pip install torch==2.1.0+cu122 torchvision==0.16.1+cu122 torchaudio==2.1.1+cu122 --index-url https://download.pytorch.org/whl/cu122

# Install PyTorch Geometric dependencies
RUN pip install torch-geometric==3.3.0 torch-sparse==0.6.17 torch-scatter==2.1.1 torch-cluster==1.6.0 torch-spline-conv==1.3.1 -f https://data.pyg.org/whl/torch-2.1.0+cu122.html

# Install remaining Python packages
RUN pip install -r requirements.txt

# -----------------------------
# 6. Copy API code
# -----------------------------
COPY . .

# -----------------------------
# 7. Expose port
# -----------------------------
EXPOSE 5000

# -----------------------------
# 8. Run Flask API
# -----------------------------
# Set environment variables for Flask
ENV FLASK_APP=flask_gnn_api_risk.py
ENV FLASK_ENV=production

CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]
