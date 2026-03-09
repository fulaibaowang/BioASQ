# syntax=docker/dockerfile:1

# CUDA runtime base (Ubuntu 22.04) for GPU usage
# If your HPC uses a different CUDA version, adjust the tag accordingly.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# -----
# System packages
# -----
# Includes:
# - python + pip
# - build tools (for hnswlib / any native wheels)
# - openjdk for PyTerrier
# - basic debugging tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential \
    gcc \
    g++ \
    make \
    git \
    curl \
    wget \
    ca-certificates \
    tini \
    procps \
    htop \
    less \
    vim-tiny \
    unzip \
    libxml2 \
    libxslt1.1 \
    zlib1g \
    openjdk-21-jre-headless \
 && rm -rf /var/lib/apt/lists/*

# Make `python` point to python3
RUN ln -sf /usr/bin/python3 /usr/bin/python

# new for reranking
RUN pip install nltk rank_bm25
RUN python -c "import nltk; nltk.download('punkt_tab'); nltk.download('punkt')"

# Make Java available for PyTerrier
ENV JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
ENV PATH="${JAVA_HOME}/bin:${PATH}"

# -----
# Python environment
# -----
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Upgrade pip tooling
RUN python -m pip install --no-cache-dir -U pip setuptools wheel

# -----
# Install PyTorch (GPU build) first
# -----
# IMPORTANT:
# - This uses the official PyTorch wheel index for CUDA 12.4.
# - If your base image CUDA tag changes, match the torch CUDA wheels accordingly.
RUN python -m pip install --no-cache-dir \
  torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# -----
# Install project dependencies
# -----
RUN python -m pip install --no-cache-dir \
  lxml \
  pubmed-parser \
  tqdm \
  requests \
  orjson \
  ujson \
  numpy \
  pandas \
  pyarrow \
  polars \
  python-terrier \
  ranx \
  beautifulsoup4 \
  accelerate \
  jupyterlab \
  matplotlib \
  scikit-learn \
  transformers==4.44.2 \
  sentence-transformers \
  hnswlib

#new for FlagEmbedding
RUN pip install --no-cache-dir -U FlagEmbedding==1.3.4
RUN python -c "from FlagEmbedding import FlagLLMReranker; print('ok')"
# -----
# Clone your repo
# -----
# RUN git clone https://github.com/fulaibaowang/BioASQ.git /app/bioasq
# WORKDIR /app/bioasq

# -----
# Optional: verify GPU is visible at container runtime
# (This is just a default command; override in your job script.)
# -----
CMD ["bash", "-lc", "python -c \"import torch; print('torch', torch.__version__); print('cuda available', torch.cuda.is_available()); print('cuda', torch.version.cuda); print('devices', torch.cuda.device_count())\""]
