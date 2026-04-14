FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64 \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/lib/jvm/java-21-openjdk-amd64/bin:/usr/local/cuda/bin:${PATH}

WORKDIR /app

# -----
# System packages
# -----
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

RUN ln -sf /usr/bin/python3 /usr/bin/python && \
    python -m pip install --upgrade pip setuptools wheel

# -----
# Small NLP extras used by your project
# -----
RUN python -m pip install \
    nltk \
    rank_bm25 \
 && python - <<'PY'
import nltk
nltk.download('punkt_tab')
nltk.download('punkt')
print('nltk data ready')
PY

# -----
# Install PyTorch first from the CUDA 12.8 wheel index
# This is the key B200-related change compared with your old CUDA 12.4 image.
# -----
RUN python -m pip install \
    torch \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cu128

# -----
# Project dependencies
# -----
RUN python -m pip install \
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

# -----
# FlagEmbedding / reranking
# -----
RUN python -m pip install -U FlagEmbedding==1.3.4 && \
    python - <<'PY'
from FlagEmbedding import FlagLLMReranker
print('FlagEmbedding import ok:', FlagLLMReranker is not None)
PY

# -----
# Build-time import smoke test
# -----
RUN python - <<'PY'
import sys
import torch
import transformers
print('Python:', sys.version)
print('Torch:', torch.__version__)
print('Torch CUDA build:', torch.version.cuda)
print('Transformers:', transformers.__version__)
PY

ENTRYPOINT ["tini", "--"]
CMD ["bash", "-lc", "python - <<'PY'\nimport torch\nprint('torch', torch.__version__)\nprint('cuda available', torch.cuda.is_available())\nprint('cuda build', torch.version.cuda)\nprint('device count', torch.cuda.device_count())\nif torch.cuda.is_available():\n    print('device 0', torch.cuda.get_device_name(0))\n    print('capability', torch.cuda.get_device_capability(0))\nPY"]
