# Separate image for listwise reranking (RankLLM + LiT5).
# Use this to run notebooks/listwise_rarank_test.ipynb without conflicting with
# the main Dockerfile's PyTorch/transformers stack. RankLLM's [all] extra pulls
# vLLM, which pins torch==2.6.0 and can fail on current indices; we install
# rank-llm without [all] so LiT5 (RankFiDDistill) works with the PyTorch we install.
#
# Build: docker build -f Dockerfile.listwise -t bioasq-listwise .
# Run:   docker run --gpus all -v $(pwd):/app -it bioasq-listwise bash

# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# Avoid interactive prompts (e.g. tzdata) during apt install in CI
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# -----
# System packages + Python 3.11 (RankLLM recommends 3.11)
# -----
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    build-essential \
    gcc \
    g++ \
    make \
    git \
    curl \
    wget \
    ca-certificates \
    openjdk-21-jre-headless \
    && rm -rf /var/lib/apt/lists/*

# Prefer Python 3.11
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1
ENV JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
ENV PATH="${JAVA_HOME}/bin:${PATH}"

# -----
# Python env
# -----
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
WORKDIR /app

RUN python3 -m pip install --no-cache-dir -U pip setuptools wheel

# -----
# PyTorch (GPU) first – use same CUDA as base image
# -----
RUN python3 -m pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# -----
# Install all HuggingFace ecosystem packages together so pip resolves them consistently.
# transformers 4.44.x needs tokenizers ~0.19.x and huggingface-hub <1.0;
# rank-llm uses find_pruneable_heads_and_indices which was removed in transformers >= 4.45.
# -----
RUN python3 -m pip install --no-cache-dir \
    "transformers==4.44.2" \
    "tokenizers>=0.19,<0.20" \
    "huggingface-hub>=0.23,<1.0" \
    "accelerate<1.0" \
    sentencepiece

# -----
# RankLLM without [all] to avoid vLLM and its torch==2.6.0 pin.
# LiT5 (RankFiDDistill) does not require vLLM.
# Install with --no-deps so it doesn't override our pinned transformers/tokenizers.
# -----
RUN python3 -m pip install --no-cache-dir --no-deps rank-llm

# Install rank-llm's remaining deps (excluding HF ecosystem which we already installed).
# Use --ignore-installed to avoid failures on distutils-installed packages (e.g. blinker).
RUN python3 -m pip install --no-cache-dir --ignore-installed \
    dacite \
    ftfy \
    tiktoken \
    openai \
    python-dotenv

# -----
# Rest of deps for the listwise notebook (paths, eval, tokenizer, plots)
# -----
RUN python3 -m pip install --no-cache-dir \
    pandas \
    numpy \
    matplotlib \
    tqdm \
    jupyterlab

# -----
# Diagnostic verification: fail the build if critical imports don't work.
# This catches dependency issues at build time, not runtime.
# -----
RUN python3 -c "\
import sys; \
print('Python:', sys.version); \
import torch; print('torch:', torch.__version__); \
import transformers; print('transformers:', transformers.__version__); \
import tokenizers; print('tokenizers:', tokenizers.__version__); \
import huggingface_hub; print('huggingface_hub:', huggingface_hub.__version__); \
import accelerate; print('accelerate:', accelerate.__version__); \
from transformers import AutoTokenizer; print('AutoTokenizer: ok'); \
from rank_llm.data import Request, Candidate, Query; print('rank_llm.data: ok'); \
from rank_llm.rerank import Reranker; print('rank_llm.rerank.Reranker: ok'); \
from rank_llm.rerank.listwise.rank_fid import RankFiDDistill; print('RankFiDDistill: ok'); \
print('All imports OK')"

# Optional: project code is mounted at /app; ensure shared_scripts can be used
# by running from repo root: docker run -v $(pwd):/app -w /app ...
CMD ["bash", "-lc", "python3 -c \"import torch; print('torch', torch.__version__); print('cuda available', torch.cuda.is_available())\""]
