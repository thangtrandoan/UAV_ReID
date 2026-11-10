# syntax=docker/dockerfile:1.5
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

LABEL org.opencontainers.image.title="uav-reid-h100" \
      org.opencontainers.image.description="Unified GASNet + UAVReIDNet + Optuna HPO for NVIDIA H100"

ARG PIP_NO_CACHE_DIR=1

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=${PIP_NO_CACHE_DIR} \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor \
    TORCH_HOME=/opt/torch_cache \
    HF_HOME=/opt/huggingface_cache \
    PYTHONPATH=/workspace/UAV_ReID

WORKDIR /workspace/UAV_ReID

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p ${TORCHINDUCTOR_CACHE_DIR} ${TORCH_HOME} ${HF_HOME}

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Pre-cache backbone weights so H100 server can run 100% offline
RUN python3 - <<'PY'
from torchvision import models
_ = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
print("Cached ResNet-50 weights")
_ = models.swin_t(weights=models.Swin_T_Weights.DEFAULT)
print("Cached Swin-T weights")
try:
    from transformers import AutoModel
    _ = AutoModel.from_pretrained("facebook/dinov3-convnext-small-pretrain-lvd1689m")
    print("Cached DINOv3 ConvNeXt-Small weights")
except Exception as e:
    print(f"Skipping DINOv3 pre-cache (may require HF_TOKEN): {e}")
PY

COPY uav_reid/ ./uav_reid/
COPY configs/ ./configs/
COPY scripts/ ./scripts/
COPY docker/entrypoint.sh /workspace/entrypoint.sh
RUN chmod +x /workspace/entrypoint.sh ./scripts/*.py

ENTRYPOINT ["/workspace/entrypoint.sh"]
CMD ["--help"]
