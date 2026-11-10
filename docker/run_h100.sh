#!/usr/bin/env bash
# ==============================================================================
# Script chạy trên server NVIDIA H100 (hỗ trợ cả Docker lẫn Bare-metal/Conda/Slurm)
# ==============================================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Tối ưu biến môi trường cho H100 (Hopper架构, TF32, PyTorch Inductor)
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/tmp/torchinductor_$USER}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=8

ACTION="${1:-help}"
shift || true

case "$ACTION" in
  build-docker)
    echo "[INFO] Building Docker image uav-reid-h100:latest..."
    docker build -f docker/Dockerfile -t uav-reid-h100:latest .
    ;;

  stage1-gasnet)
    echo "[INFO] Running Stage 1: GASNet on VRU (H100 BF16 + torch.compile)..."
    python3 scripts/train_gasnet.py --config configs/gasnet_vru.yaml "$@"
    ;;

  stage2-reid)
    echo "[INFO] Running Stage 2: UAVReIDNet on UAV-Anti-UAV..."
    python3 scripts/train_uav_reid.py --config configs/uav_reid.yaml "$@"
    ;;

  hpo-gasnet)
    echo "[INFO] Running Optuna Hyperparameter Search for GASNet..."
    python3 scripts/run_hpo.py --stage gasnet --config configs/hpo_gasnet.yaml "$@"
    ;;

  hpo-reid)
    echo "[INFO] Running Optuna Hyperparameter Search for UAVReIDNet..."
    python3 scripts/run_hpo.py --stage uav_reid --config configs/hpo_uav_reid.yaml "$@"
    ;;

  eval-all)
    echo "[INFO] Calibrating Fixed-FAR threshold & Evaluating UAVReIDNet..."
    python3 scripts/calibrate_threshold.py --config configs/uav_reid.yaml "$@"
    python3 scripts/eval_uav_reid.py --config configs/uav_reid.yaml "$@"
    ;;

  *)
    echo "Usage: ./docker/run_h100.sh <ACTION> [EXTRA_ARGS...]"
    echo ""
    echo "Actions:"
    echo "  build-docker     Build Docker image for H100 server"
    echo "  stage1-gasnet    Train & evaluate GASNet on VRU (--set paths.data_root=/path/to/data)"
    echo "  stage2-reid      Train 2-stage UAVReIDNet (--set paths.data_dir=/path/to/processed)"
    echo "  hpo-gasnet       Run Optuna HPO for GASNet (--n-trials 20)"
    echo "  hpo-reid         Run Optuna HPO for UAVReIDNet (--n-trials 25)"
    echo "  eval-all         Run threshold calibration + full test evaluation"
    ;;
esac
