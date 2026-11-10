#!/usr/bin/env bash
set -euo pipefail

TASK="${1:-help}"
if [[ $# -gt 0 ]]; then
  shift
fi

python3 - <<'PY'
import torch
print("============================================================")
print("  UAV_ReID Container Environment")
print(f"  PyTorch : {torch.__version__}")
print(f"  CUDA    : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print(f"  BF16    : {torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False}")
print("============================================================")
PY

case "$TASK" in
  gasnet-train)
    exec python3 scripts/train_gasnet.py "$@"
    ;;
  gasnet-eval)
    exec python3 scripts/train_gasnet.py --eval-only "$@"
    ;;
  prepare-data)
    exec python3 scripts/prepare_uav_data.py "$@"
    ;;
  reid-train)
    exec python3 scripts/train_uav_reid.py "$@"
    ;;
  reid-eval)
    exec python3 scripts/eval_uav_reid.py "$@"
    ;;
  reid-calibrate)
    exec python3 scripts/calibrate_threshold.py "$@"
    ;;
  hpo)
    exec python3 scripts/run_hpo.py "$@"
    ;;
  full-pipeline)
    echo ">>> [Step 1/4] Training GASNet on VRU..."
    python3 scripts/train_gasnet.py --config configs/gasnet_vru.yaml "$@"
    echo ">>> [Step 2/4] Training UAVReIDNet on UAV-Anti-UAV..."
    python3 scripts/train_uav_reid.py --config configs/uav_reid.yaml "$@"
    echo ">>> [Step 3/4] Calibrating Fixed-FAR Threshold..."
    python3 scripts/calibrate_threshold.py --config configs/uav_reid.yaml
    echo ">>> [Step 4/4] Final Evaluation on Test Set..."
    exec python3 scripts/eval_uav_reid.py --config configs/uav_reid.yaml
    ;;
  bash|sh)
    exec /bin/bash "$@"
    ;;
  *)
    echo "Usage: docker run --gpus all ... uav-reid-h100:latest <TASK> [ARGS...]"
    echo ""
    echo "Available tasks:"
    echo "  gasnet-train     Train & evaluate GASNet on VRU (configs/gasnet_vru.yaml)"
    echo "  gasnet-eval      Evaluate a saved GASNet checkpoint on VRU"
    echo "  prepare-data     Crop & generate before/after clips from raw UAV-Anti-UAV videos"
    echo "  reid-train       Train 2-stage UAVReIDNet (GASNet + Temporal Encoder)"
    echo "  reid-calibrate   Calibrate Fixed-FAR threshold on test calibration split"
    echo "  reid-eval        Evaluate UAVReIDNet on test split"
    echo "  hpo              Run Optuna TPE + Hyperband hyperparameter search (--stage gasnet|uav_reid)"
    echo "  full-pipeline    Run Stage 1 (VRU) -> Stage 2 (UAV) -> Calibrate -> Eval end-to-end"
    exit 0
    ;;
esac
