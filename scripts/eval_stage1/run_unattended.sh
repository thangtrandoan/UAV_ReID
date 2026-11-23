#!/usr/bin/env bash
# Unattended Stage 1 (GASNet) Jetson Orin runner with per-stage GitHub auto-push.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

CONFIG_PATH="scripts/eval_stage1/eval_config.yaml"
OUTPUT_DIR="eval_results_stage1"
REMOTE="origin"
TARGET_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
LOCK_CLOCKS=0
STRICT=0
SKIP_TRT=0
NO_PUSH=0
INCLUDE_CODE=0
USE_TMUX=0
TMUX_SESSION="eval_stage1"
MAKE_ZIP=1

usage() {
    cat <<'EOF'
Usage: scripts/eval_stage1/run_unattended.sh [OPTIONS]

Run Stage 1 (GASNet) evaluation unattended on Jetson Orin and automatically
commit & push CSV/JSON/PNG/PDF results + progress logs to GitHub after each stage.

Options:
  --config <path>       Path to eval_config.yaml (default: scripts/eval_stage1/eval_config.yaml)
  --output-dir <dir>    Output directory (default: eval_results_stage1)
  --branch <name>       Target git branch to push to (default: current branch)
  --remote <name>       Git remote name (default: origin)
  --lock-clocks         Run `sudo nvpmodel -m 0 && sudo jetson_clocks` up front
  --strict              Fail if any checkpoint or dataset is missing
  --skip-trt            Skip TensorRT export/benchmark (stages 03c, 01_trt)
  --no-push             Disable automatic git commit & push
  --include-code        Also commit & push scripts/eval_stage1, .gitignore, uav_reid/gasnet/model.py
  --no-zip              Do not create evaluation_bundle_stage1.zip at the end
  --tmux                Run pre-flight checks now, then detach into tmux session `eval_stage1`
  --session <name>      Custom tmux session name (default: eval_stage1)
  -h, --help            Show this help message
EOF
}

RAW_ARGS=("$@")
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) CONFIG_PATH="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --branch) TARGET_BRANCH="$2"; shift 2 ;;
        --remote) REMOTE="$2"; shift 2 ;;
        --lock-clocks) LOCK_CLOCKS=1; shift ;;
        --strict) STRICT=1; shift ;;
        --skip-trt) SKIP_TRT=1; shift ;;
        --no-push) NO_PUSH=1; shift ;;
        --include-code) INCLUDE_CODE=1; shift ;;
        --no-zip) MAKE_ZIP=0; shift ;;
        --tmux) USE_TMUX=1; shift ;;
        --session) TMUX_SESSION="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "[FAIL] Unknown option: $1"; usage; exit 2 ;;
    esac
done

export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib_cache}"

# --- Pre-flight 1: Git & Branch Setup ---
if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "[FAIL] Not inside a git repository: $REPO_ROOT"
    exit 1
fi

CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
if [[ "$TARGET_BRANCH" != "$CURRENT_BRANCH" ]]; then
    if git show-ref --verify --quiet "refs/heads/$TARGET_BRANCH"; then
        git checkout "$TARGET_BRANCH"
    else
        git checkout -b "$TARGET_BRANCH"
    fi
fi

# --- Pre-flight 2: Lock Jetson Clocks (interactive while user is present) ---
if [[ "$LOCK_CLOCKS" -eq 1 ]]; then
    echo "[INFO] Locking Jetson Orin clocks (nvpmodel -m 0 && jetson_clocks)..."
    sudo nvpmodel -m 0 && sudo jetson_clocks
    echo "[OK] Jetson clocks locked."
fi

# --- Pre-flight 3: Verify non-interactive GitHub push auth BEFORE long GPU work ---
if [[ "$NO_PUSH" -eq 0 ]]; then
    if ! git remote get-url "$REMOTE" >/dev/null 2>&1; then
        echo "[FAIL] Git remote '$REMOTE' not found."
        exit 1
    fi
    echo "[INFO] Testing non-interactive push auth to $REMOTE ($TARGET_BRANCH)..."
    if ! GIT_TERMINAL_PROMPT=0 git push --dry-run "$REMOTE" "HEAD:refs/heads/$TARGET_BRANCH" >/dev/null 2>&1; then
        echo "[FAIL] Non-interactive git push to '$REMOTE' failed (credentials not cached or network/auth error)."
        echo "       Run these 2 commands once while at the keyboard, then re-run this script:"
        echo "         git config --global credential.helper store"
        echo "         git push --dry-run $REMOTE $TARGET_BRANCH"
        exit 1
    fi
    echo "[OK] Non-interactive GitHub push verified."
fi

# --- Detached tmux launcher ---
if [[ "$USE_TMUX" -eq 1 ]]; then
    if ! command -v tmux >/dev/null 2>&1; then
        echo "[FAIL] tmux is not installed."
        exit 1
    fi
    if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
        echo "[FAIL] tmux session '$TMUX_SESSION' already exists. Attach with: tmux attach -t $TMUX_SESSION"
        exit 1
    fi
    CHILD_ARGS=()
    for a in "${RAW_ARGS[@]}"; do
        if [[ "$a" != "--tmux" && "$a" != "--lock-clocks" ]]; then
            CHILD_ARGS+=("$a")
        fi
    done
    QUOTED_CMD="cd $(printf '%q' "$REPO_ROOT") && bash $(printf '%q' "$SCRIPT_DIR/run_unattended.sh")"
    for a in "${CHILD_ARGS[@]}"; do
        QUOTED_CMD+=" $(printf '%q' "$a")"
    done
    tmux new-session -d -s "$TMUX_SESSION" "$QUOTED_CMD"
    echo "[OK] Launched unattended Stage 1 evaluation in detached tmux session: $TMUX_SESSION"
    echo "     Monitor live log : tail -f $OUTPUT_DIR/00_env_and_manifest/run_stage1.log"
    echo "     Attach to session: tmux attach -t $TMUX_SESSION"
    echo "     Remote status    : $OUTPUT_DIR/00_env_and_manifest/RUN_STATUS.json on GitHub ($REMOTE/$TARGET_BRANCH)"
    exit 0
fi

# --- Main unattended runner ---
MANIFEST_DIR="$OUTPUT_DIR/00_env_and_manifest"
mkdir -p "$MANIFEST_DIR"
LOG_FILE="$MANIFEST_DIR/run_stage1.log"
STATUS_FILE="$MANIFEST_DIR/RUN_STATUS.json"

log_msg() {
    local msg="$1"
    printf '%s\n' "$msg" | tee -a "$LOG_FILE"
}

write_status_json() {
    local state="$1"
    local cur_stage="$2"
    local completed_csv="$3"
    local failed_stage="$4"
    python3 - "$STATUS_FILE" "$state" "$cur_stage" "$completed_csv" "$failed_stage" "$TARGET_BRANCH" <<'PY'
import datetime as dt
import json
import os
import socket
import subprocess
import sys

status_file, state, cur_stage, completed_csv, failed_stage, branch = sys.argv[1:]
completed = [s for s in completed_csv.split(",") if s]
try:
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
except Exception:
    sha = "unknown"
payload = {
    "status": state,
    "current_stage": cur_stage or None,
    "completed_stages": completed,
    "failed_stage": failed_stage or None,
    "branch": branch,
    "hostname": socket.gethostname(),
    "base_commit_sha": sha,
    "updated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
}
os.makedirs(os.path.dirname(os.path.abspath(status_file)), exist_ok=True)
with open(status_file, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)
PY
}

push_progress() {
    local commit_msg="$1"
    if [[ "$NO_PUSH" -eq 1 ]]; then
        return 0
    fi
    if [[ "$INCLUDE_CODE" -eq 1 ]]; then
        git add .gitignore uav_reid/gasnet/model.py scripts/eval_stage1/ 2>&1 | tee -a "$LOG_FILE" || true
    fi
    git add "$OUTPUT_DIR" 2>&1 | tee -a "$LOG_FILE" || true

    if git diff --cached --quiet; then
        log_msg "[INFO] No new files to commit for: $commit_msg"
    else
        git commit -m "$commit_msg" 2>&1 | tee -a "$LOG_FILE"
    fi

    local attempt=1
    while [[ "$attempt" -le 3 ]]; do
        if GIT_TERMINAL_PROMPT=0 git push "$REMOTE" "HEAD:refs/heads/$TARGET_BRANCH" 2>&1 | tee -a "$LOG_FILE"; then
            log_msg "[OK] Pushed progress to $REMOTE/$TARGET_BRANCH: $commit_msg"
            return 0
        fi
        log_msg "[WARNING] Push attempt $attempt/3 failed; attempting pull --rebase and retry in 10s..."
        GIT_TERMINAL_PROMPT=0 git pull --rebase --autostash "$REMOTE" "$TARGET_BRANCH" 2>&1 | tee -a "$LOG_FILE" || true
        sleep 10
        attempt=$((attempt + 1))
    done
    log_msg "[WARNING] Could not push to $REMOTE/$TARGET_BRANCH after 3 attempts; continuing evaluation."
    return 0
}

STAGES=(00 01 02 03a 03b 03c 01_trt 04 05)
if [[ "$SKIP_TRT" -eq 1 ]]; then
    STAGES=(00 01 02 03a 03b 04 05)
fi

COMPLETED_CSV=""
CURRENT_STAGE=""
RUN_FINISHED=0

on_exit() {
    local rc=$?
    if [[ "$RUN_FINISHED" -eq 0 && "$rc" -ne 0 ]]; then
        log_msg "[FAIL] Evaluation aborted with exit code $rc at stage '${CURRENT_STAGE:-init}'."
        write_status_json "FAILED" "$CURRENT_STAGE" "$COMPLETED_CSV" "${CURRENT_STAGE:-init}"
        push_progress "results(stage1): FAILED at stage ${CURRENT_STAGE:-init}"
    fi
}
trap on_exit EXIT

log_msg "================================================================================"
log_msg "[INFO] Starting unattended Stage 1 evaluation at $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
log_msg "[INFO] Output dir: $OUTPUT_DIR | Branch: $TARGET_BRANCH | Remote: $REMOTE | Skip TRT: $SKIP_TRT"
log_msg "================================================================================"

write_status_json "RUNNING" "init" "" ""
push_progress "results(stage1): start unattended evaluation"

COMMON_FLAGS=(--config "$CONFIG_PATH" --output-dir "$OUTPUT_DIR")
if [[ "$STRICT" -eq 1 ]]; then
    COMMON_FLAGS+=(--strict)
fi

for st in "${STAGES[@]}"; do
    CURRENT_STAGE="$st"
    write_status_json "RUNNING" "$st" "$COMPLETED_CSV" ""
    log_msg ""
    log_msg ">>> [STAGE $st] Started at $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
    python3 -u scripts/eval_stage1/run_all.py "${COMMON_FLAGS[@]}" --stages "$st" 2>&1 | tee -a "$LOG_FILE"
    rc=${PIPESTATUS[0]}
    if [[ "$rc" -ne 0 ]]; then
        exit "$rc"
    fi
    if [[ -z "$COMPLETED_CSV" ]]; then
        COMPLETED_CSV="$st"
    else
        COMPLETED_CSV="$COMPLETED_CSV,$st"
    fi
    write_status_json "RUNNING" "$st" "$COMPLETED_CSV" ""
    push_progress "results(stage1): completed stage $st"
done

log_msg ""
log_msg ">>> [CHECKLIST] Verifying all 5 Stage 1 output requirements..."
python3 - "$OUTPUT_DIR" 2>&1 | tee -a "$LOG_FILE" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, "scripts/eval_stage1")
from run_all import verify_checklist

ok = verify_checklist(Path(sys.argv[1]))
if not ok:
    sys.exit(1)
PY
chk_rc=${PIPESTATUS[0]}
if [[ "$chk_rc" -ne 0 ]]; then
    CURRENT_STAGE="checklist"
    exit "$chk_rc"
fi

if [[ "$MAKE_ZIP" -eq 1 ]] && command -v zip >/dev/null 2>&1; then
    ZIP_PATH="$OUTPUT_DIR/evaluation_bundle_stage1.zip"
    rm -f "$ZIP_PATH"
    zip -r "$ZIP_PATH" "$OUTPUT_DIR" -x "*/engines/*" "*.onnx" "*.engine" "*.engine.meta.json" "*.cache" "*.zip" >/dev/null
    log_msg "[OK] Created lightweight local bundle: $ZIP_PATH ($(du -h "$ZIP_PATH" | cut -f1))"
fi

RUN_FINISHED=1
CURRENT_STAGE=""
write_status_json "COMPLETED" "" "$COMPLETED_CSV" ""
push_progress "results(stage1): COMPLETED all stages on Jetson Orin"
log_msg "[OK] Unattended Stage 1 evaluation COMPLETED at $(date -u +'%Y-%m-%dT%H:%M:%SZ')."
