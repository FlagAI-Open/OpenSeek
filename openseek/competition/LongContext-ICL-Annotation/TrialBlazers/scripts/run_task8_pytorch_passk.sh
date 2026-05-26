#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="${REPO_ROOT:-$EXPERIMENT_ROOT}"
SRC_DIR="${SRC_DIR:-$EXPERIMENT_ROOT/src}"
cd "$REPO_ROOT"

STAMP="${1:-20260506_t8_pytorch_passk_c1}"
RUN_ROOT="outputs/task8_pytorch_passk_${STAMP}"
LOG_DIR="outputs/logs/${STAMP}"
mkdir -p "$RUN_ROOT" "$LOG_DIR"
exec > >(tee -a "${LOG_DIR}/run.log") 2>&1

echo "[task8-pytorch-passk] start $(date -Is)"
echo "[task8-pytorch-passk] run_root=${RUN_ROOT}"
npu-smi info || true

PYTHON_BIN="${PYTHON_BIN:-/usr/local/python3.11.13/bin/python}"
API_BASE="${API_BASE:-http://127.0.0.1:2026}"
MODEL_NAME="${MODEL_NAME:-Qwen3-4B}"
MAX_WORKERS="${MAX_WORKERS:-4}"
REPEATS="${REPEATS:-2}"
SAMPLE_COUNT="${SAMPLE_COUNT:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
TOP_P="${TOP_P:-0.92}"
TEMPERATURES="${TEMPERATURES:-0.2,0.55}"
BASE_READY="${BASE_READY:-}"
ALT_TASK8_PATHS="${ALT_TASK8_PATHS:-__none__}"
TARGET_IDS_FILE="${TARGET_IDS_FILE:-}"
SEED_TASK8_PATH="${SEED_TASK8_PATH:-}"
ACCEPT_ONLY_SMOKE_OK_IMPROVEMENTS="${ACCEPT_ONLY_SMOKE_OK_IMPROVEMENTS:-0}"
OFFICIAL_MIN_CONTEXT_TASK8="${OFFICIAL_MIN_CONTEXT_TASK8:-0}"
TOKENIZER_PATH="${TOKENIZER_PATH:-$REPO_ROOT/models/Qwen3-4B}"

if [ -z "$BASE_READY" ]; then
  echo "[task8-pytorch-passk] BASE_READY is required. Pass a fresh package directory from the current run."
  exit 2
fi
if [ ! -f "${BASE_READY}/openseek-8-v1.jsonl" ]; then
  echo "[task8-pytorch-passk] missing ${BASE_READY}/openseek-8-v1.jsonl"
  exit 2
fi

EXTRA_ARGS=()
if [ -n "$TARGET_IDS_FILE" ]; then
  EXTRA_ARGS+=(--target-ids-file "$TARGET_IDS_FILE")
fi
if [ -n "$SEED_TASK8_PATH" ]; then
  EXTRA_ARGS+=(--seed-task8-path "$SEED_TASK8_PATH")
fi
if [ "$ACCEPT_ONLY_SMOKE_OK_IMPROVEMENTS" = "1" ]; then
  EXTRA_ARGS+=(--accept-only-smoke-ok-improvements)
fi
if [ "$OFFICIAL_MIN_CONTEXT_TASK8" = "1" ]; then
  EXTRA_ARGS+=(--official-min-context --tokenizer-path "$TOKENIZER_PATH")
fi

"$PYTHON_BIN" "$SRC_DIR/task8_pytorch_passk.py" \
  --base-ready "$BASE_READY" \
  --run-root "$RUN_ROOT" \
  --api-base "$API_BASE" \
  --model-name "$MODEL_NAME" \
  --max-workers "$MAX_WORKERS" \
  --repeats "$REPEATS" \
  --sample-count "$SAMPLE_COUNT" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --top-p "$TOP_P" \
  --temperatures "$TEMPERATURES" \
  --candidate-task8-paths "$ALT_TASK8_PATHS" \
  "${EXTRA_ARGS[@]}"

CAND_RAW="${RUN_ROOT}/candidate_t8_pytorch_passk_raw"
CAND_READY="${RUN_ROOT}/upload_ready_t2multi_t5guarded_t6anchor_t8pytorchpassk_t7router_c6"
mkdir -p "$CAND_RAW"
cp "${BASE_READY}"/openseek-*.jsonl "$CAND_RAW"/
cp "${RUN_ROOT}/selected_task8/openseek-8-v1.jsonl" "$CAND_RAW/openseek-8-v1.jsonl"

"$PYTHON_BIN" "$SRC_DIR/prepare_submission.py" \
  --source-dir "$CAND_RAW" \
  --output-dir "$CAND_READY" \
  --zip-path "${CAND_READY}.zip"

echo "[task8-pytorch-passk] zip entries"
"$PYTHON_BIN" - "${CAND_READY}.zip" <<'PY'
import sys
import zipfile

with zipfile.ZipFile(sys.argv[1]) as zf:
    for name in zf.namelist():
        print(name)
PY

echo "[task8-pytorch-passk] done $(date -Is)"
