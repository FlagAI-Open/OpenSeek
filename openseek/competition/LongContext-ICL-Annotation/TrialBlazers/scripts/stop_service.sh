#!/usr/bin/env bash
set -euo pipefail

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$THIS_DIR/.." && pwd)}"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-/usr/local/python3.11.13/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="python3"
fi

FLAGSCALE_DIR="${FLAGSCALE_DIR:-$REPO_ROOT/FlagScale}"
FLAGSCALE_CONFIG_DIR="${FLAGSCALE_CONFIG_DIR:-$REPO_ROOT/configs}"
FLAGSCALE_CONFIG_NAME="${FLAGSCALE_CONFIG_NAME:-llm_config}"
FLAGSCALE_MODEL_PATH="${FLAGSCALE_MODEL_PATH:-$REPO_ROOT/models/Qwen3-4B}"
FLAGSCALE_EXP_DIR="${FLAGSCALE_EXP_DIR:-$REPO_ROOT/outputs/runs/qwen3_4b_ascend_vllm}"

LOG_DIR="$REPO_ROOT/outputs/logs"
PID_FILES=(
  "$LOG_DIR/flagscale_server.pid"
  "$LOG_DIR/vllm_server.pid"
  "$LOG_DIR/llama_server.pid"
)
STOP_LOG="$LOG_DIR/flagscale_stop.log"
mkdir -p "$LOG_DIR"

is_live_pid() {
  local pid="$1"
  local stat
  if ! kill -0 "$pid" >/dev/null 2>&1; then
    return 1
  fi
  stat="$(ps -p "$pid" -o stat= 2>/dev/null | tr -d '[:space:]')"
  if [ -z "$stat" ] || [[ "$stat" == Z* ]]; then
    return 1
  fi
  return 0
}

export FLAGSCALE_MODEL_PATH
export FLAGSCALE_EXP_DIR
export PYTHONPATH="$FLAGSCALE_DIR:${PYTHONPATH:-}"

if [ -d "$FLAGSCALE_DIR" ]; then
  "$PYTHON_BIN" "$FLAGSCALE_DIR/run.py" \
    --config-path "$FLAGSCALE_CONFIG_DIR" \
    --config-name "$FLAGSCALE_CONFIG_NAME" \
    action=stop >>"$STOP_LOG" 2>&1 || true
fi

for pid_file in "${PID_FILES[@]}"; do
  if [ ! -f "$pid_file" ]; then
    continue
  fi
  PID="$(cat "$pid_file")"
  if is_live_pid "$PID"; then
    kill "$PID" || true
    for _ in $(seq 1 30); do
      if ! is_live_pid "$PID"; then
        break
      fi
      sleep 1
    done
    if is_live_pid "$PID"; then
      kill -9 "$PID" || true
    fi
  fi
  rm -f "$pid_file"
done

pkill -f 'flagscale/serve/run_inference_engine.py' >/dev/null 2>&1 || true
pkill -f 'flagscale/serve/run_fs_serve_vllm.py' >/dev/null 2>&1 || true
pkill -f 'vllm serve' >/dev/null 2>&1 || true
pkill -f 'vllm.entrypoints.openai.api_server' >/dev/null 2>&1 || true

echo "FlagScale/vLLM 服务已停止。"
