#!/usr/bin/env bash
set -euo pipefail

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd "$THIS_DIR/.." && pwd)"
REPO_ROOT="${REPO_ROOT:-$CODE_ROOT}"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-/usr/local/python3.11.13/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="python3"
fi

FLAGSCALE_DIR="${FLAGSCALE_DIR:-$REPO_ROOT/FlagScale}"
FLAGSCALE_CONFIG_DIR="${FLAGSCALE_CONFIG_DIR:-$CODE_ROOT/configs}"
FLAGSCALE_CONFIG_NAME="${FLAGSCALE_CONFIG_NAME:-llm_config}"
FLAGSCALE_MODEL_PATH="${FLAGSCALE_MODEL_PATH:-$REPO_ROOT/models/Qwen3-4B}"
FLAGSCALE_EXP_DIR="${FLAGSCALE_EXP_DIR:-$REPO_ROOT/outputs/runs/qwen3_4b_ascend_vllm}"
API_BASE="${API_BASE:-http://127.0.0.1:2026}"

LOG_DIR="$REPO_ROOT/outputs/logs"
PID_FILE="$LOG_DIR/flagscale_server.pid"
START_LOG="$LOG_DIR/flagscale_start.log"
mkdir -p "$LOG_DIR" "$FLAGSCALE_EXP_DIR"

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

check_api() {
  "$PYTHON_BIN" - "$API_BASE" <<'PY' >/dev/null 2>&1
import json
import sys
import urllib.request

api_base = sys.argv[1].rstrip("/")
with urllib.request.urlopen(api_base + "/v1/models", timeout=5) as resp:
    json.loads(resp.read().decode("utf-8"))
PY
}

check_api_stable() {
  local checks="${1:-5}"
  local delay="${2:-2}"
  local i
  for i in $(seq 1 "$checks"); do
    if ! check_api; then
      return 1
    fi
    sleep "$delay"
  done
  return 0
}

if [ ! -d "$FLAGSCALE_DIR" ]; then
  echo "未找到 FlagScale 目录: $FLAGSCALE_DIR"
  exit 1
fi

if [ ! -e "$FLAGSCALE_MODEL_PATH" ]; then
  echo "未找到模型目录或文件: $FLAGSCALE_MODEL_PATH"
  exit 1
fi

if [ -f "$PID_FILE" ]; then
  PID="$(cat "$PID_FILE")"
  if is_live_pid "$PID" && check_api; then
    echo "FlagScale 服务已在运行，PID=$PID"
    exit 0
  fi
  rm -f "$PID_FILE"
fi

export FLAGSCALE_MODEL_PATH
export FLAGSCALE_EXP_DIR
export PYTHONPATH="$FLAGSCALE_DIR:${PYTHONPATH:-}"

echo "[FlagScale] start $(date -Is)" | tee -a "$START_LOG"
echo "[FlagScale] python=$PYTHON_BIN" | tee -a "$START_LOG"
echo "[FlagScale] config=$FLAGSCALE_CONFIG_DIR/$FLAGSCALE_CONFIG_NAME.yaml" | tee -a "$START_LOG"
echo "[FlagScale] model=$FLAGSCALE_MODEL_PATH" | tee -a "$START_LOG"
echo "[FlagScale] exp_dir=$FLAGSCALE_EXP_DIR" | tee -a "$START_LOG"

"$PYTHON_BIN" "$FLAGSCALE_DIR/run.py" \
  --config-path "$FLAGSCALE_CONFIG_DIR" \
  --config-name "$FLAGSCALE_CONFIG_NAME" \
  action=run >>"$START_LOG" 2>&1

FLAGSCALE_PID=""
for _ in $(seq 1 90); do
  if [ -d "$FLAGSCALE_EXP_DIR/serve_logs/pids" ]; then
    FLAGSCALE_PID="$(find "$FLAGSCALE_EXP_DIR/serve_logs/pids" -type f -name '*.pid' -print 2>/dev/null | sort | tail -n 1 | xargs -r cat 2>/dev/null || true)"
  fi
  if [ -n "$FLAGSCALE_PID" ] && is_live_pid "$FLAGSCALE_PID" && check_api_stable 5 2; then
    echo "$FLAGSCALE_PID" > "$PID_FILE"
    echo "FlagScale 服务已启动，PID=$FLAGSCALE_PID"
    echo "API: $API_BASE"
    echo "日志文件: $FLAGSCALE_EXP_DIR/serve_logs/host_0_localhost.output"
    exit 0
  fi
  sleep 2
done

echo "FlagScale 服务启动超时。启动日志: $START_LOG"
if [ -f "$FLAGSCALE_EXP_DIR/serve_logs/host_0_localhost.output" ]; then
  echo "最近服务日志:"
  tail -n 80 "$FLAGSCALE_EXP_DIR/serve_logs/host_0_localhost.output" || true
fi
exit 1
