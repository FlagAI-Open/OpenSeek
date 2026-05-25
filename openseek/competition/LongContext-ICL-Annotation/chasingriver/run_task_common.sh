#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${TASK_ID:-}" ]]; then
  echo "错误：缺少 TASK_ID。请使用 run_task1.sh 到 run_task8.sh。" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ROOT="${ROOT:-$SCRIPT_DIR}"
FLAGSCALE_DIR="${FLAGSCALE_DIR:-$ROOT/FlagScale}"
VENV_DIR="${VENV_DIR:-$ROOT/.venv}"
if [[ ! -x "$VENV_DIR/bin/python" && -x "$ROOT/myenv/bin/python" ]]; then
  VENV_DIR="$ROOT/myenv"
fi
VENV="$VENV_DIR/bin/activate"
ENV_PYTHON="$VENV_DIR/bin/python"

PORT="${PORT:-2026}"
BASE_URL="http://127.0.0.1:${PORT}"
GPU_ID="${GPU_ID:-}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-}"

MAX_INPUT_LENGTH="${MAX_INPUT_LENGTH:-32768}"
TOKENIZER_PATH="${TOKENIZER_PATH:-Qwen3-4B}"
DEFAULT_CONFIG_NAME="${DEFAULT_CONFIG_NAME:-llm_config}"
PEER_CONFIG_NAME="${PEER_CONFIG_NAME:-llm_config_peer}"
TASK8_CONFIG_NAME="${TASK8_CONFIG_NAME:-llm_config_task8}"

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUT_BASE="${OUT_BASE:-outputs/run_task${TASK_ID}_${RUN_TAG}}"
TASK_OUT_DIR="$ROOT/$OUT_BASE/task_${TASK_ID}"

DEFAULT_SERVE_LOG="$FLAGSCALE_DIR/outputs/qwen3_4b/serve_logs/host_0_localhost.output"
PEER_SERVE_LOG="$FLAGSCALE_DIR/outputs/qwen3_4b_peer/serve_logs/host_0_localhost.output"
TASK8_SERVE_LOG="$FLAGSCALE_DIR/outputs/qwen3_4b_task8/serve_logs/host_0_localhost.output"
CURRENT_SERVE_LOG="$DEFAULT_SERVE_LOG"

HYDRA_OVERRIDES=()
if [[ -n "$GPU_ID" ]]; then
  HYDRA_OVERRIDES+=("experiment.envs.CUDA_VISIBLE_DEVICES=$GPU_ID")
fi
if [[ -n "$VLLM_GPU_MEMORY_UTILIZATION" ]]; then
  HYDRA_OVERRIDES+=("serve.0.engine_args.gpu_memory_utilization=$VLLM_GPU_MEMORY_UTILIZATION")
fi

activate_env() {
  if [[ -f "$VENV" ]]; then
    # shellcheck disable=SC1090
    source "$VENV"
    return 0
  fi
  if [[ -x "$ENV_PYTHON" ]]; then
    export PATH="$VENV_DIR/bin:$PATH"
    return 0
  fi
  echo "错误：没有找到 Python 环境：$VENV_DIR" >&2
  echo '如果使用 Conda，请先执行：conda activate flagos_chasingriver && VENV_DIR="$CONDA_PREFIX" bash run_task'"$TASK_ID"'.sh' >&2
  return 1
}

check_runtime_paths() {
  if [[ ! -d "$FLAGSCALE_DIR" ]]; then
    echo "错误：没有找到 FlagScale 目录：$FLAGSCALE_DIR" >&2
    echo "如果 FlagScale 不在当前仓库内，请设置 FLAGSCALE_DIR=/path/to/FlagScale。" >&2
    return 1
  fi
  if [[ ! -f "$FLAGSCALE_DIR/run.py" ]]; then
    echo "错误：没有找到 FlagScale 启动文件：$FLAGSCALE_DIR/run.py" >&2
    echo "请将 FlagScale 克隆到 ./FlagScale，或设置 FLAGSCALE_DIR=/path/to/FlagScale。" >&2
    return 1
  fi
  activate_env
}

get_config_for_task() {
  case "$TASK_ID" in
    1|3|4|6|7) echo "$DEFAULT_CONFIG_NAME" ;;
    2|5) echo "$PEER_CONFIG_NAME" ;;
    8) echo "$TASK8_CONFIG_NAME" ;;
    *)
      echo "错误：不支持的任务编号：${TASK_ID}" >&2
      return 1
      ;;
  esac
}

get_main_script_for_task() {
  case "$TASK_ID" in
    1) echo "src/main_task1.py" ;;
    2) echo "src/main_task2.py" ;;
    3) echo "src/main_task3.py" ;;
    4) echo "src/main_task4.py" ;;
    5) echo "src/main_task5.py" ;;
    6) echo "src/main_task6.py" ;;
    7) echo "src/main_task7.py" ;;
    8) echo "src/main_task8.py" ;;
    *)
      echo "错误：不支持的任务编号：${TASK_ID}" >&2
      return 1
      ;;
  esac
}

stop_vllm() {
  cd "$FLAGSCALE_DIR" || return 0
  [[ -f run.py ]] || return 0
  activate_env || true
  python run.py --config-path ../src --config-name "$DEFAULT_CONFIG_NAME" action=stop "${HYDRA_OVERRIDES[@]}" || true
  python run.py --config-path ../src --config-name "$PEER_CONFIG_NAME" action=stop "${HYDRA_OVERRIDES[@]}" || true
  python run.py --config-path ../src --config-name "$TASK8_CONFIG_NAME" action=stop "${HYDRA_OVERRIDES[@]}" || true
}

cleanup() {
  echo
  echo "[清理] 正在停止 vLLM，释放 GPU 显存..."
  stop_vllm
}
start_vllm() {
  local config_name="$1"
  echo "[1/6] 启动 vLLM（${config_name}）"
  if [[ -n "$GPU_ID" ]]; then
    echo "使用 GPU：$GPU_ID"
  fi
  if [[ -n "$VLLM_GPU_MEMORY_UTILIZATION" ]]; then
    echo "vLLM 显存占用比例：$VLLM_GPU_MEMORY_UTILIZATION"
  fi
  if [[ "$config_name" == "$TASK8_CONFIG_NAME" ]]; then
    CURRENT_SERVE_LOG="$TASK8_SERVE_LOG"
  elif [[ "$config_name" == "$PEER_CONFIG_NAME" ]]; then
    CURRENT_SERVE_LOG="$PEER_SERVE_LOG"
  else
    CURRENT_SERVE_LOG="$DEFAULT_SERVE_LOG"
  fi

  cd "$FLAGSCALE_DIR"
  activate_env
  stop_vllm
  rm -f "$CURRENT_SERVE_LOG"
  python run.py --config-path ../src --config-name "$config_name" action=run "${HYDRA_OVERRIDES[@]}"
}

wait_ready() {
  echo "[2/6] 等待端口 ${PORT} 和 /health 就绪"

  for _ in $(seq 1 120); do
    if ss -ltnp 2>/dev/null | grep -q ":${PORT}"; then
      echo "就绪：端口 ${PORT} 已开始监听"
      break
    fi
    sleep 1
  done

  for i in $(seq 1 240); do
    if [[ -f "$CURRENT_SERVE_LOG" ]] && grep -Eq "unrecognized arguments|Traceback \\(most recent call last\\)|NVMLError_|RuntimeError:|ValueError:" "$CURRENT_SERVE_LOG"; then
      echo "错误：vLLM 启动失败，请查看日志：$CURRENT_SERVE_LOG" >&2
      tail -n 40 "$CURRENT_SERVE_LOG" >&2 || true
      return 1
    fi

    code="$(curl -s -o /dev/null -w "%{http_code}" "${BASE_URL}/health" || true)"
    if [[ "$code" == "200" ]]; then
      echo "就绪：/health 返回 200"
      return 0
    fi
    echo "等待 /health... $i/240（http=$code）"
    sleep 2
  done

  echo "错误：vLLM 未就绪（/health 未返回 200）" >&2
  return 1
}

health_info_once() {
  echo "[3/6] 查看 vLLM 模型列表"
  curl -sS "${BASE_URL}/v1/models"
  echo
}

warmup_once() {
  echo "[4/6] 发送一次短请求预热"
  curl -sS "${BASE_URL}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
      "model": "./Qwen3-4B",
      "messages": [{"role": "user", "content": "Hello"}],
      "max_tokens": 8,
      "temperature": 0
    }' >/dev/null 2>&1 || true
}

run_task() {
  local main_script
  main_script="$(get_main_script_for_task)"

  echo "[5/6] 运行任务 ${TASK_ID}"
  cd "$ROOT"
  activate_env

  mkdir -p "$TASK_OUT_DIR"
  echo "使用入口脚本：${main_script}"
  echo "输出目录：${TASK_OUT_DIR}"

  PYTHONUNBUFFERED=1 python -u "$main_script" \
    --task_id "$TASK_ID" \
    --max_input_length "$MAX_INPUT_LENGTH" \
    --log_path_prefix "${OUT_BASE}/task_${TASK_ID}" \
    --tokenizer_path "$TOKENIZER_PATH"
}

show_results() {
  echo
  echo "[6/6] 完成"
  echo "任务 ${TASK_ID} 输出目录："
  echo "$TASK_OUT_DIR"
  echo
  echo "JSONL 结果文件："
  find "$TASK_OUT_DIR" -type f -name "*.jsonl" | sort || true
}

main() {
  local config_name
  config_name="$(get_config_for_task)"

  check_runtime_paths
  trap cleanup EXIT INT TERM
  start_vllm "$config_name"
  wait_ready
  health_info_once
  warmup_once
  run_task
  stop_vllm
  trap - EXIT
  show_results
}

main
