#!/usr/bin/env bash
set -euo pipefail

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

MAX_INPUT_LENGTH=32768
TOKENIZER_PATH="Qwen3-4B"
DEFAULT_CONFIG_NAME="llm_config"
PEER_CONFIG_NAME="llm_config_peer"
TASK8_CONFIG_NAME="llm_config_task8"
TASKS_DEFAULT=(1 3 4 6 7)
TASKS_PEER=(2 5)
TASKS_TASK8=(8)

RUN_TAG="$(date +%Y%m%d_%H%M%S)"
OUT_BASE="outputs/run_all_${RUN_TAG}"

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
  if [[ ! -x "$ENV_PYTHON" ]]; then
    echo "错误：没有找到 Python 环境：$VENV_DIR" >&2
    echo '如果使用 Conda，请先执行：conda activate flagos_chasingriver && VENV_DIR="$CONDA_PREFIX" bash run_1to8_full.sh' >&2
    return 1
  fi
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
  local config_label="$2"
  echo "[1/8] 启动 vLLM（${config_label}）"
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
  echo "[2/8] 等待端口 ${PORT} 和 /health 就绪"

  for i in $(seq 1 120); do
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
  echo "[3/8] 查看 vLLM 模型列表"
  curl -sS "${BASE_URL}/v1/models"
  echo
}

warmup_once() {
  echo "[4/8] 发送一次短请求预热"
  curl -sS "${BASE_URL}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
      "model": "../Qwen3-4B",
      "messages": [{"role": "user", "content": "Hello"}],
      "max_tokens": 8,
      "temperature": 0
    }' >/dev/null 2>&1 || true
}

get_main_script_for_task() {
  local task_id="$1"
  case "$task_id" in
    1) echo "src/main_task1.py" ;;
    2) echo "src/main_task2.py" ;;
    3) echo "src/main_task3.py" ;;
    4) echo "src/main_task4.py" ;;
    5) echo "src/main_task5.py" ;;
    6) echo "src/main_task6.py" ;;
    7) echo "src/main_task7.py" ;;
    8) echo "src/main_task8.py" ;;
    *)
      echo "错误：不支持的任务编号：${task_id}" >&2
      return 1
      ;;
  esac
}

start_watchdog() {
  return 0
  local interval="${1:-20}"
  local outdir="$ROOT/$OUT_BASE"

  echo "[监控] 启动监控（间隔 ${interval}s），pid 将保存到 $outdir/watchdog.pid"
  mkdir -p "$outdir"

  (
    while true; do
      echo
      echo "========== 监控 $(date '+%F %T') =========="

      echo "[监控] vLLM 吞吐信息（最新日志）："
      if [[ -f "$SERVE_LOG" ]]; then
        tail -n 30 "$SERVE_LOG" | grep -E "Avg prompt throughput|Avg generation throughput|Running:|Waiting:" | tail -n 5 || true
      else
        echo "未找到服务日志：$SERVE_LOG"
      fi

      echo "[监控] GPU 快照："
      if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu \
          --format=csv,noheader,nounits || true
      else
        echo "未找到 nvidia-smi"
      fi

      echo "[监控] jsonl 输出增长情况（按大小取前 10）："
      if [[ -d "$outdir" ]]; then
        find "$outdir" -type f -name "*.jsonl" -printf "%s %p\n" 2>/dev/null | sort -nr | head -n 10 || true
      else
        echo "输出目录尚不存在：$outdir"
      fi

      sleep "$interval"
    done
  ) &
  echo $! > "$outdir/watchdog.pid"
}

stop_watchdog() {
  return 0
  local outdir="$ROOT/$OUT_BASE"
  if [[ -f "$outdir/watchdog.pid" ]]; then
    local pid
    pid="$(cat "$outdir/watchdog.pid" || true)"
    if [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null; then
      echo "[监控] 停止监控 pid=$pid"
      kill "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
    fi
    rm -f "$outdir/watchdog.pid" || true
  fi
}

run_tasks() {
  local task_group_label="$1"
  shift
  local tasks=("$@")

  echo "[5/8] 运行任务组（${task_group_label}）：${tasks[*]}"
  cd "$ROOT"
  activate_env

  mkdir -p "$OUT_BASE"
  echo "输出目录：$ROOT/$OUT_BASE"

  start_watchdog 20

  for t in "${tasks[@]}"; do
    echo
    echo "==== 任务 ${t} ===="
    mkdir -p "${OUT_BASE}/task_${t}"

    MAIN_SCRIPT="$(get_main_script_for_task "$t")"
    echo "使用入口脚本：${MAIN_SCRIPT}"

    PYTHONUNBUFFERED=1 python -u "${MAIN_SCRIPT}" \
      --task_id "$t" \
      --max_input_length "$MAX_INPUT_LENGTH" \
      --log_path_prefix "${OUT_BASE}/task_${t}" \
      --tokenizer_path "$TOKENIZER_PATH"
  done

  stop_watchdog
}

run_group() {
  local config_name="$1"
  local config_label="$2"
  shift 2
  local tasks=("$@")

  if [[ "${#tasks[@]}" -eq 0 ]]; then
    echo "[跳过] ${config_label} 没有配置任务"
    return 0
  fi

  start_vllm "$config_name" "$config_label"
  wait_ready
  health_info_once
  warmup_once
  run_tasks "$config_label" "${tasks[@]}"
  stop_vllm
}

show_results() {
  echo
  echo "[6/8] jsonl 结果文件："
  find "$ROOT/$OUT_BASE" -type f -name "*.jsonl" | sort || true
}

main() {
  check_runtime_paths
  trap cleanup EXIT INT TERM
  run_group "$DEFAULT_CONFIG_NAME" "默认配置" "${TASKS_DEFAULT[@]}"
  run_group "$PEER_CONFIG_NAME" "备用配置" "${TASKS_PEER[@]}"
  run_group "$TASK8_CONFIG_NAME" "Task 8 专用配置" "${TASKS_TASK8[@]}"
  stop_vllm
  trap - EXIT
  show_results
  echo "[7/8] 全部完成"
}

main
