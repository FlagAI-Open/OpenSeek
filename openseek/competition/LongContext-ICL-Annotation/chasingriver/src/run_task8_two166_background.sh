#!/usr/bin/env bash
set -u

ROOT="/home/cenzihan/OpenSeek/openseek/competition/LongContext-ICL-Annotation/flagos"
PY="/home/cenzihan/conda_envs/flagscale311/bin/python"
PORT="${PORT:-2031}"
GPU="${GPU:-1}"
UTIL="${UTIL:-0.35}"
SERVICE_URL="http://127.0.0.1:${PORT}/v1/completions"
VLLM_LOG="${ROOT}/outputs/service_logs/qwen3_4b_two166_${PORT}.log"
RUN_LOG="${ROOT}/outputs/two166_generate.log"

mkdir -p "${ROOT}/outputs/service_logs"
cd "${ROOT}/src" || exit 1

vllm_pid=""

stop_vllm() {
  if [[ -n "${vllm_pid}" ]] && kill -0 "${vllm_pid}" 2>/dev/null; then
    kill -TERM "${vllm_pid}" 2>/dev/null || true
    sleep 3
    kill -KILL "${vllm_pid}" 2>/dev/null || true
  fi
}

trap stop_vllm EXIT

start_vllm_once() {
  : > "${VLLM_LOG}"
  CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" -m vllm.entrypoints.openai.api_server \
    --model ../../Qwen3-4B \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --dtype bfloat16 \
    --gpu-memory-utilization "${UTIL}" \
    --max-model-len 32768 \
    --max-num-batched-tokens 32768 \
    --max-num-seqs 1 \
    --enforce-eager \
    --trust-remote-code \
    --served-model-name ../Qwen3-4B Qwen3-4B \
    --disable-log-requests \
    > "${VLLM_LOG}" 2>&1 &
  vllm_pid="$!"
}

wait_ready() {
  for _ in $(seq 1 150); do
    if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null; then
      return 0
    fi
    if ! kill -0 "${vllm_pid}" 2>/dev/null; then
      return 1
    fi
    sleep 2
  done
  return 1
}

{
  echo "=== two166 background start $(date) ==="
  echo "GPU=${GPU} PORT=${PORT} UTIL=${UTIL}"
  for attempt in $(seq 1 8); do
    echo "=== vLLM attempt ${attempt} ==="
    start_vllm_once
    if wait_ready; then
      echo "=== vLLM ready on ${PORT} ==="
      OPENSEEK_TASK8_SERVICE_URL="${SERVICE_URL}" PYTHONPATH="${ROOT}/src" "${PY}" "${ROOT}/src/run_task8_two166.py"
      status="$?"
      echo "=== generation exit ${status} $(date) ==="
      exit "${status}"
    fi
    echo "=== vLLM attempt ${attempt} failed; tail log ==="
    tail -80 "${VLLM_LOG}" || true
    stop_vllm
    vllm_pid=""
    sleep 10
  done
  echo "=== all vLLM attempts failed ==="
  exit 1
} >> "${RUN_LOG}" 2>&1
