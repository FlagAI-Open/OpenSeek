#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMP_DIR="$SCRIPT_DIR"
OPENSEEK_ROOT="$(cd "$COMP_DIR/../../.." && pwd)"
FLAGOS_ROOT="$(cd "$OPENSEEK_ROOT/.." && pwd)"
FLAGSCALE_DIR="${FLAGSCALE_DIR:-$FLAGOS_ROOT/FlagScale}"
ENV_FILE="${ENV_FILE:-$COMP_DIR/environment_h20.yml}"
ENV_NAME="${ENV_NAME:-openseek-h20}"
MODEL_DIR="${MODEL_DIR:-$FLAGOS_ROOT/models/Qwen3-4B}"
OUTPUT_DIR="${OUTPUT_DIR:-$COMP_DIR/outputs/first_submission}"
VLLM_HOST="${VLLM_HOST:-0.0.0.0}"
VLLM_PORT="${VLLM_PORT:-2026}"
VLLM_URL="${VLLM_URL:-http://127.0.0.1:${VLLM_PORT}/v1/completions}"
CONTEXT_BUDGET="${CONTEXT_BUDGET:-8192}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-0}"
TORCH_EXTRA_INDEX_URL="${TORCH_EXTRA_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
TORCH_VERSION="${TORCH_VERSION:-2.9.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.24.0}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.9.0}"
VLLM_WHEEL_URL="${VLLM_WHEEL_URL:-https://resource.flagos.net/repository/flagos-pypi-hosted/packages/vllm/0.13.0%2Bfl.0.1.cu128.g72506c983/vllm-0.13.0%2Bfl.0.1.cu128.g72506c983-cp312-cp312-linux_x86_64.whl}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"

usage() {
  cat <<EOF
Usage: $(basename "$0") <command>

Commands:
  setup      Create the conda env, install runtime deps, and print verification info.
  start      Export runtime env vars and start the FlagScale vLLM service.
  api-test   Run src/api_test.py against the configured service endpoint.
  smoke      Run task 1 with sample_limit=5.
  full       Generate the first full submission package.
  env        Print the export commands used by this workflow.

Optional environment overrides:
  ENV_NAME, ENV_FILE, FLAGSCALE_DIR, MODEL_DIR, OUTPUT_DIR
  VLLM_HOST, VLLM_PORT, VLLM_URL, CONTEXT_BUDGET, CUDA_VISIBLE_DEVICES_VALUE
  TORCH_VERSION, TORCHVISION_VERSION, TORCHAUDIO_VERSION, TORCH_EXTRA_INDEX_URL
  VLLM_WHEEL_URL
EOF
}

ensure_conda() {
  if command -v conda >/dev/null 2>&1; then
    return 0
  fi
  echo "conda not found. Please install Miniconda or Anaconda first." >&2
  exit 1
}

activate_conda() {
  ensure_conda
  local conda_base
  conda_base="$(conda info --base)"
  # shellcheck disable=SC1090
  source "${conda_base}/etc/profile.d/conda.sh"
  conda activate "$ENV_NAME"
}

export_runtime_env() {
  export OPENSEEK_TOKENIZER_PATH="$MODEL_DIR"
  export OPENSEEK_MODEL_NAME="$MODEL_DIR"
  export OPENSEEK_VLLM_HOST="$VLLM_HOST"
  export OPENSEEK_VLLM_PORT="$VLLM_PORT"
  export OPENSEEK_VLLM_URL="$VLLM_URL"
  export OPENSEEK_CONTEXT_BUDGET="$CONTEXT_BUDGET"
  export CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_VALUE"
}

print_runtime_env() {
  cat <<EOF
export OPENSEEK_TOKENIZER_PATH="$MODEL_DIR"
export OPENSEEK_MODEL_NAME="$MODEL_DIR"
export OPENSEEK_VLLM_HOST="$VLLM_HOST"
export OPENSEEK_VLLM_PORT="$VLLM_PORT"
export OPENSEEK_VLLM_URL="$VLLM_URL"
export OPENSEEK_CONTEXT_BUDGET="$CONTEXT_BUDGET"
export CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_VALUE"
EOF
}

create_env() {
  ensure_conda
  if [[ ! -f "$ENV_FILE" ]]; then
    echo "Environment file not found: $ENV_FILE" >&2
    exit 1
  fi
  conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME" || conda env create -f "$ENV_FILE"
  activate_conda
  python -V
  pip install -U pip setuptools wheel
}

install_runtime() {
  if [[ ! -d "$FLAGSCALE_DIR" ]]; then
    echo "FlagScale directory not found: $FLAGSCALE_DIR" >&2
    exit 1
  fi
  activate_conda
  cd "$FLAGOS_ROOT"

  pip install -r "$FLAGSCALE_DIR/requirements/common.txt"
  pip install --extra-index-url "$TORCH_EXTRA_INDEX_URL" \
    "torch==${TORCH_VERSION}" \
    "torchaudio==${TORCHAUDIO_VERSION}" \
    "torchvision==${TORCHVISION_VERSION}"
  pip install "vllm @ ${VLLM_WHEEL_URL}"
  pip install transformers==4.57.6 openai==2.29.0
  pip install -e "$FLAGSCALE_DIR"
}

verify_runtime() {
  activate_conda
  python - <<'PY'
import openai
import requests
import torch
import transformers

print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu_count:", torch.cuda.device_count())
    print("gpu0:", torch.cuda.get_device_name(0))
print("transformers:", transformers.__version__)
print("requests:", requests.__version__)
print("openai:", openai.__version__)

import flagscale

print("flagscale ok")
PY
}

start_service() {
  activate_conda
  export_runtime_env
  if [[ ! -d "$MODEL_DIR" ]]; then
    echo "Model directory not found: $MODEL_DIR" >&2
    echo "Download Qwen3-4B first or override MODEL_DIR." >&2
    exit 1
  fi
  cd "$FLAGSCALE_DIR"
  python run.py \
    --config-path ../OpenSeek/openseek/competition/LongContext-ICL-Annotation/src \
    --config-name llm_config \
    action=run
}

run_api_test() {
  activate_conda
  export_runtime_env
  cd "$COMP_DIR/src"
  python api_test.py
}

run_smoke_test() {
  activate_conda
  export_runtime_env
  cd "$COMP_DIR/src"
  python main.py \
    --task_id 1 \
    --sample_limit 5 \
    --tokenizer_path "$MODEL_DIR"
}

run_full_submission() {
  activate_conda
  export_runtime_env
  cd "$COMP_DIR/src"
  python run_all_tasks.py \
    --tokenizer_path "$MODEL_DIR" \
    --output_dir "$OUTPUT_DIR"
}

main() {
  local command="${1:-}"
  case "$command" in
    setup)
      create_env
      install_runtime
      verify_runtime
      print_runtime_env
      ;;
    start)
      start_service
      ;;
    api-test)
      run_api_test
      ;;
    smoke)
      run_smoke_test
      ;;
    full)
      run_full_submission
      ;;
    env)
      print_runtime_env
      ;;
    *)
      usage
      exit 1
      ;;
  esac
}

main "$@"
