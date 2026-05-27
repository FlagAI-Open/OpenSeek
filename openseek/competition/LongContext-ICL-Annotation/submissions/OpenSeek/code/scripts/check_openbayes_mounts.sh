#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${1:-/openbayes/input/input0}"
DATA_DIR="${2:-/openbayes/input/input1}"

echo "[model] ${MODEL_DIR}"
ls -lah "${MODEL_DIR}"
echo
echo "[model key files]"
find "${MODEL_DIR}" -maxdepth 1 -type f | sort | grep -E 'config.json|tokenizer.json|tokenizer_config.json|model-.*\\.safetensors$|model\\.safetensors\\.index\\.json$' || true
echo
echo "[data] ${DATA_DIR}"
ls -lah "${DATA_DIR}"
echo
echo "[data json files]"
find "${DATA_DIR}" -maxdepth 1 -type f -name 'openseek-*.json' | sort
