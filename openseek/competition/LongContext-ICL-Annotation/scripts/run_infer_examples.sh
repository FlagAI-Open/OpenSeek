#!/usr/bin/env bash
# 在 examples 集上评测准确率（带金标，用于调参）
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

if [[ -f "$REPO_ROOT/configs/env.flagos.example" ]]; then
  set -a
  # shellcheck disable=SC1091
  source <(grep -v '^\s*#' "$REPO_ROOT/configs/env.flagos.example" | grep -v '^\s*$' | sed 's/^/export /')
  set +a
fi

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

TASK_START="${TASK_START:-1}"
TASK_END="${TASK_END:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-examples_main1}"

python scripts/api_test_flagos.py

python src/infer_examples_main1.py \
  --task_start "$TASK_START" \
  --task_end "$TASK_END" \
  --output_dir "$OUTPUT_DIR" \
  ${TOKENIZER_PATH:+--tokenizer_path "$TOKENIZER_PATH"}

echo "对比结果: $REPO_ROOT/$OUTPUT_DIR/openseek-${TASK_START}-examples-main1-compare.jsonl"
