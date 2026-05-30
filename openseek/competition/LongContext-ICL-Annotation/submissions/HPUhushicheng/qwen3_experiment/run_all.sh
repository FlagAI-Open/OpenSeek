#!/bin/bash
# Batch evaluation script for all 8 tasks.
# Usage: bash run_all.sh [tokenizer_path] [output_dir]

TOKENIZER_PATH=${1:-/root/autodl-tmp/qwen3-4b}
OUTPUT_DIR=${2:-./outputs}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================"
echo "Long-Context ICL Data Annotation"
echo "Tokenizer: ${TOKENIZER_PATH}"
echo "Output:    ${OUTPUT_DIR}"
echo "============================================"

# Run all 8 tasks
for task_id in 1 2 3 4 5 6 7 8; do
    echo ""
    echo "---------- Task ${task_id} ----------"
    python "${SCRIPT_DIR}/main.py" \
        --task_id ${task_id} \
        --max_input_length 128000 \
        --tokenizer_path "${TOKENIZER_PATH}" \
        --log_path_prefix "${OUTPUT_DIR}/" \
        --max_examples 100
done

echo ""
echo "============================================"
echo "All tasks completed!"
echo "Results saved to: ${OUTPUT_DIR}"
echo "============================================"
