#!/bin/bash
# Package evaluation results into a zip file for submission.
# Usage: bash package_results.sh [input_dir] [output_zip]

INPUT_DIR=${1:-./outputs}
OUTPUT_ZIP=${2:-result.zip}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT_DIR="${SCRIPT_DIR}/${INPUT_DIR}"

if [ ! -d "${INPUT_DIR}" ]; then
    echo "Error: Input directory '${INPUT_DIR}' not found!"
    exit 1
fi

echo "Packaging results from: ${INPUT_DIR}"
echo "Output: ${OUTPUT_ZIP}"

cd "${INPUT_DIR}"
zip -j "${SCRIPT_DIR}/${OUTPUT_ZIP}" openseek-*-v1.jsonl

echo ""
echo "Package created: ${SCRIPT_DIR}/${OUTPUT_ZIP}"
echo "Contents:"
unzip -l "${SCRIPT_DIR}/${OUTPUT_ZIP}"
