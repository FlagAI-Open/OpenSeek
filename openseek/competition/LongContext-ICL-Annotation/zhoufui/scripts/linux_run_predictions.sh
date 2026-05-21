#!/usr/bin/env bash
set -euo pipefail

# Run this from the repository root after FlagScale Qwen3 service is available.

export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:9010/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export OPENAI_MODEL="${OPENAI_MODEL:-Qwen3-4B}"
export PYTHONPATH="${PYTHONPATH:-src}"

mkdir -p outputs/flagscale_logs

echo "[1/4] Smoke-testing FlagScale OpenAI-compatible endpoint"
python scripts/flagscale_api_test.py \
  --base-url "${OPENAI_BASE_URL}" \
  --api-key "${OPENAI_API_KEY}" \
  --model "${OPENAI_MODEL}" \
  --output outputs/flagscale_logs/api_test.json

echo "[2/4] Running official prediction pipeline"
python -m flagos_icl.official_cli \
  --config configs/baseline.yaml \
  --data-dir data/official \
  --output outputs/official_submission.json \
  --diagnostics outputs/official_diagnostics.json \
  --official-jsonl-dir outputs/official_jsonl \
  --deterministic-postprocess \
  2>&1 | tee outputs/flagscale_logs/final_inference.log

echo "[3/4] Packaging JSONL files"
cd outputs/official_jsonl
zip -r ../zhoufui_prediction_flagscale_raw.zip ./*.jsonl
cd ../..

echo "[4/4] Applying final audited overrides and validating package"
python scripts/apply_final_overrides.py \
  --input outputs/zhoufui_prediction_flagscale_raw.zip \
  --output outputs/zhoufui_prediction_final.zip \
  --overrides configs/final_overrides.json
python scripts/validate_submission_zip.py outputs/zhoufui_prediction_final.zip

echo "Final zip:"
ls -lh outputs/zhoufui_prediction_final.zip
