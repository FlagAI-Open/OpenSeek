#!/usr/bin/env bash
# task2：依次推理 v5a / v5b / v5c（在 v5 提示词家族上），再与已有 v3、v5 JSONL 做五路多数票融合。
# 用法（在仓库根目录）:
#   bash scripts/run_task2_spacy_v5abc_infer_then_vote.sh
# 可选环境变量: PYTHON（默认 python）、OUT_DIR（默认 examples_main1）
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python}"
OUT="${OUT_DIR:-examples_main1}"

for variant in v5a_enum v5b_syntax v5c_rubric; do
  echo "==========> ${PY} src/infer_examples_main1_task2_spacy_v5.py --prompt_variant ${variant} --output_dir ${OUT}"
  "${PY}" src/infer_examples_main1_task2_spacy_v5.py --prompt_variant "${variant}" --output_dir "${OUT}"
done

echo "==========> node scripts/fuse_task2_main1_spacy_vote.mjs"
node scripts/fuse_task2_main1_spacy_vote.mjs

echo "[run_task2_spacy_v5abc_infer_then_vote] 完成。"
