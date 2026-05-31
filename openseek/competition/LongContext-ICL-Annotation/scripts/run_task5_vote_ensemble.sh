#!/usr/bin/env bash
# =============================================================================
# 本文件是 Bash 脚本，必须用 bash 执行，不要用：python run_task5_vote_ensemble.sh
# 若要用 Python 一键跑，请用：python scripts/run_task5_vote_ensemble.py
# =============================================================================
# 一键顺序跑 Task5 四条投票变体（strip × postemoji，emoji_mode 固定 off）。
# 用法（在任意目录）:
#   bash scripts/run_task5_vote_ensemble.sh
#   bash scripts/run_task5_vote_ensemble.sh --resume
#   PYTHON=python3 bash scripts/run_task5_vote_ensemble.sh --output_dir examples --retrieval_batch_size 4
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
EXTRA=("$@")

echo "[run_task5_vote_ensemble] ROOT=$ROOT PYTHON=$PYTHON"

run_one() {
  local script_relpath="$1"
  echo "==========> ${script_relpath} ${EXTRA[*]}"
  "$PYTHON" "$script_relpath" "${EXTRA[@]}"
}

run_one "src/infer_examples_compare_task5_vote_strip_on_postemoji_on.py"
run_one "src/infer_examples_compare_task5_vote_strip_on_postemoji_off.py"
run_one "src/infer_examples_compare_task5_vote_strip_off_postemoji_on.py"
run_one "src/infer_examples_compare_task5_vote_strip_off_postemoji_off.py"

echo "[run_task5_vote_ensemble] 全部完成。"
