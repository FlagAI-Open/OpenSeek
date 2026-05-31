#!/usr/bin/env bash
# 一键：加载环境变量 → 测 API → 跑 run.sh 完整提交流水线（Task 1–8）
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

echo "==> [1/2] API 连通性测试"
python scripts/api_test_flagos.py

OUTPUT_DIR="${OUTPUT_DIR:-outputs}"
echo "==> [2/2] run.sh 完整提交 -> ${OUTPUT_DIR}"
bash "$REPO_ROOT/run.sh"

echo "完成。提交文件见: $REPO_ROOT/$OUTPUT_DIR/openseek-*-v1.jsonl"
