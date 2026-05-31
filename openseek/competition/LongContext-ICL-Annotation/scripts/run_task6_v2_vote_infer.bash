#!/usr/bin/env bash
# =============================================================================
# Task6-v2 原版 + 两个投票融合变体的一键推理（纯 Bash 入口）。
#
# 用法：
#   bash scripts/run_task6_v2_vote_infer.bash
#
# 若你在 IDE 里习惯点「Run Python」打开了 .sh 文件，请改用：
#   python scripts/run_task6_v2_vote_infer.py
#   或运行与本仓库同目录的 scripts/run_task6_v2_vote_infer.sh（该文件为 Python 转发器）。
#
# 运行前可 export 覆盖任意变量，例如：
#   export LIMIT=200 RESUME=1 THINKING=on
#   bash scripts/run_task6_v2_vote_infer.bash
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# ---------------------------------------------------------------------------
# 可调参数（默认值可直接改本文件；也可用环境变量覆盖）
# ---------------------------------------------------------------------------

# Python 解释器
PYTHON="${PYTHON:-python3}"
if ! command -v "${PYTHON}" >/dev/null 2>&1; then
  PYTHON="${PYTHON_CMD:-python}"
fi

# data/ 下 task6 文件名
DATA_FILE="${DATA_FILE:-openseek-6_mnli_same_genre_classification.json}"

# 最多推理条数；<=0 表示全量
LIMIT="${LIMIT:-0}"

# 各版本输出根目录（相对仓库根目录，与 infer_* 内 _resolve_output_dir 一致）
OUT_BASE="${OUT_BASE:-outputs/task6_vote_fusion}"
OUT_V2="${OUT_V2:-${OUT_BASE}/v2_baseline}"
OUT_PAIR_CROSS="${OUT_PAIR_CROSS:-${OUT_BASE}/pair_cross}"
OUT_TASKDEF_JOINT="${OUT_TASKDEF_JOINT:-${OUT_BASE}/taskdef_joint}"

# 重试
RETRIES="${RETRIES:-3}"
RETRY_WAIT_SECONDS="${RETRY_WAIT_SECONDS:-2.0}"

# 是否断点续跑：1 开启，0 关闭
RESUME="${RESUME:-0}"

# 模型 thinking：on / off
THINKING="${THINKING:-off}"

# 分别是否跑三个脚本：1 跑，0 跳过
RUN_V2="${RUN_V2:-1}"
RUN_PAIR_CROSS="${RUN_PAIR_CROSS:-1}"
RUN_TASKDEF_JOINT="${RUN_TASKDEF_JOINT:-1}"

# 变体 pair_cross：交叉句注入最大字符数
CROSS_CONTEXT_MAX_CHARS="${CROSS_CONTEXT_MAX_CHARS:-900}"

# 变体 taskdef_joint：Definition 注入最大字符数
TASKDEF_MAX_CHARS="${TASKDEF_MAX_CHARS:-1200}"

# ---------------------------------------------------------------------------

append_resume() {
  local -n _arr="$1"
  if [[ "${RESUME}" == "1" || "${RESUME}" == "true" || "${RESUME}" == "yes" ]]; then
    _arr+=(--resume)
  fi
}

build_common_args() {
  local -n out="$1"
  out=(
    --data_file "${DATA_FILE}"
    --limit "${LIMIT}"
    --output_dir "$2"
    --retries "${RETRIES}"
    --retry_wait_seconds "${RETRY_WAIT_SECONDS}"
    --thinking "${THINKING}"
  )
  append_resume out
}

echo "=========================================="
echo "[Task6 vote fusion infer] REPO_ROOT=${REPO_ROOT}"
echo "  DATA_FILE=${DATA_FILE}  LIMIT=${LIMIT}"
echo "  RESUME=${RESUME}  THINKING=${THINKING}"
echo "  RETRIES=${RETRIES}  RETRY_WAIT_SECONDS=${RETRY_WAIT_SECONDS}"
echo "  RUN_V2=${RUN_V2}  RUN_PAIR_CROSS=${RUN_PAIR_CROSS}  RUN_TASKDEF_JOINT=${RUN_TASKDEF_JOINT}"
echo "  OUT_V2=${OUT_V2}"
echo "  OUT_PAIR_CROSS=${OUT_PAIR_CROSS}"
echo "  OUT_TASKDEF_JOINT=${OUT_TASKDEF_JOINT}"
echo "  CROSS_CONTEXT_MAX_CHARS=${CROSS_CONTEXT_MAX_CHARS}"
echo "  TASKDEF_MAX_CHARS=${TASKDEF_MAX_CHARS}"
echo "=========================================="

if [[ "${RUN_V2}" == "1" ]]; then
  echo ">>> [1/3] infer_task6_v2.py"
  args=()
  build_common_args args "${OUT_V2}"
  "${PYTHON}" src/infer_task6_v2.py "${args[@]}"
fi

if [[ "${RUN_PAIR_CROSS}" == "1" ]]; then
  echo ">>> [2/3] infer_task6_v2_vote_pair_cross.py"
  args=()
  build_common_args args "${OUT_PAIR_CROSS}"
  args+=(--cross_context_max_chars "${CROSS_CONTEXT_MAX_CHARS}")
  "${PYTHON}" src/infer_task6_v2_vote_pair_cross.py "${args[@]}"
fi

if [[ "${RUN_TASKDEF_JOINT}" == "1" ]]; then
  echo ">>> [3/3] infer_task6_v2_vote_taskdef_joint.py"
  args=()
  build_common_args args "${OUT_TASKDEF_JOINT}"
  args+=(--taskdef_max_chars "${TASKDEF_MAX_CHARS}")
  "${PYTHON}" src/infer_task6_v2_vote_taskdef_joint.py "${args[@]}"
fi

echo "=========================================="
echo "[全部完成] 后处理与 submit 见各输出目录下 jsonl / summary*.json"
echo "=========================================="
