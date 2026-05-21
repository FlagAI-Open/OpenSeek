#!/usr/bin/env bash
# 统一启动入口：可选择性运行 task1..task8 中的若干个。
#
# 用法：
#   bash scripts/run_all.sh                      # 默认 all（task1..task8）
#   bash scripts/run_all.sh all
#   bash scripts/run_all.sh 1 3 5
#   bash scripts/run_all.sh 1,3,5
#   bash scripts/run_all.sh 3-7
#   bash scripts/run_all.sh 1 5 -- --max_samples 20      # 透传给 main.py
#   bash scripts/run_all.sh 7 -- --mode single --prompt_variant A
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "${SCRIPT_DIR}/.." && pwd )"
cd "${REPO_ROOT}"

usage() {
    cat <<'EOF'
Usage:
  bash scripts/run_all.sh [TASKS] [-- EXTRA_ARGS...]

TASKS:
  all          全部 task1..task8（默认）
  1 3 5        指定多个（空格分隔）
  1,3,5        指定多个（逗号分隔）
  3-7          区间
  组合         如 "1,3 5-7"

EXTRA_ARGS:
  位于 -- 之后的参数会原样透传给每个 task 的 main.py
  例：bash scripts/run_all.sh 1 5 -- --max_samples 20
EOF
}

# ---------- 解析参数 ----------
TASKS_RAW=()
EXTRA_ARGS=()
SAW_DD=0
for a in "$@"; do
    case "$a" in
        -h|--help) usage; exit 0 ;;
    esac
    if [ "$a" = "--" ]; then SAW_DD=1; continue; fi
    if [ "$SAW_DD" = "1" ]; then EXTRA_ARGS+=("$a"); else TASKS_RAW+=("$a"); fi
done
[ ${#TASKS_RAW[@]} -eq 0 ] && TASKS_RAW=("all")

# ---------- 展开 taskid ----------
ids=()
for s in "${TASKS_RAW[@]}"; do
    if [ "$s" = "all" ]; then
        ids+=(1 2 3 4 5 6 7 8); continue
    fi
    s=${s//,/ }
    for t in $s; do
        if [[ "$t" =~ ^[0-9]+-[0-9]+$ ]]; then
            from=${t%-*}; to=${t#*-}
            for i in $(seq "$from" "$to"); do ids+=("$i"); done
        elif [[ "$t" =~ ^[0-9]+$ ]]; then
            ids+=("$t")
        else
            echo "[error] 非法 taskid: $t" >&2
            usage; exit 2
        fi
    done
done

# ---------- 去重保序 + 范围校验 ----------
declare -A seen=()
FINAL=()
for i in "${ids[@]}"; do
    if (( i < 1 || i > 8 )); then
        echo "[error] taskid 必须在 1..8 之间: $i" >&2; exit 2
    fi
    if [ -z "${seen[$i]:-}" ]; then
        seen[$i]=1; FINAL+=("$i")
    fi
done

OUTPUT_DIR="${REPO_ROOT}/outputs"
mkdir -p "${OUTPUT_DIR}"

echo "[plan] 即将运行 tasks: ${FINAL[*]}"
[ ${#EXTRA_ARGS[@]} -gt 0 ] && echo "[plan] 透传参数:    ${EXTRA_ARGS[*]}"
echo "[plan] REPO_ROOT:    ${REPO_ROOT}"
echo "[plan] OUTPUT_DIR:   ${OUTPUT_DIR}"

# ---------- 顺序执行 ----------
for i in "${FINAL[@]}"; do
    echo
    echo "=========================================="
    echo " >>> Task ${i}"
    echo "=========================================="
    T0=$SECONDS

    # task8 首次需离线规整
    if [ "$i" = "8" ]; then
        NORMED="${REPO_ROOT}/src/task8/normalized_data/openseek-8_kernel_generation_normalized.json"
        if [ ! -f "$NORMED" ]; then
            echo "[task8] 规整数据不存在，先运行 normalize_dataset.py ..."
            ( cd "src/task8" && python normalize_dataset.py )
        else
            echo "[task8] 复用已有规整数据：$NORMED"
        fi
    fi

    pushd "src/task${i}" >/dev/null
    if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
        bash run.sh "${EXTRA_ARGS[@]}"
    else
        bash run.sh
    fi
    popd >/dev/null

    DT=$((SECONDS - T0))
    LATEST=$(ls "${OUTPUT_DIR}"/openseek-${i}-v*.jsonl 2>/dev/null \
        | sed -E 's/.*-v([0-9]+)\.jsonl$/\1 &/' | sort -n | tail -1 | cut -d' ' -f2-)
    if [ -n "${LATEST}" ] && [ -f "${LATEST}" ]; then
        echo "[ok] task${i} 完成，耗时 ${DT}s，最新产物：${LATEST}"
    else
        echo "[warn] task${i}: 未找到 openseek-${i}-v*.jsonl 于 ${OUTPUT_DIR}"
    fi
done

echo
echo "=========================================="
echo " 全部任务完成: ${FINAL[*]}"
echo "=========================================="
ls -la "${OUTPUT_DIR}"
