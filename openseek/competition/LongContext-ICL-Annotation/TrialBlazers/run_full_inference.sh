#!/usr/bin/env bash
set -euo pipefail

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$THIS_DIR}"
cd "$REPO_ROOT"
EXPERIMENT_ROOT="$THIS_DIR"
SRC_DIR="$EXPERIMENT_ROOT/src"
SCRIPT_DIR="$EXPERIMENT_ROOT/scripts"
DATA_DIR="$REPO_ROOT/data/raw"
FALLBACK_DATA_DIR="$REPO_ROOT/third_party/LongContext-ICL-Annotation/data"
if [ ! -d "$DATA_DIR" ] && [ -d "$REPO_ROOT/input" ]; then
  DATA_DIR="$REPO_ROOT/input"
fi
TOKENIZER_PATH="${TOKENIZER_PATH:-${FLAGSCALE_MODEL_PATH:-$REPO_ROOT/models/Qwen3-4B}}"
MAX_INPUT_TOKENS="${MAX_INPUT_TOKENS:-32768}"
RESERVED_GENERATION_TOKENS="${RESERVED_GENERATION_TOKENS:-1024}"

PYTHON_BIN="${PYTHON_BIN:-/usr/local/python3.11.13/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="python3"
fi

API_BASE="${API_BASE:-http://127.0.0.1:2026}"
MODEL_NAME="${MODEL_NAME:-Qwen3-4B}"
STAMP="${STAMP:-$(date +%Y%m%d_independent_repro_%H%M%S)}"
FRESH_RUN="${FRESH_RUN:-1}"
CLIENT_CONCURRENCY="${CLIENT_CONCURRENCY:-2}"
ROUTER_MAX_WORKERS="${ROUTER_MAX_WORKERS:-4}"
TASK134_BRANCH="${TASK134_BRANCH:-solver_debug_loop}"
TASK2_BRANCH="${TASK2_BRANCH:-legacy_router}"
TASK134_LONGCTX_CONCURRENCY="${TASK134_LONGCTX_CONCURRENCY:-$CLIENT_CONCURRENCY}"
TASK134_LONGCTX_MAX_INPUT_TOKENS="${TASK134_LONGCTX_MAX_INPUT_TOKENS:-30000}"
TASK134_LONGCTX_RESERVED_GENERATION_TOKENS="${TASK134_LONGCTX_RESERVED_GENERATION_TOKENS:-512}"
TASK4_DSELECT_ROUTER_WORKERS="${TASK4_DSELECT_ROUTER_WORKERS:-$ROUTER_MAX_WORKERS}"
TASK4_FINAL_ENSEMBLE_WORKERS="${TASK4_FINAL_ENSEMBLE_WORKERS:-$ROUTER_MAX_WORKERS}"
T7_MAX_WORKERS="${T7_MAX_WORKERS:-4}"
T7_BANK_SAMPLES="${T7_BANK_SAMPLES:-1500}"
T7_VAL_SAMPLES="${T7_VAL_SAMPLES:-120}"
T7_TEST_CONFIGS="${T7_TEST_CONFIGS:-4}"
T7_MAIN_CONFIGS="${T7_MAIN_CONFIGS:-2}"
T7_GRID_PRESET="${T7_GRID_PRESET:-fast}"
T7_VALIDATION_MODES="${T7_VALIDATION_MODES:-semantic_mmr}"
T8_MAX_WORKERS="${T8_MAX_WORKERS:-16}"
T8_REPEATS="${T8_REPEATS:-32}"
T8_REPAIR_REPEATS="${T8_REPAIR_REPEATS:-24}"
T8_REPAIR_ROUNDS="${T8_REPAIR_ROUNDS:-3}"
T8_HARD_REPAIR_REPEATS="${T8_HARD_REPAIR_REPEATS:-64}"
T8_HARD_REPAIR_ROUNDS="${T8_HARD_REPAIR_ROUNDS:-2}"
T8_HARD_REPAIR_TEMPERATURES="${T8_HARD_REPAIR_TEMPERATURES:-0.2,0.55,0.8}"
T8_MAX_NEW_TOKENS="${T8_MAX_NEW_TOKENS:-4096}"
MAX_TEST_SAMPLES="${MAX_TEST_SAMPLES:-0}"
TASK8_SAMPLE_COUNT="${TASK8_SAMPLE_COUNT:-$MAX_TEST_SAMPLES}"
CONTEXT_AUDIT_DIR="${CONTEXT_AUDIT_DIR:-}"
CACHE_FRIENDLY_CONTEXT="${CACHE_FRIENDLY_CONTEXT:-0}"
CACHE_PREFIX_CONTEXT_TOKENS="${CACHE_PREFIX_CONTEXT_TOKENS:-0}"
CACHE_PREFIX_CONTEXT_TOKENS_TASK2="${CACHE_PREFIX_CONTEXT_TOKENS_TASK2:-0}"
CACHE_PREFIX_CONTEXT_TOKENS_TASK5="${CACHE_PREFIX_CONTEXT_TOKENS_TASK5:-24000}"
CACHE_PREFIX_CONTEXT_TOKENS_TASK5_AB="${CACHE_PREFIX_CONTEXT_TOKENS_TASK5_AB:-$CACHE_PREFIX_CONTEXT_TOKENS_TASK5}"
CACHE_PREFIX_CONTEXT_TOKENS_TASK6="${CACHE_PREFIX_CONTEXT_TOKENS_TASK6:-24000}"
RUN_ROOT="$REPO_ROOT/outputs/${STAMP}"
if [ -z "$CONTEXT_AUDIT_DIR" ]; then
  CONTEXT_AUDIT_DIR="$RUN_ROOT/context_audit"
fi
LOG_DIR="$REPO_ROOT/outputs/logs"
LOG_FILE="$LOG_DIR/${STAMP}.log"
mkdir -p "$RUN_ROOT" "$LOG_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "[independent-repro] start $(date -Is)"
echo "[independent-repro] repo_root=$REPO_ROOT"
echo "[independent-repro] experiment_root=$EXPERIMENT_ROOT"
echo "[independent-repro] run_root=$RUN_ROOT"
echo "[independent-repro] api_base=$API_BASE model=$MODEL_NAME"
echo "[official-submission] no historical submission/final_outputs files are used as prediction inputs"
echo "[independent-repro] concurrency client=$CLIENT_CONCURRENCY router=$ROUTER_MAX_WORKERS task7=$T7_MAX_WORKERS task8=$T8_MAX_WORKERS"
echo "[independent-repro] task134_branch=$TASK134_BRANCH task2_branch=$TASK2_BRANCH task134_longctx_concurrency=$TASK134_LONGCTX_CONCURRENCY"
echo "[independent-repro] task7_grid_preset=$T7_GRID_PRESET task7_val_samples=$T7_VAL_SAMPLES task7_modes=$T7_VALIDATION_MODES task7_test_configs=$T7_TEST_CONFIGS task7_main_configs=$T7_MAIN_CONFIGS"
echo "[independent-repro] task8_repair_rounds=$T8_REPAIR_ROUNDS task8_hard_repair_rounds=$T8_HARD_REPAIR_ROUNDS task8_hard_repeats=$T8_HARD_REPAIR_REPEATS"
echo "[independent-repro] official_min_context=1 max_test_samples=$MAX_TEST_SAMPLES task8_sample_count=$TASK8_SAMPLE_COUNT context_audit_dir=$CONTEXT_AUDIT_DIR"
echo "[independent-repro] context_budget max_input_tokens=$MAX_INPUT_TOKENS reserved_generation_tokens=$RESERVED_GENERATION_TOKENS"
echo "[independent-repro] cache_friendly_context=$CACHE_FRIENDLY_CONTEXT cache_prefix_context_tokens=$CACHE_PREFIX_CONTEXT_TOKENS"
echo "[independent-repro] task_cache_prefix_tokens task2=$CACHE_PREFIX_CONTEXT_TOKENS_TASK2 task5=$CACHE_PREFIX_CONTEXT_TOKENS_TASK5 task5_ab=$CACHE_PREFIX_CONTEXT_TOKENS_TASK5_AB task6=$CACHE_PREFIX_CONTEXT_TOKENS_TASK6"
npu-smi info || true

if [ "$FRESH_RUN" = "1" ] && [ -d "$RUN_ROOT" ] && find "$RUN_ROOT" -mindepth 1 -print -quit | grep -q .; then
  echo "[independent-repro] FRESH_RUN=1 requires an empty run root, but $RUN_ROOT already has files."
  echo "[independent-repro] Use a new STAMP or set FRESH_RUN=0 only when intentionally resuming the same fresh run."
  exit 2
fi

check_api() {
  "$PYTHON_BIN" - "$API_BASE" <<'PY'
import json
import sys
import urllib.request

api_base = sys.argv[1].rstrip("/")
with urllib.request.urlopen(api_base + "/v1/models", timeout=10) as resp:
    data = json.loads(resp.read().decode("utf-8"))
ids = [item.get("id") for item in data.get("data", [])]
print(json.dumps({"models": ids}, ensure_ascii=False))
PY
}

latest_task_file() {
  local dir="$1"
  local task_id="$2"
  ls -1 "${dir}/openseek-${task_id}-v"*.jsonl | sort | tail -n 1
}

run_main_task() {
  local task_id="$1"
  local output_dir="$2"
  shift 2
  mkdir -p "$output_dir"
  if compgen -G "${output_dir}/openseek-${task_id}-v*.jsonl" >/dev/null; then
    echo "[independent-repro] skip existing task=${task_id} output_dir=${output_dir}"
    return
  fi
  echo "[independent-repro] run task=${task_id} output_dir=${output_dir}"
  extra_args=()
  if [ "$MAX_TEST_SAMPLES" -gt 0 ]; then
    extra_args+=(--max-test-samples "$MAX_TEST_SAMPLES")
  fi
  if [ "$CACHE_FRIENDLY_CONTEXT" = "1" ]; then
    extra_args+=(--cache-friendly-context)
  fi
  local task_cache_prefix_tokens="$CACHE_PREFIX_CONTEXT_TOKENS"
  case "$task_id" in
    2) task_cache_prefix_tokens="$CACHE_PREFIX_CONTEXT_TOKENS_TASK2" ;;
    5) task_cache_prefix_tokens="$CACHE_PREFIX_CONTEXT_TOKENS_TASK5" ;;
    6) task_cache_prefix_tokens="$CACHE_PREFIX_CONTEXT_TOKENS_TASK6" ;;
  esac
  if [ "$task_cache_prefix_tokens" -gt 0 ]; then
    extra_args+=(--cache-prefix-context-tokens "$task_cache_prefix_tokens")
  fi
  "$PYTHON_BIN" "$SRC_DIR/main.py" \
    --backend flagscale \
    --task-id "$task_id" \
    --output-dir "$output_dir" \
    --data-dir "$DATA_DIR" \
    --fallback-data-dir "$FALLBACK_DATA_DIR" \
    --tokenizer-path "$TOKENIZER_PATH" \
    --api-base "$API_BASE" \
    --model-name "$MODEL_NAME" \
    --no-auto-start-api \
    --no-enable-thinking \
    --client-concurrency "$CLIENT_CONCURRENCY" \
    --max-input-tokens "$MAX_INPUT_TOKENS" \
    --reserved-generation-tokens "$RESERVED_GENERATION_TOKENS" \
    --official-min-context \
    --context-audit-dir "$CONTEXT_AUDIT_DIR" \
    "${extra_args[@]}" \
	    "$@"
}

run_task134_manyshot_if_missing() {
  local tasks_csv="$1"
  local output_dir="$2"
  shift 2
  local missing=0
  local task_id
  IFS=',' read -ra task_ids <<< "$tasks_csv"
  for task_id in "${task_ids[@]}"; do
    if [ ! -f "$output_dir/openseek-${task_id}-v1.jsonl" ]; then
      missing=1
    fi
  done
  if [ "$missing" = "0" ]; then
    echo "[independent-repro] skip existing Task1/3/4 manyshot tasks=$tasks_csv output_dir=$output_dir"
    return
  fi

  local sample_args=()
  if [ "$MAX_TEST_SAMPLES" -gt 0 ]; then
    sample_args+=(--max-test-samples "$MAX_TEST_SAMPLES")
  fi

  mkdir -p "$output_dir"
  echo "[independent-repro] run Task1/3/4 manyshot tasks=$tasks_csv output_dir=$output_dir"
  "$PYTHON_BIN" "$SRC_DIR/run_manyshot.py" \
    --tasks "$tasks_csv" \
    --data-dir "$DATA_DIR" \
    --output-dir "$output_dir" \
    --tokenizer-path "$TOKENIZER_PATH" \
    --api-base "$API_BASE" \
    --model-name "$MODEL_NAME" \
    --client-concurrency "$TASK134_LONGCTX_CONCURRENCY" \
    --timeout 240 \
    --write-details \
    "${sample_args[@]}" \
    "$@"
}

run_task134_longctx_manyshot_branch() {
  local combined_out="$1"
  local t134_root="$RUN_ROOT/task134_longctx_manyshot"
  local out28="$t134_root/full_value_cover_anchored_reasoning"
  local out18="$t134_root/full_t4_18k_value_cover_anchored_reasoning"
  local out12="$t134_root/full_t4_12k_value_cover_anchored_reasoning"
  local raw="$t134_root/full_t4_raw_transcribe"
  local line="$t134_root/full_t4_line_transcribe"
  local dselect="$t134_root/full_t4_budget_vote_router_selectonly_disagree_18k12k28k"
  local final="$t134_root/full_t4_weight_vote_18k12k28k_raw_line_dselect_space_len_char_swap_555511_fb18k"

  mkdir -p "$t134_root"
  run_task134_manyshot_if_missing "1,3,4" "$out28" \
    --variant anchored_reasoning \
    --context-mode value_cover \
    --target-example-tokens 28000 \
    --max-input-tokens "$TASK134_LONGCTX_MAX_INPUT_TOKENS" \
    --reserved-generation-tokens "$TASK134_LONGCTX_RESERVED_GENERATION_TOKENS"

  run_task134_manyshot_if_missing "4" "$out18" \
    --variant anchored_reasoning \
    --context-mode value_cover \
    --target-example-tokens 18000 \
    --max-input-tokens 26000 \
    --reserved-generation-tokens "$TASK134_LONGCTX_RESERVED_GENERATION_TOKENS"

  run_task134_manyshot_if_missing "4" "$out12" \
    --variant anchored_reasoning \
    --context-mode value_cover \
    --target-example-tokens 12000 \
    --max-input-tokens 20000 \
    --reserved-generation-tokens "$TASK134_LONGCTX_RESERVED_GENERATION_TOKENS"

  run_task134_manyshot_if_missing "4" "$raw" \
    --variant task4_raw_transcribe \
    --context-mode value_cover \
    --target-example-tokens 0 \
    --max-input-tokens 512 \
    --reserved-generation-tokens "$TASK134_LONGCTX_RESERVED_GENERATION_TOKENS"

  run_task134_manyshot_if_missing "4" "$line" \
    --variant task4_line_transcribe \
    --context-mode value_cover \
    --target-example-tokens 18000 \
    --max-input-tokens 26000 \
    --reserved-generation-tokens "$TASK134_LONGCTX_RESERVED_GENERATION_TOKENS"

  if [ ! -f "$dselect/openseek-4-v1.jsonl" ]; then
    "$PYTHON_BIN" "$SRC_DIR/run_task4_budget_ensemble.py" \
      --output-dir "$dselect" \
      --data-dir "$DATA_DIR" \
      --source "18k=$out18/openseek-4-v1.jsonl" \
      --source "12k=$out12/openseek-4-v1.jsonl" \
      --source "28k=$out28/openseek-4-v1.jsonl" \
      --fallback-order 18k,12k,28k \
      --router-on-disagreement \
      --router-select-only \
      --router-max-new-tokens 32 \
      --api-base "$API_BASE" \
      --model-name "$MODEL_NAME" \
      --client-concurrency "$TASK4_DSELECT_ROUTER_WORKERS"
  fi

  if [ ! -f "$final/openseek-4-v1.jsonl" ]; then
    "$PYTHON_BIN" "$SRC_DIR/run_task4_budget_ensemble.py" \
      --output-dir "$final" \
      --data-dir "$DATA_DIR" \
      --source "18k=$out18/openseek-4-v1.jsonl" \
      --source "12k=$out12/openseek-4-v1.jsonl" \
      --source "28k=$out28/openseek-4-v1.jsonl" \
      --source "raw=$raw/openseek-4-v1.jsonl" \
      --source "line=$line/openseek-4-v1.jsonl" \
      --source "dselect=$dselect/openseek-4-v1.jsonl" \
      --source-weight 18k=5 \
      --source-weight 12k=5 \
      --source-weight 28k=5 \
      --source-weight raw=5 \
      --source-weight line=1 \
      --source-weight dselect=1 \
      --fallback-order 18k,12k,28k,raw,line,dselect \
      --meta-vote-sources 18k,12k,28k,raw,line,dselect \
      --space-free-fallback-order raw,line,12k,18k,28k,dselect \
      --expected-length-fallback-order 18k,12k,28k,raw,line,dselect \
      --char-inventory-fallback-order raw,line,12k,18k,28k,dselect \
      --adjacent-swap-fallback-order 28k,dselect \
      --api-base "$API_BASE" \
      --model-name "$MODEL_NAME" \
      --client-concurrency "$TASK4_FINAL_ENSEMBLE_WORKERS"
  fi

  mkdir -p "$combined_out"
  cp "$out28/openseek-1-v1.jsonl" "$combined_out/openseek-1-v1.jsonl"
  cp "$out28/openseek-3-v1.jsonl" "$combined_out/openseek-3-v1.jsonl"
  cp "$final/openseek-4-v1.jsonl" "$combined_out/openseek-4-v1.jsonl"
  echo "[independent-repro] Task1/3/4 longctx combined_out=$combined_out"
}

TASK5_AB_CACHE_ARGS=()
if [ "$CACHE_PREFIX_CONTEXT_TOKENS_TASK5_AB" -gt 0 ]; then
  TASK5_AB_CACHE_ARGS+=(--cache-prefix-context-tokens "$CACHE_PREFIX_CONTEXT_TOKENS_TASK5_AB")
fi

write_empty_task8() {
  local output_path="$1"
  "$PYTHON_BIN" - "$output_path" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
data = json.loads(Path("data/raw/openseek-8_kernel_generation.json").read_text(encoding="utf-8"))
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", encoding="utf-8") as writer:
    for sample in data["test_samples"]:
        row = {"test_sample_id": str(sample["id"]), "prediction": "", "code": ""}
        writer.write(json.dumps(row, ensure_ascii=False) + "\n")
print(f"[independent-repro] wrote empty Task8 bootstrap: {out}")
PY
}

extract_task8_non_ok_ids() {
  local details_path="$1"
  local ids_path="$2"
  "$PYTHON_BIN" - "$details_path" "$ids_path" <<'PY'
import json
import sys
from pathlib import Path

details_path = Path(sys.argv[1])
ids_path = Path(sys.argv[2])
ids = []
if details_path.exists():
    with details_path.open(encoding="utf-8") as reader:
        for line in reader:
            if not line.strip():
                continue
            row = json.loads(line)
            ok = (
                row.get("selected_exec_ok") is True
                and row.get("selected_smoke_status") == "ok"
                and not row.get("selected_uses_triton")
                and not row.get("selected_issues")
            )
            expected = str(row.get("expected_function_name", ""))
            names = set(row.get("selected_function_names") or [])
            if expected and expected not in names:
                ok = False
            if not ok:
                ids.append(str(row["test_sample_id"]))
ids_path.parent.mkdir(parents=True, exist_ok=True)
ids_path.write_text("\n".join(ids) + ("\n" if ids else ""), encoding="utf-8")
print(json.dumps({"non_ok": len(ids), "ids_path": str(ids_path)}, ensure_ascii=False))
PY
}

verify_task7_reasoning_bank() {
  local bank_path="$1"
  "$PYTHON_BIN" - "$bank_path" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    raise SystemExit(f"missing Task7 reasoning bank: {path}")
rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
if not rows:
    raise SystemExit(f"empty Task7 reasoning bank: {path}")
lengths = [len(str(row.get("thinking", "")).strip()) for row in rows]
none_literal = sum(1 for row in rows if str(row.get("thinking", "")).strip() == "None")
short = sum(1 for length in lengths if length < 80)
summary = {
    "bank_path": str(path),
    "rows": len(rows),
    "none_literal": none_literal,
    "short_thinking": short,
    "min_thinking_len": min(lengths),
    "median_thinking_len": sorted(lengths)[len(lengths) // 2],
    "max_thinking_len": max(lengths),
}
print(json.dumps({"event": "task7_bank_quality", **summary}, ensure_ascii=False))
if none_literal or summary["median_thinking_len"] < 120:
    raise SystemExit(f"Task7 reasoning bank failed quality gate: {json.dumps(summary, ensure_ascii=False)}")
PY
}

select_best_task7_configs() {
  local summary_tsv="$1"
  local selected_tsv="$2"
  "$PYTHON_BIN" - "$summary_tsv" "$selected_tsv" "$T7_TEST_CONFIGS" <<'PY'
import csv
import sys
from pathlib import Path

summary = Path(sys.argv[1])
selected = Path(sys.argv[2])
limit = int(sys.argv[3])
rows = list(csv.DictReader(summary.open(encoding="utf-8"), delimiter="\t"))
rows.sort(key=lambda row: (float(row["accuracy"]), int(row["correct"])), reverse=True)
selected.parent.mkdir(parents=True, exist_ok=True)
with selected.open("w", encoding="utf-8") as writer:
    for row in rows[:limit]:
        writer.write(
            "\t".join(
                [
                    row["mode"],
                    row["answer_rules"],
                    row["lambda"],
                    row["pool"],
                    row["retrieval_k"],
                    row["reasoning_char_limit"],
                ]
            )
            + "\n"
        )
print(f"[independent-repro] selected {min(limit, len(rows))} Task7 configs from validation")
PY
}

check_api

MAIN_OUT="$RUN_ROOT/main_solver_tasks"
case "$TASK134_BRANCH" in
  solver_debug_loop|"")
    echo "[independent-repro] Task1/3/4 deterministic solver tasks"
    run_main_task 1 "$MAIN_OUT" --task1-experiment solver_debug_loop --max-new-tokens 256
    run_main_task 3 "$MAIN_OUT" --task3-experiment solver_debug_loop --max-new-tokens 256
    run_main_task 4 "$MAIN_OUT" --task4-experiment solver_debug_loop --max-new-tokens 256
    ;;
  longctx_manyshot)
    echo "[independent-repro] Task1/3/4 optional long-context many-shot branch"
    MAIN_OUT="$RUN_ROOT/task134_longctx_manyshot/combined_t13_28k_t4_char_swap452"
    run_task134_longctx_manyshot_branch "$MAIN_OUT"
    ;;
  *)
    echo "[independent-repro] unknown TASK134_BRANCH=$TASK134_BRANCH; supported: solver_debug_loop, longctx_manyshot"
    exit 2
    ;;
esac

case "$TASK2_BRANCH" in
  legacy_router)
    echo "[independent-repro] Task2 legacy multi-prompt router"
    TASK2_ROOT="$RUN_ROOT/task2_fresh"
    run_main_task 2 "$TASK2_ROOT/context_rerank" --task2-experiment context_rerank --max-new-tokens 16
    run_main_task 2 "$TASK2_ROOT/noun_biased" --task2-experiment noun_biased --max-new-tokens 16
    run_main_task 2 "$TASK2_ROOT/noun_rules" --task2-experiment noun_rules --max-new-tokens 16
    run_main_task 2 "$TASK2_ROOT/bucket_balanced" --task2-experiment bucket_balanced --max-new-tokens 16
    TASK2_ROUTER="$TASK2_ROOT/router"
    if [ ! -f "$TASK2_ROUTER/openseek-2-v1.jsonl" ]; then
      "$PYTHON_BIN" "$SRC_DIR/task2_candidate_router.py" \
        --dataset-path "$DATA_DIR/openseek-2_count_nouns_verbs.json" \
        --candidate "current=$(latest_task_file "$TASK2_ROOT/noun_biased" 2)" \
        --candidate "context=$(latest_task_file "$TASK2_ROOT/context_rerank" 2)" \
        --candidate "rules=$(latest_task_file "$TASK2_ROOT/noun_rules" 2)" \
        --candidate "bucket=$(latest_task_file "$TASK2_ROOT/bucket_balanced" 2)" \
        --output-dir "$TASK2_ROUTER" \
        --api-base "$API_BASE" \
        --model-name "$MODEL_NAME" \
        --max-workers "$ROUTER_MAX_WORKERS" \
        --max-samples "$MAX_TEST_SAMPLES" \
        --max-new-tokens 8 \
        --prompt-style calibrated
    fi
    ;;
  *)
    echo "[independent-repro] unknown TASK2_BRANCH=$TASK2_BRANCH; supported: legacy_router"
    exit 2
    ;;
esac

echo "[independent-repro] Task5 fresh A/B pass@k candidates plus guarded routers"
TASK5_ROOT="$RUN_ROOT/task5_fresh"
run_main_task 5 "$TASK5_ROOT/baseline_t0" --task5-experiment baseline --max-new-tokens 8 --temperature 0.0 --top-p 0.8
run_main_task 5 "$TASK5_ROOT/strict_t0" --task5-experiment strict_dynamic --max-new-tokens 8 --temperature 0.0 --top-p 0.8
run_main_task 5 "$TASK5_ROOT/baseline_t03" --task5-experiment baseline --max-new-tokens 8 --temperature 0.3 --top-p 0.9
run_main_task 5 "$TASK5_ROOT/strict_t03" --task5-experiment strict_dynamic --max-new-tokens 8 --temperature 0.3 --top-p 0.9
if [ ! -f "$TASK5_ROOT/ab_conservative_k8/openseek-5-v1.jsonl" ]; then
  "$PYTHON_BIN" "$SRC_DIR/task5_ab_labeler.py" \
    --dataset-path "$DATA_DIR/openseek-5_semeval_2018_task1_tweet_sadness_detection.json" \
    --output-dir "$TASK5_ROOT/ab_conservative_k8" \
    --api-base "$API_BASE" \
    --model-name "$MODEL_NAME" \
    --max-workers "$ROUTER_MAX_WORKERS" \
    --max-new-tokens 4 \
    --temperature 0.0 \
    --top-p 0.8 \
    --prompt-style conservative \
    --few-shot 8 \
    --max-samples "$MAX_TEST_SAMPLES" \
    --max-input-tokens "$MAX_INPUT_TOKENS" \
    --reserved-generation-tokens "$RESERVED_GENERATION_TOKENS" \
    --official-min-context \
    --tokenizer-path "$TOKENIZER_PATH" \
    --context-audit-dir "$CONTEXT_AUDIT_DIR" \
    "${TASK5_AB_CACHE_ARGS[@]}"
fi
if [ ! -f "$TASK5_ROOT/ab_conservative_k0/openseek-5-v1.jsonl" ]; then
  "$PYTHON_BIN" "$SRC_DIR/task5_ab_labeler.py" \
    --dataset-path "$DATA_DIR/openseek-5_semeval_2018_task1_tweet_sadness_detection.json" \
    --output-dir "$TASK5_ROOT/ab_conservative_k0" \
    --api-base "$API_BASE" \
    --model-name "$MODEL_NAME" \
    --max-workers "$ROUTER_MAX_WORKERS" \
    --max-new-tokens 4 \
    --temperature 0.0 \
    --top-p 0.8 \
    --prompt-style conservative \
    --few-shot 0 \
    --max-samples "$MAX_TEST_SAMPLES" \
    --max-input-tokens "$MAX_INPUT_TOKENS" \
    --reserved-generation-tokens "$RESERVED_GENERATION_TOKENS" \
    --official-min-context \
    --tokenizer-path "$TOKENIZER_PATH" \
    --context-audit-dir "$CONTEXT_AUDIT_DIR" \
    "${TASK5_AB_CACHE_ARGS[@]}"
fi
if [ ! -f "$TASK5_ROOT/ab_conservative_k16/openseek-5-v1.jsonl" ]; then
  "$PYTHON_BIN" "$SRC_DIR/task5_ab_labeler.py" \
    --dataset-path "$DATA_DIR/openseek-5_semeval_2018_task1_tweet_sadness_detection.json" \
    --output-dir "$TASK5_ROOT/ab_conservative_k16" \
    --api-base "$API_BASE" \
    --model-name "$MODEL_NAME" \
    --max-workers "$ROUTER_MAX_WORKERS" \
    --max-new-tokens 4 \
    --temperature 0.0 \
    --top-p 0.8 \
    --prompt-style conservative \
    --few-shot 16 \
    --max-samples "$MAX_TEST_SAMPLES" \
    --max-input-tokens "$MAX_INPUT_TOKENS" \
    --reserved-generation-tokens "$RESERVED_GENERATION_TOKENS" \
    --official-min-context \
    --tokenizer-path "$TOKENIZER_PATH" \
    --context-audit-dir "$CONTEXT_AUDIT_DIR" \
    "${TASK5_AB_CACHE_ARGS[@]}"
fi
if [ ! -f "$TASK5_ROOT/ab_conservative_k24/openseek-5-v1.jsonl" ]; then
  "$PYTHON_BIN" "$SRC_DIR/task5_ab_labeler.py" \
    --dataset-path "$DATA_DIR/openseek-5_semeval_2018_task1_tweet_sadness_detection.json" \
    --output-dir "$TASK5_ROOT/ab_conservative_k24" \
    --api-base "$API_BASE" \
    --model-name "$MODEL_NAME" \
    --max-workers "$ROUTER_MAX_WORKERS" \
    --max-new-tokens 4 \
    --temperature 0.0 \
    --top-p 0.8 \
    --prompt-style conservative \
    --few-shot 24 \
    --max-samples "$MAX_TEST_SAMPLES" \
    --max-input-tokens "$MAX_INPUT_TOKENS" \
    --reserved-generation-tokens "$RESERVED_GENERATION_TOKENS" \
    --official-min-context \
    --tokenizer-path "$TOKENIZER_PATH" \
    --context-audit-dir "$CONTEXT_AUDIT_DIR" \
    "${TASK5_AB_CACHE_ARGS[@]}"
fi
if [ ! -f "$TASK5_ROOT/ab_recall_k8/openseek-5-v1.jsonl" ]; then
  "$PYTHON_BIN" "$SRC_DIR/task5_ab_labeler.py" \
    --dataset-path "$DATA_DIR/openseek-5_semeval_2018_task1_tweet_sadness_detection.json" \
    --output-dir "$TASK5_ROOT/ab_recall_k8" \
    --api-base "$API_BASE" \
    --model-name "$MODEL_NAME" \
    --max-workers "$ROUTER_MAX_WORKERS" \
    --max-new-tokens 4 \
    --temperature 0.0 \
    --top-p 0.8 \
    --prompt-style recall \
    --few-shot 8 \
    --max-samples "$MAX_TEST_SAMPLES" \
    --max-input-tokens "$MAX_INPUT_TOKENS" \
    --reserved-generation-tokens "$RESERVED_GENERATION_TOKENS" \
    --official-min-context \
    --tokenizer-path "$TOKENIZER_PATH" \
    --context-audit-dir "$CONTEXT_AUDIT_DIR" \
    "${TASK5_AB_CACHE_ARGS[@]}"
fi

TASK5_ROUTE_GUARDED="$TASK5_ROOT/router_abk8_guarded"
if [ ! -f "$TASK5_ROUTE_GUARDED/openseek-5-v1.jsonl" ]; then
  "$PYTHON_BIN" "$SRC_DIR/task5_candidate_router.py" \
    --dataset-path "$DATA_DIR/openseek-5_semeval_2018_task1_tweet_sadness_detection.json" \
    --candidate "current=$(latest_task_file "$TASK5_ROOT/ab_conservative_k8" 5)" \
    --candidate "aggressive=$(latest_task_file "$TASK5_ROOT/baseline_t0" 5)" \
    --candidate "strict=$(latest_task_file "$TASK5_ROOT/strict_t0" 5)" \
    --candidate "ab_recall=$(latest_task_file "$TASK5_ROOT/ab_recall_k8" 5)" \
    --candidate "ab_k16=$(latest_task_file "$TASK5_ROOT/ab_conservative_k16" 5)" \
    --candidate "ab_k24=$(latest_task_file "$TASK5_ROOT/ab_conservative_k24" 5)" \
    --candidate "ab_k0=$(latest_task_file "$TASK5_ROOT/ab_conservative_k0" 5)" \
    --output-dir "$TASK5_ROUTE_GUARDED" \
    --api-base "$API_BASE" \
    --model-name "$MODEL_NAME" \
    --max-workers "$ROUTER_MAX_WORKERS" \
    --max-samples "$MAX_TEST_SAMPLES" \
    --max-new-tokens 8 \
    --include-calibration-examples \
    --prompt-style balanced_recall_guarded
fi
TASK5_ROUTE_BALANCED="$TASK5_ROOT/router_abk8_balanced_recall"
if [ ! -f "$TASK5_ROUTE_BALANCED/openseek-5-v1.jsonl" ]; then
  "$PYTHON_BIN" "$SRC_DIR/task5_candidate_router.py" \
    --dataset-path "$DATA_DIR/openseek-5_semeval_2018_task1_tweet_sadness_detection.json" \
    --candidate "current=$(latest_task_file "$TASK5_ROOT/ab_conservative_k8" 5)" \
    --candidate "aggressive=$(latest_task_file "$TASK5_ROOT/baseline_t0" 5)" \
    --candidate "strict=$(latest_task_file "$TASK5_ROOT/strict_t0" 5)" \
    --candidate "ab_recall=$(latest_task_file "$TASK5_ROOT/ab_recall_k8" 5)" \
    --candidate "ab_k16=$(latest_task_file "$TASK5_ROOT/ab_conservative_k16" 5)" \
    --candidate "ab_k24=$(latest_task_file "$TASK5_ROOT/ab_conservative_k24" 5)" \
    --candidate "ab_k0=$(latest_task_file "$TASK5_ROOT/ab_conservative_k0" 5)" \
    --output-dir "$TASK5_ROUTE_BALANCED" \
    --api-base "$API_BASE" \
    --model-name "$MODEL_NAME" \
    --max-workers "$ROUTER_MAX_WORKERS" \
    --max-samples "$MAX_TEST_SAMPLES" \
    --max-new-tokens 8 \
    --include-calibration-examples \
    --prompt-style balanced_recall
fi
TASK5_ROUTE_CONSERVATIVE="$TASK5_ROOT/router_abk8_conservative"
if [ ! -f "$TASK5_ROUTE_CONSERVATIVE/openseek-5-v1.jsonl" ]; then
  "$PYTHON_BIN" "$SRC_DIR/task5_candidate_router.py" \
    --dataset-path "$DATA_DIR/openseek-5_semeval_2018_task1_tweet_sadness_detection.json" \
    --candidate "current=$(latest_task_file "$TASK5_ROOT/ab_conservative_k8" 5)" \
    --candidate "aggressive=$(latest_task_file "$TASK5_ROOT/baseline_t0" 5)" \
    --candidate "strict=$(latest_task_file "$TASK5_ROOT/strict_t0" 5)" \
    --candidate "ab_recall=$(latest_task_file "$TASK5_ROOT/ab_recall_k8" 5)" \
    --candidate "ab_k16=$(latest_task_file "$TASK5_ROOT/ab_conservative_k16" 5)" \
    --candidate "ab_k24=$(latest_task_file "$TASK5_ROOT/ab_conservative_k24" 5)" \
    --candidate "ab_k0=$(latest_task_file "$TASK5_ROOT/ab_conservative_k0" 5)" \
    --output-dir "$TASK5_ROUTE_CONSERVATIVE" \
    --api-base "$API_BASE" \
    --model-name "$MODEL_NAME" \
    --max-workers "$ROUTER_MAX_WORKERS" \
    --max-samples "$MAX_TEST_SAMPLES" \
    --max-new-tokens 8 \
    --include-calibration-examples \
    --prompt-style conservative
fi
TASK5_ROUTER="$TASK5_ROOT/router_stage2_v2"
if [ ! -f "$TASK5_ROUTER/openseek-5-v1.jsonl" ]; then
  "$PYTHON_BIN" "$SRC_DIR/task5_candidate_router.py" \
    --dataset-path "$DATA_DIR/openseek-5_semeval_2018_task1_tweet_sadness_detection.json" \
    --candidate "current=$(latest_task_file "$TASK5_ROUTE_GUARDED" 5)" \
    --candidate "alt_balanced=$(latest_task_file "$TASK5_ROUTE_BALANCED" 5)" \
    --candidate "alt_conservative=$(latest_task_file "$TASK5_ROUTE_CONSERVATIVE" 5)" \
    --candidate "ab_k8=$(latest_task_file "$TASK5_ROOT/ab_conservative_k8" 5)" \
    --output-dir "$TASK5_ROUTER" \
    --api-base "$API_BASE" \
    --model-name "$MODEL_NAME" \
    --max-workers "$ROUTER_MAX_WORKERS" \
    --max-samples "$MAX_TEST_SAMPLES" \
    --max-new-tokens 8 \
    --include-calibration-examples \
    --prompt-style balanced_recall_guarded_v2
fi

echo "[independent-repro] Task6 fresh multi-prompt candidates"
TASK6_ROOT="$RUN_ROOT/task6_fresh"
run_main_task 6 "$TASK6_ROOT/anchor" --task6-experiment genre_balanced_anchor_hypothesis --max-new-tokens 16
run_main_task 6 "$TASK6_ROOT/genrefirst" --task6-experiment genre_balanced_anchor_genrefirst --max-new-tokens 16
run_main_task 6 "$TASK6_ROOT/hypothesis" --task6-experiment genre_balanced_hypothesis --max-new-tokens 16
run_main_task 6 "$TASK6_ROOT/conservative" --task6-experiment genre_balanced_conservative --max-new-tokens 16
TASK6_ROUTER="$TASK6_ROOT/router"
if [ ! -f "$TASK6_ROUTER/openseek-6-v1.jsonl" ]; then
  "$PYTHON_BIN" "$SRC_DIR/task6_candidate_router.py" \
    --dataset-path "$DATA_DIR/openseek-6_mnli_same_genre_classification.json" \
    --candidate "current=$(latest_task_file "$TASK6_ROOT/anchor" 6)" \
    --candidate "genrefirst=$(latest_task_file "$TASK6_ROOT/genrefirst" 6)" \
    --candidate "hypothesis=$(latest_task_file "$TASK6_ROOT/hypothesis" 6)" \
    --candidate "conservative=$(latest_task_file "$TASK6_ROOT/conservative" 6)" \
    --output-dir "$TASK6_ROUTER" \
    --api-base "$API_BASE" \
    --model-name "$MODEL_NAME" \
    --max-workers "$ROUTER_MAX_WORKERS" \
    --max-samples "$MAX_TEST_SAMPLES" \
    --max-new-tokens 8
fi

echo "[independent-repro] Task7 build fresh train-derived reasoning bank"
TASK7_ROOT="$RUN_ROOT/task7_fresh"
TASK7_BANK_DIR="$TASK7_ROOT/bank"
if [ ! -f "$TASK7_BANK_DIR/successful_reasoning_bank.jsonl" ]; then
"$PYTHON_BIN" "$SRC_DIR/task7_reasoning_retrieval.py" build-bank \
    --dataset-path "$DATA_DIR/openseek-7_jeopardy_answer_generation_all.json" \
    --output-dir "$TASK7_BANK_DIR" \
    --max-train-samples "$T7_BANK_SAMPLES" \
    --max-workers "$T7_MAX_WORKERS" \
    --api-base "$API_BASE" \
    --model-name "$MODEL_NAME" \
    --timeout 240 \
    --max-new-tokens 512 \
    --temperature 0.6 \
    --top-p 0.95 \
    --enable-thinking
fi
verify_task7_reasoning_bank "$TASK7_BANK_DIR/successful_reasoning_bank.jsonl"

echo "[independent-repro] Task7 validation config search from fresh bank"
TASK7_VAL_ROOT="$TASK7_ROOT/validation"
mkdir -p "$TASK7_VAL_ROOT"
TASK7_GRID="$TASK7_VAL_ROOT/grid.tsv"
if [ -z "${T7_VALIDATION_MODES// }" ]; then
  echo "[independent-repro] T7_VALIDATION_MODES must not be empty"
  exit 1
fi
case "$T7_GRID_PRESET" in
  fast)
    cat > "$TASK7_GRID" <<'EOF_GRID'
shortest	0.70	24	3	800
preserve_articles	0.70	24	3	800
shortest	0.75	36	3	1000
preserve_articles	0.70	24	2	800
EOF_GRID
    ;;
  full)
    cat > "$TASK7_GRID" <<'EOF_GRID'
shortest	0.70	24	3	800
preserve_articles	0.70	24	3	800
preserve_articles	0.65	24	3	800
preserve_articles	0.70	36	3	800
preserve_articles	0.70	24	2	800
shortest	0.75	36	3	1000
EOF_GRID
    ;;
  probe)
    cat > "$TASK7_GRID" <<'EOF_GRID'
shortest	0.70	24	3	800
preserve_articles	0.70	24	3	800
EOF_GRID
    ;;
  *)
    if [ -f "$T7_GRID_PRESET" ]; then
      cp "$T7_GRID_PRESET" "$TASK7_GRID"
    else
      echo "[independent-repro] unknown T7_GRID_PRESET=$T7_GRID_PRESET"
      exit 1
    fi
    ;;
esac

while IFS=$'\t' read -r rules lam pool retrieval_k char_limit; do
  safe_lam="${lam/./p}"
  mode_tag="${T7_VALIDATION_MODES// /_}"
  val_dir="$TASK7_VAL_ROOT/val_v${T7_VAL_SAMPLES}_${mode_tag}_rules_${rules}_lam${safe_lam}_pool${pool}_k${retrieval_k}_c${char_limit}"
  if [ -f "$val_dir/summary.json" ]; then
    echo "[independent-repro] skip existing Task7 val $val_dir"
    continue
  fi
  # shellcheck disable=SC2086
  "$PYTHON_BIN" "$SRC_DIR/task7_semantic_tag_retrieval.py" \
    --dataset-path "$DATA_DIR/openseek-7_jeopardy_answer_generation_all.json" \
    --bank-path "$TASK7_BANK_DIR/successful_reasoning_bank.jsonl" \
    --bank-tags-path "$TASK7_VAL_ROOT/bank_semantic_tags.jsonl" \
    --val-tags-path "$TASK7_VAL_ROOT/val_semantic_tags.jsonl" \
    --output-dir "$val_dir" \
    --max-val-samples "$T7_VAL_SAMPLES" \
    --modes $T7_VALIDATION_MODES \
    --semantic-mmr-lambda "$lam" \
    --semantic-mmr-pool "$pool" \
    --semantic-rerank-pool "$pool" \
    --retrieval-k "$retrieval_k" \
    --reasoning-char-limit "$char_limit" \
    --base-few-shot 4 \
    --answer-rules "$rules" \
    --official-min-context \
    --tokenizer-path "$TOKENIZER_PATH" \
    --max-input-tokens "$MAX_INPUT_TOKENS" \
    --reserved-generation-tokens "$RESERVED_GENERATION_TOKENS" \
    --compress-answer \
    --max-workers "$T7_MAX_WORKERS" \
    --api-base "$API_BASE" \
    --model-name "$MODEL_NAME" \
    --timeout 240 \
    --max-new-tokens 1536 \
    --temperature 0.6 \
    --top-p 0.95
  if [ -f "$val_dir/bank_semantic_tags.jsonl" ] && [ ! -f "$TASK7_VAL_ROOT/bank_semantic_tags.jsonl" ]; then
    cp "$val_dir/bank_semantic_tags.jsonl" "$TASK7_VAL_ROOT/bank_semantic_tags.jsonl"
  fi
  if [ -f "$val_dir/val_semantic_tags.jsonl" ] && [ ! -f "$TASK7_VAL_ROOT/val_semantic_tags.jsonl" ]; then
    cp "$val_dir/val_semantic_tags.jsonl" "$TASK7_VAL_ROOT/val_semantic_tags.jsonl"
  fi
done < "$TASK7_GRID"

"$PYTHON_BIN" - "$TASK7_VAL_ROOT" "$T7_VAL_SAMPLES" "${T7_VALIDATION_MODES// /_}" > "$TASK7_VAL_ROOT/validation_summary.tsv" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
val_samples = sys.argv[2]
mode_tag = sys.argv[3]
rows = []
for summary_path in root.glob(f"val_v{val_samples}_{mode_tag}_*/summary.json"):
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    cfg = data["config"]
    for mode, metric in data.items():
        if mode in {"tagging", "config"}:
            continue
        if not isinstance(metric, dict) or "accuracy" not in metric:
            continue
        rows.append(
            (
                metric["accuracy"],
                metric["correct"],
                metric["total"],
                mode,
                cfg.get("answer_rules", "shortest"),
                cfg["semantic_mmr_lambda"],
                cfg["semantic_mmr_pool"],
                cfg["retrieval_k"],
                cfg["reasoning_char_limit"],
                str(summary_path.parent),
            )
        )
rows.sort(reverse=True)
print("accuracy\tcorrect\ttotal\tmode\tanswer_rules\tlambda\tpool\tretrieval_k\treasoning_char_limit\tpath")
for row in rows:
    print("\t".join(str(item) for item in row))
PY
cat "$TASK7_VAL_ROOT/validation_summary.tsv"
select_best_task7_configs "$TASK7_VAL_ROOT/validation_summary.tsv" "$TASK7_VAL_ROOT/selected_test_configs.tsv"

echo "[independent-repro] Task7 generate test candidates from best validation configs"
TASK7_TEST_ROOT="$TASK7_ROOT/test_candidates"
mkdir -p "$TASK7_TEST_ROOT"
TASK7_CANDIDATE_ARGS=()
TASK7_RESCUE_ARGS=()
candidate_idx=0
while IFS=$'\t' read -r mode rules lam pool retrieval_k char_limit; do
  if [ -z "$mode" ]; then
    continue
  fi
  candidate_idx=$((candidate_idx + 1))
  safe_lam="${lam/./p}"
  out_dir="$TASK7_TEST_ROOT/candidate${candidate_idx}_${mode}_${rules}_lam${safe_lam}_pool${pool}_k${retrieval_k}_c${char_limit}"
  run_main_task 7 "$out_dir" \
    --task7-experiment semantic_mmr \
    --task7-retrieval-mode "$mode" \
    --task7-bank-path "$TASK7_BANK_DIR/successful_reasoning_bank.jsonl" \
    --task7-bank-tags-path "$TASK7_VAL_ROOT/bank_semantic_tags.jsonl" \
    --task7-test-tags-path "$TASK7_TEST_ROOT/test_semantic_tags.jsonl" \
    --task7-answer-rules "$rules" \
    --task7-semantic-mmr-lambda "$lam" \
    --task7-semantic-mmr-pool "$pool" \
    --task7-semantic-rerank-pool "$pool" \
    --task7-retrieval-k "$retrieval_k" \
    --task7-reasoning-char-limit "$char_limit" \
    --max-new-tokens 1536 \
    --temperature 0.6 \
    --top-p 0.95 \
    --enable-thinking
  candidate_name="rank${candidate_idx}"
  if [ "$candidate_idx" -le "$T7_MAIN_CONFIGS" ]; then
    TASK7_CANDIDATE_ARGS+=("--candidate" "${candidate_name}=$(latest_task_file "$out_dir" 7)")
  else
    TASK7_RESCUE_ARGS+=("--rescue-candidate" "${candidate_name}=$(latest_task_file "$out_dir" 7)")
  fi
done < "$TASK7_VAL_ROOT/selected_test_configs.tsv"

TASK7_ROUTER="$TASK7_ROOT/router"
if [ ! -f "$TASK7_ROUTER/openseek-7-v1.jsonl" ]; then
  "$PYTHON_BIN" "$SRC_DIR/task7_candidate_router.py" \
    --dataset-path "$DATA_DIR/openseek-7_jeopardy_answer_generation_all.json" \
    "${TASK7_CANDIDATE_ARGS[@]}" \
    "${TASK7_RESCUE_ARGS[@]}" \
    --output-dir "$TASK7_ROUTER" \
    --api-base "$API_BASE" \
    --model-name "$MODEL_NAME" \
    --max-workers "$ROUTER_MAX_WORKERS" \
    --max-samples "$MAX_TEST_SAMPLES" \
    --timeout 180 \
    --max-new-tokens 4 \
    --temperature 0.0 \
    --top-p 0.8 \
    --single-policy keep \
    --router-mode letter \
    --prompt-style plain \
    --rescue-style suspicious_weak
fi

echo "[independent-repro] assemble pre-Task8 package from fresh outputs"
PRE_T8="$RUN_ROOT/pre_t8_base"
rm -rf "$PRE_T8"
mkdir -p "$PRE_T8"
cp "$(latest_task_file "$MAIN_OUT" 1)" "$PRE_T8/openseek-1-v1.jsonl"
cp "$TASK2_ROUTER/openseek-2-v1.jsonl" "$PRE_T8/openseek-2-v1.jsonl"
cp "$(latest_task_file "$MAIN_OUT" 3)" "$PRE_T8/openseek-3-v1.jsonl"
cp "$(latest_task_file "$MAIN_OUT" 4)" "$PRE_T8/openseek-4-v1.jsonl"
cp "$TASK5_ROUTER/openseek-5-v1.jsonl" "$PRE_T8/openseek-5-v1.jsonl"
cp "$TASK6_ROUTER/openseek-6-v1.jsonl" "$PRE_T8/openseek-6-v1.jsonl"
cp "$TASK7_ROUTER/openseek-7-v1.jsonl" "$PRE_T8/openseek-7-v1.jsonl"
write_empty_task8 "$PRE_T8/openseek-8-v1.jsonl"

echo "[independent-repro] Task8 full fresh PyTorch pass@k"
BASE_READY="$PRE_T8" \
ALT_TASK8_PATHS="__none__" \
MAX_WORKERS="$T8_MAX_WORKERS" \
REPEATS="$T8_REPEATS" \
SAMPLE_COUNT="$TASK8_SAMPLE_COUNT" \
MAX_NEW_TOKENS="$T8_MAX_NEW_TOKENS" \
API_BASE="$API_BASE" \
MODEL_NAME="$MODEL_NAME" \
PYTHON_BIN="$PYTHON_BIN" \
REPO_ROOT="$REPO_ROOT" \
SRC_DIR="$SRC_DIR" \
TOKENIZER_PATH="$TOKENIZER_PATH" \
OFFICIAL_MIN_CONTEXT_TASK8=1 \
bash "$SCRIPT_DIR/run_task8_pytorch_passk.sh" "${STAMP}_t8_fullpassk"

CURRENT_T8_RUN="$REPO_ROOT/outputs/task8_pytorch_passk_${STAMP}_t8_fullpassk"
CURRENT_READY="$CURRENT_T8_RUN/upload_ready_t2multi_t5guarded_t6anchor_t8pytorchpassk_t7router_c6"
CURRENT_T8="$CURRENT_T8_RUN/selected_task8/openseek-8-v1.jsonl"

round=1
while [ "$round" -le "$T8_REPAIR_ROUNDS" ]; do
  IDS_FILE="$RUN_ROOT/task8_repair_round${round}_ids.txt"
  extract_task8_non_ok_ids "$CURRENT_T8_RUN/selection_details.jsonl" "$IDS_FILE"
  if [ ! -s "$IDS_FILE" ]; then
    echo "[independent-repro] Task8 all selected candidates passed local validator after round $((round - 1))"
    break
  fi
  echo "[independent-repro] Task8 repair round=${round} ids=$(wc -l < "$IDS_FILE")"
  BASE_READY="$CURRENT_READY" \
  ALT_TASK8_PATHS="__none__" \
  SEED_TASK8_PATH="$CURRENT_T8" \
  ACCEPT_ONLY_SMOKE_OK_IMPROVEMENTS=1 \
  TARGET_IDS_FILE="$IDS_FILE" \
  MAX_WORKERS="$T8_MAX_WORKERS" \
  REPEATS="$T8_REPAIR_REPEATS" \
  SAMPLE_COUNT="$TASK8_SAMPLE_COUNT" \
  MAX_NEW_TOKENS="$T8_MAX_NEW_TOKENS" \
  API_BASE="$API_BASE" \
  MODEL_NAME="$MODEL_NAME" \
  PYTHON_BIN="$PYTHON_BIN" \
  REPO_ROOT="$REPO_ROOT" \
  SRC_DIR="$SRC_DIR" \
  TOKENIZER_PATH="$TOKENIZER_PATH" \
  OFFICIAL_MIN_CONTEXT_TASK8=0 \
  bash "$SCRIPT_DIR/run_task8_pytorch_passk.sh" "${STAMP}_t8_repair_r${round}"
  CURRENT_T8_RUN="$REPO_ROOT/outputs/task8_pytorch_passk_${STAMP}_t8_repair_r${round}"
  CURRENT_READY="$CURRENT_T8_RUN/upload_ready_t2multi_t5guarded_t6anchor_t8pytorchpassk_t7router_c6"
  CURRENT_T8="$CURRENT_T8_RUN/selected_task8/openseek-8-v1.jsonl"
  round=$((round + 1))
done

hard_round=1
while [ "$hard_round" -le "$T8_HARD_REPAIR_ROUNDS" ]; do
  IDS_FILE="$RUN_ROOT/task8_hard_repair_round${hard_round}_ids.txt"
  extract_task8_non_ok_ids "$CURRENT_T8_RUN/selection_details.jsonl" "$IDS_FILE"
  if [ ! -s "$IDS_FILE" ]; then
    echo "[independent-repro] Task8 all selected candidates passed local validator before hard round $hard_round"
    break
  fi
  echo "[independent-repro] Task8 hard repair round=${hard_round} ids=$(wc -l < "$IDS_FILE")"
  BASE_READY="$CURRENT_READY" \
  ALT_TASK8_PATHS="__none__" \
  SEED_TASK8_PATH="$CURRENT_T8" \
  ACCEPT_ONLY_SMOKE_OK_IMPROVEMENTS=1 \
  TARGET_IDS_FILE="$IDS_FILE" \
  MAX_WORKERS="$T8_MAX_WORKERS" \
  REPEATS="$T8_HARD_REPAIR_REPEATS" \
  SAMPLE_COUNT="$TASK8_SAMPLE_COUNT" \
  MAX_NEW_TOKENS="$T8_MAX_NEW_TOKENS" \
  TEMPERATURES="$T8_HARD_REPAIR_TEMPERATURES" \
  API_BASE="$API_BASE" \
  MODEL_NAME="$MODEL_NAME" \
  PYTHON_BIN="$PYTHON_BIN" \
  REPO_ROOT="$REPO_ROOT" \
  SRC_DIR="$SRC_DIR" \
  TOKENIZER_PATH="$TOKENIZER_PATH" \
  OFFICIAL_MIN_CONTEXT_TASK8=0 \
  bash "$SCRIPT_DIR/run_task8_pytorch_passk.sh" "${STAMP}_t8_hard_repair_r${hard_round}"
  CURRENT_T8_RUN="$REPO_ROOT/outputs/task8_pytorch_passk_${STAMP}_t8_hard_repair_r${hard_round}"
  CURRENT_READY="$CURRENT_T8_RUN/upload_ready_t2multi_t5guarded_t6anchor_t8pytorchpassk_t7router_c6"
  CURRENT_T8="$CURRENT_T8_RUN/selected_task8/openseek-8-v1.jsonl"
  hard_round=$((hard_round + 1))
done

if [ ! -d "$CURRENT_READY" ]; then
  echo "[independent-repro] missing final Task8 ready directory: $CURRENT_READY"
  exit 1
fi

echo "[independent-repro] Task8 final full-validator rescore"
FINAL_T8_VALIDATOR="$RUN_ROOT/task8_final_validator"
rm -rf "$FINAL_T8_VALIDATOR"
"$PYTHON_BIN" "$SRC_DIR/task8_pytorch_passk.py" \
  --base-ready "$CURRENT_READY" \
  --run-root "$FINAL_T8_VALIDATOR" \
  --api-base "$API_BASE" \
  --model-name "$MODEL_NAME" \
  --repeats 0 \
  --sample-count "$TASK8_SAMPLE_COUNT" \
  --candidate-task8-paths "__none__" \
  --no-generate
"$PYTHON_BIN" - "$FINAL_T8_VALIDATOR/summary.json" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
samples = int(summary["samples"])
ok_smoke = int(summary.get("selected_smoke_counts", {}).get("ok", 0))
ok_exec = int(summary.get("selected_exec_counts", {}).get("ok", 0))
static_bad = int(summary.get("selected_static_bad_rows", -1))
function_match = int(summary.get("selected_function_match_rows", -1))
bad = {
    "samples": samples,
    "ok_smoke": ok_smoke,
    "ok_exec": ok_exec,
    "static_bad": static_bad,
    "function_match": function_match,
}
print(json.dumps({"event": "task8_final_validator_gate", **bad}, ensure_ascii=False))
if ok_smoke != samples or ok_exec != samples or static_bad != 0 or function_match != samples:
    raise SystemExit("Task8 final validator gate failed: " + json.dumps(bad, ensure_ascii=False))
PY

FINAL_RAW="$RUN_ROOT/final_raw"
FINAL_READY="$REPO_ROOT/outputs/upload_ready_${STAMP}"
FINAL_ZIP="${FINAL_READY}.zip"
rm -rf "$FINAL_RAW" "$FINAL_READY" "$FINAL_ZIP"
mkdir -p "$FINAL_RAW"
cp "$CURRENT_READY"/openseek-*.jsonl "$FINAL_RAW"/

"$PYTHON_BIN" "$SRC_DIR/prepare_submission.py" \
  --source-dir "$FINAL_RAW" \
  --output-dir "$FINAL_READY" \
  --zip-path "$FINAL_ZIP"

echo "[independent-repro] final zip entries"
"$PYTHON_BIN" - "$FINAL_ZIP" <<'PY'
import sys
import zipfile

with zipfile.ZipFile(sys.argv[1]) as zf:
    for name in zf.namelist():
        print(name)
PY

echo "[independent-repro] sha256"
sha256sum "$FINAL_ZIP" "$FINAL_READY"/openseek-*.jsonl

echo "[independent-repro] final_ready=$FINAL_READY"
echo "[independent-repro] final_zip=$FINAL_ZIP"
echo "[independent-repro] log=$LOG_FILE"
echo "[independent-repro] done $(date -Is)"
