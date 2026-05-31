#!/usr/bin/env bash
# 一键脚本：需先启动本地 FlagOS（见 readme.md），并安装 requirements.txt
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"

# 可选：加载 FlagOS 环境变量
if [[ -f "${REPO_ROOT}/configs/env.flagos.example" ]]; then
  set -a
  # shellcheck disable=SC1091
  source <(grep -v '^\s*#' "${REPO_ROOT}/configs/env.flagos.example" | grep -v '^\s*$' | sed 's/^/export /')
  set +a
fi

mkdir -p "$OUTPUT_DIR"
# 若需从头重跑，先删除旧提交文件，否则 main1 可能写出 openseek-*-v2.jsonl：
# rm -f "$OUTPUT_DIR"/openseek-{1,2,3,4,5,7,8}-v*.jsonl
# rm -f "$OUTPUT_DIR"/openseek-2-test_samples-task2-spacy-v5-zeroshot-predictions.jsonl
# rm -f "$OUTPUT_DIR"/openseek-5-taskdata-infer-*.jsonl
# rm -f "$OUTPUT_DIR"/openseek-7-test_samples-task7opt-v3-predictions.jsonl
# rm -f "$OUTPUT_DIR"/openseek-8-test_samples-predictions-v3.jsonl

# ---------------------------------------------------------------------------
# Task 1–5 / 7–8：test_samples 推理 + 后处理（提交用 openseek-{id}-v1.jsonl）
# ---------------------------------------------------------------------------

echo "==> Task 1: main1 test_samples + postprocess"
python src/main1.py --task_start 1 --task_end 1 --log_path_prefix "$OUTPUT_DIR"
TASK1_JSONL="${OUTPUT_DIR}/openseek-1-v1.jsonl"
python src/postprocess_task1_outputs.py --input "$TASK1_JSONL" --inplace

echo "==> Task 2: test_samples spaCy v5 零样本（bs=${TASK2_BS:-1}）"
python src/infer_task2_test_samples_spacy_v5.py \
  --output_dir "$OUTPUT_DIR" \
  --bs "${TASK2_BS:-1}" \
  --resume
TASK2_RAW="${OUTPUT_DIR}/openseek-2-test_samples-task2-spacy-v5-zeroshot-predictions.jsonl"
TASK2_JSONL="${OUTPUT_DIR}/openseek-2-v1.jsonl"
cp -f "$TASK2_RAW" "$TASK2_JSONL"

echo "==> Task 3: main1 test_samples + postprocess"
python src/main1.py --task_start 3 --task_end 3 --log_path_prefix "$OUTPUT_DIR"
TASK3_JSONL="${OUTPUT_DIR}/openseek-3-v1.jsonl"
python src/postprocess_task3_outputs.py --input "$TASK3_JSONL" --inplace

echo "==> Task 4: main1 test_samples + postprocess（规则拼接覆盖）"
python src/main1.py --task_start 4 --task_end 4 --log_path_prefix "$OUTPUT_DIR"
TASK4_JSONL="${OUTPUT_DIR}/openseek-4-v1.jsonl"
TASK4_CHECKED="${OUTPUT_DIR}/openseek-4-v1-checked.jsonl"
python src/postprocess_task4_outputs.py \
  --input "$TASK4_JSONL" \
  --output "$TASK4_CHECKED" \
  --task4-data data/openseek-4_conala_concat_strings.json
mv -f "$TASK4_CHECKED" "$TASK4_JSONL"

echo "==> Task 5: test_samples 零样本推理 + postprocess_task5_outputs"
python src/infer_task5_test_samples.py \
  --output_dir "$OUTPUT_DIR" \
  --task5_emoji_mode off \
  --task5_strip_hashtag on \
  --task5_postemoji off \
  --task5_fp_postprocess off \
  --resume \
  --infer_parallelism "${TASK5_PARALLEL:-8}"
TASK5_RAW="${OUTPUT_DIR}/openseek-5-taskdata-infer-emoji-off-striphash-on.jsonl"
python src/postprocess_task5_outputs.py \
  --input "$TASK5_RAW" \
  --inplace \
  --strip-hashtag \
  --text-calib \
  --prob-ge \
  --task5-json data/openseek-5_semeval_2018_task1_tweet_sadness_detection.json
TASK5_JSONL="${OUTPUT_DIR}/openseek-5-v1.jsonl"
python - <<PY
import json
from pathlib import Path

src = Path("${TASK5_RAW}")
dst = Path("${TASK5_JSONL}")
with src.open(encoding="utf-8") as rf, dst.open("w", encoding="utf-8", newline="\n") as wf:
    for line in rf:
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        sid = str(row.get("test_sample_id") or row.get("sample_id") or "").strip()
        pred = row.get("prediction", row.get("model_output", ""))
        wf.write(
            json.dumps(
                {"test_sample_id": sid, "prediction": "" if pred is None else str(pred)},
                ensure_ascii=False,
            )
            + "\n"
        )
print(f"[Task5 提交格式] {dst}")
PY

echo "==> Task 7: test_samples V3（Jeopardy + hybrid ICL, batch=${TASK7_BATCH_SIZE:-8}）"
python src/infer_task7_test_samples_v3.py \
  --output_dir "$OUTPUT_DIR" \
  --resume \
  --batch_size "${TASK7_BATCH_SIZE:-8}"
TASK7_RAW="${OUTPUT_DIR}/openseek-7-test_samples-task7opt-v3-predictions.jsonl"
TASK7_JSONL="${OUTPUT_DIR}/openseek-7-v1.jsonl"
cp -f "$TASK7_RAW" "$TASK7_JSONL"

echo "==> Task 8: test_samples V3（AST/ReAct, batch=${TASK8_BATCH_SIZE:-8}）"
python src/infer_task8_test_samples_v3.py \
  --output_dir "$OUTPUT_DIR" \
  --resume \
  --retrieval_batch_size "${TASK8_BATCH_SIZE:-8}"
TASK8_RAW="${OUTPUT_DIR}/openseek-8-test_samples-predictions-v3.jsonl"
TASK8_JSONL="${OUTPUT_DIR}/openseek-8-v1.jsonl"
cp -f "$TASK8_RAW" "$TASK8_JSONL"

echo "[Task 1–5 / 7–8 完成] $TASK1_JSONL | $TASK2_JSONL | $TASK3_JSONL | $TASK4_JSONL | $TASK5_JSONL | $TASK7_JSONL | $TASK8_JSONL"

# ---------------------------------------------------------------------------
# Task 1 examples 调参（可选，带金标准确率）
# ---------------------------------------------------------------------------
# python src/infer_examples_main1.py --task_start 1 --task_end 1 --output_dir examples_main1
# python src/postprocess_task1_outputs.py --input examples_main1/openseek-1-examples-main1-compare.jsonl --inplace

# ---------------------------------------------------------------------------
# Task 6：test_samples 投票融合
# ---------------------------------------------------------------------------
python src/infer_task6_v2.py --batch_size 16 --flip_gate not_fiction
python src/infer_task6_v2_vote_taskdef_joint.py --batch_size 16
python src/infer_task6_v2_vote_pair_cross.py --batch_size 16
# 或一键：
python scripts/run_task6_vote_fusion.py
# 环境变量：BATCH_SIZE=16  FLIP_GATE=not_fiction  SKIP_V3=0
