set -ex

PROMPT_TYPE=$1
MODEL_NAME_OR_PATH=$2
MODEL_NAME_SHORT=$(basename $2)
OUTPUT_DIR=${MODEL_NAME_SHORT}/math_eval_sglang

SPLIT="test"
NUM_TEST_SAMPLE=-1

DATA_NAME="gsm8k,math500"
TOKENIZERS_PARALLELISM=false \

python3 -u math_eval.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --data_name ${DATA_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --split ${SPLIT} \
    --prompt_type ${PROMPT_TYPE} \
    --num_test_sample ${NUM_TEST_SAMPLE} \
    --seed 0 \
    --temperature 0.6 \
    --n_sampling 16 \
    --max_tokens_per_call 3072 \
    --top_p 0.95 \
    --start 0 \
    --end 16\
    --use_sglang \
    --pipeline_parallel_size 8 \
    --dp_size 8 \
    --save_outputs \
    --overwrite \
