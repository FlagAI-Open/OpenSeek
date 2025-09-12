set -ex

PROMPT_TYPE="prompt_gsm8k"
MODEL_NAME_OR_PATH=$1
MODEL_NAME_SHORT=$(basename $1)
OUTPUT_DIR="eval_final"

SPLIT="test"
NUM_TEST_SAMPLE=-1

DATA_NAME="gsm8k,math500,minerva_math,aime24,olympiadbench"
TOKENIZERS_PARALLELISM=false \

python3 -u math_eval.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --data_name ${DATA_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --split ${SPLIT} \
    --prompt_type ${PROMPT_TYPE} \
    --num_test_sample ${NUM_TEST_SAMPLE} \
    --seed 0 \
    --temperature 0.05 \
    --n_sampling 16 \
    --max_tokens_per_call 2784 \
    --top_p 1 \
    --start 0 \
    --end -1\
    --use_sglang \
    --pipeline_parallel_size 1 \
    --dp_size 1 \
    --save_outputs \
    --overwrite \