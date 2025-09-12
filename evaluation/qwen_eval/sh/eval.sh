set -ex
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PROMPT_TYPE='openseek_sft'
# MODEL_NAME_OR_PATH='/mnt/cfs/shanhai/qiuchenhao/model/ckpt/PPO/hfmodel_v0_v1/actor'
# MODEL_NAME_OR_PATH='/mnt/cfs/shanhai/qiuchenhao/model/ckpt/sfdp_to_hf/sft/epoch2/'
MODEL_NAME_OR_PATH=/mnt/cfs/shanhai/qiuchenhao/model/ckpt/final/global_step_450
MODEL_NAME_SHORT=$(basename $MODEL_NAME_OR_PATH)
OUTPUT_DIR=${MODEL_NAME_SHORT}/math_eval_sglang

SPLIT="test"
NUM_TEST_SAMPLE=-1

DATA_NAME="gsm8k,math500,minerva_math,amc23,aime24,olympiadbench"
# DATA_NAME='amc23'
TOKENIZERS_PARALLELISM=false \

python3 -u math_eval.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --data_name ${DATA_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --split ${SPLIT} \
    --prompt_type ${PROMPT_TYPE} \
    --num_test_sample ${NUM_TEST_SAMPLE} \
    --seed 0 \
    --temperature 0.0 \
    --n_sampling 16 \
    --max_tokens_per_call 3072 \
    --top_p 0.95 \
    --start 0 \
    --end -1\
    --use_sglang \
    --pipeline_parallel_size 1 \
    --dp_size 4 \
    --save_outputs \
    --overwrite \
    --apply_chat_template \
    --max_prompt_tokens 1000
