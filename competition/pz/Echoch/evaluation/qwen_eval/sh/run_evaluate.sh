# Evaluate Qwen2.5-Math-Instruct
PROMPT_TYPE="openseek_sft"

# Qwen2.5-Math-1.5B-Instruct
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
MODEL_NAME_OR_PATH="OpenSeek-Small-v1-SFT"

bash sh/eval.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH