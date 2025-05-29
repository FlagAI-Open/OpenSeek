set -u
  MODEL_URL=$1
set +u

## https://github.com/EleutherAI/lm-evaluation-harness
## commit e20e1ddc7963996d1c9310a7e880b5c17b54295d

#hellaswag
HF_ENDPOINT=https://hf-mirror.com lm_eval --model hf \
    --model_args trust_remote_code=True,pretrained=${MODEL_URL} \
    --tasks hellaswag \
    --device cuda:0 \
    --batch_size 8 \
    --trust_remote_code \
    --output_path output \
    --num_fewshot 5

#truthfulqa
HF_ENDPOINT=https://hf-mirror.com lm_eval --model hf \
    --model_args trust_remote_code=True,pretrained=${MODEL_URL} \
    --tasks truthfulqa \
    --device cuda:0 \
    --batch_size 8 \
    --trust_remote_code \
    --output_path output \
    --num_fewshot 0

#winogrande
HF_ENDPOINT=https://hf-mirror.com lm_eval --model hf \
    --model_args trust_remote_code=True,pretrained=${MODEL_URL} \
    --tasks winogrande \
    --device cuda:0 \
    --batch_size 8 \
    --trust_remote_code \
    --output_path output \
    --num_fewshot 5

#commonsense_qa
HF_ENDPOINT=https://hf-mirror.com lm_eval --model hf \
    --model_args trust_remote_code=True,pretrained=${MODEL_URL} \
    --tasks commonsense_qa \
    --device cuda:0 \
    --batch_size 8 \
    --trust_remote_code \
    --output_path output \
    --num_fewshot 5

#piqa
HF_ENDPOINT=https://hf-mirror.com lm_eval --model hf \
    --model_args trust_remote_code=True,pretrained=${MODEL_URL} \
    --tasks piqa \
    --device cuda:0 \
    --batch_size 8 \
    --trust_remote_code \
    --output_path output \
    --num_fewshot 5

#openbookqa
HF_ENDPOINT=https://hf-mirror.com lm_eval --model hf \
    --model_args trust_remote_code=True,pretrained=${MODEL_URL} \
    --tasks openbookqa \
    --device cuda:0 \
    --batch_size 8 \
    --trust_remote_code \
    --output_path output \
    --num_fewshot 5

#boolq
HF_ENDPOINT=https://hf-mirror.com lm_eval --model hf \
    --model_args trust_remote_code=True,pretrained=${MODEL_URL} \
    --tasks boolq \
    --device cuda:0 \
    --batch_size 8 \
    --trust_remote_code \
    --output_path output \
    --num_fewshot 5

#arc_easy
HF_ENDPOINT=https://hf-mirror.com lm_eval --model hf \
    --model_args trust_remote_code=True,pretrained=${MODEL_URL} \
    --tasks arc_easy \
    --device cuda:0 \
    --batch_size 8 \
    --trust_remote_code \
    --output_path output \
    --num_fewshot 5

#arc_challenge
HF_ENDPOINT=https://hf-mirror.com lm_eval --model hf \
    --model_args trust_remote_code=True,pretrained=${MODEL_URL} \
    --tasks arc_challenge \
    --device cuda:0 \
    --batch_size 8 \
    --trust_remote_code \
    --output_path output \
    --num_fewshot 5

#mmlu
HF_ENDPOINT=https://hf-mirror.com lm_eval --model hf \
    --model_args trust_remote_code=True,pretrained=${MODEL_URL} \
    --tasks mmlu \
    --device cuda:0 \
    --batch_size 8 \
    --trust_remote_code \
    --output_path output \
    --num_fewshot 5

#gsm8k
HF_ENDPOINT=https://hf-mirror.com lm_eval --model hf \
    --model_args trust_remote_code=True,pretrained=${MODEL_URL} \
    --tasks gsm8k \
    --device cuda:0 \
    --batch_size 8 \
    --trust_remote_code \
    --output_path output \
    --num_fewshot 5

#minerva_math
HF_ENDPOINT=https://hf-mirror.com lm_eval --model hf \
    --model_args trust_remote_code=True,pretrained=${MODEL_URL} \
    --tasks minerva_math \
    --device cuda:0 \
    --batch_size 8 \
    --trust_remote_code \
    --output_path output \
    --num_fewshot 4

#ceval-valid
HF_ENDPOINT=https://hf-mirror.com lm_eval --model hf \
    --model_args trust_remote_code=True,pretrained=${MODEL_URL} \
    --tasks ceval-valid \
    --device cuda:0 \
    --batch_size 8 \
    --trust_remote_code \
    --output_path output \
    --num_fewshot 5

#cmmlu 
HF_ENDPOINT=https://hf-mirror.com lm_eval --model hf \
    --model_args trust_remote_code=True,pretrained=${MODEL_URL} \
    --tasks cmmlu \
    --device cuda:0 \
    --batch_size 8 \
    --trust_remote_code \
    --output_path output \
    --num_fewshot 5

