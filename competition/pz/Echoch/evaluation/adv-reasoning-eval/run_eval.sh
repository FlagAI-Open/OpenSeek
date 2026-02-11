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

python eval_gsm8k_adv.py \
    --model_path ${MODEL_URL} \

python eval_gsm8k-platinum_adv.py \
    --model_path ${MODEL_URL} \

python eval_cruxeval_i_adv.py \
    --model_path ${MODEL_URL} \

python eval_cruxeval_o_adv.py \
    --model_path ${MODEL_URL} \
    