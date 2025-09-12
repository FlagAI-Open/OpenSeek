set -u
  MODEL_URL=$1
set +u

# https://github.com/EleutherAI/lm-evaluation-harness
# commit 6b3f3f7e2aece0597a0a4853d364061f93eef5ef
export VLLM_USE_FLASHINFER_SAMPLER=0

#hellaswag
HF_ENDPOINT=https://hf-mirror.com lm_eval --model vllm \
    --model_args trust_remote_code=True,pretrained=${MODEL_URL} \
    --tasks hellaswag \
    --device cuda:0 \
    --batch_size 8 \
    --trust_remote_code \
    --output_path output \
    --num_fewshot 5
hellaswag_exit_code=$?
if [ $hellaswag_exit_code -ne 0 ]; then
    exit $hellaswag_exit_code
fi

#truthfulqa
HF_ENDPOINT=https://hf-mirror.com lm_eval --model vllm \
    --model_args trust_remote_code=True,pretrained=${MODEL_URL} \
    --tasks truthfulqa \
    --device cuda:0 \
    --batch_size 8 \
    --trust_remote_code \
    --output_path output \
    --num_fewshot 0
truthfulqa_exit_code=$?
if [ $truthfulqa_exit_code -ne 0 ]; then
    exit $truthfulqa_exit_code
fi

#winogrande
HF_ENDPOINT=https://hf-mirror.com lm_eval --model vllm \
    --model_args trust_remote_code=True,pretrained=${MODEL_URL} \
    --tasks winogrande \
    --device cuda:0 \
    --batch_size 8 \
    --trust_remote_code \
    --output_path output \
    --num_fewshot 5
winogrande_exit_code=$?
if [ $winogrande_exit_code -ne 0 ]; then
    exit $winogrande_exit_code
fi

#commonsense_qa
HF_ENDPOINT=https://hf-mirror.com lm_eval --model vllm \
    --model_args trust_remote_code=True,pretrained=${MODEL_URL} \
    --tasks commonsense_qa \
    --device cuda:0 \
    --batch_size 8 \
    --trust_remote_code \
    --output_path output \
    --num_fewshot 5
commonsense_qa_exit_code=$?
if [ $commonsense_qa_exit_code -ne 0 ]; then
    exit $commonsense_qa_exit_code
fi

#piqa
HF_ENDPOINT=https://hf-mirror.com lm_eval --model vllm \
    --model_args trust_remote_code=True,pretrained=${MODEL_URL} \
    --tasks piqa \
    --device cuda:0 \
    --batch_size 8 \
    --trust_remote_code \
    --output_path output \
    --num_fewshot 5
piqa_exit_code=$?
if [ $piqa_exit_code -ne 0 ]; then
    exit $piqa_exit_code
fi

#openbookqa
HF_ENDPOINT=https://hf-mirror.com lm_eval --model vllm \
    --model_args trust_remote_code=True,pretrained=${MODEL_URL} \
    --tasks openbookqa \
    --device cuda:0 \
    --batch_size 8 \
    --trust_remote_code \
    --output_path output \
    --num_fewshot 5
openbookqa_exit_code=$?
if [ $openbookqa_exit_code -ne 0 ]; then
    exit $openbookqa_exit_code
fi

#boolq
HF_ENDPOINT=https://hf-mirror.com lm_eval --model vllm \
    --model_args trust_remote_code=True,pretrained=${MODEL_URL} \
    --tasks boolq \
    --device cuda:0 \
    --batch_size 8 \
    --trust_remote_code \
    --output_path output \
    --num_fewshot 5
boolq_exit_code=$?
if [ $boolq_exit_code -ne 0 ]; then
    exit $boolq_exit_code
fi

#arc_easy
HF_ENDPOINT=https://hf-mirror.com lm_eval --model vllm \
    --model_args trust_remote_code=True,pretrained=${MODEL_URL} \
    --tasks arc_easy \
    --device cuda:0 \
    --batch_size 8 \
    --trust_remote_code \
    --output_path output \
    --num_fewshot 5
arc_easy_exit_code=$?
if [ $arc_easy_exit_code -ne 0 ]; then
    exit $arc_easy_exit_code
fi

#arc_challenge
HF_ENDPOINT=https://hf-mirror.com lm_eval --model vllm \
    --model_args trust_remote_code=True,pretrained=${MODEL_URL} \
    --tasks arc_challenge \
    --device cuda:0 \
    --batch_size 8 \
    --trust_remote_code \
    --output_path output \
    --num_fewshot 5
arc_challenge_exit_code=$?
if [ $arc_challenge_exit_code -ne 0 ]; then
    exit $arc_challenge_exit_code
fi

#mmlu
HF_ENDPOINT=https://hf-mirror.com lm_eval --model vllm \
    --model_args trust_remote_code=True,pretrained=${MODEL_URL} \
    --tasks mmlu \
    --device cuda:0 \
    --batch_size 8 \
    --trust_remote_code \
    --output_path output \
    --num_fewshot 5
mmlu_exit_code=$?
if [ $mmlu_exit_code -ne 0 ]; then
    exit $mmlu_exit_code
fi

#gsm8k
HF_ENDPOINT=https://hf-mirror.com lm_eval --model vllm \
    --model_args trust_remote_code=True,pretrained=${MODEL_URL} \
    --tasks gsm8k \
    --device cuda:0 \
    --batch_size 8 \
    --trust_remote_code \
    --output_path output \
    --num_fewshot 5
gsm8k_exit_code=$?
if [ $gsm8k_exit_code -ne 0 ]; then
    exit $gsm8k_exit_code
fi

#minerva_math
HF_ENDPOINT=https://hf-mirror.com lm_eval --model vllm \
    --model_args trust_remote_code=True,pretrained=${MODEL_URL} \
    --tasks minerva_math \
    --device cuda:0 \
    --batch_size 8 \
    --trust_remote_code \
    --output_path output \
    --num_fewshot 4
minerva_math_exit_code=$?
if [ $minerva_math_exit_code -ne 0 ]; then
    exit $minerva_math_exit_code
fi

#ceval-valid
HF_ENDPOINT=https://hf-mirror.com lm_eval --model vllm \
    --model_args trust_remote_code=True,pretrained=${MODEL_URL} \
    --tasks ceval-valid \
    --device cuda:0 \
    --batch_size 8 \
    --trust_remote_code \
    --output_path output \
    --num_fewshot 5
ceval_valid_exit_code=$?
if [ $ceval_valid_exit_code -ne 0 ]; then
    exit $ceval_valid_exit_code
fi

#cmmlu 
HF_ENDPOINT=https://hf-mirror.com lm_eval --model vllm \
    --model_args trust_remote_code=True,pretrained=${MODEL_URL} \
    --tasks cmmlu \
    --device cuda:0 \
    --batch_size 8 \
    --trust_remote_code \
    --output_path output \
    --num_fewshot 5
cmmlu_exit_code=$?
if [ $cmmlu_exit_code -ne 0 ]; then
    exit $cmmlu_exit_code
fi

# collect results
model_url=$(echo "$MODEL_URL" | sed 's/\//__/g')
echo $model_url
python collect_results.py output/$model_url
