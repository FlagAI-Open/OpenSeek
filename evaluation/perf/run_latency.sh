
set -u
  MODEL_URL=$1
  FS_ROOT_PATH=$2
set +u

vLLM_ROOT_PATH=$FS_ROOT_PATH/third_party/vllm

export VLLM_USE_FLASHINFER_SAMPLER=0
python $vLLM_ROOT_PATH/benchmarks/benchmark_latency.py \
    --model $MODEL_URL \
    --trust-remote-code \
    --input-len 16 \
    --output-len 4000 \
    --output-json latency_results.json \
    2>&1 | tee benchmark_latency.txt
