# FlagOS 赛道三 - 双卡910B推理部署

VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 CUDA_VISIBLE_DEVICES=0,1 vllm serve \
    /root/.cache/modelscope/hub/models/Qwen/Qwen3-4B \
    --tensor-parallel-size 2 \
    --port 2026 \
    --max-model-len 131072 \
    --gpu-memory-utilization 0.85 \
    --enforce-eager