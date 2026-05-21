#!/bin/bash

export HCCL_CONNECT_TIMEOUT=180
export HCCL_IF_MTU=10240
export VLLM_PLUGINS=fl
export VLLM_ASCEND_ENABLE_CONTEXT_PARALLEL=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ASCEND_USE_TORCHAIR=0
export ASCEND_RT_VISIBLE_DEVICES=0,1
export VLLM_USE_MODELSCOPE=true
export TRITON_ALL_BLOCKS_PARALLEL=1
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256

export VLLM_ASCEND_ENABLE_FUSED_MC2=1  # 启用融合算子，支持TND
export VLLM_ASCEND_ENABLE_FLASHCOMM=1  # 多卡并行时启用FlashComm
export TASK_QUEUE_ENABLE=2             # EP (Enqueued Processing) 模式

dir=$(cd "$(dirname "$0")" && pwd)
cd $dir

# 注意：启动服务后需要耐心等一小段时间让模型服务就绪（npu-info或nputop查看显存）
# NPU卡数超过2张时，可以需要提高参数 tensor_parallel_size 的数值

ASCEND_RT_VISIBLE_DEVICES=0 VLLM_PLUGINS=fl flagscale serve qwen3_s4_0 --config $dir/llm_config_s4_0.yaml
ASCEND_RT_VISIBLE_DEVICES=0 VLLM_PLUGINS=fl flagscale serve qwen3_s4_1 --config $dir/llm_config_s4_1.yaml

ASCEND_RT_VISIBLE_DEVICES=1 VLLM_PLUGINS=fl flagscale serve qwen3_s4_2 --config $dir/llm_config_s4_2.yaml
ASCEND_RT_VISIBLE_DEVICES=1 VLLM_PLUGINS=fl flagscale serve qwen3_s4_3 --config $dir/llm_config_s4_3.yaml
