#!/bin/bash

# vllm
VLLM_TARGET_DEVICE=empty pip install vllm==0.13.0 --extra-index https://download.pytorch.org/whl/cpu
# 忽略vllm依赖torch的版本提示
pip install vllm_ascend==0.13.0rc1
pip install vllm-plugin-fl==0.1.0+vllm0.13.0 --extra-index-url https://resource.flagos.net/repository/flagos-pypi-hosted/simple

# FlagScale
git clone https://github.com/flagos-ai/FlagScale
cd FlagScale
pip install setuptools==82.0.0 scikit-build-core==0.11 pybind11==3.0.2 ninja==1.13.0 cmake==4.2.3
pip install ".[ascend-serve]" -v --no-build-isolation
cd ..

# 常用工具
pip install ascend-nputop
pip install modelscope

# 下载Qwen3-4B
pip install "setuptools<82.0.0"
modelscope download --model Qwen/Qwen3-4B
sed -i 's#"rope_scaling": null#"rope_scaling": {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 40960}#g' ~/.cache/modelscope/hub/models/Qwen/Qwen3-4B/config.json
cat ~/.cache/modelscope/hub/models/Qwen/Qwen3-4B/config.json

# 环境变量
export HCCL_CONNECT_TIMEOUT=180
export HCCL_IF_MTU=10240
export VLLM_PLUGINS=fl
export VLLM_ASCEND_ENABLE_CONTEXT_PARALLEL=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ASCEND_USE_TORCHAIR=0
export ASCEND_RT_VISIBLE_DEVICES=0,1
export ASCEND_VISIBLE_DEVICES=0,1
export VLLM_USE_MODELSCOPE=true
export TRITON_ALL_BLOCKS_PARALLEL=1
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256

export VLLM_ASCEND_ENABLE_FUSED_MC2=1  # 启用融合算子，支持TND
export VLLM_ASCEND_ENABLE_FLASHCOMM=1  # 多卡并行时启用FlashComm
export TASK_QUEUE_ENABLE=2             # EP (Enqueued Processing) 模式
