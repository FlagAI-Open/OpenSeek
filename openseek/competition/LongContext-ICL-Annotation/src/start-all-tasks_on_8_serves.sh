#!/bin/bash

# export SERVE_API_KEY=""
# export SERVE_BASE_URL="http://localhost:2026/v1"
# export SERVE_MODEL_NAME="Qwen/Qwen3-4B"
# export SERVE_TOKENIZER_PATH=$HOME/.cache/modelscope/hub/models/Qwen/Qwen3-4B
# export REQ_CONCURRENCY_NUM=2
# export ICL_REPEAT_N=1
#
# REQ_CONCURRENCY_NUM   默认2，客户端API并发请求数，如果卡很多，可以增加该数值，与服务端张量和数据并行计算协同，加快实验进度
# ICL_REPEAT_N          默认1，在提取Opration时对ICL的重复次数，如果卡很多，可以增加该数值，提高Opration的完整性
#
# 单个服务，不适合多任务并行，不同任务的防止KV缓存干扰其他任务的准确率
#

dir=$(cd "$(dirname "$0")" && pwd)

pid=$(ps -ef | grep src/main_batch.py | grep -v grep | awk '{print $2}')
if [ -n "$pid" ]; then
    echo 'not need repeat start ...'

else
    export DEBUG_PRINT_STREAM=0
    export REQ_CONCURRENCY_NUM=8

    # 8个服务地址，为了隔离KV缓存，防止不同任务的KV缓存相互干扰，可并发执行
    export SERVE_API_KEY=""
    export SERVE_BASE_URL=http://server1:2026/v1
    export SERVE_MODEL_NAME=Qwen/Qwen3-4B
    export SERVE_TOKENIZER_PATH=$HOME/.cache/modelscope/hub/models/Qwen/Qwen3-4B
    nohup python $dir/main_batch.py --task_id=1 --task_step=0 > output1.log 2>&1 &

    export SERVE_API_KEY=""
    export SERVE_BASE_URL=http://server2:2026/v1
    export SERVE_MODEL_NAME=Qwen/Qwen3-4B
    export SERVE_TOKENIZER_PATH=$HOME/.cache/modelscope/hub/models/Qwen/Qwen3-4B
    nohup python $dir/main_batch.py --task_id=2 --task_step=0 > output2.log 2>&1 &

    export SERVE_API_KEY=""
    export SERVE_BASE_URL=http://server3:2026/v1
    export SERVE_MODEL_NAME=Qwen/Qwen3-4B
    export SERVE_TOKENIZER_PATH=$HOME/.cache/modelscope/hub/models/Qwen/Qwen3-4B
    nohup python $dir/main_batch.py --task_id=3 --task_step=0 > output3.log 2>&1 &

    export SERVE_API_KEY=""
    export SERVE_BASE_URL=http://server4:2026/v1
    export SERVE_MODEL_NAME=Qwen/Qwen3-4B
    export SERVE_TOKENIZER_PATH=$HOME/.cache/modelscope/hub/models/Qwen/Qwen3-4B
    nohup python $dir/main_batch.py --task_id=4 --task_step=0 > output4.log 2>&1 &

    export SERVE_API_KEY=""
    export SERVE_BASE_URL=http://server5:2026/v1
    export SERVE_MODEL_NAME=Qwen/Qwen3-4B
    export SERVE_TOKENIZER_PATH=$HOME/.cache/modelscope/hub/models/Qwen/Qwen3-4B
    nohup python $dir/main_batch.py --task_id=5 --task_step=0 > output5.log 2>&1 &

    export SERVE_API_KEY=""
    export SERVE_BASE_URL=http://server6:2026/v1
    export SERVE_MODEL_NAME=Qwen/Qwen3-4B
    export SERVE_TOKENIZER_PATH=$HOME/.cache/modelscope/hub/models/Qwen/Qwen3-4B
    nohup python $dir/main_batch.py --task_id=6 --task_step=0 > output6.log 2>&1 &

    export SERVE_API_KEY=""
    export SERVE_BASE_URL=http://server7:2026/v1
    export SERVE_MODEL_NAME=Qwen/Qwen3-4B
    export SERVE_TOKENIZER_PATH=$HOME/.cache/modelscope/hub/models/Qwen/Qwen3-4B
    nohup python $dir/main_batch.py --task_id=7 --task_step=0 > output7.log 2>&1 &

    export SERVE_API_KEY=""
    export SERVE_BASE_URL=http://server8:2026/v1
    export SERVE_MODEL_NAME=Qwen/Qwen3-4B
    export SERVE_TOKENIZER_PATH=$HOME/.cache/modelscope/hub/models/Qwen/Qwen3-4B
    nohup python $dir/main_batch.py --task_id=8 --task_step=0 > output8.log 2>&1 &

    echo 'you can check by cmd: `ps -ef | grep main_batch.py | grep -v grep`'
fi
