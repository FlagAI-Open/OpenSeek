#!/bin/bash

# export SERVE_API_KEY=""
# export SERVE_BASE_URL="http://localhost:2026/v1"
# export SERVE_MODEL_NAME="Qwen/Qwen3-4B"
# export SERVE_TOKENIZER_PATH=~/.cache/modelscope/hub/models/Qwen/Qwen3-4B
# export REQ_CONCURRENCY_NUM=2
# export ICL_REPEAT_N=1
#
# REQ_CONCURRENCY_NUM   默认2，客户端API并发请求数，如果卡很多，可以增加该数值，与服务端张量和数据并行计算协同，加快实验进度
# ICL_REPEAT_N          默认1，在提取Opration时对ICL的重复次数，如果卡很多，可以增加该数值，提高Opration的完整性
#
# 单个服务，不适合多任务并行，防止不同任务的KV缓存干扰其他任务的准确率
#

export DEBUG_PRINT_STREAM=0
export REQ_CONCURRENCY_NUM=2
export ICL_REPEAT_N=3

dir=$(cd "$(dirname "$0")" && pwd)

pids=$(ps -ef | grep src/main_batch.py | grep -v grep | awk '{print $2}')
if [ -n "$pids" ]; then
    echo 'not need repeat start ...'

else
    export SERVE_BASE_URL="http://localhost:2026/v1"
    nohup python $dir/main_batch.py --task_id=[1,2] --task_step=0 > output0_12.log 2>&1 &

    export SERVE_BASE_URL="http://localhost:2027/v1"
    nohup python $dir/main_batch.py --task_id=[3,4] --task_step=0 > output0_34.log 2>&1 &

    export SERVE_BASE_URL="http://localhost:2028/v1"
    nohup python $dir/main_batch.py --task_id=[5,6] --task_step=0 > output0_56.log 2>&1 &

    export SERVE_BASE_URL="http://localhost:2029/v1"
    nohup python $dir/main_batch.py --task_id=[7,8] --task_step=0 > output0_78.log 2>&1 &

    echo 'you can check by cmd: `ps -ef | grep main_batch.py | grep -v grep`'
fi
