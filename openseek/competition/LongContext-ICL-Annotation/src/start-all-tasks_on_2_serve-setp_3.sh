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
# 单个服务，不适合多任务并行，防止不同任务的KV缓存干扰其他任务的准确率
#

export DEBUG_PRINT_STREAM=0
export REQ_CONCURRENCY_NUM=2
export ICL_REPEAT_N=3

dir=$(cd "$(dirname "$0")" && pwd)

# 等待第一阶段的第一个任务operation出现
while [ ! -f "$(dirname "$dir")/operations_tmp/operation-1.json" ]; do
    echo "wait for operation-1.json"
    sleep 600
done

pid=$(ps -ef | grep src/main_batch.py | grep 'task_step=3' | grep '[1,2,3,4]' | grep -v grep | awk '{print $2}')
if [ -n "$pid" ]; then
    echo 'not need repeat start tasks[1,2,3,4] ...'
else
    # 卡少的时候，须串行处理
    export SERVE_BASE_URL="http://localhost:2027/v1"
    nohup python $dir/main_batch.py --task_id=[1,2,3,4] --task_step=3 > output3_1234.log 2>&1 &
    # echo 'you can check by cmd: ps -ef | grep main_batch.py | grep -v grep'
fi


# 等待第一阶段所有operation出现
task_ids=(1 2 3 4 5 6 7 8)
for task_id in "${task_ids[@]}"
do
    while [ ! -f "$(dirname "$dir")/operations_tmp/operation-${task_id}.json" ]; do
        echo "wait for operation-${task_id}.json"
        sleep 1800
    done
done

# 在等待第一阶段完成后，其对应服务空闲下来，可以用来处理第二阶段的其他任务
pid=$(ps -ef | grep src/main_batch.py | grep 'task_step=3' | grep '[5,6,7,8]' | grep -v grep | awk '{print $2}')
if [ -n "$pid" ]; then
    echo 'not need repeat start tasks[5,6,7,8] ...'
else
    # 卡少的时候，须串行处理
    export SERVE_BASE_URL="http://localhost:2026/v1"
    nohup python $dir/main_batch.py --task_id=[5,6,7,8] --task_step=3 > output3_5678.log 2>&1 &
    # echo 'you can check by cmd: ps -ef | grep main_batch.py | grep -v grep'
fi
