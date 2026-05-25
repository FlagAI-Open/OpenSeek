#!/bin/bash

dir=$(cd "$(dirname "$0")" && pwd)
cd $dir

VLLM_PLUGINS=fl flagscale server qwen3_s2_0 --config $dir/llm_config_s2_0.yaml --stop
VLLM_PLUGINS=fl flagscale server qwen3_s2_1 --config $dir/llm_config_s2_1.yaml --stop

rm -rf s2_0
rm -rf s2_1
