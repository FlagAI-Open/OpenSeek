#!/bin/bash

dir=$(cd "$(dirname "$0")" && pwd)
cd $dir

flagscale serve qwen3 --config $dir/llm_config.yaml --stop

VLLM_PLUGINS=fl flagscale server qwen3_s4_0 --config $dir/llm_config_s4_0.yaml --stop
VLLM_PLUGINS=fl flagscale server qwen3_s4_1 --config $dir/llm_config_s4_1.yaml --stop
VLLM_PLUGINS=fl flagscale server qwen3_s4_2 --config $dir/llm_config_s4_2.yaml --stop
VLLM_PLUGINS=fl flagscale server qwen3_s4_3 --config $dir/llm_config_s4_3.yaml --stop

rm -rf s4_0
rm -rf s4_1
rm -rf s4_2
rm -rf s4_3
