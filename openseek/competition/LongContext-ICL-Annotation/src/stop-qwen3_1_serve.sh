#!/bin/bash

dir=$(cd "$(dirname "$0")" && pwd)
cd $dir

flagscale serve qwen3 --config $dir/llm_config.yaml --stop
