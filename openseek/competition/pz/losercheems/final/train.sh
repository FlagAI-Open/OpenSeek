#!/bin/bash

export HF_ENDPOINT=https://hf-mirror.com
export XDG_CACHE_HOME=cache
export WANDB_OFFLINE=true

ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/zero2.yaml ./trainer/kto.py --config recipes/openseek/config.yaml

# tmux new -s openseek
# tmux attach -t openseek