#!/bin/bash
#SBATCH -N 1
#SBATCH -n 32
#SBATCH -t 114514
#SBATCH -J openseek
#SBATCH -o out.log
#SBATCH --gpus=4

module unload compilers/cuda
module unload cudnn
module load compilers/cuda/12.2
module load cudnn/9.8.0.87_cuda12.x
conda activate train
conda init

export HF_ENDPOINT=https://hf-mirror.com
export XDG_CACHE_HOME=cache

accelerate launch --config_file recipes/accelerate_configs/ddp.yaml ./trainer/pt_dpsk.py --config recipes/openseek/config.yaml

# sbatch train.sh
