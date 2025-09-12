#!/bin/bash
#SBATCH --gpus=4
#SBATCH -x paraai-n32-h-01-agent-[1,4,8,16,17,25,27,28,29,30,31]
module load compilers/cuda/12.4
module load cudnn/8.9.5.29_cuda12.x
module load compilers/gcc/12.2.0
module load cmake/3.26.3
module load miniforge3/24.1

WANDB_RUN="OpenSeek-Small-v1_tokens-15B-math-exp2"
LOG_DIR="/home/bingxing2/home/scx7353/workspace/OpenSeek/OpenSeek-Small-v1-Baseline/logs/details/host_0_localhost/20250814_2024"
SAVE_STEPS=720
EVAL_STEPS=100

export CUDA_VISIBLE_DEVICES=0,1,2,3

export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=^docker0,lo
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

ulimit -n 1048576

source activate flagscale-train

mkdir -p /home/bingxing2/home/scx7353/workspace/OpenSeek/OpenSeek-Small-v1-Baseline/checkpoints
mkdir -p /home/bingxing2/home/scx7353/workspace/OpenSeek/OpenSeek-Small-v1-Baseline/checkpoints
mkdir -p /home/bingxing2/home/scx7353/workspace/OpenSeek/OpenSeek-Small-v1-Baseline/logs
mkdir -p /home/bingxing2/home/scx7353/workspace/OpenSeek/OpenSeek-Small-v1-Baseline/logs/pids
mkdir -p /home/bingxing2/home/scx7353/workspace/OpenSeek/OpenSeek-Small-v1-Baseline/logs/details
mkdir -p /home/bingxing2/home/scx7353/workspace/OpenSeek/OpenSeek-Small-v1-Baseline/tensorboard
mkdir -p /home/bingxing2/home/scx7353/workspace/OpenSeek/OpenSeek-Small-v1-Baseline/wandb

cd /home/bingxing2/home/scx7353/workspace/OpenSeek/FlagScale

export PYTHONPATH=/home/bingxing2/home/scx7353/workspace/OpenSeek/FlagScale/third_party/Megatron-LM:/home/bingxing2/home/scx7353/workspace/OpenSeek/FlagScale:${PYTHONPATH}

export http_proxy=http://u-cEoRwn:EDvFuZTe@172.16.4.9:3128
export https_proxy=http://u-cEoRwn:EDvFuZTe@172.16.4.9:3128

export WANDB_API_KEY={WANDB_API_KEY}  # need to config
export WANDB_MODE=online
export WANDB__SERVICE_WAIT=300
export WANDB_SILENT=false
export WANDB_START_METHOD=thread
export WANDB_GROUP=$WANDB_RUN

VISIBLE_DEVICES=0,1,2,3 DEVICE_MAX_CONNECTIONS=4 torchrun \
        --rdzv_backend static \
        --nnodes 1 \
        --nproc_per_node 4 \
        --rdzv_id default \
        --node_rank 0 \
        --rdzv_endpoint localhost:59249 \
        --log_dir $LOG_DIR \
        --redirects 3 \
        --tee 3 \
        flagscale/train/train_gpt.py \
        --no-load-optim --no-load-rng \
        --recompute-method uniform --recompute-granularity full --recompute-num-layers 6 \
        --moe-router-dtype fp32 --num-workers 1 \
        --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --expert-model-parallel-size 1 --context-parallel-size 1 \
        --disable-bias-linear --reset-position-ids --reset-attention-mask \
        --qk-layernorm --sequence-parallel --use-distributed-optimizer --overlap-grad-reduce --overlap-param-gather \
        --bf16 --attention-softmax-in-fp32 --accumulate-allreduce-grads-in-fp32 \
        --log-interval 1 --tensorboard-log-interval 1 \
        --wandb-mode online \
        --wandb-api-key 2356f969f25a7b0f375f3bcf3aff92e70d912bda \
        --wandb-project OpenSeek-Small-v1 \
        --wandb-exp-name $WANDB_RUN \
        --log-timers-to-tensorboard --log-validation-ppl-to-tensorboard \
        --log-throughput --log-params-norm --log-num-zeros-in-grad --log-memory-to-tensorboard \
        --tensorboard-dir /home/bingxing2/home/scx7353/workspace/OpenSeek/OpenSeek-Small-v1-Baseline/tensorboard \
        --wandb-save-dir /home/bingxing2/home/scx7353/workspace/OpenSeek/OpenSeek-Small-v1-Baseline/wandb \
        --save-interval $SAVE_STEPS \
        --load /home/bingxing2/home/scx7353/workspace/OpenSeek/OpenSeek-Small-v1-Baseline/checkpoints \
        --ckpt-format torch \
        --save /home/bingxing2/home/scx7353/workspace/OpenSeek/OpenSeek-Small-v1-Baseline/checkpoints \
        --transformer-impl transformer_engine --num-layers 6 --hidden-size 1280 \
        --num-attention-heads 10 --num-query-groups 10 --seq-length 4096 --max-position-embeddings 4096 --norm-epsilon 1e-06 \
        --use-rotary-position-embeddings --rotary-base 1000000 \
        --swiglu --normalization RMSNorm --init-method-std 0.006 --attention-dropout 0.0 --hidden-dropout 0.0 --clip-grad 1.0 \
        --position-embedding-type rope --no-position-embedding --no-rope-fusion --multi-latent-attention \
        --kv-lora-rank 512 --qk-head-dim 128 --qk-pos-emb-head-dim 64 --v-head-dim 128 --ffn-hidden-size 7168 \
        --moe-ffn-hidden-size 896 --moe-grouped-gemm --moe-shared-expert-intermediate-size 1792 \
        --num-experts 64 --moe-router-load-balancing-type seq_aux_loss --moe-router-score-function sigmoid \
        --moe-router-enable-expert-bias --moe-router-bias-update-rate 0.001 --moe-aux-loss-coeff 0.0001 \
        --moe-layer-freq '[0]+[1]*5' --moe-router-num-groups 1 --moe-router-group-topk 1 --moe-router-topk 6 \
        --moe-router-topk-scaling-factor 2.446 --moe-token-dispatcher-type alltoall --seed 42 \
        --micro-batch-size 2 --global-batch-size 1024 \
        --eval-iters 4 --eval-interval $EVAL_STEPS \
        --train-iters 3620 \
        --weight-decay 0.01 \
        --adam-beta1 0.9 --adam-beta2 0.95 \
        --lr 0.00001 --min-lr 0.0000001 \
        --lr-warmup-iters 50 --lr-warmup-samples 0 --lr-decay-style cosine \
        --data-path 1.1068 /home/bingxing2/home/scx7353/workspace/OpenSeek-Pretrain-100B/Nemotron-CC-high-actual-actual-high/part_142_text_document 0.5397 /home/bingxing2/home/scx7353/workspace/OpenSeek-Pretrain-100B/Nemotron-CC-high-synthetic-diverse_qa_pairs-high/part_244_text_document 0.4616 /home/bingxing2/home/scx7353/workspace/OpenSeek-Pretrain-100B/Nemotron-CC-high-synthetic-extract_knowledge-high/part_498_text_document 0.261 /home/bingxing2/home/scx7353/workspace/OpenSeek-Pretrain-100B/Nemotron-CC-high-synthetic-knowledge_list-high/part_86_text_document 0.6414 /home/bingxing2/home/scx7353/workspace/OpenSeek-Pretrain-100B/arxiv/007_00000_text_document 0.4696 /home/bingxing2/home/scx7353/workspace/OpenSeek-Pretrain-100B/books/016_00007_text_document 1.0102 /home/bingxing2/home/scx7353/workspace/OpenSeek-Pretrain-100B/code-high/part_13_text_document 0.3755 /home/bingxing2/home/scx7353/workspace/OpenSeek-Pretrain-100B/cot_synthesis2_CC-high/23_text_document 0.4598 /home/bingxing2/home/scx7353/workspace/OpenSeek-Pretrain-100B/cot_synthesis2_code-high/4_text_document 1.3135 /home/bingxing2/home/scx7353/workspace/OpenSeek-Pretrain-100B/cot_synthesis2_math-high/12_text_document 0.3536 /home/bingxing2/home/scx7353/workspace/OpenSeek-Pretrain-100B/cot_synthesis2_math-mid/5_text_document 0.6314 /home/bingxing2/home/scx7353/workspace/OpenSeek-Pretrain-100B/cot_synthesis2_wiki-high/5_text_document 0.5074 /home/bingxing2/home/scx7353/workspace/OpenSeek-Pretrain-100B/cot_synthesis_math-high/11_text_document 0.6406 /home/bingxing2/home/scx7353/workspace/OpenSeek-Pretrain-100B/cot_synthesis_math-mid/29_text_document 1.8165 /home/bingxing2/home/scx7353/workspace/OpenSeek-Pretrain-100B/math-high/part_04_text_document 1.6311 /home/bingxing2/home/scx7353/workspace/OpenSeek-Pretrain-100B/math-mid/part_07_text_document 0.4202 /home/bingxing2/home/scx7353/workspace/OpenSeek-Pretrain-100B/wiki/012_00000_text_document 1.8171 /home/bingxing2/home/scx7353/workspace/OpenSeek-Pretrain-100B/zh_cc-high-loss0/part_28_text_document \
        --split 998,1,1 \
        --no-mmap-bin-files \
        --tokenizer-type QwenTokenizerFS --tokenizer-path ../hf_openseek/tokenizer \
        --vocab-size 151851 --make-vocab-size-divisible-by 64
