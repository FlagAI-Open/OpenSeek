#!/bin/bash

# OpenSeek nanochat experiment runner
# This script runs nanochat training using OpenSeek datasets

set -e

# Default settings
export OMP_NUM_THREADS=1
export OPENSEEK_NANOCHAT_DATA_DIR="${OPENSEEK_NANOCHAT_DATA_DIR:-$HOME/.cache/openseek_nanochat}"
mkdir -p "$OPENSEEK_NANOCHAT_DATA_DIR"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENSEEK_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

echo "=========================================="
echo "OpenSeek nanochat Experiment Runner"
echo "=========================================="
echo "OpenSeek root: $OPENSEEK_ROOT"
echo "Data directory: $OPENSEEK_NANOCHAT_DATA_DIR"
echo "=========================================="
echo

# Check if nanochat is available
if ! python -c "import nanochat" 2>/dev/null; then
    echo "Error: nanochat is not installed or not in Python path"
    echo "Please install nanochat or add it to your PYTHONPATH"
    echo "You can clone nanochat from: https://github.com/karpathy/nanochat"
    exit 1
fi

# Step 1: Convert OpenSeek dataset to parquet format
echo "Step 1: Converting OpenSeek dataset to parquet format..."
echo "----------------------------------------------------------"

# Check if parquet files already exist
PARQUET_DIR="$OPENSEEK_NANOCHAT_DATA_DIR/parquet_shards"
if [ -d "$PARQUET_DIR" ] && [ "$(ls -A $PARQUET_DIR/*.parquet 2>/dev/null)" ]; then
    echo "Parquet files already exist in $PARQUET_DIR"
    read -p "Do you want to regenerate them? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipping dataset conversion..."
        SKIP_CONVERSION=true
    fi
fi

if [ "${SKIP_CONVERSION:-false}" != "true" ]; then
    # Check if OpenSeek dataset is available
    DATASET_PATH="$OPENSEEK_ROOT/OpenSeek-Pretrain-100B"
    
    if [ ! -d "$DATASET_PATH" ]; then
        echo "OpenSeek-Pretrain-100B not found at $DATASET_PATH"
        echo "Attempting to download from HuggingFace..."
        
        python -m examples.nanochat_exp.dataset \
            --dataset "BAAI/OpenSeek-Pretrain-100B" \
            --num-shards "${NUM_SHARDS:-240}" \
            --streaming
    else
        echo "Using local dataset at: $DATASET_PATH"
        # Convert local dataset
        python -m examples.nanochat_exp.dataset \
            --dataset "$DATASET_PATH" \
            --num-shards "${NUM_SHARDS:-240}"
    fi
fi

echo
echo "Step 2: Training with nanochat..."
echo "----------------------------------------------------------"

# Set nanochat data directory to use our parquet files
export NANOCHAT_BASE_DIR="$OPENSEEK_NANOCHAT_DATA_DIR"

# Number of GPUs to use
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

# Model depth (d20 = ~$100 tier, d26 = ~$300 tier, d32 = ~$800 tier)
DEPTH="${DEPTH:-20}"

# Wandb run name (optional)
WANDB_RUN="${WANDB_RUN:-openseek_d${DEPTH}}"

echo "Configuration:"
echo "  Model depth: $DEPTH"
echo "  GPUs: $NPROC_PER_NODE"
echo "  Wandb run: $WANDB_RUN"
echo

# Note: The actual training scripts need to be modified to use our dataloader
# For now, we provide instructions
echo "To train with OpenSeek data, you need to:"
echo "1. Modify nanochat scripts to use examples.nanochat_exp.dataloader"
echo "2. Or use the provided wrapper scripts (coming soon)"
echo
echo "Example training command (after setup):"
echo "  torchrun --standalone --nproc_per_node=$NPROC_PER_NODE \\"
echo "    -m scripts.base_train -- --depth=$DEPTH --run=$WANDB_RUN"
echo
echo "Note: You may need to modify nanochat's base_train.py to import:"
echo "  from examples.nanochat_exp.dataloader import tokenizing_distributed_data_loader"

