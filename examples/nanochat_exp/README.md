# OpenSeek nanochat Experiment

This directory provides integration between OpenSeek datasets and the [nanochat](https://github.com/karpathy/nanochat) training framework, allowing you to train nanochat models using OpenSeek's high-quality pretraining datasets.

## Overview

This module adapts OpenSeek datasets (specifically OpenSeek-Pretrain-100B) to work with nanochat's training pipeline. It provides:

- **Dataset conversion**: Converts OpenSeek datasets from HuggingFace format to parquet format compatible with nanochat
- **Data loader**: Provides a data loader compatible with nanochat's training interface
- **Experiment scripts**: Scripts to run training experiments with OpenSeek data

## Prerequisites

1. **nanochat**: 建议将 nanochat 仓库克隆在 OpenSeek 同一父目录下，并在运行前显式加入 `PYTHONPATH`：
   ```bash
   cd /Users/liuguang/Documents/workspace
   git clone https://github.com/karpathy/nanochat.git
   export PYTHONPATH="/Users/liuguang/Documents/workspace/nanochat:${PYTHONPATH}"
   export PYTHONPATH="/Users/liuguang/Documents/workspace/OpenSeek:${PYTHONPATH}"
   ```
   需要长期使用时，可把 `export` 写入 shell 启动脚本。

2. **Python dependencies**（推荐 Python≥3.10、PyTorch≥2.1，与 nanochat 官方示例保持一致）：
   ```bash
   pip install pyarrow datasets huggingface_hub
   ```

3. **OpenSeek dataset**: Download the OpenSeek-Pretrain-100B dataset:
   - Option 1: Download from HuggingFace (automatic)
   - Option 2: Download manually to `OpenSeek-Pretrain-100B/` directory in the project root

## Quick Start

### 0. 准备 nanochat 目录与环境

确保 nanochat 源码路径已加入 `PYTHONPATH`，并确认基础导入无误：

```bash
cd /Users/liuguang/Documents/workspace/OpenSeek
python -c "import nanochat, examples.nanochat_exp as mod; print('nanochat ready')"
```

若该命令报错，请检查 nanochat 克隆路径与 `PYTHONPATH`。

### 1. Convert OpenSeek Dataset to Parquet Format

First, convert the OpenSeek dataset to the parquet format expected by nanochat:

```bash
# From OpenSeek root directory
python -m examples.nanochat_exp.dataset \
    --dataset "BAAI/OpenSeek-Pretrain-100B" \
    --num-shards 240 \
    --streaming
```

Or if you have the dataset locally:

```bash
python -m examples.nanochat_exp.dataset \
    --dataset ./OpenSeek-Pretrain-100B \
    --num-shards 240
```

This will create parquet shards in `~/.cache/openseek_nanochat/parquet_shards/` (or the directory specified by `OPENSEEK_NANOCHAT_DATA_DIR`).

### 2. Modify nanochat Training Scripts

To use OpenSeek data with nanochat, you need to modify nanochat's training scripts to use our data loader. In `nanochat/scripts/base_train.py`, change:

```python
from nanochat.dataloader import tokenizing_distributed_data_loader
```

to:

```python
from examples.nanochat_exp.dataloader import tokenizing_distributed_data_loader
```

### 3. Run Training

After modifying the import, you can run nanochat training as usual:

```bash
# Set the data directory
export OPENSEEK_NANOCHAT_DATA_DIR="$HOME/.cache/openseek_nanochat"
export NANOCHAT_BASE_DIR="$OPENSEEK_NANOCHAT_DATA_DIR"

# Run training (example for d20 model)
torchrun --standalone --nproc_per_node=8 \
    -m scripts.base_train -- --depth=20 --run=openseek_d20
```

## Directory Structure

```
examples/nanochat_exp/
├── __init__.py              # Module initialization
├── dataset.py               # Dataset conversion and loading utilities
├── dataloader.py            # Data loader compatible with nanochat
├── run_openseek_exp.sh      # Experiment runner script
├── train_wrapper.py         # Training wrapper script
└── README.md                # This file
```

This module can be imported via `examples.nanochat_exp` when the repository root is on `PYTHONPATH`.

## Design Notes & Configuration

### 数据加载接口说明

- `examples.nanochat_exp.dataloader.tokenizing_distributed_data_loader()` 保持与 nanochat 原生接口兼容，可直接替换 import。
- 默认 tokenizer 仍由 nanochat 配置决定；若需变更 tokenizer、最大长度或 padding 规则，可在 `train_wrapper.py` 中调整对应函数，再由训练脚本调用。
- 批量大小、梯度累积等策略建议继续放在 nanochat 脚本侧统一配置，确保与数据加载逻辑一致。

### Environment Variables

- `OPENSEEK_NANOCHAT_DATA_DIR`: Base directory for OpenSeek nanochat data (default: `~/.cache/openseek_nanochat`)
- `NANOCHAT_BASE_DIR`: Base directory for nanochat (should point to where parquet files are)

> 若需放置在自定义路径，可在 shell 中提前导出上述变量，例如 `export OPENSEEK_NANOCHAT_DATA_DIR=/mnt/data/openseek_nanochat`，同时确保 nanochat 的配置文件指向相同目录。

### Dataset Conversion Options

The `dataset.py` script supports several options:

```bash
python -m examples.nanochat_exp.dataset --help
```

Key options:
- `--dataset`: HuggingFace dataset name or local path
- `--num-shards`: Number of parquet shards to create (-1 for all)
- `--shard-size`: Approximate characters per shard (default: 250M)
- `--text-column`: Name of text column in dataset (default: "text")
- `--streaming`: Use streaming mode for large datasets

## Data Format

The converted parquet files follow nanochat's expected format:
- Each parquet file contains a `text` column with document text
- Files are named `