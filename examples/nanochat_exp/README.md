# OpenSeek nanochat Experiment

This directory provides integration between OpenSeek datasets and the [nanochat](https://github.com/karpathy/nanochat) training framework, allowing you to train nanochat models using OpenSeek's high-quality pretraining datasets.

## Overview

This module adapts OpenSeek datasets (specifically OpenSeek-Pretrain-Data-Examples) to work with nanochat's training pipeline. It provides:

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
   # 推荐使用虚拟环境（避免权限警告）
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # 或 venv\Scripts\activate  # Windows
   
   # OpenSeek 使用 HuggingFace tokenizers 库（易于安装，无需 Rust 编译）
   pip install pyarrow datasets huggingface_hub tokenizers>=0.22.0
   ```
   
   > **重要**: OpenSeek 使用 HuggingFace `tokenizers` 库替代 nanochat 的 `rustbpe` 模块。
   > - **优势**: `tokenizers` 库更容易安装，只需 `pip install tokenizers`，无需 Rust 编译
   > - **兼容性**: 完全兼容 nanochat 的 tokenizer 接口
   > - **性能**: HuggingFace tokenizers 库性能优秀，基于 Rust 实现但提供预编译的 Python 包
   > 
   > 如果以 root 用户运行 pip，会收到警告。建议使用虚拟环境，或使用 `--root-user-action=ignore` 选项（仅在明确知道自己在做什么时使用）。

3. **OpenSeek dataset**: Download the OpenSeek-Pretrain-Data-Examples dataset:
   - Option 1: Download from HuggingFace (automatic)
   - Option 2: Download manually to `OpenSeek-Pretrain-Data-Examples/` directory in the project root

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
# 对于示例数据集，默认使用 -1（处理所有数据，不限制 shards 数量）
python -m examples.nanochat_exp.dataset \
    --dataset "BAAI/OpenSeek-Pretrain-Data-Examples" \
    --num-shards -1 \
    --streaming
```

Or if you have the dataset locally:

```bash
python -m examples.nanochat_exp.dataset \
    --dataset ./OpenSeek-Pretrain-Data-Examples \
    --num-shards -1
```

This will create parquet shards in `~/.cache/openseek_nanochat/parquet_shards/` (or the directory specified by `OPENSEEK_NANOCHAT_DATA_DIR`).

### 2. Train Tokenizer (Optional but Recommended)

Train a BPE tokenizer from your data using HuggingFace tokenizers (no rustbpe needed):

```bash
# From OpenSeek root directory
python -m examples.nanochat_exp.tok_train \
    --vocab-size 50257 \
    --data-dir ~/.cache/openseek_nanochat/parquet_shards
```

This will create `tokenizer.json` in the tokenizer directory. The script uses HuggingFace tokenizers library, which is much easier to install than rustbpe (no Rust compilation needed).

### 3. Modify nanochat Training Scripts

To use OpenSeek data with nanochat, you need to modify nanochat's training scripts to use our data loader and tokenizer. You can use the automated patch script:

```bash
# From OpenSeek root directory
python -m examples.nanochat_exp.patch_nanochat --nanochat-path /path/to/nanochat
```

This will automatically modify `nanochat/scripts/base_train.py` to:
- Use OpenSeek's dataloader (from `examples.nanochat_exp.dataloader`)
- Use OpenSeek's tokenizer (from `examples.nanochat_exp.tokenizer`, uses HuggingFace tokenizers, no rustbpe)

Alternatively, you can manually modify `nanochat/scripts/base_train.py`:

```python
# Change this:
from nanochat.dataloader import tokenizing_distributed_data_loader
from nanochat.tokenizer import get_tokenizer

# To this:
from examples.nanochat_exp.dataloader import tokenizing_distributed_data_loader
from examples.nanochat_exp.tokenizer import get_tokenizer
```

### 4. Run Training

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
├── tokenizer.py             # Tokenizer wrapper using HuggingFace tokenizers (easy install)
├── tok_train.py             # Tokenizer training script (uses HuggingFace tokenizers, no rustbpe)
├── patch_nanochat.py        # Script to patch nanochat's base_train.py (replaces dataloader & tokenizer)
├── run_openseek_exp.sh      # Experiment runner script
├── train_wrapper.py         # Training wrapper script
└── README.md                # This file
```

This module can be imported via `examples.nanochat_exp` when the repository root is on `PYTHONPATH`.

## Design Notes & Configuration

### Tokenizer 说明

OpenSeek 使用 **HuggingFace tokenizers 库**替代 nanochat 的 `rustbpe` 模块：

- **优势**: 
  - 更容易安装：只需 `pip install tokenizers`，无需 Rust 编译
  - 完全兼容：提供与 nanochat tokenizer 相同的接口
  - 性能优秀：基于 Rust 实现，提供预编译的 Python 包
  
- **使用方式**: 
  - `examples.nanochat_exp.tokenizer.get_tokenizer()` 会自动使用 HuggingFace tokenizers
  - 如果找不到 tokenizer.json，会尝试从标准位置加载，或创建默认 tokenizer
  - 数据加载器会自动使用新的 tokenizer，无需修改代码
  - 训练 tokenizer 使用 `python -m examples.nanochat_exp.tok_train`（无需 rustbpe）

- **训练 Tokenizer**: 
  - 使用 `examples.nanochat_exp.tok_train` 脚本训练 BPE tokenizer
  - 完全替代 nanochat 的 `tok_train.py`，无需 rustbpe
  - 支持自定义词汇表大小、最小频率等参数
  - 输出标准的 `tokenizer.json` 文件，兼容 HuggingFace tokenizers

- **兼容性**: 
  - 如果系统中已安装 rustbpe，代码会优先尝试使用 HuggingFace tokenizers
  - 如果 HuggingFace tokenizers 不可用，会回退到 nanochat 的 rustbpe（如果可用）
  - 训练脚本完全独立，不依赖 rustbpe

### 数据加载接口说明

- `examples.nanochat_exp.dataloader.tokenizing_distributed_data_loader()` 保持与 nanochat 原生接口兼容，可直接替换 import。
- Tokenizer 使用 HuggingFace tokenizers 库（易于安装），完全兼容 nanochat 的 tokenizer 接口。
- 若需变更 tokenizer、最大长度或 padding 规则，可在 `train_wrapper.py` 中调整对应函数，再由训练脚本调用。
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