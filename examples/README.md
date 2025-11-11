# OpenSeek Examples

This directory contains example code, tutorials, and use cases demonstrating how to use OpenSeek.

## Structure

```
examples/
├── baseline/          # Baseline training example
│   ├── setup.sh       #   Environment setup script
│   ├── run_exp.sh     #   Training script
│   └── README.md      #   Usage documentation
├── nanochat_exp/      # nanochat integration example
│   ├── dataset.py     #   Dataset conversion utilities
│   ├── dataloader.py  #   Data loader for nanochat
│   ├── train_wrapper.py # Training wrapper script
│   └── README.md      #   Usage documentation
└── data_mix_exp/      # Data mixture experiments example
    ├── config_deepseek_v3_16b.yaml  #   Experiment configuration
    ├── train_deepseek_v3_16b.yaml   #   Training configuration
    └── README.md      #   Usage documentation
```

## Available Examples

### Baseline Training
The baseline example provides a complete training pipeline for the OpenSeek-Small-v1-Baseline model.

**Features:**
- Environment setup script
- Standardized training scripts
- Configuration for 1.4B parameter model

**Quick Start:**
```bash
# Setup environment
bash examples/baseline/setup.sh

# Run training
bash examples/baseline/run_exp.sh start
```

See [baseline/README.md](baseline/README.md) for detailed instructions.

### nanochat Integration
The nanochat example demonstrates how to use OpenSeek datasets with the nanochat training framework.

**Features:**
- Dataset conversion from OpenSeek format to nanochat parquet format
- Data loader compatible with nanochat training pipeline
- Training wrapper scripts

**Quick Start:**
```bash
# Convert dataset
python -m examples.nanochat_exp.dataset \
    --dataset "BAAI/OpenSeek-Pretrain-100B" \
    --num-shards 240

# Use with nanochat training
# (See nanochat_exp/README.md for details)
```

See [nanochat_exp/README.md](nanochat_exp/README.md) for detailed instructions.

### Data Mixture Experiments
The data mixture example demonstrates how to conduct experiments comparing different data mixture strategies for LLM training.

**Features:**
- Configuration files for data mixture experiments
- Comparison of No-Sampling vs Phi4-Sampling strategies
- Evaluation across multiple benchmarks

**Quick Start:**
```bash
# See the README for detailed instructions
# bash stage1/algorithm/run_exp.sh start <config-path>
```

See [data_mix_exp/README.md](data_mix_exp/README.md) for detailed instructions.

## Running Examples

Each example directory contains a README with specific instructions. Generally:

```bash
# Navigate to example directory
cd examples/quickstart/

# Follow the README instructions
python example_script.py
```

## Contributing Examples

We welcome example contributions! When adding examples:

1. Create a clear directory structure
2. Include a README with:
   - Description
   - Prerequisites
   - Step-by-step instructions
   - Expected output
3. Add comments in code explaining key concepts
4. Test your examples before submitting

## Requirements

Examples may have specific dependencies. Check each example's README for requirements.

Common dependencies:
```bash
pip install torch transformers datasets
```

