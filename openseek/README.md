# OpenSeek
## Directory Structure

```
openseek/
├── algorithm/      # Algorithm innovations and experimental implementations
├── baseline/       # Training scripts for baseline model
├── data/           # Data processing, datasets, and data mixture experiments
└── system/         # System-level optimizations and distributed training
```

## Modules

### `baseline/`

The baseline directory contains scripts for setting up and running the OpenSeek baseline model experiments. These serve as standardized reference points for evaluating improvements in model architectures, training techniques, and data processing methods.

**Key components:**
- `setup.sh`: Environment setup script
- `run_exp.sh`: Training management script
- Configuration reference for the OpenSeek Small v1 Baseline (1.4B parameters)

For detailed information, see [baseline/README.md](baseline/README.md).

### `algorithm/`

This module focuses on algorithmic innovations and experiments for improving model performance, including:

**Key components:**
- `run_exp.sh`: Algorithm experiment runner
- `mtp_exp/`: Multi-task Pretraining (MTP) experiments
- `hparam_exp/`: Hyperparameter optimization experiments

These experiments explore optimizations in model architectures, attention mechanisms, and training methodologies to enhance model capabilities beyond the baseline.

### `data/`

The data module contains tools and experiments related to dataset creation, preprocessing, and optimization.

**Key components:**
- `cci4_0/`: Implementation for the CCI4.0-M2 dataset, including quality classification and evaluation
- `data_mix_exp/`: Data mixture experiments for optimizing training data composition

This directory includes code for processing raw data, implementing quality filters, and experiments with different data mixture strategies.

### `system/`

The system directory focuses on scalability, efficiency, and distributed training optimizations.

**Key components:**
- Distributed training system improvements
- Memory optimization techniques
- Multi-device coordination implementations
- Performance profiling and monitoring tools

## Getting Started

The details can be found in each sub dir.

## References

- [OpenSeek Project Repository](https://github.com/FlagAI-Open/OpenSeek)
- [FlagScale Framework](https://github.com/FlagOpen/FlagScale)
- [CCI4.0-M2 Dataset](https://huggingface.co/datasets/BAAI/CCI4.0-M2-Base-v1)
- [OpenSeek-Small-v1-Baseline Model](https://huggingface.co/BAAI/OpenSeek-Small-v1-Baseline) 