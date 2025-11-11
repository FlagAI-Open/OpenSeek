# OpenSeek Utils

This directory contains utility tools and HuggingFace implementations for data preprocessing, model checkpoint conversion, and HuggingFace model utilities as part of the OpenSeek project.

## Overview

This directory contains:

1. **Utility Scripts**: Data preprocessing and checkpoint conversion tools
2. **HuggingFace Utils** (`hf_utils/`): HuggingFace model implementations and tokenizers

The tools provided here assist with:
1. Preprocessing text data for training language models
2. Converting DeepSeek V3 model checkpoints between different formats
3. HuggingFace model implementations and tokenizer utilities

## Tools

### 1. `preprocess_data_args.py`

A Python script for preprocessing text data into the format required for training OpenSeek models.

#### Features
- Converts JSON text data into indexed datasets for efficient training
- Supports sentence splitting using NLTK
- Provides Fill-in-Middle (FIM) preprocessing strategy
- Configurable for different tokenizers and model requirements

#### Usage
```bash
python preprocess_data_args.py \
    --input <input_json_file> \
    --json-keys text \
    --split-sentences \
    --model-name <tokenizer_model_name> \
    --model-dir <tokenizer_model_directory> \
    --output-prefix <output_path_prefix> \
    --workers <num_workers> \
    --chunk-size <chunk_size>
```

#### Options
- `--input`: Path to the input JSON data file
- `--json-keys`: Keys to extract from the JSON (default: 'text')
- `--split-sentences`: Split documents into sentences
- `--keep-newlines`: Keep newlines between sentences when splitting
- `--fill-in-middle`: Apply Fill-in-Middle preprocessing strategy
- `--fill-in-middle-percentage`: Percentage of data to apply FIM (default: 10)
- `--model-name`: Tokenizer model name 
- `--model-dir`: Directory to save/load tokenizer
- `--output-prefix`: Prefix for output files
- `--workers`: Number of worker processes to launch
- `--chunk-size`: Chunk size for each worker process

### 2. `convert_deepseek_v3_ckpt.sh`

A shell script for converting DeepSeek V3 model checkpoints from the FlagScale format to the Hugging Face Transformers format.

#### Features
- Creates symbolic links to avoid affecting ongoing training
- Handles conversion between different parallel configurations
- Maintains tensor precision during conversion
- Supports full model architecture parameters

#### Usage
```bash
bash convert_deepseek_v3_ckpt.sh <experiment_name> <checkpoint_version>
```

#### Requirements
- Environment variables:
  - `$FlagScale_HOME`: Path to the FlagScale repository
  - `$Checkpoints_HOME`: Path to store converted checkpoints

#### Output
The converted checkpoint will be saved in:
```
$Checkpoints_HOME/<experiment_name>/iter_<checkpoint_version>_hf
```

## Directory Structure

```
utils/
├── hf_utils/                    # HuggingFace utilities
│   ├── deepseek_v3/            # DeepSeek V3 model implementation
│   ├── aquila/                 # Aquila model implementation
│   └── tokenizer/              # Shared tokenizer implementation
├── preprocess_data_args.py     # Data preprocessing script
├── convert_deepseek_v3_ckpt.sh # Checkpoint conversion script
└── README.md                    # This file
```

## Integration with OpenSeek

These tools are essential components of the OpenSeek training pipeline:

1. `preprocess_data_args.py` is used to prepare the CCI4.0 dataset and other training data
2. `convert_deepseek_v3_ckpt.sh` enables the conversion of trained checkpoints for evaluation and deployment
3. `hf_utils/` provides HuggingFace-compatible model implementations and tokenizers used throughout the project

## Requirements

- Python 3.8+
- NLTK (for sentence splitting)
- PyTorch
- FlagScale codebase
- Transformers library 