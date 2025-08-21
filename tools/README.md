# OpenSeek Tools

This directory contains utility tools for data preprocessing and model checkpoint conversion as part of the OpenSeek project.

## Overview

The tools provided here assist with:
1. Preprocessing text data for training language models
2. Converting DeepSeek V3 model checkpoints between different formats

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

### 2. `pyedu_dataset_utils.py`

A Python utility for downloading, preprocessing, and integrating the PyEdu dataset (educational Python code from smollm-corpus) into OpenSeek training pipelines.

#### Features
- Downloads PyEdu dataset from Hugging Face Hub
- Preprocesses data for OpenSeek training with Fill-in-Middle (FIM) strategy optimized for code
- Validates dataset quality and provides statistics
- Generates training configuration files
- Supports educational Python code with ~6GB of high-quality content

#### Usage
```bash
# Download PyEdu dataset
python pyedu_dataset_utils.py download --output-dir ./data/pyedu

# Preprocess for training
python pyedu_dataset_utils.py preprocess \
    --input ./data/pyedu/pyedu_raw.jsonl \
    --output-prefix ./data/pyedu/pyedu \
    --workers 8

# Validate dataset
python pyedu_dataset_utils.py validate --dataset-path ./data/pyedu/pyedu_raw.jsonl

# Create training configuration
python pyedu_dataset_utils.py create-config \
    --dataset-path ./data/pyedu/pyedu_text_sentence \
    --output-config ./configs/pyedu_config.yaml
```

#### Requirements
- `datasets` library for Hugging Face dataset access
- `huggingface_hub` for dataset downloads
- Existing OpenSeek preprocessing dependencies

### 3. `convert_deepseek_v3_ckpt.sh`

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

## Integration with OpenSeek

These tools are essential components of the OpenSeek training pipeline:

1. `preprocess_data_args.py` is used to prepare the CCI4.0 dataset and other training data
2. `pyedu_dataset_utils.py` enables integration of high-quality educational Python code data for enhanced code understanding
3. `convert_deepseek_v3_ckpt.sh` enables the conversion of trained checkpoints for evaluation and deployment

## Requirements

- Python 3.8+
- NLTK (for sentence splitting)
- PyTorch
- FlagScale codebase
- Transformers library 