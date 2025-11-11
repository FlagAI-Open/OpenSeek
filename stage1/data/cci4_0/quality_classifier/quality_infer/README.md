# CCI4.0-ZH-HQ-Classifiers

CCI4.0 Chinese corpus quality annotation script for quality scoring of the CCI4.0 Chinese corpus. This script implements multiple quality classifiers for comprehensive text quality assessment.

## Overview

This script is specifically designed for quality scoring of the CCI4.0 Chinese corpus. Considering that a single classifier has limited recall in identifying high-quality pre-training documents, we built three independent quality classifiers to reassess Chinese pre-training data, following the approach used by Nemotron-CC for processing English Common Crawl.

The inference code combines three quality classifiers: two Roberta classifiers trained on data obtained from Qwen and DeepSeek, and a FastText-based text quality classifier. We have also incorporated a Fineweb-edu-classifier in the inference code to provide comprehensive quality assessment for Chinese text data. Each classifier contributes to the final quality score through a specific ensemble strategy.

## Quality Classifier Training

We used two large language models to annotate Chinese samples for quality:
- Built two training datasets of 460,000 samples each based on Qwen2.5-72B and DeepSeek-V3 respectively, developing two independent quality classifiers through parameter tuning
- Implemented a FastText-based classifier trained on instruction-format data and high-scoring posts, with multiple iterations to eliminate irrelevant factors

## Features

- Model Support:
  - Qwen2.5-72B based quality classifier
  - DeepSeek-V3 based quality classifier
  - FastText based quality classifier
  - Domain classifier
- Batch processing capability
- Comprehensive text preprocessing
- Processing status tracking log system
- Support for single text and batch processing
- Support for JSONL input/output format

## Requirements

- Python 3.6+
- PyTorch
- Transformers
- FastText
- jsonlines
- jieba

## Usage

Install required dependencies:

```bash
pip install torch transformers fasttext jsonlines jieba 
```

### Download the Quality Classifiers

There are 4 quality scoring models:
- Qwen2.5-72B based quality classifier —— Scorer 1
- DeepSeek-V3 based quality classifier —— Scorer 2
- FastText-based quality classifier    —— Scorer 3
- FineWeb-Edu classifier               —— Scorer 4

### Basic Usage

```bash
python data_infer.py     --roberta-model-path1 "/path/to/roberta_model (Scorer 1)" \
                         --roberta-model-path2 "/path/to/roberta_model (Scorer 2)" \
                         --deberta-model-path "/path/to/deberta_model (Scorer 4)" \
                         --fasttext-model-path "/path/to/fasttext_model (Scorer 3)" \
                         --input-file-path "/path/to/input1.jsonl" "/path/to/input2.jsonl" \
                         --output-file-dir "/path/to/output" \
                         --batch-size 16 \
                         --model "roberta-qwen" "roberta-deepseek" "deberta" "fasttext" \
                         --cuda-device 1
                         --stopwords-path "/path/to/stopwords"
```

### Input Format

Input should be JSONL files, with each line containing a JSON object that has at least a "text" field:

```json
{"text": "Your text content"}
```

### Output Format

The output will be a JSONL file containing the original content along with quality scores from each model:

```json
{
    "text": "Original text",
    "quality": {
        "fasttext_400k_train_1gram": *,
        "fasttext_400k_train_1gram_normal": *,
        "roberta_qwen": *,
        "roberta_qwen_normal" : *,
        "roberta_deepseek" : *,
        "roberta_deepseek_normal" : *,
        "fine_web" : *,
        "fine_web_normal" : *
    }
}
```

## Model Details

### Qwen2.5-72B Based Classifier
- Uses Qwen2.5-72B as the base model
- Trained on 460,000 quality-annotated samples
- Provides quality classification scores

### DeepSeek-V3 Based Classifier
- Based on DeepSeek-V3 architecture
- Trained on 460,000 quality-annotated samples
- Supports sequence classification

### FastText Based Classifier
- Implements FastText text classification
- Trained on instruction-format data
- Includes text preprocessing

### FineWeb-Edu Classifier 