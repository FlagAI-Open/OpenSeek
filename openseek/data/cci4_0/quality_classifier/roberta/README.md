# RoBERTa Quality Classifier for CCI4.0

This directory contains the RoBERTa-based quality classifier used for assessing and filtering content in the CCI4.0-M2 dataset pipeline. The classifier helps categorize content into high, medium, and low quality tiers, which is essential for creating high-quality training data for language models.

## Overview

The quality classifier is built on the RoBERTa architecture and fine-tuned to predict the quality of text documents. It serves as a critical component in the data pipeline, enabling efficient filtering and prioritization of web-crawled content.

## Features

- Text quality classification into high/medium/low categories
- Designed specifically for bilingual (Chinese and English) content
- Optimized for web-crawled documents from diverse sources
- Trained on manually labeled quality assessment data

## Model Details

- Base architecture: RoBERTa
- Input: Text documents (with appropriate truncation/padding)
- Output: Quality classification scores
- Training data: Manually annotated subset of web documents with quality labels

## Usage

### Prerequisites

```bash
pip install transformers torch scikit-learn numpy tqdm
```

### Example Code

The details can be seen in each sub-dir.

## Training

The model was trained on a diverse corpus of labeled documents. The training process involved:

1. Manual annotation of a subset of documents based on quality criteria
2. Fine-tuning RoBERTa on this labeled dataset
3. Validation with human reviewers to ensure alignment with quality standards

## Integration with CCI4.0 Pipeline

The quality classifier is integrated into the CCI4.0 data processing pipeline to:

1. Filter out low-quality content
2. Create data subsets based on quality tiers
3. Enable quality-based weighting in training data mixtures


## References

- [CCI4.0-M2 Dataset](https://huggingface.co/datasets/BAAI/CCI4.0-M2-Base-v1)
- [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692)
