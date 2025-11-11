# Data Mixture Experiments

This directory contains configuration files for data mixture experiments in the OpenSeek project. These experiments focus on optimizing the composition and weighting of different data sources for training large language models.

## Overview

The data mixture experiments aim to identify the optimal balance of various data types, domains, and qualities to improve model performance. By adjusting the weights of different data subsets, we can systematically analyze how data composition affects model learning and capabilities.

## Configuration Files

- `config_deepseek_v3_16b.yaml`: Experiment-level configuration defining the training environment, distributed setup, and backend engine
- `train_deepseek_v3_16b.yaml`: Task-level configuration specifying model parameters, optimizer settings, and the data mixture composition

## Data Mixture Configuration

The data mixture is defined in `train_deepseek_v3_16b.yaml` under the `data.data_path` section. Each dataset is assigned a weight (the first value) followed by its path (the second value). For example:

```yaml
data:
  data_path:
    - 4.08  # Weight for the dataset
    - OpenSeek/k73_edu_qwen_text_document  # Path to the dataset
    # ... more datasets with their weights
```

The mixture includes various data categories:
- Educational content
- Code in multiple programming languages
- High-quality web content
- Synthetic question-answering pairs
- Mathematical content
- Academic papers (arXiv)

## Running Experiments

To run a data mixture experiment:

1. Ensure you have all required datasets available in the paths specified
2. Configure the experiment parameters in `config_deepseek_v3_16b.yaml` (nodes, GPUs, etc.)
3. Adjust dataset weights in `train_deepseek_v3_16b.yaml` based on your experimental design
4. Execute using the OpenSeek framework's experiment runner

## Experiment Design

When designing data mixture experiments:
1. Consider the balance between different data domains
2. Adjust weights based on data quality and relevance to target tasks
3. Monitor training metrics to assess the effectiveness of different mixtures
4. Compare performance on downstream evaluation tasks to measure improvements

## References

For more information on data mixture strategies and related experiments, refer to the documentation in the `docs/data_mixture_exp.md` and `docs/data_mixture_exp_results.md` files in the OpenSeek repository.
