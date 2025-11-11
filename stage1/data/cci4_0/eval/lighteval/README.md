# LightEval Framework for CCI4.0 Evaluation

This directory contains customized scripts and configurations for evaluating language models trained on CCI4.0 data using the LightEval framework.

## Overview

LightEval is an efficient framework for evaluating large language models across a diverse set of benchmarks. The customized implementation in this directory supports both English and Chinese evaluation benchmarks, making it particularly suitable for bilingual models trained on CCI4.0 data.

## Components

- **lighteval_tasks_v3.py**: Custom task definitions for LightEval, including:
  - Common sense reasoning tasks (hellaswag, winogrande, piqa, siqa, etc.)
  - MMLU (Massive Multitask Language Understanding)
  - MMLU-Pro (Enhanced MMLU benchmark)
  - CMMLU (Chinese MMLU benchmark)
  - C-Eval (Chinese Evaluation benchmark)
  - Various question-answering and reasoning tasks

- **run_lighteval_v3_mgpu.sh**: Multi-GPU execution script for running comprehensive evaluations

## Task Categories

The evaluation suite covers multiple categories of tasks:

1. **Common Sense Reasoning**: Tests the model's ability to understand everyday situations and make logical inferences
2. **Knowledge**: Evaluates the model's grasp of factual information across various domains
3. **Mathematics**: Tests mathematical reasoning and problem-solving abilities
4. **Chinese-specific**: Evaluates performance on Chinese language, culture, and knowledge

## Usage

```bash
# Run evaluation on a specific model
bash run_lighteval_v3_mgpu.sh <model_directory>
```

The script will run multiple evaluation tasks in parallel across multiple GPUs and save the results to an output directory.

## Key Features

- **Multi-GPU Support**: Distributes evaluation tasks across multiple GPUs for faster evaluation
- **Customized Prompts**: Modified prompts to improve performance with smaller models and non-instruction tuned models
- **Bilingual Evaluation**: Comprehensive coverage of both English and Chinese benchmarks
- **Diverse Task Coverage**: Tests abilities across reasoning, knowledge, mathematics, and more

## Integration with CCI4.0

This evaluation framework is specifically designed to assess the quality and capabilities of models trained on the CCI4.0 dataset. It helps measure how effective the dataset is at teaching:

1. Factual knowledge
2. Reasoning capabilities
3. Language understanding
4. Cross-lingual transfer

## References

- [LightEval GitHub Repository](https://github.com/huggingface/lighteval)
- [Hugging Face's Open LLM Leaderboard](https://huggingface.co/open-llm-leaderboard) 