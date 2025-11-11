# LM-Evaluation-Harness for CCI4.0 Evaluation

This directory contains scripts and configurations for evaluating language models trained on CCI4.0 data using the EleutherAI LM-Evaluation-Harness framework.

## Overview

The LM-Evaluation-Harness is a standardized framework for evaluating large language models across a wide range of benchmarks. This implementation provides a streamlined approach to evaluate models trained on CCI4.0 data on popular English and Chinese benchmarks.

## Components

- **run_eval.sh**: Script for running a comprehensive suite of evaluations on a specified model

## Supported Benchmarks

The evaluation script covers the following benchmarks:

1. **General Understanding and Reasoning**:
   - HellaSwag: Common sense reasoning about events
   - Winogrande: Commonsense reasoning and coreference resolution
   - PIQA: Physical commonsense reasoning
   - BoolQ: Reading comprehension with boolean questions

2. **Knowledge and QA**:
   - CommonsenseQA: Question answering about common sense concepts
   - TruthfulQA: Measuring factuality and avoiding falsehoods
   - OpenBookQA: Knowledge-based question answering
   - ARC (Easy and Challenge): Science question answering

3. **Mathematics and Problem Solving**:
   - GSM8K: Grade school math word problems
   - Minerva Math: Advanced mathematical problem solving

4. **Multilingual Evaluation**:
   - MMLU: Multi-task language understanding (57 subjects)
   - C-Eval: Chinese evaluation suite
   - CMMLU: Chinese massive multitask language understanding

## Usage

```bash
# Run evaluation on a specific model
bash run_eval.sh <model_directory>
```

The script will sequentially run all evaluation tasks and save the results to an output directory.

## Key Features

- **Standardized Evaluation**: Uses the widely accepted EleutherAI LM-Evaluation-Harness for comparable results
- **Few-shot Learning**: Most tasks are evaluated in a few-shot setting (typically 5 examples)
- **Comprehensive Coverage**: Tests models across reasoning, knowledge, mathematics, and multilingual capabilities
- **Detailed Metrics**: Provides accuracy, F1 scores, and other task-specific metrics

## Integration with CCI4.0

This evaluation framework helps assess the effectiveness of the CCI4.0 dataset in training language models by measuring:

1. Knowledge retention across various domains
2. Reasoning capabilities in different contexts
3. Multilingual performance (particularly English and Chinese)
4. Mathematical and problem-solving abilities

## References

- [LM-Evaluation-Harness GitHub Repository](https://github.com/EleutherAI/lm-evaluation-harness)
- [EleutherAI](https://www.eleuther.ai/)
- [CCI4.0 Dataset](https://huggingface.co/datasets/BAAI/CCI4.0-M2-Base-v1) 