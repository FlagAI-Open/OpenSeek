# Data Quality Scoring with LLMs

## Overview

We provide some scripts here to score the quality of web page extracts based on their educational value using large language models (LLMs). The scoring process evaluates how useful the content is for educational purposes, assigning scores between 0 and 5, where 0 indicates no educational value and 5 represents exceptional value suitable for teaching from primary school to college levels. These scores can be used to curate datasets for training RoBERTa-based classifiers to predict data quality or filter high-quality educational content.

## Dependencies and Installation

This project requires the following Python libraries:

- `openai`
- `jsonlines`
- `tqdm`
- `loguru`

Install them using:

```bash
pip install openai jsonlines tqdm loguru
```

## Process

The scripts facilitate the following process:

1. **Prompt Selection**: Choose a prompt template that instructs the LLM to evaluate the educational value of web page extracts.
2. **Scoring with LLMs**: Use the selected prompt with an LLM to generate quality scores (0-5) for each data sample.

The scored data can then be used to train RoBERTa-based classifiers for automated quality assessment.

## Prompt Templates

We provide four prompt templates to guide the LLM in assigning scores from 0 to 5 based on educational value:

- `en-add`: English prompt using an additive scoring system, where points accumulate based on specific educational criteria.
- `en-direct`: English prompt using a direct scoring system, assigning a single score from 0 to 5 based on overall quality and utility.
- `cn-add`: Chinese prompt using an additive scoring system, similar to `en-add` but tailored for Chinese content.
- `cn-direct`: Chinese prompt using a direct scoring system, similar to `en-direct` but for Chinese content.

These prompts were selected by testing them with various LLMs on a sample dataset and comparing the resulting scores to a ground truth established by a reliable model (e.g., GPT-4). The chosen prompts demonstrated the best alignment with the ground truth, ensuring accurate and consistent quality evaluations.

## Usage

To score a dataset using an LLM, use the `request_llm.py` script.

### Example

```bash
python request_llm.py --input-path path/to/input.jsonl --output-path path/to/output.jsonl --model Qwen2.5-72B-Instruct --prompt en-add
```

### Arguments

- `--input-path`: Path to the input JSONL file containing the data to be scored (required).
- `--output-path`: Path to save the scored data (optional; defaults to a timestamped file if not specified).
- `--model`: The LLM model to use. Supported models include `"Qwen2.5-72B-Instruct"`, `"deepseek-chat"`, and `"gpt-4o-2024-11-20"` (according to `config.py`).
- `--prompt`: The prompt template to use. Choices are `"en-add"`, `"en-direct"`, `"cn-add"`, `"cn-direct"` (according to `config.py`).
- `--max-length`: Maximum length of text to process (default: 8192).
- `--text-key`: Key in the JSON object containing the text (default: `"text"`).
- `--label-key`: Key to store the score in the output JSON (default: `"label"`).
- `--threads`: Number of threads for parallel processing (default: 8).

### Configuration

Before running the script, you must configure the API keys and URLs for the LLMs in `config.py`. The file includes placeholders for both local deployments (e.g., `"Qwen2.5-72B-Instruct"`) and cloud-based APIs (e.g., `"deepseek-chat"`, `"gpt-4o-2024-11-20"`). Update these settings to match your specific setup.

## Important Notes

- **Configuration Reminder**: Update `config.py` with your API keys and URLs before running the script.
- The script supports parallel processing to efficiently handle large datasets, producing scores between 0 and 5 for training RoBERTa-based classifiers.
