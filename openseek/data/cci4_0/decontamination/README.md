# decontamination
## Overview

This script generates a decontamination data and report for a dataset by comparing a training dataset against an evaluation dataset. It identifies potential data leakage using n-gram similarity and filters contaminated samples based on a specified threshold.

## Requirements

- Python 3.x
- Required dependencies (install via `pip install -r requirements.txt` if applicable)

## Usage

Run the script from the command line with the following arguments:

```sh
python decontaminate.py --train_dataset <path_to_train_dataset> \
                 --report_dataset_name <path_to_output_report> \
                 [--save_decontaminated] \
                 [--eval_dataset <path_to_eval_dataset>] \
                 [--decontaminated_dataset_name <path_to_decontaminated_dataset>] \
                 [--ngram_length <n>] \
                 [--diff_threshold <threshold>] \
                 [--num_proc <num_processes>]
```

## Arguments

| Argument                        | Type    | Default      | Description                                                  |
| ------------------------------- | ------- |--------------| ------------------------------------------------------------ |
| `--eval_dataset`                | string  | `eval.jsonl` | Name of the dataset with benchmark samples for decontamination. |
| `--train_dataset`               | string  | required     | Path or name of the training dataset to process.             |
| `--report_dataset_name`         | string  | None         | Path of the output dataset with the decontamination report (`report.jsonl`). |
| `--decontaminated_dataset_name` | string  | None         | Path of the decontaminated dataset (optional).               |
| `--ngram_length`                | integer | `10`         | Length of the n-grams to consider.                           |
| `--diff_threshold`              | float   | `0.85`       | Threshold for filtering based on difference ratio.           |
| `--num_proc`                    | integer | `16`         | Number of processes to use for map operations.               |
| `--save_decontaminated`         | flag    | `False`      | Whether to save the decontaminated dataset.                  |

## Example Usage

```sh
python decontaminate.py --train_dataset train.jsonl \
                 --report_dataset_name report.jsonl \
                 --ngram_length 8 \
                 --diff_threshold 0.9 \
                 --num_proc 8 \
                 --save_decontaminated
```

## Output

- (Optional) `report.jsonl`: A jsonl file containing the decontamination samples with diff rates.
- (Optional) `decontaminated_dataset.jsonl`: A cleaned dataset with contaminated samples removed.

# 📜 License Agreement
- Code is licensed under Apache 2.0
- Model weights are licensed under Apache 2.0
- Data is licensed under CC BY-SA 4.0

