# RoBERTa Quality Classifier Deploying and Scoring

- **`scorer_eval_local.py`**: Evaluates the performance of a pre-trained RoBERTa quality classifier on a labeled dataset. It computes and displays metrics such as accuracy, precision, recall, and F1 score, along with a confusion matrix and classification report.
- **`scorer_pred_local.py`**: Applies the RoBERTa quality classifier to dataset, scoring each sample and optionally filtering out low-quality samples based on a specified threshold. The scored results are saved to an output file.

Both scripts leverage the `transformers` library to load and use a RoBERTa-based sequence classification model fine-tuned for quality scoring.

## Dependencies

To run these scripts, install the following Python packages:

- `torch`
- `transformers`
- `jsonlines`
- `scikit-learn` (required only for `scorer_eval_local.py`)

Install them using pip:

```bash
pip install torch transformers jsonlines scikit-learn
```

## Usage: `scorer_eval_local.py`

This script assesses the quality classifier's performance on a labeled dataset provided in JSONL format, where each line contains a `"text"` field (the input text) and a `"score"` field (the true quality score).

### Arguments

- **`--input-file-path`** (required): Path to the input JSONL file with labeled data.
- **`--scorer-model-path`** (required): Path to the directory containing the pre-trained RoBERTa model.
- **`--score-thres`** (optional, default=3.0): Threshold for binary classification. Predicted scores below this value are labeled as 1 (low quality), otherwise 0.
- **`--max-length`** (optional, default=2048): Maximum sequence length for tokenization.

### How It Works

1. Loads the model and tokenizer from the specified `scorer-model-path`.
2. Processes the input file line-by-line, tokenizing each `"text"` and predicting a score.
3. Compares predicted scores against the threshold to assign labels (1 or 0).
4. Uses the `"score"` field to determine true labels (1 if `< 3`, else 0).
5. Outputs a confusion matrix, classification report, and metrics (accuracy, precision, recall, F1).

### Example

```bash
python evaluate_model.py --input-file-path path/to/labeled_data.jsonl --scorer-model-path path/to/model
```

## Usage: `scorer_pred_local.py`

This script scores new data using the quality classifier and writes the results to an output file, with an option to filter out samples based on their scores.

### Arguments

- **`--scorer-model-path`** (required): Path to the directory containing the pre-trained RoBERTa model.
- **`--input-file-path`** (required): Path to a single JSONL file or a directory of JSONL files to score.
- **`--output-file-path`** (required): Path to the output JSONL file where scored data will be saved.
- **`--score-thres`** (optional, default=3.0): Threshold for filtering (used only if `--do-score-filter` is set).
- **`--text-key`** (optional, default="text"): Key in the input JSONL lines containing the text to score.
- **`--output-key`** (optional, default="score"): Key under which the predicted score will be stored in the output.
- **`--do-score-filter`** (optional): If specified, only samples with scores ≥ `score-thres` are written to the output.

### How It Works

1. Loads the model and tokenizer from the specified `scorer-model-path`.
2. Processes the input file(s), tokenizing the text under `text-key` and predicting a score.
3. Adds the score to each line under `output-key`.
4. If `--do-score-filter` is enabled, skips samples with scores below `score-thres`.
5. Writes the processed lines to the output file and reports progress/performance metrics.

### Example

```bash
python score_data.py --scorer-model-path path/to/model --input-file-path path/to/data.jsonl --output-file-path path/to/scored_data.jsonl --do-score-filter --score-thres 3.0
```

## Notes

- **GPU Requirement**: Both scripts assume a CUDA-compatible GPU is available. To run on CPU, remove `.cuda()` and `.to("cuda")` calls from the code.
- **Model Compatibility**: The model should be a RoBERTa-based sequence classification model that outputs a single score per input.
- **Input Format**: Input files must be in JSONL format, with each line being a valid JSON object containing at least the specified `text-key` (default: `"text"`) for `scorer_pred_local.py`, and both `"text"` and `"score"` for `scorer_eval_local.py`.

