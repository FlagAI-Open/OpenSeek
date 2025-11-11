
# README.md

This Python script computes the loss of a transformer-based causal language model on a dataset of text inputs. It uses the Hugging Face `transformers` library to load a pre-trained model and tokenizer, processes input data from a JSON file, and calculates the loss for each example. The script outputs the results with loss values appended and computes percentile statistics (e.g., 90th, 95th, 99th) using the `tdigest` library.

## requirements.txt

```plaintext
torch
transformers
numpy
tdigest
tqdm
```

To install these dependencies, run:
```bash
pip install -r requirements.txt
```

## Usage

Run the script from the command line with the following arguments:
```bash
python script.py --gpu_id <GPU_ID> --model_path <MODEL_PATH> --data_path <DATA_PATH>
```

## Arguments

- `--gpu_id` (int, required): The ID of the GPU to use (e.g., `0` for the first GPU).
- `--model_path` (str, required): Path to the pre-trained model (e.g., a Hugging Face model directory or identifier).
- `--data_path` (str, required): Path to the input JSON file containing text data.

## Input Format

The input file (`data_path`) should be a JSON file where each line is a JSON object with at least a `"text"` field:
```json
{"text": "Example text 1"}
{"text": "Example text 2"}
```

### Output
- `<data_path>.loss`: A new file with the same data as the input, but with a `"loss"` field added to each line.
- `<data_path>.stat`: A JSON file containing loss percentiles and the total count of processed examples:
  ```json
  {
    "loss_99": 2.34,
    "loss_95": 2.10,
    "loss_90": 1.95,
    "loss_80": 1.75,
    "loss_70": 1.60,
    "count": 10000
  }
  ```

## Example

```bash
python script.py --gpu_id 0 --model_path "gpt2" --data_path "input.json"
```
- Loads the `gpt2` model.
- Processes `input.json` on GPU 0.
- Outputs `input.json.loss` and `input.json.stat`.

