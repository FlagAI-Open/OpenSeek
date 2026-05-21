# Reproduction Guide

[中文版 (Chinese Version)](README_zh.md)

This document is intended for **organizers / reproducers**, providing complete steps to reproduce inference results for all 8 tasks from scratch.

---

## 1. Clone the Repository

```bash
git clone https://github.com/FlagAI-Open/OpenSeek.git
cd OpenSeek/openseek/competition/LongContext-ICL-Annotation
# This submission is under the COM/ directory
cd COM
```

---

## 2. Set Up the Environment

Reproducers may adjust according to their own environment:

```bash
# Create conda environment (optional, skip if Python 3.11 is already available)
conda init bash && source ~/.bashrc
conda create -n flagscale python=3.11.11 -y
conda activate flagscale

# Install Python dependencies
pip install -r requirements.txt -i https://pypi.mirrors.ustc.edu.cn/simple/

# Download Qwen3-4B model weights
modelscope download --model Qwen/Qwen3-4B --cache_dir ./models
```

After download, model weights will be located at `COM/models/Qwen/Qwen3-4B`.

---

## 3. Configure Paths

Edit `src/common/paths.py` and modify the following variables to match your local environment:

```python
# Modify these lines for reproduction ↓↓↓
COM_ROOT = '/your/path/to/COM'                        # Absolute path to the COM directory
MODEL_DIR = COM_ROOT + '/models/Qwen/Qwen3-4B'       # Model weights location
VLLM_MODEL_ID = '../models/Qwen/Qwen3-4B'            # Must match the model field in llm_config.yaml
```

Other paths (`DATA_DIR` / `FINAL_OUTPUT_DIR` / `SRC_DIR`, etc.) are automatically derived from `COM_ROOT` and do not need modification.

Run the self-check to verify paths are correct:

```bash
cd src && python -m common.paths
```

---

## 4. Start the Model Service

```bash
git clone https://github.com/FlagOpen/FlagScale.git
cd FlagScale
python run.py --config-path ../env --config-name llm_config action=run
```

Once started, the service listens on `localhost:2026` and provides an OpenAI-compatible API.

Verify deployment with the API connectivity test:

```bash
cd ..  # Back to COM/
python env/api_test.py
```

If a normal response is returned, the model service is ready.

---

## 5. Prepare Data

The `data/` directory contains the official competition data, with files named as:

```
data/
├── openseek-1_closest_integers.json
├── openseek-2_count_nouns_verbs.json
├── openseek-3_collatz_conjecture.json
├── openseek-4_conala_concat_strings.json
├── openseek-5_semeval_2018_task1_tweet_sadness_detection.json
├── openseek-6_mnli_same_genre_classification.json
├── openseek-7_jeopardy_answer_generation_all.json
└── openseek-8_kernel_generation.json
```

---

## 5.1 (Optional) Regenerate CoT Data & Normalize Task 8 Data

> The following steps are **NOT required** — pre-generated CoT data (`src/taskN/cot_data/`) and normalized Task 8 data (`src/task8/normalized_data/`) are already included in the submission. If you wish to reproduce from scratch, run the commands below (model service must be running).

```bash
# Generate CoT for each task (Task 1/2/3/4/6, ~1-3 hours each)
cd src/task1 && python generate_cot.py && cd ../..
cd src/task2 && python generate_cot.py && cd ../..
cd src/task3 && python generate_cot.py && cd ../..
cd src/task4 && python generate_cot.py && cd ../..
cd src/task6 && python generate_cot.py && cd ../..

# Normalize Task 8 raw data (completes in seconds)
cd src/task8 && python normalize_dataset.py && cd ../..
```

---

## 6. Run All Tasks

```bash
bash scripts/run_all.sh          # Run all 8 tasks
bash scripts/run_all.sh 1 3 5    # Selectively run specified tasks
bash scripts/run_all.sh 3-7      # Run a range of tasks
```

**Runtime Reference** (single 4090D GPU):

| Task | Total Time | Avg Speed | Notes |
|------|-----------|-----------|-------|
| Task 1 | ~2h 14min | 16.1s/sample | Single-round inference |
| Task 2 | ~1h 30min | 10.8s/sample | Single-round inference |
| Task 3 | ~2h 11min | 15.7s/sample | Single-round inference |
| Task 4 | ~1h 50min | 13.2s/sample | Single-round inference |
| Task 5 | ~3h 19min | 23.8s/sample | 5-round voting |
| Task 6 | ~4h 31min | 32.5s/sample | 3-round voting |
| Task 7 | ~6h 59min | 50.3s/sample | 3 strategies + verifier + retry |
| Task 8 | ~4h 26min | 96.0s/sample | 6-stage pipeline |

---

## 7. Verify Outputs

```bash
bash scripts/verify_outputs.sh
```

The final 8 prediction files are located at `outputs/openseek-{1..8}-v1.jsonl`.

---

## 8. Stop the Model Service

```bash
cd FlagScale
python run.py --config-path ../env --config-name llm_config action=stop
```
