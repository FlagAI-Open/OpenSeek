# Local Baseline Workflow

This note turns the official single-task example into a practical local workflow for producing a first submission.

## 1. Prepare dependencies

At minimum, you need:

- `requests`
- `tqdm`
- `transformers` (optional, only for prompt length checks)
- a serving stack that exposes `POST /v1/completions`

## 2. Download Qwen3-4B

Put the model in a local folder, for example:

```bash
hf download Qwen/Qwen3-4B --local-dir /path/to/models/Qwen3-4B
```

If you use long-context YaRN scaling, update `config.json` as described in the official README.

## 3. Start the model service

Export the runtime variables first. `llm_config.yaml` reads them automatically, so you usually do not need to edit the file by hand:

```bash
export OPENSEEK_MODEL_NAME=/path/to/models/Qwen3-4B
export OPENSEEK_TOKENIZER_PATH=/path/to/models/Qwen3-4B
export OPENSEEK_VLLM_URL=http://127.0.0.1:2026/v1/completions
export OPENSEEK_VLLM_HOST=0.0.0.0
export OPENSEEK_VLLM_PORT=2026
```

Then start the service from the `FlagScale` repo:

```bash
cd /path/to/FlagScale
python run.py --config-path ../OpenSeek/openseek/competition/LongContext-ICL-Annotation/src --config-name llm_config action=run
```

## 4. Smoke test one task

From the competition `src` directory:

```bash
cd /path/to/OpenSeek/openseek/competition/LongContext-ICL-Annotation/src
python main.py --task_id 1 --sample_limit 5
```

## 5. Generate a first full submission

```bash
cd /path/to/OpenSeek/openseek/competition/LongContext-ICL-Annotation/src
python run_all_tasks.py --output_dir ../outputs/first_submission
```

This creates:

- `openseek-1-v*.jsonl` through `openseek-8-v*.jsonl`
- `baseline_submission.zip`
- `run_summary.json`

## 6. Recommended next edits

Customize `method.py`:

- improve example selection
- tighten output extraction
- add retries / validation for malformed labels
- vary prompts by task instead of using one generic template
