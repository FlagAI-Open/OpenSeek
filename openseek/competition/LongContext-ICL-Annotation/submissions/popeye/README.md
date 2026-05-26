# 大力水手 — LongContext-ICL-Annotation Submission

This directory contains the open-source submission of team **大力水手 (Popeye)** for the
[FlagOS Open Computing Global Challenge](https://www.kaggle.com/competitions/flag-os-open-computing-global-challenge),
track **LongContext-ICL-Annotation**.

## 1. Files in this directory

| File | Description |
|------|-------------|
| `技术报告-大力水手.pdf` | Final technical report (Chinese). |
| `final_submission_v48.zip` | The 8-task `.jsonl` prediction archive. Contains `openseek-1-v1.jsonl` … `openseek-8-v1.jsonl` directly at the zip root, conforming to the format described in `outputs/README.md` of this repo. |
| `code/` | Full reproducible source code, data, environment files, deployment script, and supporting documentation. |

## 2. Quick reproduction guide

The `code/` directory is self-contained. The full instructions are in `code/README.md`;
the abridged path is:

```bash
cd code
conda env create -f environment_h20.yml
conda activate openseek-h20
pip install -r requirements.txt

# Download Qwen3-4B locally and edit src/llm_config.yaml (model + tokenizer paths).
bash bootstrap_h20.sh start         # launch vLLM service
bash bootstrap_h20.sh api-test      # smoke-check the API

# Per-task smoke test
python src/main.py --task_id 1 --sample_limit 5 --tokenizer_path /path/to/Qwen3-4B

# Full run (8 tasks)
python src/run_all_tasks.py --tokenizer_path /path/to/Qwen3-4B \
                            --output_dir ../outputs/first_submission

# Reproduce the v48 task7 candidate (the version this submission corresponds to)
python src/build_task7_v48_fact2_candidate.py
python src/check_task7_v48_candidate_smoke.py
python src/write_v48_task7_online_readout.py
```

## 3. Method summary

The full method is documented in `技术报告-大力水手.pdf`. In short, the solution
stays on the official Qwen3-4B + YaRN long-context baseline and gains score
through a controlled "main-task method line + changed-row audit + ultra-narrow
factual patch" workflow. Per-version evidence (`verified_patch_notes`,
`candidate_build`, `candidate_smoke`, `online_readout`, and the
`official_score_ledger`) is preserved under `code/docs/phase-summaries/`.

## 4. Team

- Team name: **大力水手 (Popeye)**
- Track: LongContext-ICL-Annotation
