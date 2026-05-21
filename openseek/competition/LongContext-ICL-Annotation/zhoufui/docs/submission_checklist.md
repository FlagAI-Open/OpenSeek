# Submission Checklist

## Before 2026-05-20

- Submit prediction results on the competition platform.
- Submit a technical report matching the final prediction method.
- Submit complete source code matching the final prediction method.
- Ensure the submitted code uses FlagScale for model loading/inference.
- Include environment setup, model loading command, inference command, and output format.
- Keep logs or command records for the final run.

## Risk Exclusion Checklist

- Do not upload an old prediction package after `82.72`; the locked best package is `outputs/zhoufui_prediction_8272_task2_noun_last_push20.zip`.
- Do not confuse the prediction zip with the source-code zip: platform prediction upload uses `zhoufui_prediction_8272_task2_noun_last_push20.zip`, while source-code upload uses `flagos_source_package.zip`.
- Do not describe WSL/Transformers as the final compliant runtime; final reproducibility must be described as FlagScale + Qwen3-4B + official data only.
- Do not mention any external data, self-built data, extra LLM, external embedding model, or fine-tuning path as part of the final method.
- Final package beat `82.00` on 2026-05-20; README, `docs/score_ledger.json`, report markdown, report PDF, reproduction guide, overrides, and source package must match `82.72` before uploading final materials.
- 
## Required Evidence For Reproducibility

- Config used for final run.
- Dataset file names and checksums if allowed.
- Exact model name: Qwen3-4B.
- Exact framework: FlagScale.
- Hardware or runtime description.
- Command used to generate final predictions.
- Output file path and upload time.
- Current best local package: `outputs/zhoufui_prediction_8272_task2_noun_last_push20.zip`.
- Current best platform score: 82.72.
- Use no-BOM UTF-8 JSONL files in uploaded zip packages.
- Run `python scripts/validate_submission_zip.py <candidate.zip>` before every platform upload.
- Final reproduction command: `bash scripts/linux_run_predictions.sh`.
- Final task 2 audit manifest: `configs/final_overrides.json`.
- Final task 8 fallback source: `src/flagos_icl/task8_torch_reference.py`.

## After 2026-05-20

- Fork `https://github.com/FlagAI-Open/OpenSeek`.
- Add the final technical report and source code in the required location.
- Open a Pull Request between 2026-05-21 and 2026-05-31.
- Submit the PR link back to the FlagOS competition platform.
