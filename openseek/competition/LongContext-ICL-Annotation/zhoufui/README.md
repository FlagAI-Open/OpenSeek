# zhoufui Final Submission

Final open-source submission for FlagOS Track 3 / OpenSeek LongContext-ICL-Annotation.

- Team: zhoufui
- Final platform score: 82.72
- Final prediction package used during the competition phase: outputs/zhoufui_prediction_8272_task2_noun_last_push20.zip
- Technical report: technical_report_zhoufui.pdf / technical_report_zhoufui.md
- Complete reproducible source: flagos_source_package.zip

The source zip contains the FlagScale/Qwen3-4B reproduction entrypoint, configuration files, task 8 PyTorch fallback, final task 2 override manifest, documentation, tests, and scripts. It was audited before upload: no model weights, no official test data, no outputs directory, no .env/credentials, and no pycache.

Primary reproduction command after extracting flagos_source_package.zip:

```bash
bash scripts/linux_run_predictions.sh
```

Final task 2 audit manifest after extraction:

```text
configs/final_overrides.json
```
