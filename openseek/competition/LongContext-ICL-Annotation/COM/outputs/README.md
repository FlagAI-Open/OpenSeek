# Final Predictions

This directory holds the **final 8 prediction JSONL files** produced by each task's `main.py`.
每个 task 的 `main.py` 通过 `common.paths.final_output_file(task_id, version)` 直接写到本目录，文件名带版本号 `v{N}`，每跑一次自动递增。

```
outputs/
├── openseek-1-v{N}.jsonl
├── openseek-2-v{N}.jsonl
├── openseek-3-v{N}.jsonl
├── openseek-4-v{N}.jsonl
├── openseek-5-v{N}.jsonl
├── openseek-6-v{N}.jsonl
├── openseek-7-v{N}.jsonl
└── openseek-8-v{N}.jsonl
```

Each line is a JSON object with at least the `prediction` field. 首次跑即 `openseek-{i}-v1.jsonl`；提交时取每个 task 的最新 v{N} 即可，无需重命名/汇聚。
