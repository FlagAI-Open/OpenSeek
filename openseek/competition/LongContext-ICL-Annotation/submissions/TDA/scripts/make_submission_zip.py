#!/usr/bin/env python3
import zipfile
from pathlib import Path

SRC_DIR = Path("outputs/final_submission")
OUT = SRC_DIR / "submission_repacked.zip"

with zipfile.ZipFile(OUT, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for task_id in range(1, 9):
        name = f"openseek-{task_id}-v1.jsonl"
        zf.write(SRC_DIR / name, arcname=name)
print(OUT)
