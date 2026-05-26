#!/usr/bin/env python3
import json
import re
import sys
import zipfile
from pathlib import Path

EXPECTED = {1: 500, 2: 500, 3: 500, 4: 500, 5: 500, 6: 500, 7: 500, 8: 166}


def validate(zip_path: Path) -> None:
    if not zip_path.exists():
        raise FileNotFoundError(zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        names = sorted(zf.namelist())
        expected_names = [f"openseek-{i}-v1.jsonl" for i in range(1, 9)]
        if names != expected_names:
            raise ValueError(f"Unexpected zip members: {names}")
        for name in names:
            m = re.fullmatch(r"openseek-(\d+)-v1\.jsonl", name)
            task_id = int(m.group(1))
            rows = [json.loads(line) for line in zf.read(name).decode("utf-8").splitlines() if line.strip()]
            if len(rows) != EXPECTED[task_id]:
                raise ValueError(f"{name}: expected {EXPECTED[task_id]} rows, got {len(rows)}")
            for idx, row in enumerate(rows, 1):
                if "test_sample_id" not in row or "prediction" not in row:
                    raise ValueError(f"{name}:{idx}: missing test_sample_id or prediction")
                if row["prediction"] is None:
                    raise ValueError(f"{name}:{idx}: null prediction")
    print(f"OK: {zip_path}")


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("outputs/final_submission/submission_0520v11.zip")
    validate(target)
