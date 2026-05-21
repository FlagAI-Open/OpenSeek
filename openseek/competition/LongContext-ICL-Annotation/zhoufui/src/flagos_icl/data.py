from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class Record:
    record_id: str
    text: str
    label: str | None
    raw: dict[str, Any]


def read_jsonl(path: str | Path, id_field: str, text_field: str, label_field: str) -> list[Record]:
    records: list[Record] = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            if text_field not in item:
                raise ValueError(f"{path}:{line_no} is missing text field '{text_field}'")
            record_id = str(item.get(id_field, f"row-{line_no}"))
            label = item.get(label_field)
            records.append(
                Record(record_id=record_id, text=str(item[text_field]), label=None if label is None else str(label), raw=item)
            )
    return records


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
