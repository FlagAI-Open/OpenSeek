from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .data import Record


@dataclass(frozen=True)
class OfficialTask:
    task_id: str
    task_name: str
    definition: str
    examples: list[Record]
    test_samples: list[Record]
    source_path: Path


def _join_definition(value: Any) -> str:
    if isinstance(value, list):
        return "\n".join(str(item) for item in value)
    return str(value)


def normalize_output(value: Any) -> str:
    if isinstance(value, list):
        if len(value) == 1:
            return str(value[0])
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def load_official_task(path: str | Path) -> OfficialTask:
    source = Path(path)
    data = json.loads(source.read_text(encoding="utf-8"))
    task_id = str(data["task_id"])
    task_name = str(data.get("task_name", task_id))
    examples = [
        Record(
            record_id=str(item.get("id", f"example-{index}")),
            text=str(item["input"]),
            label=normalize_output(item.get("output", "")),
            raw=item,
        )
        for index, item in enumerate(data.get("examples", []), start=1)
    ]
    test_samples = [
        Record(
            record_id=str(item.get("id", f"test-{index}")),
            text=str(item["input"]),
            label=None,
            raw=item,
        )
        for index, item in enumerate(data.get("test_samples", []), start=1)
    ]
    return OfficialTask(
        task_id=task_id,
        task_name=task_name,
        definition=_join_definition(data.get("Definition", "")),
        examples=examples,
        test_samples=test_samples,
        source_path=source,
    )


def load_official_tasks(data_dir: str | Path) -> list[OfficialTask]:
    files = sorted(Path(data_dir).glob("openseek-*.json"))
    if not files:
        raise FileNotFoundError(f"No official openseek-*.json files found in {data_dir}")
    return [load_official_task(path) for path in files]
