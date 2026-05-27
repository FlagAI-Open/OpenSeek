from dataclasses import dataclass
from typing import Any, Dict, List

from src.ai_lab.datasets import load_official_task


@dataclass
class SampleRecord:
    sample_id: str
    task_id: int
    task_name: str
    task_type: str
    instruction: str
    text: str
    label_space: List[str]


@dataclass
class TaskDataset:
    task_id: int
    task_name: str
    task_type: str
    definition: str
    examples: List[Dict[str, Any]]
    test_samples: List[SampleRecord]
    file_name: str


def _extract_label_space(examples: List[Dict[str, Any]], limit: int = 32) -> List[str]:
    seen: List[str] = []
    for example in examples:
        output = example.get("output", "")
        if isinstance(output, list):
            output = output[0] if output else ""
        output = str(output).strip()
        if not output or output in seen:
            continue
        seen.append(output)
        if len(seen) >= limit:
            break
    return seen


def load_task_dataset(data_dir: str, registry_entry: Dict[str, Any]) -> TaskDataset:
    raw_task = load_official_task(data_dir, registry_entry["file_name"])
    definition_list = raw_task.get("Definition", [])
    definition = definition_list[0] if definition_list else ""
    examples = raw_task.get("examples", [])
    label_space = _extract_label_space(examples)

    records: List[SampleRecord] = []
    for sample in raw_task.get("test_samples", []):
        records.append(
            SampleRecord(
                sample_id=str(sample["id"]),
                task_id=int(registry_entry["task_id"]),
                task_name=str(registry_entry["task_name"]),
                task_type=str(registry_entry["task_type"]),
                instruction=definition,
                text=str(sample.get("input", "")),
                label_space=label_space,
            )
        )

    return TaskDataset(
        task_id=int(registry_entry["task_id"]),
        task_name=str(registry_entry["task_name"]),
        task_type=str(registry_entry["task_type"]),
        definition=definition,
        examples=examples,
        test_samples=records,
        file_name=str(registry_entry["file_name"]),
    )
