import json
from pathlib import Path
from typing import Any, Dict, List

from src.ai_lab.adapters.official_reader import load_task_dataset


def validate_prediction_dir(
    prediction_root: str, registry: List[Dict[str, Any]], official_data_dir: str
) -> Dict[str, Dict[str, Any]]:
    root = Path(prediction_root)
    report: Dict[str, Dict[str, Any]] = {}

    for dataset in registry:
        task = load_task_dataset(official_data_dir, dataset)
        expected_ids = {record.sample_id for record in task.test_samples}
        file_name = f"openseek-{dataset['task_id']}-v1.jsonl"
        prediction_file = root / file_name
        status: Dict[str, Any] = {
            "prediction_file": str(prediction_file),
            "exists": prediction_file.exists(),
            "expected_count": len(expected_ids),
            "predicted_count": 0,
            "missing_ids": 0,
            "extra_ids": 0,
            "duplicate_ids": 0,
            "empty_predictions": 0,
            "valid": False,
        }

        if not prediction_file.exists():
            report[f"openseek-{dataset['task_id']}"] = status
            continue

        rows = []
        with prediction_file.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))

        predicted_ids: List[str] = [str(row.get("test_sample_id", "")) for row in rows]
        predicted_id_set = set(predicted_ids)
        missing_ids = expected_ids - predicted_id_set
        extra_ids = predicted_id_set - expected_ids
        duplicate_ids = len(predicted_ids) - len(predicted_id_set)
        empty_predictions = sum(
            1
            for row in rows
            if row.get("prediction") is None or str(row.get("prediction")).strip() == ""
        )

        status["predicted_count"] = len(rows)
        status["missing_ids"] = len(missing_ids)
        status["extra_ids"] = len(extra_ids)
        status["duplicate_ids"] = duplicate_ids
        status["empty_predictions"] = empty_predictions
        status["valid"] = (
            len(rows) == len(expected_ids)
            and not missing_ids
            and not extra_ids
            and duplicate_ids == 0
            and empty_predictions == 0
        )
        report[f"openseek-{dataset['task_id']}"] = status
    return report
