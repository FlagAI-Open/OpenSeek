import json
from pathlib import Path
from typing import Any, Dict, List

from src.ai_lab.config import load_yaml


def load_registry(registry_path: str) -> List[Dict[str, Any]]:
    config = load_yaml(registry_path)
    return config.get("datasets", [])


def load_official_task(data_dir: str, file_name: str) -> Dict[str, Any]:
    task_path = Path(data_dir) / file_name
    with task_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)
