from pathlib import Path
from typing import Any, Dict

from src.ai_lab.config import load_yaml


def load_protocol(path: str) -> Dict[str, Any]:
    return load_yaml(path)


def render_protocol(template: str, values: Dict[str, Any]) -> str:
    return template.format(**values)


def protocol_exists(path: str) -> bool:
    return Path(path).exists()
