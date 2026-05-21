from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    framework: str = "flagscale"
    provider: str = "openai_compatible"
    model_name: str = "Qwen3-4B"
    local_model_path: str = ""
    load_in_4bit: bool = False
    base_url: str = ""
    api_key: str = ""
    temperature: float = 0.1
    max_tokens: int = 512
    timeout_seconds: int = 120


@dataclass
class PipelineConfig:
    task_name: str = "long-context-annotation"
    label_field: str = "label"
    text_field: str = "text"
    id_field: str = "id"
    k_examples: int = 6
    chunk_chars: int = 1800
    max_context_chars: int = 16000
    vote_rounds: int = 3
    abstain_label: str = "unknown"


@dataclass
class PromptConfig:
    system: str = ""
    instruction: str = ""


@dataclass
class AppConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)


def _expand_env(value: Any) -> Any:
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        return os.getenv(value[2:-1], "")
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    return value


def _simple_yaml(path: str | Path) -> dict[str, Any]:
    data: dict[str, Any] = {}
    current_section: str | None = None
    current_key: str | None = None
    block_lines: list[str] = []

    def flush_block() -> None:
        nonlocal current_key, block_lines
        if current_section and current_key is not None:
            data[current_section][current_key] = "\n".join(block_lines).rstrip()
        current_key = None
        block_lines = []

    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()
        if current_key is not None and indent >= 4:
            block_lines.append(raw_line[4:])
            continue
        flush_block()
        if indent == 0 and line.endswith(":"):
            current_section = line[:-1]
            data[current_section] = {}
            continue
        if current_section and indent == 2 and ":" in line:
            key, value = line.split(":", 1)
            value = value.strip()
            if value == "|":
                current_key = key
                block_lines = []
            else:
                data[current_section][key] = _coerce_scalar(value)
    flush_block()
    return data


def _coerce_scalar(value: str) -> Any:
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    if value.startswith("'") and value.endswith("'"):
        return value[1:-1]
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def load_config(path: str | Path) -> AppConfig:
    data = _expand_env(_simple_yaml(path))
    return AppConfig(
        model=ModelConfig(**data.get("model", {})),
        pipeline=PipelineConfig(**data.get("pipeline", {})),
        prompt=PromptConfig(**data.get("prompt", {})),
    )
