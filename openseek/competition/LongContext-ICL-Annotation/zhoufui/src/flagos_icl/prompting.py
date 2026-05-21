from __future__ import annotations

from .config import AppConfig
from .data import Record
from .long_context import compress_text


def render_examples(examples: list[Record], text_field: str) -> str:
    parts: list[str] = []
    for index, example in enumerate(examples, start=1):
        label = example.label if example.label is not None else "unknown"
        parts.append(f"Example {index}\n{text_field}: {example.text}\nlabel: {label}")
    return "\n\n".join(parts)


def build_messages(config: AppConfig, target: Record, examples: list[Record]) -> list[dict[str, str]]:
    pipeline = config.pipeline
    target_text = compress_text(target.text, pipeline.max_context_chars, pipeline.chunk_chars)
    user_prompt = "\n\n".join(
        [
            f"Task: {pipeline.task_name}",
            f"Instruction:\n{config.prompt.instruction}",
            f"Few-shot examples:\n{render_examples(examples, pipeline.text_field)}",
            f"Target {pipeline.text_field}:\n{target_text}",
            "Return JSON only. Example: {\"label\":\"technology\",\"confidence\":0.82,\"rationale\":\"short reason\"}",
        ]
    )
    return [
        {"role": "system", "content": config.prompt.system},
        {"role": "user", "content": user_prompt},
    ]
