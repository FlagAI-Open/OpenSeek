from __future__ import annotations

import os
import re


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
MODEL_PATH = os.path.join(PROJECT_ROOT, "Qwen3-4B")
SERVICE_URL = os.environ.get("OPENSEEK_SERVICE_URL", "http://127.0.0.1:2026/v1/completions")


def normalize_text(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_]+", text.lower())


def extract_count_target(text: str) -> str | None:
    lowered = text.lower()
    if "number of nouns" in lowered:
        return "nouns"
    if "number of verbs" in lowered:
        return "verbs"
    return None


def extract_genre(text: str) -> str | None:
    match = re.search(r"Genre:\s*([^\n]+)", text)
    return match.group(1).strip().lower() if match else None


def extract_category(text: str) -> str | None:
    match = re.search(r"Category:\s*(.+?)\s*[\r\n]+Clue:", text, re.DOTALL)
    return match.group(1).strip().lower() if match else None


def extract_clue(text: str) -> str | None:
    match = re.search(r"Clue:\s*(.*)", text, re.DOTALL)
    return match.group(1).strip() if match else None


def split_task6_input(text: str) -> tuple[str | None, str | None, str | None]:
    match1 = re.search(r"Sentence 1:\s*(.*?)\s*Sentence 2:", text, re.DOTALL)
    match2 = re.search(r"Sentence 2:\s*(.*?)\s*Genre:", text, re.DOTALL)
    match3 = re.search(r"Genre:\s*(.*)", text, re.DOTALL)
    sent1 = match1.group(1).strip() if match1 else None
    sent2 = match2.group(1).strip() if match2 else None
    genre = match3.group(1).strip().lower() if match3 else None
    return sent1, sent2, genre


def task8_signature_hint(text: str) -> str:
    patterns = (
        r"Wrapper Entry Information:\s*([^\n]+)",
        r"The Python function ['`\"]?([A-Za-z_][A-Za-z0-9_]*)['`\"]?",
        r"The wrapper function ['`\"]?([A-Za-z_][A-Za-z0-9_]*)['`\"]?",
        r"\bdef\s+([A-Za-z_][A-Za-z0-9_]*\s*\([^\n)]*\))",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            value = match.group(1).strip()
            if "(" not in value and pattern != patterns[0]:
                value = f"{value}(...)"
            return value
    return "Use the wrapper information from the request exactly."


def example_score(example_input: str, text2annotate: str) -> tuple[int, int]:
    example_tokens = set(normalize_text(example_input))
    target_tokens = set(normalize_text(text2annotate))
    overlap = len(example_tokens & target_tokens)
    length_gap = abs(len(example_input) - len(text2annotate))
    return overlap, -length_gap


def serialize_python_like_list(values: list) -> str:
    return "[" + ", ".join(str(v) for v in values) + "]"


def extract_first_label(text: str) -> str | None:
    match = re.search(r"<label>\s*(.*?)\s*</label>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    trailing_open_only = re.search(r"<label>\s*([^\n\r<].*?)\s*$", text, re.DOTALL)
    if trailing_open_only:
        return trailing_open_only.group(1).strip()
    return None


def strip_code_fences(text: str) -> str:
    stripped = text.strip()
    stripped = re.sub(r"^```[a-zA-Z0-9_+-]*\n", "", stripped)
    stripped = re.sub(r"\n```$", "", stripped)
    return stripped.strip()
