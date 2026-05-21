from __future__ import annotations

import ast
import re


def solve_task(task_id: str, text: str) -> str | None:
    if task_id == "openseek-1":
        return closest_integers(text)
    if task_id == "openseek-3":
        return collatz(text)
    if task_id == "openseek-4":
        return concat_strings(text)
    return None


def _literal(text: str) -> object | None:
    try:
        return ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return None


def closest_integers(text: str) -> str | None:
    value = _literal(text)
    if not isinstance(value, list) or len(value) < 2:
        return None
    try:
        numbers = sorted(int(item) for item in value)
    except (TypeError, ValueError):
        return None
    best = min(abs(right - left) for left, right in zip(numbers, numbers[1:]))
    return str(best)


def collatz(text: str) -> str | None:
    value = _literal(text)
    if not isinstance(value, list):
        return None
    try:
        numbers = [int(item) for item in value]
    except (TypeError, ValueError):
        return None
    result = [item // 2 if item % 2 == 0 else item * 3 + 1 for item in numbers]
    return str(result)


def concat_strings(text: str) -> str | None:
    value = _literal(text)
    if not isinstance(value, list):
        return None
    if not all(isinstance(item, str) for item in value):
        return None
    return "".join(value)


def extract_answer_tag(text: str) -> str:
    matches = re.findall(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    return text.strip()
