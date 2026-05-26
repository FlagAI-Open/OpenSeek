from __future__ import annotations

import re


def normalize_text(text: str | None) -> list[str]:
    if not text:
        return []
    return re.findall(r"[a-z0-9']+", text.lower())


def extract_category(text: str | None) -> str | None:
    if not text:
        return None
    match = re.search(r"Category:\s*(.*?)\s*(?:\n|$)", text, flags=re.IGNORECASE)
    if not match:
        return None
    category = re.sub(r"\s+", " ", match.group(1)).strip()
    return category or None


def extract_clue(text: str | None) -> str | None:
    if not text:
        return None
    match = re.search(r"Clue:\s*(.*)", text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    clue = re.sub(r"\s+", " ", match.group(1)).strip()
    return clue or None


def extract_first_label(text: str | None) -> str | None:
    if not text:
        return None
    match = re.search(r"<label>\s*(.*?)\s*</label>", text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    value = re.sub(r"\s+", " ", match.group(1)).strip()
    return value or None
