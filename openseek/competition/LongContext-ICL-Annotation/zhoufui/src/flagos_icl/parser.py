from __future__ import annotations

import json
import re
from typing import Any

JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_prediction(text: str, abstain_label: str) -> dict[str, Any]:
    match = JSON_RE.search(text)
    candidate = match.group(0) if match else text
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return {"label": abstain_label, "confidence": 0.0, "rationale": "failed to parse model JSON", "raw": text}

    label = str(parsed.get("label") or abstain_label).strip()
    try:
        confidence = float(parsed.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    return {
        "label": label or abstain_label,
        "confidence": max(0.0, min(1.0, confidence)),
        "rationale": str(parsed.get("rationale", ""))[:500],
        "raw": text,
    }
