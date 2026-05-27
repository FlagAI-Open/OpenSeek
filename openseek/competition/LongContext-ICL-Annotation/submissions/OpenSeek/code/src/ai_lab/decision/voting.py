from collections import defaultdict
from typing import Any, Dict, List

from src.ai_lab.output_parser import normalize_answer


def finalize_prediction(predictions: List[Dict[str, Any]], confidence: Dict[str, Any]) -> Dict[str, Any]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for prediction in predictions:
        label = normalize_answer(str(prediction.get("label", "")))
        if not label:
            continue
        grouped[label].append(prediction)

    if not grouped:
        return {
            "prediction": "",
            "confidence": confidence.get("score", 0.0),
            "strategy": "empty",
            "evidence": [],
        }

    best_label = max(
        grouped.items(),
        key=lambda item: (
            len(item[1]),
            sum(int(pred.get("confidence", 0)) for pred in item[1]),
        ),
    )[0]
    source_predictions = grouped[best_label]
    pretty_label = str(source_predictions[0].get("label", "")).strip()
    evidence: List[str] = []
    for prediction in source_predictions:
        for item in prediction.get("evidence", []):
            if item not in evidence:
                evidence.append(item)

    return {
        "prediction": pretty_label,
        "confidence": confidence.get("score", 0.0),
        "strategy": "vote",
        "evidence": evidence[:3],
    }
