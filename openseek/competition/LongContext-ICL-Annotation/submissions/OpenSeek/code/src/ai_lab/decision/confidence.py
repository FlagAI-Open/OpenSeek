import math
from collections import Counter
from typing import Any, Dict, List

from src.ai_lab.output_parser import evidence_overlap, normalize_answer


def compute_confidence(predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid_predictions = [pred for pred in predictions if pred.get("valid")]
    if not valid_predictions:
        return {
            "score": 0.0,
            "agreement": 0.0,
            "entropy": 1.0,
            "schema": 0.0,
            "evidence": 0.0,
            "margin": 0.0,
            "need_retry": True,
        }

    labels = [normalize_answer(str(pred.get("label", ""))) for pred in valid_predictions]
    counts = Counter(labels)
    total = len(labels)
    top_label, top_count = counts.most_common(1)[0]
    agreement = top_count / total

    if len(counts) <= 1:
        entropy = 0.0
    else:
        entropy_value = 0.0
        for count in counts.values():
            prob = count / total
            entropy_value -= prob * math.log(prob)
        entropy = entropy_value / math.log(len(counts))

    schema = sum(1 for pred in valid_predictions if pred.get("schema_ok")) / total
    evidence = evidence_overlap([pred.get("evidence", []) for pred in valid_predictions])

    top_counts = counts.most_common(2)
    if len(top_counts) == 1:
        margin = 1.0
    else:
        margin = (top_counts[0][1] - top_counts[1][1]) / total

    score = 0.35 * agreement + 0.20 * (1.0 - entropy) + 0.15 * schema + 0.15 * evidence + 0.15 * margin
    return {
        "score": score,
        "agreement": agreement,
        "entropy": entropy,
        "schema": schema,
        "evidence": evidence,
        "margin": margin,
        "top_label": top_label,
        "need_retry": score < 0.82,
    }
