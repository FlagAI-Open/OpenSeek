from collections import Counter
from typing import Any, Dict, List, Tuple

from src.ai_lab.output_parser import normalize_answer


def select_top2_labels(predictions: List[Dict[str, Any]]) -> Tuple[str, str]:
    labels = [str(pred.get("label", "")).strip() for pred in predictions if pred.get("valid")]
    if not labels:
        return "", ""
    counts = Counter(normalize_answer(label) for label in labels)
    normalized_to_original: Dict[str, str] = {}
    for label in labels:
        normalized_to_original.setdefault(normalize_answer(label), label)
    winners = counts.most_common(2)
    if len(winners) == 1:
        value = normalized_to_original[winners[0][0]]
        return value, value
    return normalized_to_original[winners[0][0]], normalized_to_original[winners[1][0]]


def build_judge_prompt_values(
    task_definition: str,
    task_type: str,
    label_desc: str,
    examples_block: str,
    context: str,
    label_a: str,
    label_b: str,
) -> Dict[str, Any]:
    return {
        "task_definition": task_definition,
        "task_type": task_type,
        "label_desc": label_desc,
        "examples_block": examples_block,
        "context": context,
        "label_a": label_a,
        "label_b": label_b,
    }
