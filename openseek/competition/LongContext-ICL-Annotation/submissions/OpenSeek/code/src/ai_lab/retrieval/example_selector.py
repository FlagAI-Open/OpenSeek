import math
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Sequence

from src.ai_lab.adapters import SampleRecord


WORD_RE = re.compile(r"[A-Za-z0-9_]+")


def _tokenize(text: str) -> List[str]:
    return [token.lower() for token in WORD_RE.findall(text)]


def _example_input(example: Dict[str, Any]) -> str:
    return str(example.get("input", example.get("text", "")))


def _example_output(example: Dict[str, Any]) -> str:
    output = example.get("output", example.get("label", ""))
    if isinstance(output, list):
        output = output[0] if output else ""
    return str(output).strip()


def _score_tokens(query_tokens: Sequence[str], text: str) -> float:
    tokens = _tokenize(text)
    if not tokens:
        return 0.0
    counts = Counter(tokens)
    overlap = sum(counts[token] for token in query_tokens if token in counts)
    return overlap / math.sqrt(len(tokens))


def _score_examples(record: SampleRecord, examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    query_parts = [record.instruction, " ".join(record.label_space[:16]), record.text]
    query_tokens = _tokenize("\n".join(part for part in query_parts if part))
    scored: List[Dict[str, Any]] = []
    for index, example in enumerate(examples):
        scored.append(
            {
                "index": index,
                "label": _example_output(example),
                "score": _score_tokens(query_tokens, _example_input(example)),
                "input_len": len(_example_input(example)),
                "example": example,
            }
        )
    scored.sort(key=lambda item: (float(item["score"]), -int(item["index"])), reverse=True)
    return scored


def _unique_examples(scored_items: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    seen = set()
    for item in scored_items:
        index = int(item["index"])
        if index in seen:
            continue
        selected.append(item)
        seen.add(index)
        if len(selected) >= limit:
            break
    selected.sort(key=lambda item: int(item["index"]))
    return [dict(item["example"]) for item in selected]


def _lexical_topk(
    record: SampleRecord,
    examples: List[Dict[str, Any]],
    num_examples: int,
) -> List[Dict[str, Any]]:
    return _unique_examples(_score_examples(record, examples), num_examples)


def _is_balanceable(record: SampleRecord, scored: List[Dict[str, Any]], max_labels: int) -> bool:
    labels = {str(item["label"]) for item in scored if str(item["label"]).strip()}
    if record.task_type == "generation":
        return False
    if not labels:
        return False
    return len(labels) <= max_labels


def _balanced_similarity(
    record: SampleRecord,
    examples: List[Dict[str, Any]],
    icl_cfg: Dict[str, Any],
    num_examples: int,
) -> List[Dict[str, Any]]:
    scored = _score_examples(record, examples)
    max_labels = int(icl_cfg.get("max_balanced_labels", 16))
    if not _is_balanceable(record, scored, max_labels):
        return _unique_examples(scored, num_examples)

    by_label: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in scored:
        label = str(item["label"])
        if label:
            by_label[label].append(item)

    selected: List[Dict[str, Any]] = []
    selected_indexes = set()

    def add(item: Dict[str, Any]) -> None:
        if len(selected) >= num_examples:
            return
        index = int(item["index"])
        if index in selected_indexes:
            return
        selected.append(item)
        selected_indexes.add(index)

    similarity_quota = int(icl_cfg.get("similarity_quota", max(1, num_examples // 2)))
    for item in scored[:similarity_quota]:
        add(item)

    min_per_label = int(icl_cfg.get("label_balance_min_per_label", 1))
    labels_by_relevance = sorted(
        by_label,
        key=lambda label: float(by_label[label][0]["score"]) if by_label[label] else 0.0,
        reverse=True,
    )
    if record.label_space:
        relevance_rank = {label: rank for rank, label in enumerate(labels_by_relevance)}
        label_rank = {label: rank for rank, label in enumerate(record.label_space)}
        labels_by_relevance.sort(key=lambda label: (label_rank.get(label, 10_000), relevance_rank[label]))

    for label in labels_by_relevance:
        for item in by_label[label][:min_per_label]:
            add(item)

    boundary_quota = int(icl_cfg.get("boundary_quota", 0))
    if boundary_quota > 0:
        boundary_pool = sorted(scored, key=lambda item: int(item["input_len"]))
        for item in boundary_pool[:boundary_quota]:
            add(item)
        for item in reversed(boundary_pool[-boundary_quota:]):
            add(item)

    for item in scored:
        add(item)

    selected.sort(key=lambda item: int(item["index"]))
    return [dict(item["example"]) for item in selected]


def select_examples(
    record: SampleRecord,
    examples: List[Dict[str, Any]],
    icl_cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    num_examples = max(0, int(icl_cfg.get("num_examples", 0)))
    if num_examples <= 0 or not examples:
        return []

    selector = str(icl_cfg.get("example_selector", icl_cfg.get("selector", "static_first")))
    if selector in {"static_first", "first", "head"}:
        return [dict(example) for example in examples[:num_examples]]
    if selector in {"lexical_topk", "similarity", "similarity_topk"}:
        return _lexical_topk(record, examples, num_examples)
    if selector in {"balanced_similarity", "label_balanced_similarity"}:
        return _balanced_similarity(record, examples, icl_cfg, num_examples)

    return [dict(example) for example in examples[:num_examples]]
