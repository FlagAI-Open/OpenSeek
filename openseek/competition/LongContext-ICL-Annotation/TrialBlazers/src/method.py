from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests

try:
    from transformers import AutoTokenizer
except ImportError:  # pragma: no cover
    AutoTokenizer = None


LABEL_PATTERN = re.compile(r"<label>\s*(.*?)\s*</label>", re.IGNORECASE | re.DOTALL)
INT_PATTERN = re.compile(r"-?\d+")
LIST_PATTERN = re.compile(r"\[[\s\-\d,]+\]")
WORD_PATTERN = re.compile(r"[a-zA-Z0-9_]+")
_CODE_FENCE = re.compile(r"```(?:python)?\n(.*?)```", re.DOTALL | re.IGNORECASE)
_ANSWER_LINE_PATTERN = re.compile(r"(?im)^\s*(?:\*\*)?\s*(?:final\s+answer|answer)\s*(?:\*\*)?\s*[:：]\s*(.+?)\s*$")
_ANSWER_INLINE_PATTERN = re.compile(r"(?i)\bthe\s+answer\s+is\s*[:：]?\s*([^\n\r]+)")
_THINK_BLOCK_PATTERN = re.compile(r"(?is)<think>.*?</think>")

_TOKENIZER_CACHE: Dict[str, Any] = {}
_TASK2_PARSE_CACHE: Dict[str, Tuple[str, str]] = {}
_TASK2_FEATURE_CACHE: Dict[str, Dict[str, Any]] = {}


@dataclass
class InferenceConfig:
    api_base: str
    model_name: str
    timeout: float = 180.0
    max_new_tokens: int = 4096
    temperature: float = 0.0
    top_p: float = 1.0
    stop: Optional[List[str]] = None
    enable_thinking: Optional[bool] = None


@dataclass
class ResponseDetails:
    raw_text: str
    visible_text: str
    reasoning_text: str


def load_tokenizer(tokenizer_path: Optional[str]):
    if not tokenizer_path or AutoTokenizer is None:
        return None
    if tokenizer_path in _TOKENIZER_CACHE:
        return _TOKENIZER_CACHE[tokenizer_path]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    _TOKENIZER_CACHE[tokenizer_path] = tokenizer
    return tokenizer


def _estimate_tokens(text: str, tokenizer: Any) -> int:
    if tokenizer is None:
        # Lightweight fallback when tokenizer is unavailable.
        return max(1, len(text) // 4)
    return len(tokenizer.encode(text, add_special_tokens=False))


def _normalize_example_output(example_output: Any) -> str:
    if isinstance(example_output, list) and example_output:
        return str(example_output[0]).strip()
    if example_output is None:
        return ""
    return str(example_output).strip()


def _overlap_score(query_text: str, candidate_text: str) -> float:
    query_tokens = set(token.lower() for token in WORD_PATTERN.findall(query_text))
    cand_tokens = set(token.lower() for token in WORD_PATTERN.findall(candidate_text))
    return _token_set_overlap(query_tokens, cand_tokens)


def _token_set_overlap(query_tokens: set[str], cand_tokens: set[str]) -> float:
    if not query_tokens or not cand_tokens:
        return 0.0
    return len(query_tokens & cand_tokens) / (len(query_tokens) ** 0.5 * len(cand_tokens) ** 0.5)


def _parse_task4_segments(text: str) -> Optional[List[str]]:
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return None
    if not isinstance(parsed, list):
        return None
    if not all(isinstance(item, str) for item in parsed):
        return None
    return [str(item) for item in parsed]


def _task4_char_stats_from_segments(segments: Sequence[str]) -> Dict[str, int]:
    joined = "".join(segments)
    return {
        "length": len(joined),
        "num_items": len(segments),
        "uppercase": sum(ch.isupper() for ch in joined),
        "lowercase": sum(ch.islower() for ch in joined),
        "digits": sum(ch.isdigit() for ch in joined),
        "punct": sum(not ch.isalnum() for ch in joined),
        "single_char_items": sum(len(item) == 1 for item in segments),
        "multi_char_items": sum(len(item) > 1 for item in segments),
    }


def _task4_char_stats_from_prediction(text: str) -> Dict[str, int]:
    return {
        "length": len(text),
        "uppercase": sum(ch.isupper() for ch in text),
        "lowercase": sum(ch.islower() for ch in text),
        "digits": sum(ch.isdigit() for ch in text),
        "punct": sum(not ch.isalnum() for ch in text),
    }


def _task4_structure_score(query_text: str, candidate_text: str) -> float:
    query_segments = _parse_task4_segments(query_text)
    candidate_segments = _parse_task4_segments(candidate_text)
    if not query_segments or not candidate_segments:
        return _overlap_score(query_text, candidate_text)

    query_stats = _task4_char_stats_from_segments(query_segments)
    candidate_stats = _task4_char_stats_from_segments(candidate_segments)
    score = 0.0
    score += 4.0 / (1.0 + abs(query_stats["num_items"] - candidate_stats["num_items"]))
    score += 3.0 / (1.0 + abs(query_stats["length"] - candidate_stats["length"]))
    score += 1.5 / (1.0 + abs(query_stats["single_char_items"] - candidate_stats["single_char_items"]))
    score += 1.5 / (1.0 + abs(query_stats["multi_char_items"] - candidate_stats["multi_char_items"]))
    for key in ("uppercase", "lowercase", "digits", "punct"):
        score += 1.0 / (1.0 + abs(query_stats[key] - candidate_stats[key]))
    return score


def _parse_task2_prompt(text: str) -> Tuple[str, str]:
    cached = _TASK2_PARSE_CACHE.get(text)
    if cached is not None:
        return cached

    stripped = text.strip()
    match = re.match(
        r"^Sentence:\s*'(.*)'\.\s*Count the number of (nouns|verbs) in this sentence\.\s*$",
        stripped,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match:
        sentence = match.group(1).strip()
        target = match.group(2).lower()
        parsed = (sentence, target)
        _TASK2_PARSE_CACHE[text] = parsed
        return parsed

    lower = stripped.lower()
    if "count the number of nouns" in lower:
        target = "nouns"
    elif "count the number of verbs" in lower:
        target = "verbs"
    else:
        target = "unknown"
    sentence = stripped
    if stripped.lower().startswith("sentence:"):
        sentence = stripped.split(":", 1)[1].strip()
    sentence = re.sub(r"\.\s*count the number of (nouns|verbs) in this sentence\.\s*$", "", sentence, flags=re.IGNORECASE)
    sentence = sentence.strip().strip("'").strip()
    parsed = (sentence, target)
    _TASK2_PARSE_CACHE[text] = parsed
    return parsed


def _task2_sentence_features(text: str) -> Dict[str, Any]:
    cached = _TASK2_FEATURE_CACHE.get(text)
    if cached is not None:
        return cached

    tokens = [token.lower() for token in WORD_PATTERN.findall(text)]
    lower = text.lower()
    features: Dict[str, Any] = {
        "tokens": set(tokens),
        "token_len": len(tokens),
        "punct": sum(not ch.isalnum() and not ch.isspace() for ch in text),
        "quotes": text.count('"') + text.count("'"),
        "hyphen": text.count("-"),
        "comma": text.count(","),
        "digits": sum(ch.isdigit() for ch in text),
        "ing": sum(token.endswith("ing") for token in tokens),
        "ed": sum(token.endswith("ed") for token in tokens),
        "caps": sum(token[:1].isupper() for token in text.split()),
        "be": sum(token in {"am", "is", "are", "was", "were", "be", "been", "being"} for token in tokens),
        "have": sum(token in {"have", "has", "had"} for token in tokens),
        "do": sum(token in {"do", "does", "did"} for token in tokens),
    }
    _TASK2_FEATURE_CACHE[text] = features
    return features


def _task2_example_score(query_text: str, candidate_text: str, noun_biased: bool = False) -> float:
    query_sentence, query_target = _parse_task2_prompt(query_text)
    cand_sentence, cand_target = _parse_task2_prompt(candidate_text)
    query_feat = _task2_sentence_features(query_sentence)
    cand_feat = _task2_sentence_features(cand_sentence)

    score = 0.0
    if query_target == cand_target:
        score += 6.0
        if noun_biased and query_target == "nouns":
            score += 5.0
    elif query_target != "unknown" and cand_target != "unknown":
        score -= 2.5
        if noun_biased and query_target == "nouns":
            score -= 5.0

    token_overlap = _token_set_overlap(query_feat["tokens"], cand_feat["tokens"])
    score += (10.0 if noun_biased and query_target == "nouns" else 8.0) * token_overlap

    score += (4.0 if noun_biased and query_target == "nouns" else 3.0) / (
        1.0 + abs(query_feat["token_len"] - cand_feat["token_len"])
    )
    for key in ("punct", "quotes", "hyphen", "comma", "digits", "ing", "ed", "caps", "be", "have", "do"):
        score += 0.75 / (1.0 + abs(query_feat[key] - cand_feat[key]))
    if noun_biased and query_target == "nouns":
        for key in ("quotes", "hyphen", "comma", "caps"):
            score += 0.75 / (1.0 + abs(query_feat[key] - cand_feat[key]))
    return score


def _task2_candidate_similarity(text_a: str, text_b: str) -> float:
    sent_a, _ = _parse_task2_prompt(text_a)
    sent_b, _ = _parse_task2_prompt(text_b)
    feat_a = _task2_sentence_features(sent_a)
    feat_b = _task2_sentence_features(sent_b)
    score = _token_set_overlap(feat_a["tokens"], feat_b["tokens"])
    score += 0.3 / (1.0 + abs(feat_a["token_len"] - feat_b["token_len"]))
    score += 0.2 / (1.0 + abs(feat_a["punct"] - feat_b["punct"]))
    score += 0.2 / (1.0 + abs(feat_a["ing"] - feat_b["ing"]))
    score += 0.2 / (1.0 + abs(feat_a["ed"] - feat_b["ed"]))
    return score


def _parse_task6_prompt(text: str) -> Tuple[str, str, str]:
    stripped = text.strip()
    match = re.match(
        r"^Sentence 1:\s*(.*?)\s*Sentence 2:\s*(.*?)\s*Genre:\s*([^.]+)\.?\s*$",
        stripped,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match:
        return match.group(1).strip(), match.group(2).strip(), match.group(3).strip().lower()
    lower = stripped.lower()
    genre = ""
    genre_match = re.search(r"genre:\s*([^.]+)\.?", lower, flags=re.IGNORECASE)
    if genre_match:
        genre = genre_match.group(1).strip().lower()
    return stripped, "", genre


def _task6_example_score(query_text: str, candidate_text: str) -> float:
    query_s1, query_s2, query_genre = _parse_task6_prompt(query_text)
    cand_s1, cand_s2, cand_genre = _parse_task6_prompt(candidate_text)
    query_body = f"{query_s1} {query_s2}".strip()
    cand_body = f"{cand_s1} {cand_s2}".strip()
    score = 8.0 * _overlap_score(query_body, cand_body)
    if query_genre and query_genre == cand_genre:
        score += 8.0
    elif query_genre and cand_genre:
        score -= 2.0
    score += 1.0 / (1.0 + abs(len(WORD_PATTERN.findall(query_body)) - len(WORD_PATTERN.findall(cand_body))))
    return score


def _task6_label(example: Dict[str, Any]) -> str:
    label = _normalize_example_output(example.get("output", "")).strip().upper()
    return "Y" if label == "Y" else "N"


def _task6_genre_balanced_examples(
    all_examples: Sequence[Dict[str, Any]],
    query_text: str,
) -> List[Dict[str, Any]]:
    _, _, query_genre = _parse_task6_prompt(query_text)
    scored = [
        (example, _task6_example_score(query_text, str(example.get("input", ""))))
        for example in all_examples
    ]
    same_genre = []
    other_genre = []
    for example, score in scored:
        _, _, cand_genre = _parse_task6_prompt(str(example.get("input", "")))
        if query_genre and cand_genre == query_genre:
            same_genre.append((example, score))
        else:
            other_genre.append((example, score))

    ranked: List[Dict[str, Any]] = []
    for pool in (same_genre, other_genre):
        by_label = {"N": [], "Y": []}
        for item in sorted(pool, key=lambda pair: pair[1], reverse=True):
            by_label[_task6_label(item[0])].append(item[0])
        while by_label["N"] or by_label["Y"]:
            for label in ("N", "N", "Y"):
                if by_label[label]:
                    ranked.append(by_label[label].pop(0))
            if not by_label["N"] and by_label["Y"]:
                ranked.append(by_label["Y"].pop(0))
    return ranked


def _task5_label(example: Dict[str, Any]) -> str:
    label = _normalize_example_output(example.get("output", "")).strip().lower()
    return "Not sad" if "not sad" in label else "Sad"


def _task5_example_score(query_text: str, candidate_text: str) -> float:
    query_lower = query_text.lower()
    cand_lower = candidate_text.lower()
    score = 6.0 * _overlap_score(query_text, candidate_text)
    cue_groups = (
        ("quote", "joke", "lol", "haha", "😂", "sarcas", "song", "lyrics"),
        ("lost", "miss", "missing", "forgot", "can't find", "cant find"),
        ("hurt", "pain", "sick", "tired", "sleep", "insomnia", "weary"),
        ("pissed", "fuming", "furious", "angry", "annoyed", "frustrat"),
        ("sad", "depress", "lonely", "alone", "cry", "miserable", "unhappy"),
        ("politic", "trump", "government", "news", "police", "obama"),
    )
    for cues in cue_groups:
        if any(cue in query_lower for cue in cues) and any(cue in cand_lower for cue in cues):
            score += 2.0
    score += 1.0 / (1.0 + abs(len(WORD_PATTERN.findall(query_text)) - len(WORD_PATTERN.findall(candidate_text))))
    return score


def _task5_balanced_examples(
    all_examples: Sequence[Dict[str, Any]],
    query_text: str,
    conservative: bool = False,
) -> List[Dict[str, Any]]:
    scored = sorted(
        (
            (example, _task5_example_score(query_text, str(example.get("input", ""))))
            for example in all_examples
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    by_label = {"Sad": [], "Not sad": []}
    for example, _ in scored:
        by_label[_task5_label(example)].append(example)

    cycle = ("Not sad", "Not sad", "Sad") if conservative else ("Not sad", "Sad")
    ranked: List[Dict[str, Any]] = []
    while by_label["Sad"] or by_label["Not sad"]:
        progressed = False
        for label in cycle:
            if by_label[label]:
                ranked.append(by_label[label].pop(0))
                progressed = True
        if not progressed:
            break
        if not by_label["Not sad"] and by_label["Sad"]:
            ranked.append(by_label["Sad"].pop(0))
        if not by_label["Sad"] and by_label["Not sad"]:
            ranked.append(by_label["Not sad"].pop(0))
    return ranked


def _task2_output_bucket(example_output: Any) -> str:
    output = _normalize_example_output(example_output)
    try:
        value = int(output)
    except (TypeError, ValueError):
        return "unknown"
    if value <= 1:
        return "0-1"
    if value <= 3:
        return "2-3"
    if value <= 5:
        return "4-5"
    return "6+"


def _task2_output_value(example_output: Any) -> Optional[int]:
    output = _normalize_example_output(example_output)
    try:
        return int(output)
    except (TypeError, ValueError):
        return None


def _task2_order_selected_examples(
    selected: Sequence[Tuple[Dict[str, Any], float]],
) -> List[Dict[str, Any]]:
    if not selected:
        return []

    ordered = sorted(selected, key=lambda item: item[1], reverse=True)
    if len(ordered) <= 4:
        return [item[0] for item in ordered]

    anchors = ordered[:4]
    middle = ordered[4:]

    bucket_groups: Dict[str, List[Tuple[Dict[str, Any], float]]] = {}
    for item in middle:
        bucket = _task2_output_bucket(item[0].get("output"))
        bucket_groups.setdefault(bucket, []).append(item)

    bucket_keys = sorted(bucket_groups.keys())
    interleaved_middle: List[Dict[str, Any]] = []
    while any(bucket_groups.values()):
        for bucket in bucket_keys:
            group = bucket_groups[bucket]
            if group:
                interleaved_middle.append(group.pop(0)[0])

    return [anchors[0][0], anchors[2][0], *interleaved_middle, anchors[3][0], anchors[1][0]]


def _append_unselected_examples(
    ranked: Sequence[Dict[str, Any]],
    all_examples: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    ordered: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for example in ranked:
        key = str(example.get("input", ""))
        if not key or key in seen:
            continue
        seen.add(key)
        ordered.append(example)
    for example in all_examples:
        key = str(example.get("input", ""))
        if not key or key in seen:
            continue
        seen.add(key)
        ordered.append(example)
    return ordered


def _task2_pop_diverse(
    pool: List[Tuple[Dict[str, Any], float, int]],
    selected: Sequence[Dict[str, Any]],
    lookahead: int = 48,
) -> Optional[Tuple[Dict[str, Any], float, int]]:
    if not pool:
        return None
    if not selected:
        return pool.pop(0)

    best_idx = 0
    best_value: Optional[float] = None
    selected_inputs = [str(example.get("input", "")) for example in selected]
    for idx, (example, relevance, value) in enumerate(pool[:lookahead]):
        candidate_input = str(example.get("input", ""))
        similarity_penalty = max(
            _task2_candidate_similarity(candidate_input, selected_input)
            for selected_input in selected_inputs
        )
        # Keep label coverage while avoiding near-duplicate caption templates.
        adjusted = relevance - 0.8 * similarity_penalty + 0.02 * value
        if best_value is None or adjusted > best_value:
            best_value = adjusted
            best_idx = idx
    return pool.pop(best_idx)


def _task2_bucket_balanced_examples(
    all_examples: Sequence[Dict[str, Any]],
    query_text: str,
) -> List[Dict[str, Any]]:
    _, query_target = _parse_task2_prompt(query_text)
    noun_target = query_target == "nouns"

    candidates: List[Tuple[Dict[str, Any], float, int]] = []
    for example in all_examples:
        candidate_input = str(example.get("input", ""))
        _, candidate_target = _parse_task2_prompt(candidate_input)
        if query_target != "unknown" and candidate_target != query_target:
            continue
        value = _task2_output_value(example.get("output"))
        if value is None:
            continue
        if noun_target and value <= 0:
            continue
        score = _task2_example_score(query_text, candidate_input, noun_biased=noun_target)
        candidates.append((example, score, value))

    if not candidates:
        return sorted(
            all_examples,
            key=lambda ex: _task2_example_score(query_text, str(ex.get("input", "")), noun_biased=noun_target),
            reverse=True,
        )

    by_label: Dict[int, List[Tuple[Dict[str, Any], float, int]]] = {}
    for item in sorted(candidates, key=lambda row: row[1], reverse=True):
        by_label.setdefault(item[2], []).append(item)

    if noun_target:
        quotas = {4: 18, 3: 14, 5: 12, 2: 8, 6: 6, 1: 2, 7: 2, 8: 2}
        cycle = [4, 3, 5, 2, 4, 6, 3, 5]
        tail_labels = [2, 3, 5, 4, 6, 4]
        target_total = 64
    else:
        quotas = {1: 20, 0: 12, 2: 12, 3: 4, 4: 2, 5: 2}
        cycle = [1, 0, 2, 1, 3, 2]
        tail_labels = [0, 2, 1, 3, 1]
        target_total = 52

    for label in sorted(by_label):
        quotas.setdefault(label, 1)
    cycle = [label for label in cycle if label in by_label]
    if not cycle:
        cycle = sorted(by_label)

    selected_body: List[Dict[str, Any]] = []
    selected_tail: List[Dict[str, Any]] = []
    selected_keys: set[str] = set()
    selected_counts: Dict[int, int] = {}

    def add_item(item: Tuple[Dict[str, Any], float, int], target: List[Dict[str, Any]]) -> None:
        example, _, label = item
        key = str(example.get("input", ""))
        if key in selected_keys:
            return
        selected_keys.add(key)
        selected_counts[label] = selected_counts.get(label, 0) + 1
        target.append(example)

    for label in tail_labels:
        item = _task2_pop_diverse(by_label.get(label, []), selected_body + selected_tail)
        if item is not None:
            add_item(item, selected_tail)

    top_candidates = sorted(candidates, key=lambda row: row[1], reverse=True)
    for example, score, label in top_candidates:
        if len(selected_body) >= 8:
            break
        if selected_counts.get(label, 0) >= min(2, quotas.get(label, 1)):
            continue
        key = str(example.get("input", ""))
        if key in selected_keys:
            continue
        add_item((example, score, label), selected_body)

    while len(selected_body) + len(selected_tail) < target_total:
        progressed = False
        for label in cycle:
            if len(selected_body) + len(selected_tail) >= target_total:
                break
            if selected_counts.get(label, 0) >= quotas.get(label, 1):
                continue
            item = _task2_pop_diverse(by_label.get(label, []), selected_body + selected_tail)
            if item is None:
                continue
            add_item(item, selected_body)
            progressed = True
        if not progressed:
            break

    if len(selected_body) + len(selected_tail) < target_total:
        leftovers = sorted(
            [item for pool in by_label.values() for item in pool],
            key=lambda row: row[1],
            reverse=True,
        )
        for item in leftovers:
            if len(selected_body) + len(selected_tail) >= target_total:
                break
            if str(item[0].get("input", "")) in selected_keys:
                continue
            add_item(item, selected_body)

    return _append_unselected_examples(selected_body + selected_tail, all_examples)


def select_examples(
    all_examples: Sequence[Dict[str, Any]],
    query_text: str,
    tokenizer: Any,
    max_input_tokens: int,
    target_context_tokens: int,
    reserved_generation_tokens: int,
    task_id: Optional[int] = None,
    strategy: str = "semantic",
) -> Tuple[str, int, int]:
    budget_for_examples = max(0, max_input_tokens - reserved_generation_tokens)

    if task_id == 4 and strategy == "structured":
        ranked = sorted(
            all_examples,
            key=lambda ex: _task4_structure_score(query_text, str(ex.get("input", ""))),
            reverse=True,
        )
    elif task_id == 2 and strategy == "task2_bucket_balanced":
        ranked = _task2_bucket_balanced_examples(all_examples, query_text)
    elif task_id == 2 and strategy in {"task2_aware", "task2_noun_biased"}:
        query_sentence, query_target = _parse_task2_prompt(query_text)
        noun_biased = strategy == "task2_noun_biased" and query_target == "nouns"
        scored_candidates = [
            (example, _task2_example_score(query_text, str(example.get("input", "")), noun_biased=noun_biased))
            for example in all_examples
        ]
        scored_candidates.sort(key=lambda item: item[1], reverse=True)
        if noun_biased:
            same_target = [
                item
                for item in scored_candidates
                if _parse_task2_prompt(str(item[0].get("input", "")))[1] == query_target
            ]
            other_target = [
                item
                for item in scored_candidates
                if _parse_task2_prompt(str(item[0].get("input", "")))[1] != query_target
            ]
            scored_candidates = same_target + other_target

        selected_scored: List[Tuple[Dict[str, Any], float]] = []
        remaining = scored_candidates[:]
        same_target_quota = max(24 if noun_biased else 16, min(64 if noun_biased else 48, len(remaining)))
        while remaining and len(selected_scored) < same_target_quota:
            best_idx = 0
            best_value = None
            for idx, (example, relevance) in enumerate(remaining[:160 if noun_biased else 120]):
                similarity_penalty = 0.0
                if selected_scored:
                    similarity_penalty = max(
                        _task2_candidate_similarity(
                            str(example.get("input", "")),
                            str(existing[0].get("input", "")),
                        )
                        for existing in selected_scored
                    )
                value = relevance - (1.1 if noun_biased else 1.4) * similarity_penalty
                if best_value is None or value > best_value:
                    best_value = value
                    best_idx = idx
            selected_scored.append(remaining.pop(best_idx))
        ranked = _append_unselected_examples(_task2_order_selected_examples(selected_scored), all_examples)
    elif task_id == 5 and strategy in {"task5_balanced", "task5_conservative"}:
        ranked = _task5_balanced_examples(
            all_examples,
            query_text,
            conservative=strategy == "task5_conservative",
        )
    elif task_id == 6 and strategy == "task6_genre_balanced":
        ranked = _task6_genre_balanced_examples(all_examples, query_text)
    else:
        ranked = sorted(
            all_examples,
            key=lambda ex: _overlap_score(query_text, str(ex.get("input", ""))),
            reverse=True,
        )

    selected_lines: List[str] = []
    used_tokens = 0
    selected_count = 0

    for idx, example in enumerate(ranked, start=1):
        example_input = str(example.get("input", "")).strip()
        example_output = _normalize_example_output(example.get("output", ""))
        if not example_input:
            continue

        block = (
            f"[Example {idx}]\n"
            f"Input:\n{example_input}\n"
            f"Output:\n{example_output}\n"
        )
        block_tokens = _estimate_tokens(block, tokenizer)
        if used_tokens + block_tokens > budget_for_examples:
            continue

        selected_lines.append(block)
        used_tokens += block_tokens
        selected_count += 1

        if used_tokens >= target_context_tokens:
            break

    return "\n".join(selected_lines).strip(), selected_count, used_tokens


def build_prompt(
    task_id: int,
    task_description: str,
    examples_text: str,
    text_to_annotate: str,
    prompt_variant: str = "default",
) -> str:
    if task_id == 8:
        if prompt_variant == "task8_compact":
            return (
                "You are writing a complete Triton + PyTorch implementation for one target operator.\n\n"
                "Use the reference implementations only as coding patterns and API style references. "
                "Do not copy task instructions, field names, or natural-language descriptions into the answer.\n\n"
                "[Reference Implementations]\n"
                f"{examples_text}\n\n"
                "[Target Specification]\n"
                f"{text_to_annotate}\n\n"
                "[Strict Output Constraints]\n"
                "1. Output only Python code. No markdown fences, no explanation, no prose.\n"
                "2. The first non-empty line must be a Python import, decorator, or function definition.\n"
                "3. Never output any of these strings: [Sample To Annotate], Functional Description:, Wrapper Entry Information:, Final answer:.\n"
                "4. Never output placeholder comments such as 'Implement the ... here', and never output pass.\n"
                "5. If some details are uncertain, still provide the best complete implementation you can, but do not leave placeholders.\n\n"
                "Python code only:\n"
            )
        output_rule = (
            "Return only executable Python/Triton code for the requested operator. "
            "Do not include markdown fences and do not add extra explanation text. "
            "Think silently. The first non-empty line must be Python code."
        )
    elif task_id == 4 and prompt_variant == "strict":
        output_rule = (
            "Return exactly one line in XML tag format: <label>YOUR_FINAL_ANSWER</label>. "
            "This is a deterministic character-copying task: concatenate every list element from left to right, "
            "preserving every character exactly, including uppercase/lowercase, punctuation, and digits. "
            "Do not add, drop, normalize, explain, or paraphrase anything."
        )
    elif task_id == 2 and prompt_variant == "task2_strict":
        output_rule = (
            "Return exactly one integer as plain text. "
            "First determine whether the instruction asks for nouns or verbs, then count only that part of speech in the sentence. "
            "Do not output reasoning, labels, JSON, or extra text."
        )
    elif task_id == 2 and prompt_variant == "task2_count_rules":
        output_rule = (
            "Return exactly one integer as plain text. "
            "First identify whether the instruction asks for nouns or verbs, then count only that target part of speech. "
            "Use the caption-counting conventions shown by the examples, and do not output reasoning or extra text."
        )
    elif task_id == 5 and prompt_variant == "task5_sadness_strict":
        output_rule = (
            "Return exactly one label: Sad or Not sad. "
            "Label Sad only when the tweet author expresses personal sadness, grief, loneliness, disappointment, emotional pain, or clear distress. "
            "Do not output reasoning, JSON, markdown, or any extra text."
        )
    elif task_id == 5 and prompt_variant == "task5_sadness_balanced":
        output_rule = (
            "Return exactly one label: Sad or Not sad. "
            "Label Sad when the tweet author personally expresses negative affect or a direct bad situation affecting them, including sadness, disappointment, frustration, anger, loneliness, pain, insomnia, loss, discouragement, stress, or clear distress. "
            "Label Not sad when negative words appear only in a quote, joke, lyric, sarcasm, insult, news/commentary, third-person event, or ambiguous fragment. "
            "Do not output reasoning, JSON, markdown, or any extra text."
        )
    elif task_id == 5 and prompt_variant == "task5_sadness_conservative":
        output_rule = (
            "Return exactly one label: Sad or Not sad. "
            "Default to Not sad unless the author clearly states their own sadness, emotional pain, frustration, anger, physical distress, loss, loneliness, insomnia, discouragement, or stress. "
            "A sad-sounding word, quotation, joke, political comment, insult toward someone else, third-person problem, or general complaint is Not sad without the author's own feeling. "
            "Do not output reasoning, JSON, markdown, or any extra text."
        )
    elif task_id == 6 and prompt_variant == "task6_genre_strict":
        output_rule = (
            "Return exactly one character: Y or N. "
            "Judge whether both sentences plausibly belong to the provided genre, not whether the two sentences are semantically similar. "
            "Do not output reasoning, labels, JSON, or extra text."
        )
    elif task_id == 6 and prompt_variant == "task6_genre_conservative":
        output_rule = (
            "Return exactly one character: Y or N. "
            "Use Y only when both sentences independently show clear writing-style evidence for the provided genre. "
            "Same topic, related entities, or general plausibility is not enough. "
            "Do not output reasoning, labels, JSON, or extra text."
        )
    elif task_id == 6 and prompt_variant == "task6_genre_hypothesis":
        output_rule = (
            "Return exactly one character: Y or N. "
            "Treat this as a same-source genre/hypothesis check: Y means sentence 2 could be a short human-written hypothesis, "
            "paraphrase, consequence, or contradiction derived from sentence 1 in the stated genre. "
            "N means sentence 2 looks like an unrelated source, event, entity, or writing style. "
            "Do not output reasoning, labels, JSON, or extra text."
        )
    elif task_id == 6 and prompt_variant == "task6_genre_anchor_hypothesis":
        output_rule = (
            "Return exactly one character: Y or N. "
            "Y means sentence 2 is anchored to the same main entity, event, situation, or claim as sentence 1 and could be a "
            "human-written hypothesis, paraphrase, consequence, generalization, or contradiction in the stated genre. "
            "N means sentence 2 switches to a different main subject, entity, event, source, or writing style. "
            "Do not output reasoning, labels, JSON, or extra text."
        )
    elif task_id == 6 and prompt_variant == "task6_genre_anchor_genrefirst":
        output_rule = (
            "Return exactly one character: Y or N. "
            "Judge this as a genre-first same-source hypothesis task. Y requires both genre compatibility and a clear shared "
            "main anchor between the two sentences. Same topic words alone are not enough. "
            "Do not output reasoning, labels, JSON, or extra text."
        )
    else:
        output_rule = (
            "Return exactly one line in XML tag format: <label>YOUR_FINAL_ANSWER</label>. "
            "No rationale, no extra text, no markdown."
        )

    extra_rules = []
    if task_id == 4 and prompt_variant == "strict":
        extra_rules = [
            "Treat the input as a Python list of string literals.",
            "Remove only the list syntax, quotes, commas, and separator spaces between elements.",
            "Keep the content of each string element unchanged and in the original order.",
            "Before answering, verify that the final string length matches the total character count of all elements.",
        ]
    if task_id == 2 and prompt_variant == "task2_strict":
        extra_rules = [
            "Read the sentence and the count target carefully before answering.",
            "Count words by their role in the sentence, not by suffix heuristics alone.",
            "Do not count punctuation marks or quote symbols as words.",
            "Output the final count as digits only.",
        ]
    if task_id == 2 and prompt_variant == "task2_count_rules":
        extra_rules = [
            "Read the sentence and the count target carefully before answering.",
            "For nouns, count head nouns and coordinated nouns; do not count articles, determiners, color words, ordinary adjectives, or numbers.",
            "For compound nouns, count the noun words that function as nouns, but avoid double-counting one object because it has multiple modifiers.",
            "For verbs, count finite verbs, copulas, auxiliaries, and verbal participles that describe actions in the caption; do not count adjectival participles such as dirty, broken, colored, or shaped.",
            "Do not count punctuation marks or quote symbols as words.",
            "Output the final count as digits only.",
        ]
    if task_id == 5 and prompt_variant == "task5_sadness_strict":
        extra_rules = [
            "Use the tweet author's own emotional state as the primary signal.",
            "Do not mark Sad for jokes, quotes, sarcasm, insults, anger, neutral complaints, or third-person bad events unless the author's sadness is clear.",
            "Hashtags and emojis are supporting evidence only.",
            "Output exactly Sad or Not sad.",
        ]
    if task_id == 5 and prompt_variant == "task5_sadness_balanced":
        extra_rules = [
            "Use the author's own state as the primary signal; casual wording, hashtags, or laughter do not cancel a direct personal bad feeling.",
            "Direct first-person anger or frustration can be Sad for this dataset when the author is personally pissed, fuming, annoyed, discouraged, stressed, hurt, unable to sleep, or missing/lost something important.",
            "Keep Not sad for quote marks, song/literary lines, jokes, sports or political commentary, insults aimed at others, third-person sadness, and isolated words such as dark, grim, dull, sober, pout, hell, or rage.",
            "If both labels are plausible and the author's own feeling is not explicit, choose Not sad.",
            "Output exactly Sad or Not sad.",
        ]
    if task_id == 5 and prompt_variant == "task5_sadness_conservative":
        extra_rules = [
            "First ask: is the tweet author themself feeling bad right now? If not, choose Not sad.",
            "Choose Sad for clear personal distress: I am sad/upset/pissed/fuming/frustrated/lonely, my body/head hurts, I cannot sleep, I lost or miss something important, or something clearly ruined the author's day.",
            "Choose Not sad for quotations, lyrics, jokes, sarcasm, memes, motivational advice, political/religious commentary, third-person bad events, insults, or sad words used as descriptions rather than feelings.",
            "Words such as sober, dark, gritty, grim, dreadful, pout, hell, blood, rage, suspicion, panic, or stinks are not enough by themselves.",
            "Output exactly Sad or Not sad.",
        ]
    if task_id == 6 and prompt_variant == "task6_genre_strict":
        extra_rules = [
            "Return Y only if both sentences are strong matches for the stated genre.",
            "Return N if either sentence is generic, mismatched, or clearly from another style/source.",
            "Genre cues: telephone is disfluent spoken dialogue; fiction is narrative or dialogue prose; government is formal institutional prose; travel is guidebook-style description; slate is magazine-style commentary or opinion.",
            "Output exactly Y or N.",
        ]
    if task_id == 6 and prompt_variant == "task6_genre_conservative":
        extra_rules = [
            "Default to N if either sentence is short, generic, caption-like, encyclopedic, news-like, merely topical, or could fit many genres.",
            "Do not answer Y because the two sentences are semantically related; judge each sentence's source style separately.",
            "Y requires both sentences to strongly fit the named style: telephone = informal spoken dialogue or disfluencies; fiction = narrative or character dialogue; government = formal institutional, legal, administrative, or policy prose; travel = guidebook or tourist destination description; slate = magazine-style opinion, critique, or commentary.",
            "Return N when only one sentence has genre evidence, even if the other sentence is not obviously wrong.",
            "Output exactly Y or N.",
        ]
    if task_id == 6 and prompt_variant == "task6_genre_hypothesis":
        extra_rules = [
            "Sentence 2 may be much shorter, simpler, or less stylistically marked than sentence 1 and can still be Y if it is clearly about sentence 1.",
            "Contradiction does not make N by itself: a wrong detail can still be a same-genre hypothesis when it is derived from sentence 1.",
            "Return N when sentence 2 introduces unrelated people, places, events, institutions, topics, or a different source style, even if both sentences are plausible for the genre.",
            "Use genre cues as a guardrail: telephone = spoken disfluency or conversational utterance; fiction = narrative or character action/dialogue; government = institutional, legal, policy, or administrative prose; travel = guidebook/location description; slate = magazine-style commentary or criticism.",
            "Default to N only when the relation to sentence 1 or the stated genre is weak.",
            "Output exactly Y or N.",
        ]
    if task_id == 6 and prompt_variant == "task6_genre_anchor_hypothesis":
        extra_rules = [
            "First identify the main subject, named entity, event, or situation in sentence 1, then check whether sentence 2 is about that same anchor.",
            "A changed attribute, wrong detail, negation, or contradiction about the same anchor can still be Y.",
            "Return N if sentence 2 mainly introduces a new person, place, institution, object, policy, or scene that is not recoverable from sentence 1.",
            "Return N for generic fragments or discourse markers such as 'Furthermore' unless they clearly continue sentence 1.",
            "Use the stated genre as a guardrail, but same genre alone is not enough without the anchor relation.",
            "Output exactly Y or N.",
        ]
    if task_id == 6 and prompt_variant == "task6_genre_anchor_genrefirst":
        extra_rules = [
            "Step 1: decide whether each sentence can plausibly come from the stated genre: telephone = spoken conversation; fiction = narrative or character dialogue; government = formal policy, legal, administrative, or institutional prose; travel = guidebook/location description; slate = magazine-style commentary, review, or opinion.",
            "Step 2: identify the main anchor in sentence 1, such as a person, place, institution, event, object, policy, or situation.",
            "Answer Y only when sentence 2 is about that same anchor and is a plausible hypothesis, paraphrase, consequence, contradiction, or continuation from the same source style.",
            "Answer N when sentence 2 changes to a different anchor, only shares generic topic words, sounds like another source style, or is a vague discourse fragment.",
            "For travel, a short factual description can still be Y when it clearly continues the same place or attraction; do not require opinionated wording.",
            "For fiction, slate, telephone, and government, be stricter: related wording without the same concrete anchor should be N.",
            "Output exactly Y or N.",
        ]

    prompt = (
        "You are a data annotation assistant for the OpenSeek long-context competition.\n\n"
        "[Task Description]\n"
        f"{task_description}\n\n"
        "[In-Context Examples]\n"
        f"{examples_text}\n\n"
        "[Sample To Annotate]\n"
        f"{text_to_annotate}\n\n"
        "[Output Rules]\n"
        f"1. {output_rule}\n"
        "2. Follow the same answer format and style as the examples.\n"
        "3. If uncertain, still provide the single best answer.\n\n"
        "Final answer:\n"
    )
    if extra_rules:
        rules = "\n".join(f"{idx + 4}. {rule}" for idx, rule in enumerate(extra_rules))
        prompt = prompt.replace("\n\nFinal answer:\n", f"{rules}\n\nFinal answer:\n")
    return prompt


def build_task4_review_prompt(
    task_description: str,
    text_to_annotate: str,
    draft_answer: Optional[str],
) -> str:
    draft = draft_answer if draft_answer is not None else "<empty>"
    return (
        "You are reviewing a string concatenation answer for the OpenSeek competition.\n\n"
        "[Task Description]\n"
        f"{task_description}\n\n"
        "[Input List]\n"
        f"{text_to_annotate}\n\n"
        "[Draft Answer]\n"
        f"{draft}\n\n"
        "[Review Instructions]\n"
        "1. Check whether the draft is exactly the left-to-right concatenation of all list elements.\n"
        "2. Preserve every character in each element exactly, including case, punctuation, and digits.\n"
        "3. If the draft is wrong, output the corrected concatenation.\n"
        "4. Return exactly one line in XML tag format: <label>YOUR_FINAL_ANSWER</label>.\n"
        "5. No explanation, no markdown, no extra text.\n\n"
        "Final answer:\n"
    )


def build_task4_retry_prompt(
    task_description: str,
    examples_text: str,
    text_to_annotate: str,
    issues: Sequence[str],
) -> str:
    issue_text = "; ".join(issues) if issues else "previous answer was invalid"
    return (
        "You are retrying a deterministic string concatenation annotation.\n\n"
        "[Task Description]\n"
        f"{task_description}\n\n"
        "[In-Context Examples]\n"
        f"{examples_text}\n\n"
        "[Sample To Annotate]\n"
        f"{text_to_annotate}\n\n"
        "[Previous Failure Signal]\n"
        f"{issue_text}\n\n"
        "[Retry Rules]\n"
        "1. Return exactly one line in XML tag format: <label>YOUR_FINAL_ANSWER</label>.\n"
        "2. Concatenate every list element from left to right with zero edits.\n"
        "3. Preserve every character exactly, including case, punctuation, and digits.\n"
        "4. Do not omit any element and do not output any explanation.\n"
        "5. Re-check the final character count before answering.\n\n"
        "Final answer:\n"
    )


def build_task8_repair_prompt(
    text_to_annotate: str,
    draft_code: Optional[str],
    issues: Sequence[str],
) -> str:
    draft = draft_code if draft_code is not None else "<empty>"
    issue_text = "; ".join(issues) if issues else "draft is incomplete or invalid"
    return (
        "You are repairing a CPU-safe PyTorch fallback implementation draft.\n"
        "Prefer a simple executable PyTorch implementation over any Triton or CUDA-specific code.\n\n"
        "[Target Specification]\n"
        f"{text_to_annotate}\n\n"
        "[Current Draft]\n"
        f"{draft}\n\n"
        "[Detected Issues]\n"
        f"{issue_text}\n\n"
        "[Repair Instructions]\n"
        "1. Rewrite and return the full corrected Python code only.\n"
        "2. Do not output markdown fences, explanation, or natural-language notes.\n"
        "3. Remove any prompt leakage such as [Sample To Annotate], Functional Description:, Wrapper Entry Information:, or Final answer:.\n"
        "4. Replace every pass, TODO, or placeholder comment with concrete code.\n"
        "5. Do not import triton, do not use @triton.jit, and do not call CUDA-only APIs.\n"
        "6. Keep any already-correct code structure when useful, but return one clean final implementation.\n\n"
        "Python code only:\n"
    )


def build_task8_rewrite_prompt(
    text_to_annotate: str,
    issues: Sequence[str],
) -> str:
    issue_text = "; ".join(issues) if issues else "previous answers remained incomplete"
    return (
        "Write a complete Python implementation for the requested Triton/PyTorch operator from scratch.\n\n"
        "[Target Specification]\n"
        f"{text_to_annotate}\n\n"
        "[Previous Failure Signal]\n"
        f"{issue_text}\n\n"
        "[Hard Rules]\n"
        "1. Output one complete Python file only. No markdown, no explanation, no prose.\n"
        "2. The first non-empty line must be Python code, such as import/from/@triton.jit/def/class.\n"
        "3. Zero pass statements. Zero TODO. Zero placeholder comments. Do not leak any prompt text.\n"
        "4. Preserve the required public wrapper function name and signature from the target specification.\n"
        "5. If exact Triton details are uncertain, prefer a fully executable PyTorch fallback implementation over an incomplete Triton skeleton.\n\n"
        "Python code only:\n"
    )


def _format_solver_examples(examples: Sequence[Dict[str, Any]]) -> str:
    blocks: List[str] = []
    for idx, example in enumerate(examples, start=1):
        example_input = str(example.get("input", "")).strip()
        example_output = _normalize_example_output(example.get("output", ""))
        if not example_input or not example_output:
            continue
        blocks.append(
            f"[Example {idx}]\n"
            f"Input:\n{example_input}\n"
            f"Expected Output:\n{example_output}\n"
        )
    return "\n".join(blocks).strip()


def build_task_solver_synthesis_prompt(
    task_id: int,
    task_description: str,
    examples: Sequence[Dict[str, Any]],
    function_name: str = "solve",
) -> str:
    examples_text = _format_solver_examples(examples)
    task_specific_rules: List[str] = []
    if task_id == 1:
        task_specific_rules = [
            "For this task, parse `input_text` as a Python list literal of integers.",
            "Compute the minimum absolute difference between any two integers in the list.",
            "Return exactly one integer as a string, for example `8`.",
            "Do not echo the input list, the first element, or any explanation.",
            "Do not leave `pass`, TODO, placeholder comments, or partial code.",
        ]
    if task_id == 3:
        task_specific_rules = [
            "For this task, parse `input_text` as a Python list literal of integers.",
            "Use this exact rule per integer: if even return n // 2, else return 3 * n + 1.",
            "Return the final answer with Python's standard list string format, for example `str(result)`, so commas are followed by one space.",
            "Do not remove spaces after commas; the benchmark examples use strings like `[36, 88, 148]`.",
            "Do not leave `pass`, TODO, placeholder comments, or partial code.",
        ]
    if task_id == 4:
        task_specific_rules = [
            "For this task, parse `input_text` as a Python list literal of strings.",
            "The input uses Python literal syntax with single quotes, so `json.loads` is invalid here; use `ast.literal_eval` or an equivalent Python-literal parser.",
            "Concatenate all string elements from left to right with zero edits.",
            "Preserve every character exactly, including case, punctuation, digits, and spaces inside elements.",
            "Return only the final concatenated string.",
            "Do not leave `pass`, TODO, placeholder comments, or partial code.",
        ]
    extra_rule_text = ""
    if task_specific_rules:
        extra_rule_text = "\n".join(f"{idx + 6}. {rule}" for idx, rule in enumerate(task_specific_rules)) + "\n\n"
    return (
        "You are writing a deterministic Python solver for one OpenSeek task.\n\n"
        "[Task ID]\n"
        f"{task_id}\n\n"
        "[Task Description]\n"
        f"{task_description}\n\n"
        "[Reference Examples]\n"
        f"{examples_text}\n\n"
        "[Programming Instructions]\n"
        f"1. Write exactly one Python module that defines a function `{function_name}(input_text: str) -> str`.\n"
        "2. The function must parse the raw input string, compute the answer, and return the final answer string.\n"
        "3. Use deterministic program logic. Do not call external services, do not read files, and do not print anything.\n"
        "4. Output Python code only. No markdown fences, no explanation, no prose, no analysis.\n"
        "5. The returned string must match the task output format exactly.\n"
        "6. The first non-empty line must be `import ...` or `def solve(input_text: str) -> str:`.\n"
        "7. Stop immediately after the final line of code.\n"
        f"{extra_rule_text}"
        "Python code only:\n"
    )


def build_task_solver_repair_prompt(
    task_id: int,
    task_description: str,
    examples: Sequence[Dict[str, Any]],
    draft_code: str,
    failures: Sequence[Dict[str, str]],
    function_name: str = "solve",
) -> str:
    examples_text = _format_solver_examples(examples)
    failure_lines = []
    for idx, failure in enumerate(failures, start=1):
        failure_lines.append(
            f"[Failure {idx}]\n"
            f"Input:\n{failure['input']}\n"
            f"Expected Output:\n{failure['expected']}\n"
            f"Observed Output:\n{failure['predicted']}\n"
            f"Error:\n{failure['error']}\n"
        )
    failure_text = "\n".join(failure_lines).strip() or "No concrete failures captured."
    task_specific_rules: List[str] = []
    if task_id == 1:
        task_specific_rules = [
            "Parse the input list with Python code, such as `ast.literal_eval` or `json.loads`.",
            "Compute the minimum absolute difference over all pairs of integers.",
            "Return exactly one integer string.",
            "Do not leave `pass`, TODO, or explanatory prose anywhere in the module.",
        ]
    if task_id == 3:
        task_specific_rules = [
            "Parse the input list with Python code, such as `ast.literal_eval`.",
            "Apply exactly one Collatz step per element: even -> n // 2, odd -> 3 * n + 1.",
            "Return the final answer with Python's standard list string format, for example `str(result)`, so commas are followed by one space.",
            "Do not remove spaces after commas; the benchmark examples use strings like `[36, 88, 148]`.",
            "Do not leave `pass`, TODO, or explanatory prose anywhere in the module.",
        ]
    if task_id == 4:
        task_specific_rules = [
            "Parse the input list with Python code, such as `ast.literal_eval`; do not use `json.loads` because the input is a Python literal with single quotes.",
            "Join all string elements in the original order with no separator and no edits.",
            "Preserve every character exactly in the final string.",
            "Do not leave `pass`, TODO, or explanatory prose anywhere in the module.",
        ]
    extra_rule_text = ""
    if task_specific_rules:
        extra_rule_text = "\n".join(f"{idx + 5}. {rule}" for idx, rule in enumerate(task_specific_rules)) + "\n\n"
    return (
        "You are repairing a deterministic Python solver for one OpenSeek task.\n\n"
        "[Task ID]\n"
        f"{task_id}\n\n"
        "[Task Description]\n"
        f"{task_description}\n\n"
        "[Reference Examples]\n"
        f"{examples_text}\n\n"
        "[Current Solver]\n"
        f"{draft_code}\n\n"
        "[Observed Validation Failures]\n"
        f"{failure_text}\n\n"
        "[Repair Instructions]\n"
        f"1. Rewrite the full Python module and keep the public function name `{function_name}`.\n"
        "2. Fix the logic so the solver passes the failed examples exactly.\n"
        "3. Output Python code only. No markdown fences, no explanation, no prose, no analysis.\n"
        "4. Do not read files, do not use randomness, and do not call external services.\n\n"
        f"{extra_rule_text}"
        "Python code only:\n"
    )


def build_task_solver_debug_prompt(
    task_id: int,
    task_description: str,
    examples: Sequence[Dict[str, Any]],
    draft_code: str,
    failures: Sequence[Dict[str, str]],
    function_name: str = "solve",
) -> str:
    examples_text = _format_solver_examples(examples)
    failure_lines = []
    for idx, failure in enumerate(failures, start=1):
        failure_lines.append(
            f"[Observed Failure {idx}]\n"
            f"Input:\n{failure['input']}\n"
            f"Expected Output:\n{failure['expected']}\n"
            f"Observed Output:\n{failure['predicted']}\n"
            f"Runtime/Error Signal:\n{failure['error']}\n"
        )
    failure_text = "\n".join(failure_lines).strip() or "No concrete failures captured."

    debug_rules: List[str] = [
        f"Rewrite the full Python module and keep the public function name `{function_name}`.",
        "Think like a debugging engineer: identify the root cause from the failing cases, then fix the program.",
        "Use the execution failures as the primary signal, not natural-language guessing.",
        "Output Python code only. No markdown fences, no explanation, no prose, no analysis.",
        "Do not leave `pass`, TODO, or placeholder code.",
    ]
    if task_id == 4:
        debug_rules.append(
            "The input is a Python literal list with single quotes, so prefer `ast.literal_eval` over `json.loads`."
        )
    if task_id in (1, 3, 4):
        debug_rules.append("Return the exact final answer string format required by the task.")
    if task_id == 3:
        debug_rules.append("For task3, format the final result exactly as Python's standard list string, for example `str(result)`, so commas are followed by one space.")

    rule_text = "\n".join(f"{idx + 1}. {rule}" for idx, rule in enumerate(debug_rules))
    return (
        "You are debugging a deterministic Python solver for one OpenSeek task.\n\n"
        "[Task ID]\n"
        f"{task_id}\n\n"
        "[Task Description]\n"
        f"{task_description}\n\n"
        "[Reference Examples]\n"
        f"{examples_text}\n\n"
        "[Current Solver]\n"
        f"{draft_code}\n\n"
        "[Validation Failures]\n"
        f"{failure_text}\n\n"
        "[Debug Instructions]\n"
        f"{rule_text}\n\n"
        "Python code only:\n"
    )


def _completions_url(api_base: str) -> str:
    return f"{api_base.rstrip('/')}/v1/completions"


def _chat_url(api_base: str) -> str:
    return f"{api_base.rstrip('/')}/v1/chat/completions"


def _extract_chat_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts).strip()
    return str(content)


def _extract_visible_response_text(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return ""
    if "<think>" not in stripped.lower():
        return stripped
    if "</think>" not in stripped.lower():
        return ""
    visible = _THINK_BLOCK_PATTERN.sub("", stripped).strip()
    return visible or stripped


def _extract_reasoning_text(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return ""
    match = re.search(r"(?is)<think>\s*(.*?)\s*</think>", stripped)
    if not match:
        return ""
    return match.group(1).strip()


def _extract_message_text(message: Any) -> str:
    if not isinstance(message, dict):
        return _extract_visible_response_text(_extract_chat_text(message))
    content_text = _extract_visible_response_text(_extract_chat_text(message.get("content", "")))
    if content_text:
        return content_text
    reasoning_text = _extract_visible_response_text(_extract_chat_text(message.get("reasoning_content", "")))
    return reasoning_text


def _extract_message_details(message: Any) -> ResponseDetails:
    if not isinstance(message, dict):
        raw_text = _extract_chat_text(message).strip()
        return ResponseDetails(
            raw_text=raw_text,
            visible_text=_extract_visible_response_text(raw_text),
            reasoning_text=_extract_reasoning_text(raw_text),
        )

    raw_content = _extract_chat_text(message.get("content", "")).strip()
    explicit_reasoning = _extract_chat_text(message.get("reasoning_content", "")).strip()
    raw_text = raw_content or explicit_reasoning
    visible_text = _extract_visible_response_text(raw_content or raw_text)
    reasoning_text = explicit_reasoning or _extract_reasoning_text(raw_content)
    return ResponseDetails(
        raw_text=raw_text,
        visible_text=visible_text,
        reasoning_text=reasoning_text,
    )


def _extract_text_details(text: str) -> ResponseDetails:
    raw_text = text.strip()
    return ResponseDetails(
        raw_text=raw_text,
        visible_text=_extract_visible_response_text(raw_text),
        reasoning_text=_extract_reasoning_text(raw_text),
    )


def annotate_detailed(prompt: str, config: InferenceConfig) -> ResponseDetails:
    try:
        payload = {
            "model": config.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": config.max_new_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
        }
        if config.enable_thinking is not None:
            payload["chat_template_kwargs"] = {"enable_thinking": config.enable_thinking}
        if config.stop:
            payload["stop"] = config.stop
        chat_resp = requests.post(
            _chat_url(config.api_base),
            json=payload,
            timeout=config.timeout,
        )
        chat_resp.raise_for_status()
        data = chat_resp.json()
        choices = data.get("choices") or []
        if choices:
            message = choices[0].get("message", {})
            return _extract_message_details(message)
    except Exception:
        pass

    try:
        payload = {
            "model": config.model_name,
            "prompt": prompt,
            "max_tokens": config.max_new_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
        }
        if config.stop:
            payload["stop"] = config.stop
        comp_resp = requests.post(
            _completions_url(config.api_base),
            json=payload,
            timeout=config.timeout,
        )
        comp_resp.raise_for_status()
        data = comp_resp.json()
        choices = data.get("choices") or []
        if choices:
            return _extract_text_details(str(choices[0].get("text", "")))
    except Exception:
        pass

    return ResponseDetails(raw_text="", visible_text="", reasoning_text="")


def annotate(prompt: str, config: InferenceConfig) -> str:
    details = annotate_detailed(prompt, config)
    if details.visible_text:
        return details.visible_text.strip()
    raw_lower = details.raw_text.strip().lower()
    if raw_lower.startswith("<think>"):
        return ""
    if details.reasoning_text and not details.raw_text.strip():
        return ""
    return details.raw_text.strip()


def _strip_code_fence(text: str) -> str:
    match = _CODE_FENCE.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def _extract_task8_code(text: str) -> Optional[str]:
    stripped = _strip_code_fence(text)
    if not stripped:
        return None

    lines = stripped.splitlines()
    code_start = None
    code_prefixes = (
        "import ",
        "from ",
        "@triton.jit",
        "def ",
        "class ",
        "BLOCK_SIZE",
        "import\ttorch",
    )
    for idx, line in enumerate(lines):
        if line.lstrip().startswith(code_prefixes):
            code_start = idx
            break

    code_lines: List[str] = []
    stop_markers = (
        "[sample to annotate]",
        "[task description]",
        "[in-context examples]",
        "[output rules]",
        "[current task]",
        "[current failure signal]",
        "[previous failure signal]",
        "[new instructions]",
        "[new rules]",
        "[hard rules]",
        "[retry rules]",
        "[repair instructions]",
        "functional description:",
        "wrapper entry information:",
        "python code only:",
        "final answer:",
        "explanation:",
        "here's",
        "here is",
        "this code",
        "the code",
        "i hope",
        "let me know",
        "okay,",
        "ok,",
        "wait,",
        "but ",
        "the previous attempt",
        "the current code",
        "the correct implementation",
        "here is the corrected",
        "here is a corrected",
        "here is a complete",
        "i need to",
        "let me ",
    )

    def _looks_like_reasoning(line: str) -> bool:
        lowered_line = line.strip().lower()
        if not lowered_line:
            return False
        if lowered_line.startswith("#"):
            return False
        if any(lowered_line.startswith(marker) for marker in stop_markers):
            return True
        if re.match(r"^[A-Z][a-z]+(?:\s+[A-Za-z'/-]+){4,}", line.strip()):
            return True
        return False

    if code_start is None:
        if _looks_like_reasoning(lines[0]):
            return None
        if not any(token in stripped for token in ("import ", "def ", "@triton.jit", "class ", "torch", "triton")):
            return None
        return stripped

    for line in lines[code_start:]:
        if _looks_like_reasoning(line):
            break
        code_lines.append(line.rstrip())

    code = "\n".join(code_lines).strip()
    return code or None


def extract_solver_code(text: str, function_name: str = "solve") -> Optional[str]:
    stripped = _strip_code_fence(text)
    if not stripped:
        return None

    lines = stripped.splitlines()
    code_prefixes = ("import ", "from ", "def ", "class ")
    stop_prefixes = (
        "final answer:",
        "explanation:",
        "here is",
        "python code only:",
        "the solver",
    )
    assignment_pattern = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*\s*=")
    candidate_blocks: List[str] = []

    for start_idx, line in enumerate(lines):
        if not line.lstrip().startswith(code_prefixes):
            continue

        code_lines: List[str] = []
        for current_line in lines[start_idx:]:
            lowered = current_line.strip().lower()
            if lowered and any(lowered.startswith(prefix) for prefix in stop_prefixes):
                break
            if current_line.strip() and not current_line.startswith((" ", "\t")):
                current_stripped = current_line.lstrip()
                if (
                    code_lines
                    and not current_stripped.startswith(code_prefixes)
                    and not current_stripped.startswith(("@", "#"))
                    and not assignment_pattern.match(current_stripped)
                ):
                    break
            code_lines.append(current_line.rstrip())

        code = "\n".join(code_lines).strip()
        if f"def {function_name}" in code:
            candidate_blocks.append(code)

    if not candidate_blocks:
        return None
    return candidate_blocks[-1]


def _extract_tag_answer(text: str) -> Optional[str]:
    matches = LABEL_PATTERN.findall(text)
    if not matches:
        return None
    return matches[-1].strip() if matches[-1].strip() else None


def _first_nonempty_line(text: str) -> Optional[str]:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return None


def _strip_wrappers(text: str) -> str:
    value = text.strip().strip("`").strip()
    value = value.strip("*").strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        value = value[1:-1].strip()
    value = re.sub(r"(?i)^(?:\*\*)?\s*(?:final\s+answer|answer)\s*(?:\*\*)?\s*[:：]\s*", "", value).strip()
    return value


def _extract_answer_marker(text: str) -> Optional[str]:
    candidates: List[str] = []
    for match in _ANSWER_LINE_PATTERN.finditer(text):
        value = _strip_wrappers(match.group(1))
        if value:
            candidates.append(value)
    for match in _ANSWER_INLINE_PATTERN.finditer(text):
        value = _strip_wrappers(match.group(1))
        if value:
            candidates.append(value)
    if not candidates:
        return None
    return candidates[-1]


def _extract_task4_answer(raw_text: str, tagged: Optional[str]) -> Optional[str]:
    if tagged:
        value = _strip_wrappers(tagged)
        return value if value else None

    for line in raw_text.splitlines():
        value = _strip_wrappers(line)
        if not value:
            continue
        # task4 output should be a concatenated string without whitespaces.
        if re.search(r"\s", value):
            continue
        if value.lower().startswith(("okay", "now", "first", "let's")):
            continue
        return value

    marker = _extract_answer_marker(raw_text)
    if marker:
        compact = _strip_wrappers(marker)
        if compact and not re.search(r"\s", compact):
            return compact

    quoted = [
        _strip_wrappers(item)
        for item in re.findall(r"['\"]([^'\"\n]{2,200})['\"]", raw_text)
        if item and not re.search(r"\s", item)
    ]
    if quoted:
        quoted.sort(key=len, reverse=True)
        return quoted[0]

    colon_candidates = [
        _strip_wrappers(item)
        for item in re.findall(r"[:=]\s*([^\s,]+)", raw_text)
        if item
    ]
    colon_candidates = [item for item in colon_candidates if item and not re.search(r"\s", item)]
    if colon_candidates:
        colon_candidates.sort(key=len, reverse=True)
        return colon_candidates[0]

    first_line = _first_nonempty_line(raw_text)
    if not first_line:
        return None
    compact = _strip_wrappers(first_line)
    return compact if compact and not re.search(r"\s", compact) else None


def assess_task4_prediction(text_to_annotate: str, prediction: Optional[str]) -> List[str]:
    issues: List[str] = []
    if prediction is None:
        return ["prediction is null"]

    value = prediction.strip()
    if not value:
        return ["prediction is empty"]

    if re.search(r"\s", value):
        issues.append("prediction contains whitespace")
    if any(token in value for token in ("<label>", "</label>", "[", "]")):
        issues.append("prediction still contains wrapper tokens")
    if value.startswith(("'", '"')) or value.endswith(("'", '"')):
        issues.append("prediction still contains surrounding quotes")

    segments = _parse_task4_segments(text_to_annotate)
    if not segments:
        return issues

    expected_stats = _task4_char_stats_from_segments(segments)
    observed_stats = _task4_char_stats_from_prediction(value)

    if observed_stats["length"] != expected_stats["length"]:
        issues.append(
            f"length mismatch: expected {expected_stats['length']}, got {observed_stats['length']}"
        )
    for key in ("uppercase", "lowercase", "digits", "punct"):
        if observed_stats[key] != expected_stats[key]:
            issues.append(
                f"{key} count mismatch: expected {expected_stats[key]}, got {observed_stats[key]}"
            )

    return issues


def assess_task8_prediction(prediction: Optional[str]) -> List[str]:
    issues: List[str] = []
    if prediction is None:
        return ["prediction is null"]

    value = prediction.strip()
    if not value:
        return ["prediction is empty"]

    prompt_markers = (
        "[Sample To Annotate]",
        "Functional Description:",
        "Wrapper Entry Information:",
        "Final answer:",
        "[Current Task]",
        "[Current Failure Signal]",
        "[Previous Failure Signal]",
        "[New Instructions]",
        "[New Rules]",
        "[Hard Rules]",
        "[Retry Rules]",
        "[Repair Instructions]",
    )
    if any(marker in value for marker in prompt_markers):
        issues.append("prediction contains leaked prompt markers")

    if re.search(r"(^|\n)\s*pass\s*(\n|$)", value):
        issues.append("prediction contains pass placeholders")

    placeholder_markers = (
        "Implement the ",
        "TODO",
        "placeholder",
        "Your implementation here",
        "Your kernel code here",
        "Your wrapper code here",
        "Your code here",
        "... kernel code here",
        "... wrapper code here",
    )
    if any(marker in value for marker in placeholder_markers):
        issues.append("prediction contains placeholder comments")

    if not any(token in value for token in ("import torch", "@triton.jit", "def ")):
        issues.append("prediction does not look like python/triton code")

    try:
        ast.parse(value)
    except SyntaxError as exc:
        location = f"line {exc.lineno}" if exc.lineno is not None else "unknown line"
        issues.append(f"prediction has Python syntax error: {location}: {exc.msg}")
    except ValueError as exc:
        issues.append(f"prediction has Python syntax error: {type(exc).__name__}: {exc}")

    return issues


def _extract_task7_answer(raw_text: str, tagged: Optional[str]) -> Optional[str]:
    candidate = tagged or _extract_answer_marker(raw_text) or _first_nonempty_line(raw_text)
    if not candidate:
        return None

    value = _strip_wrappers(candidate)
    if not value:
        return None
    lines = value.splitlines()
    if not lines:
        return None
    value = lines[0].strip()
    if not value:
        return None
    value = value.rstrip(" .。")
    value = re.sub(r"(?i)^(?:still|just|probably|maybe|it(?:'s| is))\s+", "", value).strip()

    reasoning_markers = ("okay", "let's", "the category is", "clue:")
    if any(marker in value.lower() for marker in reasoning_markers):
        marker_value = _extract_answer_marker(raw_text)
        if marker_value:
            value = _strip_wrappers(marker_value).splitlines()[0].strip().rstrip(" .。")
        else:
            recovered = None
            for line in reversed(raw_text.splitlines()):
                line_value = _strip_wrappers(line).strip().rstrip(" .。")
                if not line_value:
                    continue
                if any(marker in line_value.lower() for marker in reasoning_markers):
                    continue
                if len(line_value.split()) <= 8:
                    recovered = line_value
                    break
            if recovered:
                value = recovered
            else:
                phrase_candidates = []
                for match in re.finditer(
                    r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\b", raw_text
                ):
                    phrase = match.group(0).strip()
                    if not phrase:
                        continue
                    if phrase.lower() in {"okay", "first", "category", "clue", "best actor"}:
                        continue
                    phrase_candidates.append(phrase)
                if phrase_candidates:
                    value = phrase_candidates[-1]
                else:
                    return None

    value = re.split(r"[.;]", value, maxsplit=1)[0].strip()
    value = re.sub(r"(?i)\s*,\s*so\b.*$", "", value).strip()
    value = re.sub(r"(?i)^(?:still|just|probably|maybe|it(?:'s| is))\s+", "", value).strip()

    return value.lower() if value else None


def _normalize_task_output(task_id: int, value: Any) -> Optional[str]:
    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return ""

    if task_id in (1, 2):
        hit = INT_PATTERN.search(text)
        return hit.group(0) if hit else text

    if task_id == 3:
        return text

    if task_id == 4:
        return text

    return text


def execute_solver_code(
    solver_code: str,
    input_text: str,
    function_name: str = "solve",
) -> str:
    namespace: Dict[str, Any] = {
        "__builtins__": __builtins__,
        "ast": ast,
        "json": json,
        "re": re,
    }
    exec(solver_code, namespace, namespace)
    solver_fn = namespace.get(function_name)
    if not callable(solver_fn):
        raise ValueError(f"generated solver does not define callable `{function_name}`")
    result = solver_fn(input_text)
    if result is None:
        return ""
    return str(result).strip()


def validate_solver_code(
    task_id: int,
    solver_code: str,
    examples: Sequence[Dict[str, Any]],
    function_name: str = "solve",
    max_failure_examples: int = 5,
) -> Tuple[int, int, List[Dict[str, str]]]:
    total = 0
    correct = 0
    failures: List[Dict[str, str]] = []

    for example in examples:
        input_text = str(example.get("input", "")).strip()
        if not input_text:
            continue
        total += 1
        expected = _normalize_task_output(task_id, _normalize_example_output(example.get("output", "")))
        try:
            predicted_raw = execute_solver_code(solver_code, input_text, function_name=function_name)
            predicted = _normalize_task_output(task_id, predicted_raw)
        except Exception as exc:
            predicted = None
            error_text = f"{type(exc).__name__}: {exc}"
        else:
            error_text = ""

        if predicted == expected:
            correct += 1
            continue

        if len(failures) < max_failure_examples:
            failures.append(
                {
                    "input": input_text,
                    "expected": expected or "",
                    "predicted": predicted or "",
                    "error": error_text or "mismatch",
                }
            )

    return correct, total, failures


def postprocess_prediction(raw_text: str, task_id: int) -> Optional[str]:
    if not raw_text:
        return None

    tagged = _extract_tag_answer(raw_text)
    candidate = tagged if tagged else raw_text.strip()

    if task_id in (1, 2):
        hit = INT_PATTERN.search(candidate)
        return hit.group(0) if hit else None

    if task_id == 3:
        hit = LIST_PATTERN.search(candidate)
        if not hit:
            return candidate.strip() or None
        try:
            parsed = ast.literal_eval(hit.group(0))
        except (SyntaxError, ValueError):
            return hit.group(0)
        return str(parsed) if isinstance(parsed, list) else hit.group(0)

    if task_id == 5:
        lowered = candidate.lower()
        if "not sad" in lowered:
            return "Not sad"
        if "sad" in lowered:
            return "Sad"
        return candidate.strip() or None

    if task_id == 6:
        upper = candidate.strip().upper()
        if upper.startswith("Y"):
            return "Y"
        if upper.startswith("N"):
            return "N"
        return candidate.strip() or None

    if task_id == 4:
        return _extract_task4_answer(raw_text, tagged)

    if task_id == 7:
        return _extract_task7_answer(raw_text, tagged)

    if task_id == 8:
        text = _extract_task8_code(candidate)
        return text if text else None

    return candidate.strip() or None
