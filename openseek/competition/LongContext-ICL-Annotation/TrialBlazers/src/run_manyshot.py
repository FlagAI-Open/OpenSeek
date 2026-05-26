from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests
from tqdm import tqdm

try:
    from transformers import AutoTokenizer
except ImportError:  # pragma: no cover
    AutoTokenizer = None


TASK_FILES = {
    1: "openseek-1_closest_integers.json",
    3: "openseek-3_collatz_conjecture.json",
    4: "openseek-4_conala_concat_strings.json",
}

INT_PATTERN = re.compile(r"-?\d+")
LIST_PATTERN = re.compile(r"\[[\s,\-0-9]+\]")
LABEL_PATTERN = re.compile(r"<label>\s*(.*?)\s*</label>", re.IGNORECASE | re.DOTALL)
ANSWER_LINE_PATTERN = re.compile(r"(?im)^\s*(?:final\s+answer|answer|output)\s*[:：]\s*(.+?)\s*$")
THINK_PATTERN = re.compile(r"(?is)<think>.*?</think>")
UNCLOSED_THINK_PATTERN = re.compile(r"(?is)<think>.*$")


@dataclass(frozen=True)
class CandidateConfig:
    profile: str
    temperature: float
    top_p: float = 0.95
    enable_thinking: Optional[bool] = None
    max_tokens: Optional[int] = None


@dataclass
class Candidate:
    config: CandidateConfig
    raw_text: str
    prediction: Optional[str]


SUPPORTED_VARIANTS = (
    "official_flat_label",
    "flat_label_direct",
    "flat_label_reasoning",
    "anchored_direct",
    "anchored_reasoning",
    "anchored_passk_router",
    "task4_chunk_reasoning",
    "task4_chunk_direct",
    "task4_chunk_passk_router",
    "task4_boundary_reasoning",
    "task4_boundary_passk_router",
    "task4_literal_direct",
    "task4_boundary_direct",
    "task4_raw_transcribe",
    "task4_line_transcribe",
    "task4_tagged_transcribe",
    "task4_tagged_noexample_transcribe",
    "task4_lenhint_transcribe",
    "task4_lenhint_noexample_transcribe",
    "passk_router",
)
SUPPORTED_CONTEXT_MODES = ("structure_cover", "value_cover")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task1/3/4 long-context many-shot direct annotation.")
    parser.add_argument("--tasks", default="1,3,4", help="任务列表，例如 1,3,4。")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tokenizer-path", default="models/Qwen3-4B")
    parser.add_argument("--api-base", default="http://127.0.0.1:2026")
    parser.add_argument("--model-name", default="Qwen3-4B")
    parser.add_argument("--variant", choices=SUPPORTED_VARIANTS, default="official_flat_label")
    parser.add_argument("--context-mode", choices=SUPPORTED_CONTEXT_MODES, default="structure_cover")
    parser.add_argument("--target-example-tokens", type=int, default=24000)
    parser.add_argument("--max-input-tokens", type=int, default=30000)
    parser.add_argument("--reserved-generation-tokens", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--router-max-new-tokens", type=int, default=96)
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--client-concurrency", type=int, default=4)
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-backoff", type=float, default=1.5)
    parser.add_argument("--write-details", action="store_true")
    parser.add_argument("--enable-thinking", dest="enable_thinking", action="store_true", default=None)
    parser.add_argument("--disable-thinking", dest="enable_thinking", action="store_false")
    return parser.parse_args()


def parse_tasks(text: str) -> List[int]:
    tasks = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        task_id = int(part)
        if task_id not in TASK_FILES:
            raise ValueError(f"本实验只支持 task1/3/4，收到: {task_id}")
        tasks.append(task_id)
    return sorted(set(tasks))


def load_task_data(data_dir: Path, task_id: int) -> Dict[str, Any]:
    path = data_dir / TASK_FILES[task_id]
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_tokenizer(path: Path):
    if AutoTokenizer is None or not path.exists():
        return None
    return AutoTokenizer.from_pretrained(path, trust_remote_code=True)


def estimate_tokens(text: str, tokenizer: Any) -> int:
    if tokenizer is None:
        return max(1, len(text) // 4)
    return len(tokenizer.encode(text, add_special_tokens=False))


def normalize_example_output(value: Any) -> str:
    if isinstance(value, list) and value:
        return str(value[0]).strip()
    if value is None:
        return ""
    return str(value).strip()


def format_example(idx: int, example: Dict[str, Any]) -> str:
    return f"# {str(example.get('input', '')).strip()} <label> {normalize_example_output(example.get('output'))} </label>\n"


def format_raw_example(idx: int, example: Dict[str, Any]) -> str:
    return f"Input: {str(example.get('input', '')).strip()}\nOutput: {normalize_example_output(example.get('output'))}\n"


def format_line_input(input_text: str) -> str:
    items = _parse_str_list(input_text)
    if items is None:
        return f"Input: {input_text.strip()}"
    lines = ["Input chunks:"]
    for idx, item in enumerate(items, start=1):
        lines.append(f"{idx}. {item}")
    return "\n".join(lines)


def format_line_example(idx: int, example: Dict[str, Any]) -> str:
    input_text = str(example.get("input", "")).strip()
    return f"{format_line_input(input_text)}\nOutput: {normalize_example_output(example.get('output'))}\n"


def format_tagged_input(input_text: str) -> str:
    items = _parse_str_list(input_text)
    if items is None:
        return f"Input: {input_text.strip()}"
    lines = ["Input chunks:"]
    for idx, item in enumerate(items, start=1):
        lines.append(f'{idx:03d}: <chunk>{item}</chunk>')
    return "\n".join(lines)


def format_tagged_example(idx: int, example: Dict[str, Any]) -> str:
    input_text = str(example.get("input", "")).strip()
    return f"{format_tagged_input(input_text)}\nOutput: {normalize_example_output(example.get('output'))}\n"


def format_lenhint_input(input_text: str) -> str:
    items = _parse_str_list(input_text)
    if items is None:
        return f"Input: {input_text.strip()}"
    lines = [f"Expected final length: {sum(len(item) for item in items)}", "Chunks:"]
    for idx, item in enumerate(items, start=1):
        lines.append(f"{idx:03d} | len={len(item)} | {item}")
    return "\n".join(lines)


def format_lenhint_example(idx: int, example: Dict[str, Any]) -> str:
    input_text = str(example.get("input", "")).strip()
    return f"{format_lenhint_input(input_text)}\nOutput: {normalize_example_output(example.get('output'))}\n"


def _parse_int_list(text: str) -> Optional[List[int]]:
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return None
    if not isinstance(parsed, list):
        return None
    values: List[int] = []
    for item in parsed:
        if isinstance(item, bool) or not isinstance(item, int):
            return None
        values.append(int(item))
    return values


def _parse_str_list(text: str) -> Optional[List[str]]:
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return None
    if not isinstance(parsed, list):
        return None
    values: List[str] = []
    for item in parsed:
        if not isinstance(item, str):
            return None
        values.append(str(item))
    return values


def _task1_input_stats(example: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    nums = _parse_int_list(str(example.get("input", "")))
    if not nums:
        return None
    output_text = normalize_example_output(example.get("output"))
    try:
        output_value = int(output_text)
    except ValueError:
        return None
    has_negative = any(num < 0 for num in nums)
    has_nonnegative = any(num >= 0 for num in nums)
    return {
        "nums": nums,
        "length": len(nums),
        "output": output_value,
        "has_duplicate": len(set(nums)) < len(nums),
        "has_negative": has_negative,
        "has_nonnegative": has_nonnegative,
        "range": max(nums) - min(nums),
        "min_pair_diff": min(abs(nums[i] - nums[j]) for i in range(len(nums)) for j in range(i + 1, len(nums))),
    }


def _task3_input_stats(example: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    nums = _parse_int_list(str(example.get("input", "")))
    output_nums = _parse_int_list(normalize_example_output(example.get("output")))
    if not nums or not output_nums:
        return None
    return {
        "nums": nums,
        "output_nums": output_nums,
        "length": len(nums),
        "output_length": len(output_nums),
        "max_output": max(output_nums),
        "min_output": min(output_nums),
        "has_duplicate": len(set(nums)) < len(nums),
        "all_even": all(num % 2 == 0 for num in nums),
        "all_odd": all(num % 2 != 0 for num in nums),
        "mixed_parity": any(num % 2 == 0 for num in nums) and any(num % 2 != 0 for num in nums),
        "contains_one": 1 in nums,
        "contains_two": 2 in nums,
    }


def _task4_input_stats(example: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    items = _parse_str_list(str(example.get("input", "")))
    output_text = normalize_example_output(example.get("output"))
    if not items or output_text is None:
        return None
    joined = "".join(items)
    return {
        "items": items,
        "item_count": len(items),
        "joined": joined,
        "joined_length": len(output_text),
        "has_punct": any(any(not ch.isalnum() and ch != " " for ch in item) for item in items),
        "has_digit": any(any(ch.isdigit() for ch in item) for item in items),
        "mixed_case": any(any(ch.islower() for ch in item) for item in items) and any(
            any(ch.isupper() for ch in item) for item in items
        ),
        "has_multi_char_item": any(len(item) > 1 for item in items),
        "has_duplicate": len(set(items)) < len(items),
    }


def _task1_length_bucket(length: int) -> int:
    return max(0, min(length - 3, 7))


def _task1_output_bucket(value: int) -> int:
    if value <= 0:
        return 0
    if value == 1:
        return 1
    if value == 2:
        return 2
    if value <= 5:
        return 3
    if value <= 10:
        return 4
    if value <= 20:
        return 5
    return 6


def _task1_feature_bucket(stats: Dict[str, Any]) -> int:
    if stats["has_duplicate"] and stats["has_negative"] and stats["has_nonnegative"]:
        return 0
    if stats["has_duplicate"]:
        return 1
    if stats["has_negative"] and stats["has_nonnegative"]:
        return 2
    if stats["has_negative"]:
        return 3
    if stats["has_nonnegative"]:
        return 4
    if stats["range"] <= 10:
        return 5
    return 6


def _task1_priority(stats: Dict[str, Any]) -> Tuple[int, int, int, int, int]:
    output = int(stats["output"])
    return (
        0 if output <= 2 else 1 if output <= 5 else 2 if output <= 10 else 3,
        0 if stats["has_duplicate"] else 1,
        0 if stats["has_negative"] and stats["has_nonnegative"] else 1 if stats["has_negative"] else 2,
        abs(stats["length"] - 6),
        stats["output"],
    )


def _task1_similarity(query: Dict[str, Any], candidate: Dict[str, Any]) -> Tuple[int, int, int, int, int]:
    return (
        abs(int(query["length"]) - int(candidate["length"])),
        0 if bool(query["has_negative"]) == bool(candidate["has_negative"]) else 1,
        0 if bool(query["has_nonnegative"]) == bool(candidate["has_nonnegative"]) else 1,
        0 if bool(query["has_duplicate"]) == bool(candidate["has_duplicate"]) else 1,
        abs(int(query["range"]) - int(candidate["range"])),
    )


def _task3_length_bucket(length: int) -> int:
    return max(0, min(length - 2, 7))


def _task3_output_bucket(max_output: int) -> int:
    if max_output < 10:
        return 0
    if max_output < 100:
        return 1
    if max_output < 200:
        return 2
    if max_output < 400:
        return 3
    return 4


def _task3_feature_bucket(stats: Dict[str, Any]) -> int:
    if stats["contains_one"] and stats["contains_two"]:
        return 0
    if stats["contains_one"]:
        return 1
    if stats["contains_two"]:
        return 2
    if stats["has_duplicate"]:
        return 3
    if stats["mixed_parity"]:
        return 4
    if stats["all_even"]:
        return 5
    if stats["all_odd"]:
        return 6
    return 7


def _task3_priority(stats: Dict[str, Any]) -> Tuple[int, int, int, int, int]:
    return (
        0 if stats["has_duplicate"] else 1,
        0 if stats["contains_one"] or stats["contains_two"] else 1,
        -int(stats["max_output"]),
        abs(stats["length"] - 5),
        abs(stats["output_length"] - stats["length"]),
    )


def _task3_similarity(query: Dict[str, Any], candidate: Dict[str, Any]) -> Tuple[int, int, int, int, int, int]:
    return (
        abs(int(query["length"]) - int(candidate["length"])),
        0 if bool(query["mixed_parity"]) == bool(candidate["mixed_parity"]) else 1,
        0 if bool(query["all_even"]) == bool(candidate["all_even"]) else 1,
        0 if bool(query["all_odd"]) == bool(candidate["all_odd"]) else 1,
        abs(max(query["nums"]) - max(candidate["nums"])),
        abs(min(query["nums"]) - min(candidate["nums"])),
    )


def _task4_item_bucket(count: int) -> int:
    if count <= 3:
        return 0
    if count <= 5:
        return 1
    if count <= 7:
        return 2
    if count <= 10:
        return 3
    if count <= 13:
        return 4
    return 5


def _task4_output_bucket(length: int) -> int:
    if length <= 7:
        return 0
    if length <= 15:
        return 1
    if length <= 23:
        return 2
    if length <= 31:
        return 3
    return 4


def _task4_feature_bucket(stats: Dict[str, Any]) -> int:
    if stats["has_punct"] and stats["has_digit"]:
        return 0
    if stats["has_punct"]:
        return 1
    if stats["has_digit"]:
        return 2
    if stats["mixed_case"]:
        return 3
    if stats["has_duplicate"]:
        return 4
    if stats["has_multi_char_item"]:
        return 5
    return 6


def _task4_priority(stats: Dict[str, Any]) -> Tuple[int, int, int, int, int]:
    return (
        0 if stats["has_punct"] or stats["has_digit"] or stats["mixed_case"] else 1,
        0 if stats["has_duplicate"] else 1,
        0 if stats["has_multi_char_item"] else 1,
        -int(stats["joined_length"]),
        abs(stats["item_count"] - 9),
    )


def _task4_similarity(query: Dict[str, Any], candidate: Dict[str, Any]) -> Tuple[int, int, int, int, int, int]:
    return (
        abs(int(query["item_count"]) - int(candidate["item_count"])),
        abs(int(query["joined_length"]) - int(candidate["joined_length"])),
        0 if bool(query["has_punct"]) == bool(candidate["has_punct"]) else 1,
        0 if bool(query["has_digit"]) == bool(candidate["has_digit"]) else 1,
        0 if bool(query["mixed_case"]) == bool(candidate["mixed_case"]) else 1,
        0 if bool(query["has_duplicate"]) == bool(candidate["has_duplicate"]) else 1,
    )


def _bucketed_round_robin(
    examples: Sequence[Dict[str, Any]],
    key_fn,
    stats_fn,
    priority_fn,
) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[int, ...], List[Tuple[Dict[str, Any], Dict[str, Any]]]] = defaultdict(list)
    for example in examples:
        stats = stats_fn(example)
        if stats is None:
            continue
        groups[key_fn(stats)].append((example, stats))

    for key in groups:
        groups[key].sort(key=lambda item: priority_fn(item[1]))

    ranked: List[Dict[str, Any]] = []
    ordered_keys = sorted(groups)
    while True:
        progressed = False
        for key in ordered_keys:
            bucket = groups.get(key)
            if bucket:
                ranked.append(bucket.pop(0)[0])
                progressed = True
        if not progressed:
            break
    return ranked


def structure_rank_examples(task_id: int, examples: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if task_id == 1:
        return _bucketed_round_robin(
            examples,
            key_fn=lambda stats: (
                _task1_length_bucket(int(stats["length"])),
                _task1_output_bucket(int(stats["output"])),
                _task1_feature_bucket(stats),
            ),
            stats_fn=_task1_input_stats,
            priority_fn=_task1_priority,
        )
    if task_id == 3:
        return _bucketed_round_robin(
            examples,
            key_fn=lambda stats: (
                _task3_length_bucket(int(stats["length"])),
                _task3_output_bucket(int(stats["max_output"])),
                _task3_feature_bucket(stats),
            ),
            stats_fn=_task3_input_stats,
            priority_fn=_task3_priority,
        )
    if task_id == 4:
        return _bucketed_round_robin(
            examples,
            key_fn=lambda stats: (
                _task4_item_bucket(int(stats["item_count"])),
                _task4_output_bucket(int(stats["joined_length"])),
                _task4_feature_bucket(stats),
            ),
            stats_fn=_task4_input_stats,
            priority_fn=_task4_priority,
        )
    return [example for example in examples if str(example.get("input", "")).strip()]


def query_stats_for_task(task_id: int, query_text: str) -> Optional[Dict[str, Any]]:
    if task_id == 1:
        nums = _parse_int_list(query_text)
        if not nums:
            return None
        has_negative = any(num < 0 for num in nums)
        has_nonnegative = any(num >= 0 for num in nums)
        return {
            "nums": nums,
            "length": len(nums),
            "has_duplicate": len(set(nums)) < len(nums),
            "has_negative": has_negative,
            "has_nonnegative": has_nonnegative,
            "range": max(nums) - min(nums),
        }
    if task_id == 3:
        nums = _parse_int_list(query_text)
        if not nums:
            return None
        return {
            "nums": nums,
            "length": len(nums),
            "has_duplicate": len(set(nums)) < len(nums),
            "all_even": all(num % 2 == 0 for num in nums),
            "all_odd": all(num % 2 != 0 for num in nums),
            "mixed_parity": any(num % 2 == 0 for num in nums) and any(num % 2 != 0 for num in nums),
        }
    if task_id == 4:
        items = _parse_str_list(query_text)
        if not items:
            return None
        joined = "".join(items)
        return {
            "items": items,
            "item_count": len(items),
            "joined_length": len(joined),
            "has_punct": any(any(not ch.isalnum() and ch != " " for ch in item) for item in items),
            "has_digit": any(any(ch.isdigit() for ch in item) for item in items),
            "mixed_case": any(any(ch.islower() for ch in item) for item in items) and any(
                any(ch.isupper() for ch in item) for item in items
            ),
            "has_duplicate": len(set(items)) < len(items),
        }
    return None


def query_tail_examples(task_id: int, examples: Sequence[Dict[str, Any]], query_text: str, limit: int = 24) -> List[Dict[str, Any]]:
    query_stats = query_stats_for_task(task_id, query_text)
    if query_stats is None:
        return []

    scored: List[Tuple[Tuple[int, ...], Dict[str, Any]]] = []
    for example in examples:
        if task_id == 1:
            stats = _task1_input_stats(example)
            if stats is None:
                continue
            scored.append((_task1_similarity(query_stats, stats), example))
        elif task_id == 3:
            stats = _task3_input_stats(example)
            if stats is None:
                continue
            scored.append((_task3_similarity(query_stats, stats), example))
        elif task_id == 4:
            stats = _task4_input_stats(example)
            if stats is None:
                continue
            scored.append((_task4_similarity(query_stats, stats), example))
    scored.sort(key=lambda item: item[0])
    return [example for _, example in scored[:limit]]


def _task1_value_similarity(query: Dict[str, Any], candidate: Dict[str, Any]) -> Tuple[int, int, int, int, int, int]:
    query_nums = sorted(int(num) for num in query["nums"])
    cand_nums = sorted(int(num) for num in candidate["nums"])
    pair_count = min(len(query_nums), len(cand_nums))
    aligned_distance = sum(abs(query_nums[idx] - cand_nums[idx]) for idx in range(pair_count))
    try:
        cand_output = int(candidate["output"])
    except (TypeError, ValueError):
        cand_output = 999
    return (
        abs(int(query["length"]) - int(candidate["length"])),
        aligned_distance,
        0 if bool(query["has_negative"]) == bool(candidate["has_negative"]) else 1,
        0 if bool(query["has_nonnegative"]) == bool(candidate["has_nonnegative"]) else 1,
        0 if bool(query["has_duplicate"]) == bool(candidate["has_duplicate"]) else 1,
        cand_output,
    )


def _task3_value_similarity(query: Dict[str, Any], candidate: Dict[str, Any]) -> Tuple[int, int, int, int, int, int]:
    query_nums = [int(num) for num in query["nums"]]
    cand_nums = [int(num) for num in candidate["nums"]]
    pair_count = min(len(query_nums), len(cand_nums))
    aligned_distance = sum(abs(query_nums[idx] - cand_nums[idx]) for idx in range(pair_count))
    query_odds = sum(num % 2 != 0 for num in query_nums)
    cand_odds = sum(num % 2 != 0 for num in cand_nums)
    query_evens = len(query_nums) - query_odds
    cand_evens = len(cand_nums) - cand_odds
    value_overlap = len(set(query_nums) & set(cand_nums))
    return (
        abs(len(query_nums) - len(cand_nums)),
        -value_overlap,
        abs(query_odds - cand_odds),
        abs(query_evens - cand_evens),
        aligned_distance,
        abs(max(query_nums) - max(cand_nums)),
    )


def _task4_value_similarity(query: Dict[str, Any], candidate: Dict[str, Any]) -> Tuple[int, int, int, int, int, int, int]:
    query_items = [str(item) for item in query["items"]]
    cand_items = [str(item) for item in candidate["items"]]
    exact_overlap = len(set(query_items) & set(cand_items))
    char_overlap = len(set("".join(query_items)) & set("".join(cand_items)))
    return (
        abs(len(query_items) - len(cand_items)),
        -exact_overlap,
        abs(int(query["joined_length"]) - int(candidate["joined_length"])),
        -char_overlap,
        0 if bool(query["has_punct"]) == bool(candidate["has_punct"]) else 1,
        0 if bool(query["has_digit"]) == bool(candidate["has_digit"]) else 1,
        0 if bool(query["mixed_case"]) == bool(candidate["mixed_case"]) else 1,
    )


def query_value_tail_examples(
    task_id: int,
    examples: Sequence[Dict[str, Any]],
    query_text: str,
    limit: int = 48,
) -> List[Dict[str, Any]]:
    query_stats = query_stats_for_task(task_id, query_text)
    if query_stats is None:
        return []

    scored: List[Tuple[Tuple[int, ...], Dict[str, Any]]] = []
    for example in examples:
        if task_id == 1:
            stats = _task1_input_stats(example)
            if stats is None:
                continue
            scored.append((_task1_value_similarity(query_stats, stats), example))
        elif task_id == 3:
            stats = _task3_input_stats(example)
            if stats is None:
                continue
            scored.append((_task3_value_similarity(query_stats, stats), example))
        elif task_id == 4:
            stats = _task4_input_stats(example)
            if stats is None:
                continue
            scored.append((_task4_value_similarity(query_stats, stats), example))
    scored.sort(key=lambda item: item[0])
    return [example for _, example in scored[:limit]]


def build_examples_text(
    task_id: int,
    examples: Sequence[Dict[str, Any]],
    tokenizer: Any,
    max_input_tokens: int,
    target_example_tokens: int,
    reserved_generation_tokens: int,
    context_mode: str,
    query_text: str = "",
) -> Tuple[str, int, int]:
    if context_mode not in SUPPORTED_CONTEXT_MODES:
        raise ValueError(f"不支持的上下文模式: {context_mode}")
    ranked = structure_rank_examples(task_id, examples)
    tail_examples: List[Dict[str, Any]] = []
    if query_text:
        if context_mode == "value_cover":
            structure_tail = query_tail_examples(task_id, examples, query_text, limit=24)
            value_tail = query_value_tail_examples(task_id, examples, query_text, limit=48)
            seen_tail: set[str] = set()
            for example in value_tail + structure_tail:
                key = str(example.get("input", ""))
                if key in seen_tail:
                    continue
                seen_tail.add(key)
                tail_examples.append(example)
        else:
            tail_examples = query_tail_examples(task_id, examples, query_text, limit=24)
        tail_keys = {str(example.get("input", "")) for example in tail_examples}
        ranked = [example for example in ranked if str(example.get("input", "")) not in tail_keys]

    budget = max(0, max_input_tokens - reserved_generation_tokens)
    tail_blocks = [format_example(0, example) for example in tail_examples]
    tail_tokens = sum(estimate_tokens(block, tokenizer) for block in tail_blocks)
    body_target_tokens = max(0, target_example_tokens - tail_tokens)
    selected_blocks: List[str] = []
    used = 0
    count = 0
    for example in ranked:
        block = format_example(count + 1, example)
        block_tokens = estimate_tokens(block, tokenizer)
        if used + block_tokens > budget:
            continue
        selected_blocks.append(block)
        used += block_tokens
        count += 1
        if used >= body_target_tokens:
            break
    for block in tail_blocks:
        block_tokens = estimate_tokens(block, tokenizer)
        if used + block_tokens > budget:
            continue
        selected_blocks.append(block)
        used += block_tokens
        count += 1
    return "\n".join(selected_blocks).strip(), count, used


def task_rules(task_id: int, variant: str) -> List[str]:
    common = [
        "Read the official examples carefully and imitate their flat label format.",
        "Return exactly one answer wrapped in <label> and </label>.",
        "Do not output markdown, code, JSON, or extra explanation text.",
    ]
    anchored = (
        variant.startswith("anchored_")
        or variant.startswith("task4_chunk_")
        or variant.startswith("task4_boundary_")
        or variant == "task4_literal_direct"
    )
    if task_id == 1:
        specific = [
            "The label content is the smallest absolute difference between any two integers in the list.",
            "Output digits only inside the label.",
        ]
        if anchored:
            specific.extend(
                [
                    "For the sample to annotate, internally sort the integers from smallest to largest.",
                    "Only compare adjacent integers after sorting; the smallest adjacent gap is the answer.",
                    "Do not guess common labels such as 0, 1, or 2 unless that exact adjacent gap is present.",
                ]
            )
    elif task_id == 3:
        specific = [
            "The label content is the list after applying exactly one Collatz step to each input integer.",
            "Keep Python list formatting inside the label, including commas and spaces in the normal list string form.",
        ]
        if anchored:
            specific.extend(
                [
                    "Transform each position independently and preserve the original list length and order.",
                    "Even integer n becomes n/2; odd integer n becomes 3*n+1.",
                    "Never divide odd integers by two. Check every odd position before writing the final list.",
                ]
            )
    else:
        specific = [
            "The label content is the concatenation of the list elements from left to right with no separator.",
            "Preserve every character exactly, including case, punctuation, digits, and spacing inside each string element.",
        ]
        if anchored:
            specific.extend(
                [
                    "Copy each quoted string element exactly once in order.",
                    "Do not change case, spelling, punctuation, digits, or spaces inside a quoted string.",
                    "Before finalizing, internally verify the output length equals the sum of the element lengths.",
                ]
            )
        if variant.startswith("task4_chunk_"):
            specific = [
                "The label content is the concatenation of the list elements from left to right with no separator.",
                "Treat each quoted list element as an indivisible raw chunk.",
                "Append the chunks directly next to each other: previous chunk immediately followed by next chunk.",
                "Never insert a space or separator between two chunks unless that space is inside one quoted chunk.",
                "Do not count character lengths; do not rewrite chunks as words; do not normalize case or spelling.",
                "Copy every one-letter chunk exactly as its own adjacent character.",
                "Return exactly one answer wrapped in <label> and </label>.",
                "Do not output markdown, code, JSON, length checks, or extra explanation text.",
            ]
        if variant.startswith("task4_boundary_"):
            specific = [
                "The label content is the concatenation of the list elements from left to right with no separator.",
                "Copy each quoted string element exactly once in order.",
                "Do not add spaces at chunk boundaries, even when two adjacent chunks look like separate words or letters.",
                "Do not remove, add, lowercase, uppercase, translate, or spell-correct any character from a quoted string.",
                "If a one-letter chunk appears between words, keep that one letter directly adjacent to its neighbors.",
                "Use a short boundary scan only: item1+item2+item3...; do not do a character-by-character length count.",
                "Return exactly one answer wrapped in <label> and </label>.",
                "Do not output markdown, code, JSON, length checks, or extra explanation text.",
            ]
        if variant == "task4_literal_direct":
            specific.extend(
                [
                    "Treat every quoted string as raw text, not as a word to translate, expand, correct, or normalize.",
                    "Write the concatenated text once inside the label; do not include a checklist, length count, or any explanation.",
                ]
            )
        if variant == "task4_boundary_direct":
            specific = [
                "The label content is the concatenation of the list elements from left to right with no separator.",
                "Copy each quoted string element exactly once in order.",
                "Do not add spaces at chunk boundaries, even when adjacent chunks look like separate words.",
                "Do not remove, add, lowercase, uppercase, translate, or spell-correct any character from a quoted string.",
                "Return exactly one answer wrapped in <label> and </label>.",
                "Do not output markdown, code, JSON, length checks, or extra explanation text.",
            ]
        if variant == "task4_raw_transcribe":
            specific = [
                "The output is the concatenation of the quoted string list elements from left to right with no separator.",
                "Copy every character exactly as raw text.",
                "Do not add spaces at boundaries unless that space is inside a quoted string element.",
                "Do not use XML, labels, markdown, JSON, code, quotes, explanations, or length checks.",
                "Return only the raw concatenated string on one line.",
            ]
        if variant == "task4_line_transcribe":
            specific = [
                "The output is the concatenation of the numbered raw chunks from top to bottom with no separator.",
                "Copy every character in each numbered chunk exactly once.",
                "Do not add spaces at boundaries unless a space is already inside a numbered chunk.",
                "Do not use XML, labels, markdown, JSON, code, quotes, explanations, or length checks.",
                "Return only the raw concatenated string on one line.",
            ]
        if variant in {"task4_tagged_transcribe", "task4_tagged_noexample_transcribe"}:
            specific = [
                "The output is the concatenation of the raw text inside each <chunk>...</chunk> block from top to bottom.",
                "Copy only the chunk contents; do not copy line numbers, tags, separators, or spaces around the tags.",
                "Append chunk contents directly with no separator.",
                "Do not add, remove, lowercase, uppercase, translate, or spell-correct any chunk character.",
                "Return only the raw concatenated string on one line.",
            ]
        if variant in {"task4_lenhint_transcribe", "task4_lenhint_noexample_transcribe"}:
            specific = [
                "The output is the concatenation of the chunk text after each len marker from top to bottom.",
                "Use the shown length values only as copy checks; do not copy line numbers, len markers, separators, or spaces around separators.",
                "Append chunk text directly with no separator.",
                "Return a raw string whose character count equals the expected final length.",
                "Return only the raw concatenated string on one line.",
            ]

    if variant == "flat_label_direct":
        specific.append("Reason carefully before answering, but keep the final output flat and minimal.")
    elif variant == "flat_label_reasoning":
        specific.append("You may reason silently before answering, but the final answer must still be only the flat label.")
    elif variant == "anchored_reasoning":
        specific.append("Think through the checklist internally, then output only the final flat label.")
    elif variant == "task4_chunk_reasoning":
        specific.append("Use private chunk-copy scratch work only if needed, then output only the final flat label.")
    elif variant == "task4_chunk_direct":
        specific.append("Answer immediately with only the final flat label.")
    elif variant == "task4_chunk_passk_router":
        specific.append("Use the chunk-copy rule, then output only the final flat label.")
    elif variant == "task4_boundary_reasoning":
        specific.append("Use short private boundary-copy scratch work, then output only the final flat label.")
    elif variant == "task4_boundary_passk_router":
        specific.append("Use the boundary-copy rule, then output only the final flat label.")
    elif variant == "task4_literal_direct":
        specific.append("Answer immediately with only the final flat label.")
    elif variant == "task4_boundary_direct":
        specific.append("Answer directly with only the final flat label.")
    elif variant == "task4_raw_transcribe":
        specific.append("Answer directly with only the raw output string.")
    elif variant == "task4_line_transcribe":
        specific.append("Answer directly with only the raw output string.")
    elif variant in {"task4_tagged_transcribe", "task4_tagged_noexample_transcribe"}:
        specific.append("Answer directly with only the raw output string.")
    elif variant in {"task4_lenhint_transcribe", "task4_lenhint_noexample_transcribe"}:
        specific.append("Answer directly with only the raw output string.")
    elif variant in {"anchored_direct", "anchored_passk_router"}:
        specific.append("Use the checklist internally, then output only the final flat label.")
    else:
        specific.append("Use the examples as the main annotation pattern and keep the final answer minimal.")
    return common + specific


def build_prompt(
    task_id: int,
    task_description: str,
    examples_text: str,
    sample_input: str,
    profile: str,
) -> str:
    rules = "\n".join(f"{idx + 1}. {rule}" for idx, rule in enumerate(task_rules(task_id, profile)))
    variant_intro = {
        "official_flat_label": (
            "You are solving an OpenSeek long-context annotation sample.\n"
            "Follow the official examples exactly and return only the final flat label.\n"
        ),
        "flat_label_direct": (
            "You are solving an OpenSeek long-context annotation sample.\n"
            "Use the examples to infer the rule, then return only the final flat label.\n"
        ),
        "flat_label_reasoning": (
            "You are solving an OpenSeek long-context annotation sample.\n"
            "You may think internally, but do not reveal reasoning and return only the final flat label.\n"
        ),
        "anchored_direct": (
            "You are solving an OpenSeek long-context annotation sample.\n"
            "Use the long example bank and the deterministic checklist, then return only the final flat label.\n"
        ),
        "anchored_reasoning": (
            "You are solving an OpenSeek long-context annotation sample.\n"
            "You may use private scratch work, but the visible answer must be only the final flat label.\n"
        ),
        "anchored_passk_router": (
            "You are solving an OpenSeek long-context annotation sample.\n"
            "Use the long example bank and the deterministic checklist, then return only the final flat label.\n"
        ),
        "task4_chunk_reasoning": (
            "You are solving an OpenSeek literal string concatenation sample.\n"
            "Copy raw chunks directly adjacent to each other. Do not count lengths or insert separators.\n"
        ),
        "task4_chunk_direct": (
            "You are solving an OpenSeek literal string concatenation sample.\n"
            "Copy raw chunks directly adjacent to each other and return only the final flat label.\n"
        ),
        "task4_chunk_passk_router": (
            "You are solving an OpenSeek literal string concatenation sample.\n"
            "Use chunk-copy candidates and choose the final flat label.\n"
        ),
        "task4_boundary_reasoning": (
            "You are solving an OpenSeek literal string concatenation sample.\n"
            "Copy chunks in order, scan boundaries for invented separators, and return only the final flat label.\n"
        ),
        "task4_boundary_passk_router": (
            "You are solving an OpenSeek literal string concatenation sample.\n"
            "Use boundary-copy candidates and choose the final flat label.\n"
        ),
        "task4_literal_direct": (
            "You are solving an OpenSeek literal string concatenation sample.\n"
            "Copy the quoted string elements exactly in order, with no separators, and return only the final flat label.\n"
        ),
        "task4_boundary_direct": (
            "You are solving an OpenSeek literal string concatenation sample.\n"
            "Copy chunks in order, check boundaries for invented separators, and return only the final flat label.\n"
        ),
        "task4_raw_transcribe": (
            "You are transcribing an OpenSeek literal string concatenation sample.\n"
            "Return only the raw output string, with no wrapper or explanation.\n"
        ),
        "task4_line_transcribe": (
            "You are transcribing an OpenSeek literal string concatenation sample from numbered raw chunks.\n"
            "Return only the raw output string, with no wrapper or explanation.\n"
        ),
        "task4_tagged_transcribe": (
            "You are transcribing an OpenSeek literal string concatenation sample from tagged raw chunks.\n"
            "Return only the raw output string, with no wrapper or explanation.\n"
        ),
        "task4_tagged_noexample_transcribe": (
            "You are transcribing an OpenSeek literal string concatenation sample from tagged raw chunks.\n"
            "Return only the raw output string, with no wrapper or explanation.\n"
        ),
        "task4_lenhint_transcribe": (
            "You are transcribing an OpenSeek literal string concatenation sample from length-checked raw chunks.\n"
            "Return only the raw output string, with no wrapper or explanation.\n"
        ),
        "task4_lenhint_noexample_transcribe": (
            "You are transcribing an OpenSeek literal string concatenation sample from length-checked raw chunks.\n"
            "Return only the raw output string, with no wrapper or explanation.\n"
        ),
        "router": (
            "You are comparing candidate answers for one OpenSeek long-context annotation sample.\n"
            "Use the examples and task description to choose the best final answer only.\n"
        ),
    }.get(
        profile,
        "You are solving an OpenSeek long-context annotation sample.\n"
        "Return only the final flat label.\n",
    )
    return (
        variant_intro
        + "\n[Task Description]\n"
        + f"{task_description}\n\n"
        + "[Official In-Context Examples]\n"
        + f"{examples_text}\n\n"
        + "[Sample To Annotate]\n"
        + f"{sample_input}\n\n"
        + "[Output Rules]\n"
        + f"{rules}\n\n"
        + "Final answer:\n"
    )


def build_raw_transcribe_prompt(
    task_description: str,
    examples_text: str,
    sample_input: str,
) -> str:
    return (
        "You are transcribing an OpenSeek literal string concatenation sample.\n"
        "Concatenate the quoted string list elements from left to right with no separator.\n"
        "Copy raw characters exactly. Do not add spaces at boundaries unless the space is inside a quoted string.\n"
        "Return only the raw output string on one line. Do not use XML labels, quotes, markdown, JSON, code, or explanation.\n\n"
        "[Task Description]\n"
        f"{task_description}\n\n"
        "[Examples]\n"
        f"{examples_text}\n\n"
        "[Sample]\n"
        f"Input: {sample_input}\n"
        "Output:"
    )


def build_line_transcribe_prompt(
    task_description: str,
    examples_text: str,
    sample_input: str,
) -> str:
    return (
        "You are transcribing an OpenSeek literal string concatenation sample.\n"
        "The sample is shown as numbered raw chunks. Concatenate chunk 1, then chunk 2, and so on, with no separator.\n"
        "Copy raw characters exactly. Do not add spaces at boundaries unless the space is inside a numbered chunk.\n"
        "Return only the raw output string on one line. Do not use XML labels, quotes, markdown, JSON, code, or explanation.\n\n"
        "[Task Description]\n"
        f"{task_description}\n\n"
        "[Examples]\n"
        f"{examples_text}\n\n"
        "[Sample]\n"
        f"{format_line_input(sample_input)}\n"
        "Output:"
    )


def build_tagged_transcribe_prompt(
    task_description: str,
    examples_text: str,
    sample_input: str,
    include_examples: bool,
) -> str:
    examples_block = f"\n[Examples]\n{examples_text}\n" if include_examples and examples_text.strip() else ""
    return (
        "You are transcribing an OpenSeek literal string concatenation sample.\n"
        "The sample is shown as tagged raw chunks. Concatenate only the text inside each <chunk>...</chunk> block, top to bottom, with no separator.\n"
        "Do not copy line numbers, tags, or separators. Do not add spaces at boundaries.\n"
        "Copy raw characters exactly. Return only the raw output string on one line. Do not use XML labels, quotes, markdown, JSON, code, or explanation.\n\n"
        "[Task Description]\n"
        f"{task_description}\n"
        f"{examples_block}\n"
        "[Sample]\n"
        f"{format_tagged_input(sample_input)}\n"
        "Output:"
    )


def build_lenhint_transcribe_prompt(
    task_description: str,
    examples_text: str,
    sample_input: str,
    include_examples: bool,
) -> str:
    examples_block = f"\n[Examples]\n{examples_text}\n" if include_examples and examples_text.strip() else ""
    return (
        "You are transcribing an OpenSeek literal string concatenation sample.\n"
        "The sample is shown as length-checked raw chunks. Concatenate only the chunk text after each len marker, top to bottom, with no separator.\n"
        "Use the expected final length and per-chunk lengths only to check your copy. Do not copy line numbers, len markers, or separator bars.\n"
        "Do not add spaces at boundaries. Copy raw characters exactly. Return only the raw output string on one line.\n\n"
        "[Task Description]\n"
        f"{task_description}\n"
        f"{examples_block}\n"
        "[Sample]\n"
        f"{format_lenhint_input(sample_input)}\n"
        "Output:"
    )


def strip_think(text: str) -> str:
    cleaned = THINK_PATTERN.sub("", text or "")
    cleaned = UNCLOSED_THINK_PATTERN.sub("", cleaned)
    return cleaned.strip()


def extract_answer(raw_text: str, task_id: int) -> Optional[str]:
    raw_text = strip_think(raw_text or "")
    if not raw_text:
        return None
    answer_matches = list(ANSWER_LINE_PATTERN.finditer(raw_text))
    if answer_matches:
        text = answer_matches[-1].group(1).strip()
    else:
        text = raw_text
    if not text:
        return None
    label = LABEL_PATTERN.search(text)
    if label:
        text = label.group(1).strip()
    else:
        line = ANSWER_LINE_PATTERN.search(text)
        if line:
            text = line.group(1).strip()
    text = text.strip().strip("`").strip()
    if task_id == 1:
        hit = INT_PATTERN.search(text)
        return hit.group(0) if hit else None
    if task_id == 3:
        hit = LIST_PATTERN.search(text)
        if hit:
            compact = hit.group(0)
            try:
                parsed = ast.literal_eval(compact)
            except (SyntaxError, ValueError):
                return compact
            if isinstance(parsed, list):
                return str(parsed)
            return compact
        return text if text else None
    if task_id == 4:
        # 最小格式提取：只去掉常见模型包装，不根据输入计算或修复字符。
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return None
        if len(lines) == 1:
            return lines[0]
        return lines[-1]
    return text if text else None


def extract_prediction(raw_text: str, task_id: int, profile: str) -> Optional[str]:
    if task_id == 4 and profile in {
        "task4_raw_transcribe",
        "task4_line_transcribe",
        "task4_tagged_transcribe",
        "task4_tagged_noexample_transcribe",
        "task4_lenhint_transcribe",
        "task4_lenhint_noexample_transcribe",
    }:
        raw_text = strip_think(raw_text or "")
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        if not lines:
            return None
        text = lines[-1].strip().strip("`").strip()
        if text.lower().startswith("output:"):
            text = text.split(":", 1)[1].strip()
        if text.startswith("<label>") and text.lower().endswith("</label>"):
            label = LABEL_PATTERN.search(text)
            if label:
                text = label.group(1).strip()
        return text or None
    return extract_answer(raw_text, task_id)


def normalize_for_vote(task_id: int, prediction: Optional[str]) -> Optional[str]:
    if prediction is None:
        return None
    prediction = prediction.strip()
    if task_id == 3:
        return re.sub(r"\s+", "", prediction)
    return prediction


def chat_completion(
    api_base: str,
    model_name: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    timeout: float,
    enable_thinking: Optional[bool],
) -> str:
    payload: Dict[str, Any] = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    if enable_thinking is not None:
        payload["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
    response = requests.post(
        f"{api_base.rstrip('/')}/v1/chat/completions",
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()
    choices = data.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message", {})
    content = message.get("content")
    if content:
        return str(content).strip()
    return ""


def call_with_retries(
    args: argparse.Namespace,
    prompt: str,
    cfg: CandidateConfig,
    max_tokens: Optional[int] = None,
) -> str:
    for attempt in range(args.retries + 1):
        try:
            effective_thinking = args.enable_thinking if args.enable_thinking is not None else cfg.enable_thinking
            return chat_completion(
                api_base=args.api_base,
                model_name=args.model_name,
                prompt=prompt,
                max_tokens=max_tokens or cfg.max_tokens or args.max_new_tokens,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                timeout=args.timeout,
                enable_thinking=effective_thinking,
            )
        except Exception as exc:  # noqa: BLE001
            if attempt < args.retries:
                time.sleep(args.retry_backoff * (attempt + 1))
    return f""


def configs_for(task_id: int, variant: str) -> List[CandidateConfig]:
    if variant == "passk_router":
        return [
            CandidateConfig(profile="official_flat_label", temperature=0.0, top_p=1.0, enable_thinking=False),
            CandidateConfig(profile="flat_label_direct", temperature=0.0, top_p=1.0, enable_thinking=False),
            CandidateConfig(profile="flat_label_reasoning", temperature=0.0, top_p=1.0, enable_thinking=True, max_tokens=768),
        ]
    if variant == "anchored_passk_router":
        return [
            CandidateConfig(profile="anchored_direct", temperature=0.0, top_p=1.0, enable_thinking=False),
            CandidateConfig(profile="anchored_direct", temperature=0.2, top_p=0.9, enable_thinking=False),
            CandidateConfig(profile="anchored_reasoning", temperature=0.0, top_p=1.0, enable_thinking=True, max_tokens=4096),
        ]
    if variant == "task4_chunk_passk_router":
        if task_id != 4:
            raise ValueError("task4_chunk_passk_router 只用于 Task4。")
        return [
            CandidateConfig(profile="task4_chunk_direct", temperature=0.0, top_p=1.0, enable_thinking=False, max_tokens=192),
            CandidateConfig(profile="task4_chunk_direct", temperature=0.2, top_p=0.9, enable_thinking=False, max_tokens=192),
            CandidateConfig(profile="task4_chunk_reasoning", temperature=0.0, top_p=1.0, enable_thinking=True, max_tokens=2048),
            CandidateConfig(profile="anchored_reasoning", temperature=0.0, top_p=1.0, enable_thinking=True, max_tokens=4096),
        ]
    if variant == "task4_boundary_passk_router":
        if task_id != 4:
            raise ValueError("task4_boundary_passk_router 只用于 Task4。")
        return [
            CandidateConfig(profile="task4_boundary_reasoning", temperature=0.0, top_p=1.0, enable_thinking=True, max_tokens=2048),
            CandidateConfig(profile="task4_boundary_reasoning", temperature=0.2, top_p=0.9, enable_thinking=True, max_tokens=2048),
            CandidateConfig(profile="anchored_reasoning", temperature=0.0, top_p=1.0, enable_thinking=True, max_tokens=4096),
        ]
    if variant == "official_flat_label":
        return [CandidateConfig(profile=variant, temperature=0.0, top_p=1.0, enable_thinking=False)]
    if variant == "flat_label_direct":
        return [CandidateConfig(profile=variant, temperature=0.0, top_p=1.0, enable_thinking=False)]
    if variant == "flat_label_reasoning":
        return [CandidateConfig(profile=variant, temperature=0.0, top_p=1.0, enable_thinking=True, max_tokens=768)]
    if variant == "anchored_direct":
        return [CandidateConfig(profile=variant, temperature=0.0, top_p=1.0, enable_thinking=False)]
    if variant == "anchored_reasoning":
        return [CandidateConfig(profile=variant, temperature=0.0, top_p=1.0, enable_thinking=True, max_tokens=4096)]
    if variant == "task4_chunk_reasoning":
        if task_id != 4:
            raise ValueError("task4_chunk_reasoning 只用于 Task4。")
        return [CandidateConfig(profile=variant, temperature=0.0, top_p=1.0, enable_thinking=True, max_tokens=2048)]
    if variant == "task4_chunk_direct":
        if task_id != 4:
            raise ValueError("task4_chunk_direct 只用于 Task4。")
        return [CandidateConfig(profile=variant, temperature=0.0, top_p=1.0, enable_thinking=False, max_tokens=192)]
    if variant == "task4_boundary_reasoning":
        if task_id != 4:
            raise ValueError("task4_boundary_reasoning 只用于 Task4。")
        return [CandidateConfig(profile=variant, temperature=0.0, top_p=1.0, enable_thinking=True, max_tokens=2048)]
    if variant == "task4_literal_direct":
        if task_id != 4:
            raise ValueError("task4_literal_direct 只用于 Task4；Task1/3 请继续使用 anchored_reasoning。")
        return [CandidateConfig(profile=variant, temperature=0.0, top_p=1.0, enable_thinking=False, max_tokens=128)]
    if variant == "task4_boundary_direct":
        if task_id != 4:
            raise ValueError("task4_boundary_direct 只用于 Task4。")
        return [CandidateConfig(profile=variant, temperature=0.0, top_p=1.0, enable_thinking=False, max_tokens=192)]
    if variant == "task4_raw_transcribe":
        if task_id != 4:
            raise ValueError("task4_raw_transcribe 只用于 Task4。")
        return [CandidateConfig(profile=variant, temperature=0.0, top_p=1.0, enable_thinking=False, max_tokens=192)]
    if variant == "task4_line_transcribe":
        if task_id != 4:
            raise ValueError("task4_line_transcribe 只用于 Task4。")
        return [CandidateConfig(profile=variant, temperature=0.0, top_p=1.0, enable_thinking=False, max_tokens=192)]
    if variant == "task4_tagged_transcribe":
        if task_id != 4:
            raise ValueError("task4_tagged_transcribe 只用于 Task4。")
        return [CandidateConfig(profile=variant, temperature=0.0, top_p=1.0, enable_thinking=False, max_tokens=192)]
    if variant == "task4_tagged_noexample_transcribe":
        if task_id != 4:
            raise ValueError("task4_tagged_noexample_transcribe 只用于 Task4。")
        return [CandidateConfig(profile=variant, temperature=0.0, top_p=1.0, enable_thinking=False, max_tokens=192)]
    if variant == "task4_lenhint_transcribe":
        if task_id != 4:
            raise ValueError("task4_lenhint_transcribe 只用于 Task4。")
        return [CandidateConfig(profile=variant, temperature=0.0, top_p=1.0, enable_thinking=False, max_tokens=192)]
    if variant == "task4_lenhint_noexample_transcribe":
        if task_id != 4:
            raise ValueError("task4_lenhint_noexample_transcribe 只用于 Task4。")
        return [CandidateConfig(profile=variant, temperature=0.0, top_p=1.0, enable_thinking=False, max_tokens=192)]
    raise ValueError(f"不支持的 variant: {variant}")


def majority_prediction(task_id: int, candidates: Sequence[Candidate]) -> Tuple[Optional[str], Dict[str, Any]]:
    valid = [cand for cand in candidates if cand.prediction is not None]
    if not valid:
        return None, {"strategy": "majority", "reason": "no_valid_candidate"}

    counts = Counter(normalize_for_vote(task_id, cand.prediction) for cand in valid)
    best_norm, best_count = counts.most_common(1)[0]
    tied = [norm for norm, count in counts.items() if count == best_count]
    for cand in valid:
        if normalize_for_vote(task_id, cand.prediction) == best_norm:
            return cand.prediction, {
                "strategy": "majority",
                "best_count": best_count,
                "valid_candidates": len(valid),
                "unique_candidates": len(counts),
                "tied": len(tied) > 1,
            }
    return valid[0].prediction, {"strategy": "majority_fallback"}


def build_router_prompt(
    task_id: int,
    task_description: str,
    examples_text: str,
    sample_input: str,
    candidates: Sequence[Candidate],
    rule_variant: str = "anchored_direct",
) -> str:
    option_lines = []
    label_ord = ord("A")
    seen: set[str] = set()
    for cand in candidates:
        norm = normalize_for_vote(task_id, cand.prediction)
        if cand.prediction is None or norm in seen:
            continue
        seen.add(norm or "")
        label = chr(label_ord)
        label_ord += 1
        option_lines.append(f"Option {label}: {cand.prediction}")
    options = "\n".join(option_lines) if option_lines else "No valid options."
    rules = "\n".join(f"{idx + 1}. {rule}" for idx, rule in enumerate(task_rules(task_id, rule_variant)))
    return (
        "You are routing candidate answers for one OpenSeek long-context annotation sample.\n"
        "Use only the task description, official examples, and candidate answers. Do not write code or execute tools.\n\n"
        "[Task Description]\n"
        f"{task_description}\n\n"
        "[Official In-Context Examples]\n"
        f"{examples_text}\n\n"
        "[Sample To Annotate]\n"
        f"{sample_input}\n\n"
        "[Candidate Answers]\n"
        f"{options}\n\n"
        "[Output Rules]\n"
        f"{rules}\n"
        "If one candidate is correct, output the candidate's final answer exactly. If all candidates are wrong, output your corrected final answer only.\n\n"
        "Final answer:\n"
    )


def route_prediction(
    args: argparse.Namespace,
    task_id: int,
    task_description: str,
    examples_text: str,
    sample_input: str,
    candidates: Sequence[Candidate],
) -> Tuple[Optional[str], Dict[str, Any], str]:
    majority, meta = majority_prediction(task_id, candidates)
    if args.variant not in {
        "passk_router",
        "anchored_passk_router",
        "task4_chunk_passk_router",
        "task4_boundary_passk_router",
    }:
        return majority, meta, ""

    if not meta.get("tied") and meta.get("unique_candidates", 0) <= 1:
        return majority, meta, ""

    if args.variant == "task4_chunk_passk_router":
        router_rule_variant = "task4_chunk_direct"
    elif args.variant == "task4_boundary_passk_router":
        router_rule_variant = "task4_boundary_reasoning"
    else:
        router_rule_variant = "anchored_direct"
    router_prompt = build_router_prompt(
        task_id,
        task_description,
        examples_text,
        sample_input,
        candidates,
        rule_variant=router_rule_variant,
    )
    router_cfg = CandidateConfig(profile="router", temperature=0.0, top_p=1.0)
    raw = call_with_retries(args, router_prompt, router_cfg, max_tokens=args.router_max_new_tokens)
    routed = extract_answer(raw, task_id)
    if routed is not None:
        route_meta = dict(meta)
        route_meta.update({"strategy": "llm_router", "router_used": True})
        return routed, route_meta, raw
    meta = dict(meta)
    meta["router_used"] = True
    meta["router_failed"] = True
    return majority, meta, raw


def run_task(task_id: int, args: argparse.Namespace, tokenizer: Any) -> Path:
    data = load_task_data(Path(args.data_dir), task_id)
    task_description = str((data.get("Definition") or [""])[0])
    examples: List[Dict[str, Any]] = data.get("examples", [])
    test_samples: List[Dict[str, Any]] = data.get("test_samples", [])
    if args.max_test_samples > 0:
        test_samples = test_samples[: args.max_test_samples]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"openseek-{task_id}-v1.jsonl"
    detail_file = output_dir / f"openseek-{task_id}-v1-detail.jsonl"

    static_examples, static_count, static_tokens = build_examples_text(
        task_id=task_id,
        examples=examples,
        tokenizer=tokenizer,
        max_input_tokens=args.max_input_tokens,
        target_example_tokens=args.target_example_tokens,
        reserved_generation_tokens=args.reserved_generation_tokens,
        context_mode=args.context_mode,
    )
    manifest = {
        "task_id": task_id,
        "variant": args.variant,
        "context_mode": args.context_mode,
        "static_examples": static_count,
        "static_example_tokens_est": static_tokens,
        "max_test_samples": args.max_test_samples,
        "no_solver_generation": True,
        "no_code_execution_for_task_solution": True,
        "no_programmatic_task_answer_computation": True,
        "candidate_configs": [cfg.__dict__ for cfg in configs_for(task_id, args.variant)],
    }
    (output_dir / f"openseek-{task_id}-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(
        f"[Task {task_id}] samples={len(test_samples)} examples={static_count} "
        f"example_tokens~{static_tokens} output={output_file}"
    )

    def infer_one(index: int, sample: Dict[str, Any]) -> Tuple[int, Dict[str, Any], Dict[str, Any]]:
        sample_id = sample.get("id")
        sample_input = str(sample.get("input", "")).strip()
        examples_text, example_count, example_tokens = build_examples_text(
            task_id=task_id,
            examples=examples,
            tokenizer=tokenizer,
            max_input_tokens=args.max_input_tokens,
            target_example_tokens=args.target_example_tokens,
            reserved_generation_tokens=args.reserved_generation_tokens,
            context_mode=args.context_mode,
            query_text=sample_input,
        )

        candidates: List[Candidate] = []
        for cfg in configs_for(task_id, args.variant):
            if task_id == 4 and cfg.profile == "task4_raw_transcribe":
                raw_examples_text = "\n".join(format_raw_example(idx + 1, example) for idx, example in enumerate(examples[:16]))
                prompt = build_raw_transcribe_prompt(task_description, raw_examples_text, sample_input)
            elif task_id == 4 and cfg.profile == "task4_line_transcribe":
                line_examples_text = "\n".join(format_line_example(idx + 1, example) for idx, example in enumerate(examples[:16]))
                prompt = build_line_transcribe_prompt(task_description, line_examples_text, sample_input)
            elif task_id == 4 and cfg.profile in {"task4_tagged_transcribe", "task4_tagged_noexample_transcribe"}:
                tagged_examples_text = "\n".join(
                    format_tagged_example(idx + 1, example) for idx, example in enumerate(examples[:16])
                )
                prompt = build_tagged_transcribe_prompt(
                    task_description,
                    tagged_examples_text,
                    sample_input,
                    include_examples=cfg.profile == "task4_tagged_transcribe",
                )
            elif task_id == 4 and cfg.profile in {"task4_lenhint_transcribe", "task4_lenhint_noexample_transcribe"}:
                lenhint_examples_text = "\n".join(
                    format_lenhint_example(idx + 1, example) for idx, example in enumerate(examples[:16])
                )
                prompt = build_lenhint_transcribe_prompt(
                    task_description,
                    lenhint_examples_text,
                    sample_input,
                    include_examples=cfg.profile == "task4_lenhint_transcribe",
                )
            else:
                prompt = build_prompt(task_id, task_description, examples_text, sample_input, cfg.profile)
            raw = call_with_retries(args, prompt, cfg)
            candidates.append(Candidate(config=cfg, raw_text=raw, prediction=extract_prediction(raw, task_id, cfg.profile)))

        prediction, route_meta, router_raw = route_prediction(
            args=args,
            task_id=task_id,
            task_description=task_description,
            examples_text=examples_text,
            sample_input=sample_input,
            candidates=candidates,
        )
        row = {"test_sample_id": sample_id, "prediction": prediction}
        detail = {
            "test_sample_id": sample_id,
            "input": sample_input,
            "prediction": prediction,
            "route": route_meta,
            "example_count": example_count,
            "example_tokens_est": example_tokens,
            "candidates": [
                {
                    "profile": cand.config.profile,
                    "temperature": cand.config.temperature,
                    "top_p": cand.config.top_p,
                    "prediction": cand.prediction,
                    "raw_text": cand.raw_text,
                }
                for cand in candidates
            ],
            "router_raw_text": router_raw,
        }
        return index, row, detail

    rows: List[Optional[Dict[str, Any]]] = [None] * len(test_samples)
    details: List[Optional[Dict[str, Any]]] = [None] * len(test_samples)
    client_concurrency = max(1, min(args.client_concurrency, len(test_samples) or 1))
    with ThreadPoolExecutor(max_workers=client_concurrency) as executor:
        futures = {
            executor.submit(infer_one, idx, sample): idx
            for idx, sample in enumerate(test_samples)
        }
        with tqdm(total=len(futures), desc=f"Task {task_id}", unit="sample") as pbar:
            for future in as_completed(futures):
                idx, row, detail = future.result()
                rows[idx] = row
                details[idx] = detail
                pbar.update(1)
                done = pbar.n
                missing = sum(1 for item in rows[:done] if item is not None and item.get("prediction") is None)
                pbar.set_postfix_str(f"done={done} missing~{missing}")

    with output_file.open("w", encoding="utf-8") as f:
        for row in rows:
            if row is not None:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    if args.write_details:
        with detail_file.open("w", encoding="utf-8") as f:
            for detail in details:
                if detail is not None:
                    f.write(json.dumps(detail, ensure_ascii=False) + "\n")
    return output_file


def main() -> None:
    args = parse_args()
    tasks = parse_tasks(args.tasks)
    tokenizer = load_tokenizer(Path(args.tokenizer_path))
    print(
        f"[manyshot] tasks={tasks} variant={args.variant} context={args.context_mode} "
        f"api={args.api_base} model={args.model_name}"
    )
    for task_id in tasks:
        run_task(task_id, args, tokenizer)


if __name__ == "__main__":
    main()
