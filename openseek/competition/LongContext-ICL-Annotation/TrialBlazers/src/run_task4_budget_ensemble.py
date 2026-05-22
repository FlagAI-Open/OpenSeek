from __future__ import annotations

import argparse
import ast
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from tqdm import tqdm

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from run_manyshot import chat_completion, extract_answer, load_task_data  # noqa: E402


DEFAULT_SOURCES = [
    (
        "18k",
        "outputs/task134_longctx_manyshot_20260516/full_t4_18k_value_cover_anchored_reasoning/openseek-4-v1.jsonl",
    ),
    (
        "12k",
        "outputs/task134_longctx_manyshot_20260516/full_t4_12k_value_cover_anchored_reasoning/openseek-4-v1.jsonl",
    ),
    (
        "28k",
        "outputs/task134_longctx_manyshot_20260516/full_value_cover_anchored_reasoning/openseek-4-v1.jsonl",
    ),
]


@dataclass(frozen=True)
class Source:
    name: str
    path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task4 budget-candidate exact vote / optional Qwen router.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--api-base", default="http://127.0.0.1:2026")
    parser.add_argument("--model-name", default="Qwen3-4B")
    parser.add_argument("--source", action="append", default=[], help="候选源，格式 name=path；默认使用 18k/12k/28k。")
    parser.add_argument("--source-weight", action="append", default=[], help="候选源权重，格式 name=正整数；默认每路权重为 1。")
    parser.add_argument("--fallback-order", default="18k,12k,28k", help="无多数票时的固定回退优先级。")
    parser.add_argument("--router-on-no-majority", action="store_true", help="仅在候选无多数票时调用 Qwen router。")
    parser.add_argument("--router-on-disagreement", action="store_true", help="只要候选不完全一致就调用 Qwen router。")
    parser.add_argument("--router-on-tied-majority", action="store_true", help="仅在 exact majority 存在并列最高票时调用 Qwen router。")
    parser.add_argument(
        "--router-select-only",
        action="store_true",
        help="router 只返回候选编号，脚本按编号选原始候选文本；不允许 router 自由生成新标签。",
    )
    parser.add_argument(
        "--router-prompt-style",
        choices=("standard", "chunk_audit"),
        default="standard",
        help="router prompt 风格；chunk_audit 会把 Task4 输入拆成编号片段供模型审查候选。",
    )
    parser.add_argument(
        "--meta-vote-sources",
        default="",
        help="可选二级投票源列表，例如 18k,router,select；这些名称必须来自 --source。",
    )
    parser.add_argument(
        "--space-free-fallback-order",
        default="",
        help=(
            "可选空格保护候选优先级，例如 12k,raw,line,18k。"
            "当输入字符串片段本身均不含空格、当前入选候选却含空格时，"
            "只从已有候选中改选一个不含空格的候选；不生成或修复字符串。"
        ),
    )
    parser.add_argument(
        "--expected-length-fallback-order",
        default="",
        help=(
            "可选长度保护候选优先级，例如 18k,12k,28k,raw,line。"
            "当当前入选候选长度不等于输入片段长度之和时，只从已有候选中改选一个长度匹配的候选；"
            "不拼接或修复字符串。"
        ),
    )
    parser.add_argument(
        "--char-inventory-fallback-order",
        default="",
        help=(
            "可选字符库存保护候选优先级，例如 raw,line,12k。"
            "当当前入选候选的字符 multiset 与输入片段字符 multiset 不一致时，"
            "只从已有候选中改选一个字符 multiset 完全一致的候选；不拼接或修复字符串。"
        ),
    )
    parser.add_argument(
        "--adjacent-swap-fallback-order",
        default="",
        help=(
            "可选相邻交换保护候选优先级，例如 28k,dselect。"
            "当某个已有候选和当前入选候选仅相差一次相邻字符交换、且该候选至少有两路来源支持时，"
            "按优先级改选该候选；不生成或修复字符串。"
        ),
    )
    parser.add_argument("--router-enable-thinking", action="store_true")
    parser.add_argument("--router-max-new-tokens", type=int, default=192)
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--client-concurrency", type=int, default=4)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-backoff", type=float, default=1.5)
    return parser.parse_args()


def parse_sources(items: Sequence[str]) -> List[Source]:
    raw_items = list(items) if items else [f"{name}={path}" for name, path in DEFAULT_SOURCES]
    sources: List[Source] = []
    for item in raw_items:
        if "=" not in item:
            raise ValueError(f"--source 必须是 name=path 格式: {item}")
        name, path_text = item.split("=", 1)
        name = name.strip()
        path = Path(path_text.strip())
        if not name:
            raise ValueError(f"--source name 为空: {item}")
        if not path.exists():
            raise FileNotFoundError(path)
        sources.append(Source(name=name, path=path))
    return sources


def parse_source_weights(items: Sequence[str], sources: Sequence[Source]) -> Dict[str, int]:
    weights = {source.name: 1 for source in sources}
    known = set(weights)
    for item in items:
        if "=" not in item:
            raise ValueError(f"--source-weight 必须是 name=weight 格式: {item}")
        name, value_text = item.split("=", 1)
        name = name.strip()
        if name not in known:
            raise ValueError(f"--source-weight 包含未知候选源: {name}")
        try:
            value = int(value_text.strip())
        except ValueError as exc:
            raise ValueError(f"--source-weight 权重必须是正整数: {item}") from exc
        if value <= 0:
            raise ValueError(f"--source-weight 权重必须是正整数: {item}")
        weights[name] = value
    return weights


def load_predictions(path: Path) -> Tuple[List[str], Dict[str, Optional[str]]]:
    order: List[str] = []
    predictions: Dict[str, Optional[str]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            sample_id = str(row["test_sample_id"])
            order.append(sample_id)
            predictions[sample_id] = row.get("prediction")
    return order, predictions


def choose_by_vote(
    sample_id: str,
    sources: Sequence[Source],
    predictions_by_source: Dict[str, Dict[str, Optional[str]]],
    fallback_order: Sequence[str],
    source_weights: Dict[str, int],
) -> Tuple[Optional[str], Dict[str, Any]]:
    candidates = {source.name: predictions_by_source[source.name].get(sample_id) for source in sources}
    valid_values = [value for value in candidates.values() if value is not None]
    if not valid_values:
        return None, {"strategy": "no_valid_candidate", "candidates": candidates}

    counts: Counter[str] = Counter()
    raw_counts = Counter(valid_values)
    for source in sources:
        value = candidates[source.name]
        if value is not None:
            counts[value] += source_weights.get(source.name, 1)
    best_count = max(counts.values())
    majority_values = {value for value, count in counts.items() if count == best_count and raw_counts[value] >= 2}
    if majority_values:
        selected_source = None
        for name in fallback_order:
            value = candidates.get(name)
            if value in majority_values:
                selected_source = name
                break
        if selected_source is None:
            for source in sources:
                value = candidates[source.name]
                if value in majority_values:
                    selected_source = source.name
                    break
        if selected_source is not None:
            return candidates[selected_source], {
                "strategy": "exact_majority",
                "majority_count": best_count,
                "raw_majority_count": raw_counts[candidates[selected_source]],
                "majority_tied": len(majority_values) > 1,
                "selected_source": selected_source,
                "source_weights": source_weights,
                "candidates": candidates,
            }

    for name in fallback_order:
        value = candidates.get(name)
        if value is not None:
            return value, {
                "strategy": "fallback_priority",
                "selected_source": name,
                "candidates": candidates,
            }

    for source in sources:
        value = candidates[source.name]
        if value is not None:
            return value, {
                "strategy": "fallback_first_valid",
                "selected_source": source.name,
                "candidates": candidates,
            }
    return None, {"strategy": "fallback_failed", "candidates": candidates}


def input_chunks_contain_space(sample_input: str) -> Optional[bool]:
    try:
        parsed = ast.literal_eval(sample_input)
    except (SyntaxError, ValueError):
        return None
    if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
        return None
    return any(" " in item for item in parsed)


def input_chunks_expected_length(sample_input: str) -> Optional[int]:
    try:
        parsed = ast.literal_eval(sample_input)
    except (SyntaxError, ValueError):
        return None
    if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
        return None
    return sum(len(item) for item in parsed)


def input_chunks_char_inventory(sample_input: str) -> Optional[Counter[str]]:
    try:
        parsed = ast.literal_eval(sample_input)
    except (SyntaxError, ValueError):
        return None
    if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
        return None
    inventory: Counter[str] = Counter()
    for item in parsed:
        inventory.update(item)
    return inventory


def maybe_choose_space_free_candidate(
    prediction: Optional[str],
    meta: Dict[str, Any],
    sample_input: str,
    source_order: Sequence[str],
) -> Tuple[Optional[str], Dict[str, Any]]:
    if not source_order or prediction is None or " " not in prediction:
        return prediction, meta

    chunks_have_space = input_chunks_contain_space(sample_input)
    if chunks_have_space is not False:
        return prediction, meta

    candidates = meta.get("candidates", {})
    if not isinstance(candidates, dict):
        return prediction, meta

    for source_name in source_order:
        value = candidates.get(source_name)
        if value is None or value == prediction or " " in value:
            continue
        updated = dict(meta)
        updated.update(
            {
                "space_free_candidate_rule_used": True,
                "space_free_previous_strategy": meta.get("strategy"),
                "space_free_previous_prediction": prediction,
                "strategy": "space_free_existing_candidate",
                "selected_source": source_name,
                "input_chunks_contain_space": False,
            }
        )
        return value, updated

    return prediction, meta


def maybe_choose_char_inventory_candidate(
    prediction: Optional[str],
    meta: Dict[str, Any],
    sample_input: str,
    source_order: Sequence[str],
) -> Tuple[Optional[str], Dict[str, Any]]:
    if not source_order or prediction is None:
        return prediction, meta

    expected_inventory = input_chunks_char_inventory(sample_input)
    if expected_inventory is None or Counter(prediction) == expected_inventory:
        return prediction, meta

    candidates = meta.get("candidates", {})
    if not isinstance(candidates, dict):
        return prediction, meta

    for source_name in source_order:
        value = candidates.get(source_name)
        if value is None or value == prediction or Counter(value) != expected_inventory:
            continue
        updated = dict(meta)
        updated.update(
            {
                "char_inventory_candidate_rule_used": True,
                "char_inventory_previous_strategy": meta.get("strategy"),
                "char_inventory_previous_prediction": prediction,
                "strategy": "char_inventory_existing_candidate",
                "selected_source": source_name,
            }
        )
        return value, updated

    return prediction, meta


def maybe_choose_expected_length_candidate(
    prediction: Optional[str],
    meta: Dict[str, Any],
    sample_input: str,
    source_order: Sequence[str],
) -> Tuple[Optional[str], Dict[str, Any]]:
    if not source_order or prediction is None:
        return prediction, meta

    expected_length = input_chunks_expected_length(sample_input)
    if expected_length is None or len(prediction) == expected_length:
        return prediction, meta

    candidates = meta.get("candidates", {})
    if not isinstance(candidates, dict):
        return prediction, meta

    for source_name in source_order:
        value = candidates.get(source_name)
        if value is None or value == prediction or len(value) != expected_length:
            continue
        updated = dict(meta)
        updated.update(
            {
                "expected_length_candidate_rule_used": True,
                "expected_length_previous_strategy": meta.get("strategy"),
                "expected_length_previous_prediction": prediction,
                "strategy": "expected_length_existing_candidate",
                "selected_source": source_name,
                "expected_length": expected_length,
                "previous_length": len(prediction),
            }
        )
        return value, updated

    return prediction, meta


def is_single_adjacent_swap(left: str, right: str) -> bool:
    if len(left) != len(right):
        return False
    diff_positions = [idx for idx, (left_ch, right_ch) in enumerate(zip(left, right)) if left_ch != right_ch]
    if len(diff_positions) != 2:
        return False
    first, second = diff_positions
    return second == first + 1 and left[first] == right[second] and left[second] == right[first]


def maybe_choose_adjacent_swap_candidate(
    prediction: Optional[str],
    meta: Dict[str, Any],
    source_order: Sequence[str],
) -> Tuple[Optional[str], Dict[str, Any]]:
    if not source_order or prediction is None:
        return prediction, meta

    candidates = meta.get("candidates", {})
    if not isinstance(candidates, dict):
        return prediction, meta

    raw_counts = Counter(value for value in candidates.values() if value is not None)
    for source_name in source_order:
        value = candidates.get(source_name)
        if value is None or value == prediction or raw_counts[value] < 2:
            continue
        if not is_single_adjacent_swap(prediction, value):
            continue
        updated = dict(meta)
        updated.update(
            {
                "adjacent_swap_candidate_rule_used": True,
                "adjacent_swap_previous_strategy": meta.get("strategy"),
                "adjacent_swap_previous_prediction": prediction,
                "strategy": "adjacent_swap_existing_candidate",
                "selected_source": source_name,
                "adjacent_swap_raw_support": raw_counts[value],
            }
        )
        return value, updated

    return prediction, meta


def build_router_prompt(sample_input: str, candidates: Dict[str, Optional[str]]) -> str:
    option_lines = []
    for name, value in candidates.items():
        if value is None:
            continue
        option_lines.append(f"{name}: <label>{value}</label>")
    options = "\n".join(option_lines) if option_lines else "No valid candidates."
    return (
        "You are choosing the final label for one OpenSeek Task4 sample.\n"
        "Task4 rule: concatenate the quoted string list elements from left to right with no separator.\n"
        "Preserve every character exactly, including case, punctuation, digits, and spaces inside each quoted string.\n"
        "Use the candidate labels as model-generated alternatives. If one candidate is exactly correct, output it exactly.\n"
        "If all candidates are wrong, output the corrected final label. Do not explain.\n\n"
        "[Sample To Annotate]\n"
        f"{sample_input}\n\n"
        "[Candidate Labels]\n"
        f"{options}\n\n"
        "Return exactly one answer wrapped in <label> and </label>.\n"
        "Final answer:\n"
    )


def build_select_only_router_prompt(sample_input: str, candidates: Dict[str, Optional[str]]) -> Tuple[str, Dict[str, str]]:
    option_lines = []
    option_to_source: Dict[str, str] = {}
    for idx, (name, value) in enumerate(candidates.items(), start=1):
        if value is None:
            continue
        option = chr(ord("A") + len(option_to_source))
        option_to_source[option] = name
        option_lines.append(f"Option {option} ({name}, length={len(value)}): <label>{value}</label>")
    options = "\n".join(option_lines) if option_lines else "No valid candidates."
    prompt = (
        "You are verifying candidate labels for one OpenSeek Task4 sample.\n"
        "Task4 rule: concatenate the quoted string list elements from left to right with no separator.\n"
        "Preserve every character exactly, including case, punctuation, digits, and spaces inside each quoted string.\n"
        "Choose exactly one candidate option. Do not create, correct, rewrite, or normalize any label text.\n"
        "If unsure, choose the candidate that best preserves raw chunks and boundary spacing.\n\n"
        "[Sample To Annotate]\n"
        f"{sample_input}\n\n"
        "[Candidate Labels]\n"
        f"{options}\n\n"
        "Return only the option letter wrapped in <choice> and </choice>, for example <choice>A</choice>.\n"
        "Final choice:\n"
    )
    return prompt, option_to_source


def build_chunk_audit_select_only_router_prompt(
    sample_input: str,
    candidates: Dict[str, Optional[str]],
) -> Tuple[str, Dict[str, str]]:
    option_lines = []
    option_to_source: Dict[str, str] = {}
    for name, value in candidates.items():
        if value is None:
            continue
        option = chr(ord("A") + len(option_to_source))
        option_to_source[option] = name
        option_lines.append(f"Option {option} ({name}, length={len(value)}): <label>{value}</label>")
    options = "\n".join(option_lines) if option_lines else "No valid candidates."

    chunk_block = sample_input
    try:
        import ast

        parsed = ast.literal_eval(sample_input)
    except (SyntaxError, ValueError):
        parsed = None
    if isinstance(parsed, list) and all(isinstance(item, str) for item in parsed):
        chunk_lines = ["Input chunks:"]
        for idx, item in enumerate(parsed, start=1):
            chunk_lines.append(f"{idx}. {item}")
        chunk_block = "\n".join(chunk_lines)

    prompt = (
        "You are verifying candidate labels for one OpenSeek Task4 sample.\n"
        "Task4 rule: the correct label is chunk 1 followed by chunk 2 followed by chunk 3 and so on, with no separator.\n"
        "Choose exactly one candidate option. Do not create, correct, rewrite, or normalize any label text.\n"
        "Audit the candidates by checking that every numbered chunk appears exactly once in order, and that no boundary spaces or spelling/case changes were invented.\n"
        "If no candidate is perfect, choose the candidate with the fewest raw chunk-copy errors.\n\n"
        "[Sample To Annotate]\n"
        f"{chunk_block}\n\n"
        "[Candidate Labels]\n"
        f"{options}\n\n"
        "Return only the option letter wrapped in <choice> and </choice>, for example <choice>A</choice>.\n"
        "Final choice:\n"
    )
    return prompt, option_to_source


def extract_choice(text: str, valid_options: Sequence[str]) -> Optional[str]:
    valid = set(valid_options)
    cleaned = (text or "").strip()
    start = cleaned.find("<choice>")
    end = cleaned.find("</choice>", start + len("<choice>")) if start >= 0 else -1
    if start >= 0 and end >= 0:
        choice = cleaned[start + len("<choice>") : end].strip().upper()
        if choice in valid:
            return choice
    pattern = r"(?<![A-Z])(" + "|".join(re.escape(option) for option in valid) + r")(?![A-Z])"
    match = re.search(pattern, cleaned.upper())
    if match:
        return match.group(1)
    return None


def call_router(args: argparse.Namespace, prompt: str) -> str:
    for attempt in range(args.retries + 1):
        try:
            return chat_completion(
                api_base=args.api_base,
                model_name=args.model_name,
                prompt=prompt,
                max_tokens=args.router_max_new_tokens,
                temperature=0.0,
                top_p=1.0,
                timeout=args.timeout,
                enable_thinking=args.router_enable_thinking,
            )
        except Exception:
            if attempt < args.retries:
                time.sleep(args.retry_backoff * (attempt + 1))
    return ""


def main() -> None:
    args = parse_args()
    sources = parse_sources(args.source)
    source_weights = parse_source_weights(args.source_weight, sources)
    fallback_order = [part.strip() for part in args.fallback_order.split(",") if part.strip()]
    if not fallback_order:
        fallback_order = [source.name for source in sources]
    meta_vote_sources = [part.strip() for part in args.meta_vote_sources.split(",") if part.strip()]
    space_free_fallback_order = [part.strip() for part in args.space_free_fallback_order.split(",") if part.strip()]
    expected_length_fallback_order = [
        part.strip() for part in args.expected_length_fallback_order.split(",") if part.strip()
    ]
    char_inventory_fallback_order = [
        part.strip() for part in args.char_inventory_fallback_order.split(",") if part.strip()
    ]
    adjacent_swap_fallback_order = [
        part.strip() for part in args.adjacent_swap_fallback_order.split(",") if part.strip()
    ]

    sample_order, first_predictions = load_predictions(sources[0].path)
    predictions_by_source = {sources[0].name: first_predictions}
    for source in sources[1:]:
        order, predictions = load_predictions(source.path)
        if order != sample_order:
            raise ValueError(f"候选源样本顺序不一致: {source.name}")
        predictions_by_source[source.name] = predictions

    task_data = load_task_data(Path(args.data_dir), 4)
    sample_inputs = {str(row["id"]): str(row.get("input", "")).strip() for row in task_data.get("test_samples", [])}

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "openseek-4-v1.jsonl"
    detail_file = output_dir / "openseek-4-v1-detail.jsonl"
    manifest_file = output_dir / "openseek-4-manifest.json"

    manifest = {
        "task_id": 4,
        "method": "budget_exact_vote_with_optional_qwen_router",
        "sources": [{"name": source.name, "path": str(source.path)} for source in sources],
        "source_weights": source_weights,
        "fallback_order": fallback_order,
        "router_on_no_majority": bool(args.router_on_no_majority),
        "router_on_disagreement": bool(args.router_on_disagreement),
        "router_on_tied_majority": bool(args.router_on_tied_majority),
        "router_select_only": bool(args.router_select_only),
        "router_prompt_style": args.router_prompt_style,
        "meta_vote_sources": meta_vote_sources,
        "space_free_fallback_order": space_free_fallback_order,
        "space_free_rule_selects_existing_candidate_only": bool(space_free_fallback_order),
        "space_free_rule_requires_input_chunks_without_spaces": bool(space_free_fallback_order),
        "expected_length_fallback_order": expected_length_fallback_order,
        "expected_length_rule_selects_existing_candidate_only": bool(expected_length_fallback_order),
        "char_inventory_fallback_order": char_inventory_fallback_order,
        "char_inventory_rule_selects_existing_candidate_only": bool(char_inventory_fallback_order),
        "char_inventory_rule_does_not_check_order": bool(char_inventory_fallback_order),
        "adjacent_swap_fallback_order": adjacent_swap_fallback_order,
        "adjacent_swap_rule_selects_existing_candidate_only": bool(adjacent_swap_fallback_order),
        "adjacent_swap_rule_requires_two_source_support": bool(adjacent_swap_fallback_order),
        "router_enable_thinking": bool(args.router_enable_thinking),
        "router_max_new_tokens": args.router_max_new_tokens,
        "no_reference_labels_used": True,
        "no_programmatic_task_answer_computation": True,
        "no_input_based_string_repair": True,
        "router_cannot_create_new_label_when_select_only": bool(args.router_select_only),
    }
    manifest_file.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    if meta_vote_sources:
        known_sources = {source.name for source in sources}
        unknown = [name for name in meta_vote_sources if name not in known_sources]
        if unknown:
            raise ValueError(f"--meta-vote-sources 包含未知候选源: {unknown}")
    if space_free_fallback_order:
        known_sources = {source.name for source in sources}
        unknown = [name for name in space_free_fallback_order if name not in known_sources]
        if unknown:
            raise ValueError(f"--space-free-fallback-order 包含未知候选源: {unknown}")
    if expected_length_fallback_order:
        known_sources = {source.name for source in sources}
        unknown = [name for name in expected_length_fallback_order if name not in known_sources]
        if unknown:
            raise ValueError(f"--expected-length-fallback-order 包含未知候选源: {unknown}")
    if char_inventory_fallback_order:
        known_sources = {source.name for source in sources}
        unknown = [name for name in char_inventory_fallback_order if name not in known_sources]
        if unknown:
            raise ValueError(f"--char-inventory-fallback-order 包含未知候选源: {unknown}")
    if adjacent_swap_fallback_order:
        known_sources = {source.name for source in sources}
        unknown = [name for name in adjacent_swap_fallback_order if name not in known_sources]
        if unknown:
            raise ValueError(f"--adjacent-swap-fallback-order 包含未知候选源: {unknown}")

    def process(sample_id: str) -> Tuple[int, Dict[str, Any], Dict[str, Any]]:
        index = sample_order.index(sample_id)
        active_sources = [source for source in sources if not meta_vote_sources or source.name in meta_vote_sources]
        active_weights = {source.name: source_weights.get(source.name, 1) for source in active_sources}
        prediction, meta = choose_by_vote(sample_id, active_sources, predictions_by_source, fallback_order, active_weights)
        prediction, meta = maybe_choose_space_free_candidate(
            prediction,
            meta,
            sample_inputs.get(sample_id, ""),
            space_free_fallback_order,
        )
        prediction, meta = maybe_choose_expected_length_candidate(
            prediction,
            meta,
            sample_inputs.get(sample_id, ""),
            expected_length_fallback_order,
        )
        prediction, meta = maybe_choose_char_inventory_candidate(
            prediction,
            meta,
            sample_inputs.get(sample_id, ""),
            char_inventory_fallback_order,
        )
        prediction, meta = maybe_choose_adjacent_swap_candidate(
            prediction,
            meta,
            adjacent_swap_fallback_order,
        )
        router_raw = ""
        router_should_run = (
            (args.router_on_no_majority and meta.get("strategy") == "fallback_priority")
            or (args.router_on_tied_majority and bool(meta.get("majority_tied")))
            or (args.router_on_disagreement and len({value for value in meta.get("candidates", {}).values() if value is not None}) > 1)
        )
        if router_should_run:
            if args.router_select_only:
                if args.router_prompt_style == "chunk_audit":
                    prompt, option_to_source = build_chunk_audit_select_only_router_prompt(
                        sample_inputs.get(sample_id, ""),
                        meta["candidates"],
                    )
                else:
                    prompt, option_to_source = build_select_only_router_prompt(sample_inputs.get(sample_id, ""), meta["candidates"])
                router_raw = call_router(args, prompt)
                choice = extract_choice(router_raw, option_to_source.keys())
                if choice is not None:
                    selected_source = option_to_source[choice]
                    prediction = meta["candidates"].get(selected_source)
                    meta = dict(meta)
                    meta.update(
                        {
                            "strategy": "qwen_select_only_router_disagreement"
                            if args.router_on_disagreement
                            else (
                                "qwen_select_only_router_tied_majority"
                                if args.router_on_tied_majority
                                else "qwen_select_only_router_no_majority"
                            ),
                            "router_used": True,
                            "selected_source": selected_source,
                            "selected_option": choice,
                        }
                    )
                else:
                    meta = dict(meta)
                    meta.update({"router_used": True, "router_failed": True})
            else:
                prompt = build_router_prompt(sample_inputs.get(sample_id, ""), meta["candidates"])
                router_raw = call_router(args, prompt)
                routed = extract_answer(router_raw, 4)
                if routed is not None:
                    prediction = routed
                    meta = dict(meta)
                    meta.update(
                        {
                            "strategy": "qwen_router_disagreement"
                            if args.router_on_disagreement
                            else (
                                "qwen_router_tied_majority"
                                if args.router_on_tied_majority
                                else "qwen_router_no_majority"
                            ),
                            "router_used": True,
                        }
                    )
                else:
                    meta = dict(meta)
                    meta.update({"router_used": True, "router_failed": True})
        row = {"test_sample_id": sample_id, "prediction": prediction}
        detail = {
            "test_sample_id": sample_id,
            "input": sample_inputs.get(sample_id, ""),
            "prediction": prediction,
            "selection": meta,
            "router_raw_text": router_raw,
        }
        return index, row, detail

    rows: List[Optional[Dict[str, Any]]] = [None] * len(sample_order)
    details: List[Optional[Dict[str, Any]]] = [None] * len(sample_order)
    concurrency = max(1, min(args.client_concurrency, len(sample_order)))
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {executor.submit(process, sample_id): sample_id for sample_id in sample_order}
        with tqdm(total=len(futures), desc="Task4 budget ensemble", unit="sample") as pbar:
            for future in as_completed(futures):
                index, row, detail = future.result()
                rows[index] = row
                details[index] = detail
                pbar.update(1)

    with output_file.open("w", encoding="utf-8") as f:
        for row in rows:
            if row is not None:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with detail_file.open("w", encoding="utf-8") as f:
        for detail in details:
            if detail is not None:
                f.write(json.dumps(detail, ensure_ascii=False) + "\n")

    print(f"输出目录: {output_dir}")


if __name__ == "__main__":
    main()
