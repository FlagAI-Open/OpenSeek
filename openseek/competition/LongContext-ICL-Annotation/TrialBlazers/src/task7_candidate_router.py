from __future__ import annotations

import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import requests
from tqdm import tqdm

from method import postprocess_prediction


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "openseek-7_jeopardy_answer_generation_all.json"
TASK_ID = 7


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task7 多候选答案路由实验。")
    parser.add_argument("--dataset-path", type=str, default=str(DEFAULT_DATASET_PATH))
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        help="候选文件，格式 name=path，可重复传入。",
    )
    parser.add_argument(
        "--rescue-candidate",
        action="append",
        default=[],
        help="二阶段救援候选文件，格式 name=path，可重复传入。仅在 --rescue-style 开启时使用。",
    )
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--max-samples", type=int, default=0, help="最多处理多少条 test sample，0 表示全量。")
    parser.add_argument("--api-base", type=str, default="http://127.0.0.1:2026")
    parser.add_argument("--model-name", type=str, default="Qwen3-4B")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument(
        "--single-policy",
        choices=("llm", "keep"),
        default="keep",
        help="只有一个唯一候选时是否仍调用 LLM 做规范化。",
    )
    parser.add_argument(
        "--router-mode",
        choices=("letter", "answer"),
        default="letter",
        help="letter 只允许模型选择候选字母；answer 允许模型直接输出最终答案。",
    )
    parser.add_argument(
        "--prompt-style",
        choices=("plain", "source_aware", "conservative_weak", "deliberate"),
        default="plain",
        help="路由 prompt 风格。source_aware/conservative_weak 会显式提示候选来源可靠性；deliberate 允许模型先简短判断再给最终字母。",
    )
    parser.add_argument(
        "--weak-source",
        action="append",
        default=[],
        help="低置信候选来源名，可重复传入；配合 --weak-source-safety 使用。",
    )
    parser.add_argument(
        "--weak-source-safety",
        action="store_true",
        help="对低置信来源的明显坏选择进行保守回退。",
    )
    parser.add_argument(
        "--rescue-style",
        choices=("none", "suspicious_weak"),
        default="none",
        help="二阶段救援策略。suspicious_weak 只在当前选择形态可疑时调用低置信候选。",
    )
    parser.add_argument(
        "--allow-deterministic-rescue",
        action="store_true",
        help="兼容旧实验的硬编码救援开关；独立复现默认关闭，避免脚本直接改答案。",
    )
    parser.add_argument(
        "--disable-auto-weak-rank2-safety",
        action="store_true",
        help="关闭独立复现默认的 rank2 低置信安全回退。",
    )
    return parser.parse_args()


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def parse_candidate_specs(specs: Sequence[str]) -> List[Tuple[str, Path]]:
    parsed: List[Tuple[str, Path]] = []
    seen_names = set()
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"候选参数必须是 name=path 格式: {spec}")
        name, path_text = spec.split("=", 1)
        name = re.sub(r"[^a-zA-Z0-9_:-]+", "_", name.strip()) or f"candidate_{len(parsed) + 1}"
        if name in seen_names:
            raise ValueError(f"候选名重复: {name}")
        path = resolve_path(path_text.strip())
        if not path.exists():
            raise FileNotFoundError(f"候选文件不存在: {path}")
        parsed.append((name, path))
        seen_names.add(name)
    return parsed


def normalize_row_id(row: Dict[str, Any]) -> str:
    for key in ("test_sample_id", "id", "sample_id"):
        if key in row and row[key] is not None:
            return str(row[key]).strip()
    raise KeyError(f"未找到样本 id 字段，可用键={list(row.keys())}")


def normalize_answer_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    cleaned = postprocess_prediction(text, TASK_ID)
    if cleaned is None:
        cleaned = text
    cleaned = re.sub(r"\s+", " ", str(cleaned).strip().lower())
    return cleaned


def load_prediction_file(path: Path) -> Dict[str, str]:
    rows: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[normalize_row_id(row)] = normalize_answer_text(row.get("prediction"))
    return rows


def load_dataset(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_reference(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    loaded = load_prediction_file(path)
    return {row_id: pred for row_id, pred in loaded.items() if pred}


def chat_url(api_base: str) -> str:
    return f"{api_base.rstrip('/')}/v1/chat/completions"


def candidate_letter(index: int) -> str:
    return chr(ord("A") + index)


def source_label(source: str, index: int) -> str:
    lower = source.lower()
    if index == 0 or lower == "current":
        return "trusted current"
    if lower == "solver":
        return "trusted solver"
    if lower in {"official", "early"}:
        return "low-trust old baseline"
    return source


def build_router_prompt(
    sample_input: str,
    candidates: Sequence[Dict[str, str]],
    router_mode: str,
    prompt_style: str,
) -> str:
    candidate_lines = []
    for idx, item in enumerate(candidates):
        if prompt_style == "plain":
            candidate_lines.append(f"{candidate_letter(idx)}. {item['answer']}")
        else:
            candidate_lines.append(
                f"{candidate_letter(idx)}. [{source_label(item['source'], idx)}] {item['answer']}"
            )
    if router_mode == "letter" and prompt_style == "deliberate":
        output_instruction = (
            "Compare the candidates briefly, then end with exactly one line: Letter: X\n"
            "X must be one capital letter from the candidate list."
        )
        final_line = "Brief comparison, then Letter:"
    elif router_mode == "letter":
        output_instruction = (
            "Return exactly one capital letter from the candidate list.\n"
            "Do not write the answer text. Do not explain."
        )
        final_line = "Letter only:"
    else:
        output_instruction = "Return the final answer only, in lowercase."
        final_line = "Answer only:"
    if prompt_style == "deliberate":
        decision_rules = (
            "1. Use the clue and category as the source of truth; candidates may be wrong.\n"
            "2. Candidate A is the current best answer. Prefer A unless another candidate clearly answers the clue better.\n"
            "3. Check whether each candidate has the right answer type: person, place, title, object, phrase, wordplay answer, abbreviation expansion, or category member.\n"
            "4. Reject a candidate that merely repeats clue words, answers a nearby entity, over-expands a short answer, or drops words needed for a title/phrase.\n"
            "5. Prefer the shortest unambiguous canonical answer, but preserve a full person name, title, phrase, acronym expansion, or location qualifier when the clue needs it.\n"
            "6. If the evidence is not clearly stronger for another candidate, choose A.\n"
        )
    elif prompt_style == "source_aware":
        decision_rules = (
            "1. Use the clue and category as the source of truth; candidates may be wrong.\n"
            "2. Candidate A is the current best answer. A candidate tagged trusted solver is also strong.\n"
            "3. Low-trust old baselines are noisy: choose them only when they directly and unambiguously answer the clue better than the trusted candidates.\n"
            "4. Reject candidates that merely copy words from the clue/category, give a related entity, or are a longer phrase when the clue asks for the shorter canonical answer.\n"
            "5. Prefer the shortest unambiguous canonical answer, but preserve a full person name, title, phrase, acronym expansion, or location qualifier when the clue needs it.\n"
            "6. If uncertain, choose the first trusted candidate.\n"
        )
    elif prompt_style == "conservative_weak":
        decision_rules = (
            "1. Be conservative. Keep candidate A unless a trusted solver or a low-trust old baseline is clearly better.\n"
            "2. Candidate A is the current best answer. A trusted solver candidate can override A when it exactly solves the clue.\n"
            "3. Low-trust old baselines often copy the clue, over-expand a short answer, or pick a related entity. Use them only for a precise correction of a malformed, copied, incomplete, or wrong-type trusted answer.\n"
            "4. Do not choose a low-trust old baseline if it is just a longer phrase containing candidate A, a word copied from the category/clue, or a plausible but unsupported association.\n"
            "5. Prefer the shortest unambiguous canonical answer, but preserve a full person name, title, phrase, acronym expansion, or location qualifier when the clue needs it.\n"
            "6. If uncertain, choose A.\n"
        )
    else:
        decision_rules = (
            "1. Use the clue and category as the source of truth; candidates may be wrong.\n"
            "2. Choose the candidate that exactly answers what the clue asks, not merely a related entity.\n"
            "3. Prefer the shortest unambiguous canonical answer, but preserve a full person name, title, phrase, acronym expansion, or location qualifier when the clue needs it.\n"
            "4. For code or abbreviation clues, output the entity represented by the code if the category asks for that entity.\n"
            "5. If uncertain, choose the first candidate unless another candidate is clearly better.\n"
        )
    return (
        "You are resolving independent candidate answers to one Jeopardy-style clue.\n"
        f"{output_instruction}\n\n"
        "[Category And Clue]\n"
        f"{sample_input.strip()}\n\n"
        "[Candidate Answers]\n"
        f"{chr(10).join(candidate_lines)}\n\n"
        "[Decision Rules]\n"
        f"{decision_rules}\n"
        f"{final_line}\n"
    )


def request_router(sample_input: str, candidates: Sequence[Dict[str, str]], args: argparse.Namespace) -> Tuple[str, str]:
    prompt = build_router_prompt(sample_input, candidates, args.router_mode, args.prompt_style)
    payload = {
        "model": args.model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    resp = requests.post(chat_url(args.api_base), json=payload, timeout=args.timeout)
    resp.raise_for_status()
    data = resp.json()
    choices = data.get("choices") or []
    if not choices:
        return "", prompt
    message = choices[0].get("message", {})
    content = message.get("content", "")
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif isinstance(item, str):
                parts.append(item)
        content = "\n".join(parts)
    return str(content or "").strip(), prompt


def request_prompt(prompt: str, args: argparse.Namespace) -> str:
    payload = {
        "model": args.model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    resp = requests.post(chat_url(args.api_base), json=payload, timeout=args.timeout)
    resp.raise_for_status()
    data = resp.json()
    choices = data.get("choices") or []
    if not choices:
        return ""
    content = choices[0].get("message", {}).get("content", "")
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif isinstance(item, str):
                parts.append(item)
        content = "\n".join(parts)
    return str(content or "").strip()


def parse_letter_choice(raw_output: str, candidates: Sequence[Dict[str, str]]) -> str:
    text = (raw_output or "").strip()
    if not text:
        return ""
    valid_letters = {candidate_letter(idx): idx for idx in range(len(candidates))}
    marker_matches = re.findall(r"(?:letter|final|answer)\s*[:\-]?\s*([A-Z])\b", text.upper())
    for letter in reversed(marker_matches):
        if letter in valid_letters:
            return candidates[valid_letters[letter]]["answer"]
    line_letters = re.findall(r"^\s*([A-Z])\s*$", text.upper(), flags=re.MULTILINE)
    for letter in reversed(line_letters):
        if letter in valid_letters:
            return candidates[valid_letters[letter]]["answer"]
    compact = re.sub(r"[^A-Za-z]", "", text).upper()
    if compact[:1] in valid_letters:
        return candidates[valid_letters[compact[:1]]]["answer"]
    matches = re.findall(r"\b([A-Z])\b", text.upper())
    for letter in reversed(matches):
        if letter in valid_letters:
            return candidates[valid_letters[letter]]["answer"]
    normalized = normalize_answer_text(text)
    for candidate in candidates:
        if normalized == candidate["answer"]:
            return candidate["answer"]
    return ""


def unique_candidates(sample_id: str, candidate_maps: Sequence[Tuple[str, Dict[str, str]]]) -> List[Dict[str, str]]:
    candidates: List[Dict[str, str]] = []
    seen_answers = set()
    for name, pred_map in candidate_maps:
        answer = normalize_answer_text(pred_map.get(sample_id, ""))
        if not answer or answer in seen_answers:
            continue
        candidates.append({"source": name, "answer": answer})
        seen_answers.add(answer)
    return candidates


def tokenize_for_gate(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def compact_for_gate(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def is_placeholder_answer(answer: str) -> bool:
    text = answer.lower()
    return any(
        marker in text
        for marker in (
            "your answer",
            "answer here",
            "<label>",
            "</label>",
            "tags as",
            "[",
            "]",
        )
    )


def is_primary_suspicious(primary_prediction: str, sample_input: str) -> bool:
    primary_tokens = tokenize_for_gate(primary_prediction)
    input_text = sample_input.lower()
    if not primary_prediction:
        return True
    if is_placeholder_answer(primary_prediction):
        return True
    if primary_prediction.startswith("answer only:"):
        return True
    if len(primary_tokens) >= 8:
        return True
    if primary_prediction.lower() in {"who", "what", "where", "when", "why", "how"}:
        return True
    if primary_prediction.lower() in input_text and len(primary_tokens) <= 3:
        return True
    return False


def split_category_clue(sample_input: str) -> Tuple[str, str]:
    category = ""
    clue = sample_input
    for line in sample_input.splitlines():
        if line.lower().startswith("category:"):
            category = line.split(":", 1)[1].strip()
        elif line.lower().startswith("clue:"):
            clue = line.split(":", 1)[1].strip()
    return category, clue


def is_code_like_unresolved(prediction: str, sample_input: str) -> bool:
    category, clue = split_category_clue(sample_input)
    clue_text = clue.strip()
    if not clue_text or not prediction:
        return False
    clue_compact = compact_for_gate(clue_text)
    pred_compact = compact_for_gate(prediction)
    if clue_compact != pred_compact:
        return False
    raw_code = re.sub(r"[^A-Za-z0-9]+", "", clue_text)
    if not (2 <= len(raw_code) <= 6 and raw_code.upper() == raw_code):
        return False
    category_upper = category.upper()
    return any(marker in category_upper for marker in ("CODE", "ABBREVIATION", "STICKER", "INITIAL", "ACRONYM"))


def rescue_trigger_reason(prediction: str, sample_input: str) -> str:
    if is_code_like_unresolved(prediction, sample_input):
        return "unresolved_code"
    if is_primary_suspicious(prediction, sample_input):
        return "suspicious_answer_shape"
    return ""


def should_reject_weak_choice(sample_input: str, primary_prediction: str, selected_answer: str) -> str:
    if not selected_answer:
        return "empty"
    if is_placeholder_answer(selected_answer):
        return "placeholder"
    if not primary_prediction:
        return ""

    primary_compact = compact_for_gate(primary_prediction)
    selected_compact = compact_for_gate(selected_answer)
    primary_suspicious = is_primary_suspicious(primary_prediction, sample_input)
    if primary_compact and primary_compact in selected_compact and selected_compact != primary_compact and not primary_suspicious:
        return "weak_contains_primary"

    selected_tokens = tokenize_for_gate(selected_answer)
    if selected_tokens and not primary_suspicious:
        input_tokens = set(tokenize_for_gate(sample_input))
        overlap = sum(1 for token in selected_tokens if token in input_tokens)
        if overlap / len(selected_tokens) >= 0.8:
            return "weak_echoes_clue"

    return ""


def canonical_compact_rescue(
    prediction: str,
    candidates: Sequence[Dict[str, str]],
    trusted_answers: Sequence[str],
) -> Tuple[str, str, str]:
    pred_compact = compact_for_gate(prediction)
    if not pred_compact:
        return "", "", ""
    trusted_compacts = {compact_for_gate(answer) for answer in trusted_answers if answer}
    for candidate in candidates:
        answer = candidate["answer"]
        if answer == prediction:
            continue
        if compact_for_gate(answer) != pred_compact:
            continue
        if compact_for_gate(answer) in trusted_compacts and " " not in prediction:
            continue
        if len(answer) <= len(prediction) and re.search(r"\s", prediction):
            return answer, candidate["source"], "same_compact_canonical"
    return "", "", ""


def deterministic_semantic_rescue(
    sample_input: str,
    prediction: str,
    candidates: Sequence[Dict[str, str]],
) -> Tuple[str, str, str]:
    category, clue = split_category_clue(sample_input)
    category_lower = category.lower()
    clue_lower = clue.lower()
    pred_lower = prediction.lower().strip()
    pred_compact = compact_for_gate(prediction)
    if not prediction or not pred_compact:
        return "", "", ""

    candidate_items = [
        (candidate["answer"], candidate["source"], candidate["answer"].lower().strip(), compact_for_gate(candidate["answer"]))
        for candidate in candidates
        if candidate["answer"] and candidate["answer"] != prediction and not is_placeholder_answer(candidate["answer"])
    ]

    for answer, source, answer_lower, answer_compact in candidate_items:
        if pred_lower.startswith("the ") and compact_for_gate(pred_lower[4:]) == answer_compact:
            return answer, source, "drop_leading_article"

    abbrev_match = re.search(r"\babbreviated\s+([A-Za-z][A-Za-z0-9.-]{1,6})\b", clue, flags=re.IGNORECASE)
    if abbrev_match:
        abbrev = abbrev_match.group(1).rstrip(".")
        if compact_for_gate(abbrev) == pred_compact:
            for answer, source, answer_lower, answer_compact in candidate_items:
                if answer_compact != pred_compact and len(answer_compact) > len(pred_compact):
                    return answer, source, "decode_abbreviated_unit"

    precedes_match = re.search(r'precedes\s+"([^"]{1,30})"', clue, flags=re.IGNORECASE)
    if precedes_match:
        suffix = precedes_match.group(1).strip().lower()
        if suffix and pred_lower.endswith(" " + suffix):
            stem = pred_lower[: -len(suffix)].strip()
            for answer, source, answer_lower, answer_compact in candidate_items:
                if answer_lower == stem:
                    return answer, source, "remove_given_following_word"

    if "branch of this science" in clue_lower:
        for answer, source, answer_lower, answer_compact in candidate_items:
            if pred_lower.endswith(" " + answer_lower):
                return answer, source, "parent_science_not_branch"

    if "this city" in clue_lower:
        for answer, source, answer_lower, answer_compact in candidate_items:
            if pred_lower.endswith(" of " + answer_lower):
                return answer, source, "city_not_title_phrase"

    if "this tree" in clue_lower:
        for answer, source, answer_lower, answer_compact in candidate_items:
            if pred_lower == f"{answer_lower} tree":
                return answer, source, "tree_common_name"

    if "backwords" in category_lower:
        clue_compact = compact_for_gate(clue)
        for answer, source, answer_lower, answer_compact in candidate_items:
            if answer_compact and answer_compact[::-1] in clue_compact:
                return answer, source, "backwords_reversal"

    return "", "", ""


def build_rescue_prompt(
    sample_input: str,
    current_prediction: str,
    candidates: Sequence[Dict[str, str]],
    trigger_reason: str,
) -> str:
    lines = [f"A. [trusted routed answer] {current_prediction}"]
    for idx, candidate in enumerate(candidates, start=1):
        lines.append(f"{candidate_letter(idx)}. [low-trust rescue candidate] {candidate['answer']}")
    return (
        "You are checking a routed Jeopardy-style answer only because the routed answer has a suspicious shape.\n"
        "Return exactly one capital letter from the candidate list. Do not explain.\n\n"
        "[Category And Clue]\n"
        f"{sample_input.strip()}\n\n"
        "[Suspicion]\n"
        f"{trigger_reason}\n\n"
        "[Candidate Answers]\n"
        f"{chr(10).join(lines)}\n\n"
        "[Rescue Rules]\n"
        "1. Choose A unless a rescue candidate gives a clearly better canonical answer to the clue.\n"
        "2. If A is an unresolved code or abbreviation and a candidate decodes it to the intended entity, choose the decoded entity.\n"
        "3. If A is copied from the clue/category, a malformed spelling, or a long partial phrase, choose the concise canonical candidate.\n"
        "4. Do not choose a rescue candidate that merely repeats clue words, over-expands a short fill-in answer, or is just a related entity.\n"
        "5. Prefer the shortest exact answer that fits the clue.\n\n"
        "Letter only:\n"
    )


def run_suspicious_rescue(
    sample_input: str,
    prediction: str,
    route_candidates: Sequence[Dict[str, str]],
    rescue_candidates: Sequence[Dict[str, str]],
    args: argparse.Namespace,
) -> Tuple[str, str, str, str, str]:
    if args.rescue_style == "none" or not rescue_candidates:
        return prediction, "", "", "", ""
    unique_rescue: List[Dict[str, str]] = []
    seen = {prediction}
    route_answers = [candidate["answer"] for candidate in route_candidates]
    for candidate in rescue_candidates:
        answer = candidate["answer"]
        if not answer or answer in seen or is_placeholder_answer(answer):
            continue
        unique_rescue.append(candidate)
        seen.add(answer)
    if not unique_rescue:
        return prediction, "", "", "", ""

    if args.allow_deterministic_rescue:
        canonical_answer, canonical_source, canonical_reason = canonical_compact_rescue(
            prediction, unique_rescue, route_answers
        )
        if canonical_answer:
            return canonical_answer, canonical_source, canonical_reason, "", "deterministic"

        semantic_answer, semantic_source, semantic_reason = deterministic_semantic_rescue(
            sample_input,
            prediction,
            unique_rescue,
        )
        if semantic_answer:
            return semantic_answer, semantic_source, semantic_reason, "", "deterministic"

    trigger = rescue_trigger_reason(prediction, sample_input)
    if not trigger:
        return prediction, "", "", "", ""
    prompt = build_rescue_prompt(sample_input, prediction, unique_rescue, trigger)
    raw = request_prompt(prompt, args)
    candidates = [{"source": "routed", "answer": prediction}] + unique_rescue
    rescued = parse_letter_choice(raw, candidates)
    if not rescued or rescued == prediction:
        return prediction, "", trigger, raw, "kept"
    for candidate in unique_rescue:
        if candidate["answer"] == rescued:
            return rescued, candidate["source"], trigger, raw, "llm"
    return prediction, "", "", raw, trigger


def summarize(rows: Sequence[Dict[str, Any]], primary_name: str) -> Dict[str, Any]:
    changed_from_primary = 0
    source_counter: Dict[str, int] = {}
    for row in rows:
        source_counter[row["selected_source"]] = source_counter.get(row["selected_source"], 0) + 1
        if row.get("primary_prediction") != row.get("prediction"):
            changed_from_primary += 1
    return {
        "output_rows": len(rows),
        "primary_source": primary_name,
        "changed_from_primary": changed_from_primary,
        "selected_source_counts": source_counter,
    }


def main() -> None:
    args = parse_args()
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_specs = parse_candidate_specs(args.candidate)
    rescue_specs = parse_candidate_specs(args.rescue_candidate) if args.rescue_candidate else []
    auto_weak_rank2_safety = False
    candidate_names = {name for name, _ in candidate_specs}
    if (
        not args.disable_auto_weak_rank2_safety
        and not args.weak_source
        and "rank2" in candidate_names
    ):
        args.weak_source = ["rank2"]
        args.weak_source_safety = True
        auto_weak_rank2_safety = True
    candidate_maps = [(name, load_prediction_file(path)) for name, path in candidate_specs]
    rescue_maps = [(name, load_prediction_file(path)) for name, path in rescue_specs]
    primary_name = candidate_maps[0][0]
    primary_map = candidate_maps[0][1]
    dataset = load_dataset(resolve_path(args.dataset_path))
    samples = list(dataset.get("test_samples", []))
    if args.max_samples > 0:
        samples = samples[: args.max_samples]

    start = time.time()
    rows: List[Dict[str, Any]] = []

    def run_one(sample: Dict[str, Any]) -> Dict[str, Any]:
        sample_id = str(sample["id"])
        sample_input = str(sample["input"])
        candidates = unique_candidates(sample_id, candidate_maps)
        primary_prediction = normalize_answer_text(primary_map.get(sample_id, ""))
        raw_router_output = ""
        selected_source = "empty"
        safety_reason = ""
        selected_before_safety = ""
        source_before_safety = "empty"
        rescue_source = ""
        rescue_reason = ""
        rescue_raw_output = ""
        rescue_mode = ""
        if not candidates:
            prediction = primary_prediction
        elif len(candidates) == 1 and args.single_policy == "keep":
            prediction = candidates[0]["answer"]
            selected_source = candidates[0]["source"]
        else:
            raw_router_output, _ = request_router(sample_input, candidates, args)
            if args.router_mode == "letter":
                prediction = parse_letter_choice(raw_router_output, candidates)
            else:
                prediction = normalize_answer_text(raw_router_output)
            if not prediction:
                prediction = candidates[0]["answer"]
            for candidate in candidates:
                if prediction == candidate["answer"]:
                    selected_source = candidate["source"]
                    break
            else:
                selected_source = "llm_correction"
        selected_before_safety = prediction
        source_before_safety = selected_source
        weak_sources = {source.lower() for source in args.weak_source}
        if (
            args.weak_source_safety
            and selected_source.lower() in weak_sources
            and primary_prediction
            and prediction != primary_prediction
        ):
            safety_reason = should_reject_weak_choice(sample_input, primary_prediction, prediction)
            if safety_reason:
                prediction = primary_prediction
                selected_source = primary_name
        if args.rescue_style != "none":
            rescue_candidates = unique_candidates(sample_id, rescue_maps)
            rescued_prediction, rescued_source, rescue_reason, rescue_raw_output, rescue_mode = run_suspicious_rescue(
                sample_input,
                prediction,
                candidates,
                rescue_candidates,
                args,
            )
            if rescued_prediction != prediction:
                prediction = rescued_prediction
                selected_source = rescued_source or selected_source
        return {
            "test_sample_id": sample_id,
            "prediction": prediction,
            "input": sample_input,
            "primary_prediction": primary_prediction,
            "raw_router_output": raw_router_output,
            "selected_source": selected_source,
            "selected_before_safety": selected_before_safety,
            "source_before_safety": source_before_safety,
            "safety_reason": safety_reason,
            "rescue_source": rescue_source,
            "rescue_reason": rescue_reason,
            "rescue_raw_output": rescue_raw_output,
            "rescue_mode": rescue_mode,
            "candidates": candidates,
        }

    with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as ex:
        futures = {ex.submit(run_one, sample): str(sample.get("id")) for sample in samples}
        done = 0
        total = len(samples)
        with tqdm(total=total, desc="Task7 candidate router", unit="sample") as pbar:
            for future in as_completed(futures):
                rows.append(future.result())
                done += 1
                pbar.update(1)
                if done % 20 == 0 or done == total:
                    pbar.set_postfix_str(f"elapsed={round(time.time() - start, 1)}s")
                    print(
                        json.dumps(
                            {
                                "event": "progress",
                                "done": done,
                                "total": total,
                                "elapsed_sec": round(time.time() - start, 2),
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )

    rows.sort(key=lambda item: item["test_sample_id"])
    output_rows = [{"test_sample_id": row["test_sample_id"], "prediction": row["prediction"]} for row in rows]
    detail_rows = rows
    with (output_dir / "openseek-7-v1.jsonl").open("w", encoding="utf-8") as f:
        for row in output_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (output_dir / "router_details.jsonl").open("w", encoding="utf-8") as f:
        for row in detail_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        **summarize(rows, primary_name),
        "elapsed_sec": round(time.time() - start, 2),
        "candidate_files": [{"name": name, "path": str(path)} for name, path in candidate_specs],
        "rescue_candidate_files": [{"name": name, "path": str(path)} for name, path in rescue_specs],
        "single_policy": args.single_policy,
        "model_name": args.model_name,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "router_mode": args.router_mode,
        "prompt_style": args.prompt_style,
        "weak_source": args.weak_source,
        "weak_source_safety": args.weak_source_safety,
        "auto_weak_rank2_safety": auto_weak_rank2_safety,
        "disable_auto_weak_rank2_safety": args.disable_auto_weak_rank2_safety,
        "rescue_style": args.rescue_style,
        "allow_deterministic_rescue": args.allow_deterministic_rescue,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"event": "done", "summary": summary}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
