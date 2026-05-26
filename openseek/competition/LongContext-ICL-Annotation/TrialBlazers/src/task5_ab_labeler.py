from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import requests
from tqdm import tqdm

from method import WORD_PATTERN, load_tokenizer, postprocess_prediction, select_examples as select_official_examples


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "openseek-5_semeval_2018_task1_tweet_sadness_detection.json"
TASK_ID = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task5 A/B sadness labeler.")
    parser.add_argument("--dataset-path", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--api-base", default="http://127.0.0.1:2026")
    parser.add_argument("--model-name", default="Qwen3-4B")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--few-shot", type=int, default=24)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--official-min-context", action="store_true")
    parser.add_argument("--min-context-tokens", type=int, default=30000)
    parser.add_argument(
        "--cache-prefix-context-tokens",
        type=int,
        default=0,
        help="Hybrid cache mode: use a fixed official ICL prefix, then append per-sample examples to reach min context.",
    )
    parser.add_argument("--max-input-tokens", type=int, default=60000)
    parser.add_argument("--reserved-generation-tokens", type=int, default=2048)
    parser.add_argument("--tokenizer-path", default=str(PROJECT_ROOT / "models" / "Qwen3-4B"))
    parser.add_argument("--context-audit-dir", default="")
    parser.add_argument(
        "--prompt-style",
        choices=("conservative", "balanced", "recall"),
        default="conservative",
    )
    return parser.parse_args()


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def audit_file_for(output_dir: Path, context_audit_dir: str, prompt_style: str) -> Path:
    name = f"context_audit_task5_ab_{prompt_style}_{output_dir.name}.jsonl"
    if context_audit_dir:
        return resolve_path(context_audit_dir) / name
    return output_dir / name


def normalize_example_label(value: Any) -> str:
    label = postprocess_prediction(str(value), TASK_ID)
    return "Not sad" if label == "Not sad" else "Sad"


def normalize_row_id(row: Dict[str, Any]) -> str:
    for key in ("test_sample_id", "id", "sample_id"):
        if key in row and row[key] is not None:
            return str(row[key]).strip()
    raise KeyError(f"missing sample id in keys={list(row)}")


def load_reference(path: Path) -> Dict[str, str]:
    refs: Dict[str, str] = {}
    if not path.exists():
        return refs
    with path.open(encoding="utf-8") as reader:
        for line in reader:
            if not line.strip():
                continue
            row = json.loads(line)
            refs[normalize_row_id(row)] = normalize_example_label(row.get("prediction"))
    return refs


def token_overlap(query: str, candidate: str) -> float:
    q = set(token.lower() for token in WORD_PATTERN.findall(query))
    c = set(token.lower() for token in WORD_PATTERN.findall(candidate))
    if not q or not c:
        return 0.0
    return len(q & c) / (len(q) ** 0.5 * len(c) ** 0.5)


def task5_score(query: str, candidate: str) -> float:
    query_lower = query.lower()
    cand_lower = candidate.lower()
    score = 6.0 * token_overlap(query, candidate)
    cue_groups = (
        ("quote", "joke", "lol", "haha", "😂", "song", "lyric"),
        ("lost", "miss", "missing", "forgot", "can't find", "cant find"),
        ("hurt", "pain", "sick", "tired", "sleep", "insomnia", "weary"),
        ("pissed", "fuming", "furious", "angry", "annoyed", "frustrat"),
        ("sad", "depress", "lonely", "alone", "cry", "miserable", "unhappy"),
        ("politic", "trump", "government", "news", "police", "obama"),
    )
    for cues in cue_groups:
        if any(cue in query_lower for cue in cues) and any(cue in cand_lower for cue in cues):
            score += 2.0
    score += 1.0 / (1.0 + abs(len(WORD_PATTERN.findall(query)) - len(WORD_PATTERN.findall(candidate))))
    return score


def select_few_shot(examples: Sequence[Dict[str, Any]], query: str, few_shot: int, prompt_style: str) -> List[Dict[str, Any]]:
    scored = sorted(
        ((example, task5_score(query, str(example.get("input", "")))) for example in examples),
        key=lambda item: item[1],
        reverse=True,
    )
    by_label = {"Not sad": [], "Sad": []}
    for example, _ in scored:
        by_label[normalize_example_label(example.get("output"))].append(example)

    if prompt_style == "recall":
        cycle = ("Sad", "Sad", "Not sad")
    elif prompt_style == "balanced":
        cycle = ("Not sad", "Sad")
    else:
        cycle = ("Not sad", "Not sad", "Sad")

    selected: List[Dict[str, Any]] = []
    while len(selected) < few_shot and (by_label["Not sad"] or by_label["Sad"]):
        progressed = False
        for label in cycle:
            if len(selected) >= few_shot:
                break
            if by_label[label]:
                selected.append(by_label[label].pop(0))
                progressed = True
        if not progressed:
            break
    return selected


def build_prompt(
    tweet: str,
    few_shot_examples: Sequence[Dict[str, Any]],
    prompt_style: str,
    official_examples_text: str = "",
) -> str:
    examples: List[str] = []
    for idx, example in enumerate(few_shot_examples, start=1):
        label = normalize_example_label(example.get("output"))
        letter = "A" if label == "Not sad" else "B"
        examples.append(
            f"[Example {idx}]\n"
            f"Tweet: {str(example.get('input', '')).strip()}\n"
            f"Answer: {letter}\n"
        )

    if prompt_style == "recall":
        rules = (
            "1. Choose B when the author personally expresses sadness, anger, frustration, pain, insomnia, loss, loneliness, stress, or clear distress.\n"
            "2. Casual wording, hashtags, laughter, and emojis do not cancel a direct personal bad feeling.\n"
            "3. Choose A for quotes, jokes, lyrics, third-person events, commentary, insults about others, or isolated negative words without the author's own feeling.\n"
            "4. If the author is clearly affected, choose B; otherwise choose A.\n"
        )
    elif prompt_style == "balanced":
        rules = (
            "1. Decide whether the tweet author themself is feeling bad, not whether the topic sounds negative.\n"
            "2. Choose B for clear personal sadness, disappointment, anger, frustration, pain, insomnia, loss, loneliness, stress, or discouragement.\n"
            "3. Choose A for quotations, jokes, lyrics, sarcasm, news or political commentary, third-person problems, insults, motivational advice, and ambiguous fragments.\n"
            "4. If both labels are plausible, choose A.\n"
        )
    else:
        rules = (
            "1. Choose A unless the author clearly states their own negative emotional or physical state.\n"
            "2. Choose B for clear personal sadness, being upset, pissed, fuming, frustrated, lonely, hurt, unable to sleep, discouraged, stressed, or losing/missing something important.\n"
            "3. Choose A for quotes, lyrics, jokes, sarcasm, politics, religious/news commentary, third-person bad events, insults, sports banter, or sad-sounding words used only as descriptions.\n"
            "4. Words like sober, dark, gritty, grim, dreadful, pout, hell, blood, rage, suspicion, panic, and stinks are not enough by themselves.\n"
            "5. If uncertain, choose A.\n"
        )

    official_block = ""
    if official_examples_text.strip():
        official_block = (
            "[Official Long ICL Examples]\n"
            "These examples define the task distribution. Use the decision rules and the final tweet below for the final answer.\n"
            f"{official_examples_text.strip()}\n\n"
        )

    return (
        "You are labeling one tweet for whether the author is sad or personally distressed.\n"
        "Return exactly one letter: A or B. Do not explain.\n\n"
        "[Labels]\n"
        "A. Not sad\n"
        "B. Sad\n\n"
        f"{official_block}"
        "[Decision Rules]\n"
        f"{rules}\n"
        "[Calibration Examples]\n"
        f"{''.join(examples)}\n"
        "[Tweet]\n"
        f"{tweet.strip()}\n\n"
        "Answer letter only:\n"
    )


def chat_url(api_base: str) -> str:
    return f"{api_base.rstrip('/')}/v1/chat/completions"


def extract_content(message: Any) -> str:
    if isinstance(message, dict):
        content = message.get("content", "")
    else:
        content = message
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts).strip()
    return str(content).strip()


def request_label(prompt: str, args: argparse.Namespace) -> str:
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
    return extract_content(choices[0].get("message", {}))


def parse_choice(raw_text: str, prompt_style: str) -> Tuple[str, str]:
    text = (raw_text or "").strip().upper()
    match = re.search(r"\b([AB])\b", text)
    if match:
        letter = match.group(1)
    else:
        compact = re.sub(r"[^AB]", "", text)
        letter = compact[0] if compact else ("B" if prompt_style == "recall" else "A")
    return letter, ("Not sad" if letter == "A" else "Sad")


def main() -> None:
    args = parse_args()
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(resolve_path(args.dataset_path).read_text(encoding="utf-8"))
    examples = list(data.get("examples", []))
    test_samples = list(data.get("test_samples", []))
    if args.max_samples > 0:
        test_samples = test_samples[: args.max_samples]

    tokenizer = None
    if args.official_min_context:
        tokenizer_path = resolve_path(args.tokenizer_path)
        tokenizer = load_tokenizer(str(tokenizer_path) if tokenizer_path.exists() else None)
    strategy = "task5_conservative" if args.prompt_style == "conservative" else "task5_balanced"
    cache_prefix_enabled = bool(
        args.official_min_context and args.cache_prefix_context_tokens > 0 and tokenizer is not None
    )
    cache_prefix_examples_text = ""
    cache_prefix_selected_count = 0
    cache_prefix_used_tokens = 0
    if cache_prefix_enabled and test_samples:
        cache_prefix_target = min(args.min_context_tokens, max(0, args.cache_prefix_context_tokens))
        query_seed = str(test_samples[0].get("input", ""))
        cache_prefix_examples_text, cache_prefix_selected_count, cache_prefix_used_tokens = select_official_examples(
            all_examples=examples,
            query_text=query_seed,
            tokenizer=tokenizer,
            max_input_tokens=args.max_input_tokens,
            target_context_tokens=cache_prefix_target,
            reserved_generation_tokens=args.reserved_generation_tokens,
            task_id=TASK_ID,
            strategy=strategy,
        )
        print(
            json.dumps(
                {
                    "cache_prefix_context_tokens": args.cache_prefix_context_tokens,
                    "cache_prefix_selected_count": cache_prefix_selected_count,
                    "cache_prefix_used_tokens": cache_prefix_used_tokens,
                    "remaining_target": max(0, args.min_context_tokens - cache_prefix_used_tokens),
                },
                ensure_ascii=False,
            )
        )
    output_file = output_dir / "openseek-5-v1.jsonl"
    details_file = output_dir / "task5_ab_details.jsonl"
    summary_file = output_dir / "task5_ab_summary.json"
    audit_file = audit_file_for(output_dir, args.context_audit_dir, args.prompt_style)

    def run_one(sample: Dict[str, Any]) -> Dict[str, Any]:
        row_id = str(sample.get("id"))
        tweet = str(sample.get("input", ""))
        few_shot_examples = select_few_shot(examples, tweet, args.few_shot, args.prompt_style)
        official_examples_text = ""
        official_selected_count = 0
        official_used_tokens = 0
        if args.official_min_context:
            if cache_prefix_enabled:
                official_examples_text = cache_prefix_examples_text
                official_selected_count = cache_prefix_selected_count
                official_used_tokens = cache_prefix_used_tokens
                remaining_target = max(0, args.min_context_tokens - cache_prefix_used_tokens)
                if remaining_target > 0:
                    suffix_max_input_tokens = max(
                        args.reserved_generation_tokens,
                        args.max_input_tokens - cache_prefix_used_tokens,
                    )
                    suffix_examples_text, suffix_selected_count, suffix_used_tokens = select_official_examples(
                        all_examples=examples,
                        query_text=tweet,
                        tokenizer=tokenizer,
                        max_input_tokens=suffix_max_input_tokens,
                        target_context_tokens=remaining_target,
                        reserved_generation_tokens=args.reserved_generation_tokens,
                        task_id=TASK_ID,
                        strategy=strategy,
                    )
                    if suffix_examples_text:
                        official_examples_text = (official_examples_text + "\n\n" + suffix_examples_text).strip()
                    official_selected_count += suffix_selected_count
                    official_used_tokens += suffix_used_tokens
            else:
                official_examples_text, official_selected_count, official_used_tokens = select_official_examples(
                    all_examples=examples,
                    query_text=tweet,
                    tokenizer=tokenizer,
                    max_input_tokens=args.max_input_tokens,
                    target_context_tokens=args.min_context_tokens,
                    reserved_generation_tokens=args.reserved_generation_tokens,
                    task_id=TASK_ID,
                    strategy=strategy,
                )
        prompt = build_prompt(tweet, few_shot_examples, args.prompt_style, official_examples_text)
        raw = ""
        try:
            raw = request_label(prompt, args)
            letter, label = parse_choice(raw, args.prompt_style)
        except Exception as exc:
            raw = f"<error>{type(exc).__name__}: {exc}"
            letter, label = ("B", "Sad") if args.prompt_style == "recall" else ("A", "Not sad")
        return {
            "test_sample_id": row_id,
            "prediction": label,
            "_detail": {
                "id": row_id,
                "input": tweet,
                "letter": letter,
                "prediction": label,
                "raw_response": raw,
                "prompt": prompt,
                "official_context_tokens": official_used_tokens,
            },
            "_audit": {
                "task_id": TASK_ID,
                "sample_id": row_id,
                "stage": "candidate_generation",
                "mode": f"task5_ab_{args.prompt_style}",
                "target_min_context_tokens": args.min_context_tokens,
                "used_example_tokens": official_used_tokens,
                "selected_example_count": official_selected_count,
                "pass_min_context": official_used_tokens >= args.min_context_tokens,
                "cache_prefix_context_tokens": int(args.cache_prefix_context_tokens),
                "cache_prefix_used_tokens": int(cache_prefix_used_tokens if cache_prefix_enabled else 0),
            },
        }

    indexed_rows: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as executor:
        futures = {executor.submit(run_one, sample): idx for idx, sample in enumerate(test_samples)}
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Task5 AB {args.prompt_style}", unit="sample"):
            indexed_rows.append({"idx": futures[future], "row": future.result()})

    indexed_rows.sort(key=lambda item: item["idx"])
    details = []
    audit_rows = []
    audit_file.parent.mkdir(parents=True, exist_ok=True)
    with (
        output_file.open("w", encoding="utf-8") as writer,
        details_file.open("w", encoding="utf-8") as detail_writer,
        audit_file.open("w", encoding="utf-8") as audit_writer,
    ):
        for item in indexed_rows:
            row = item["row"]
            writer.write(
                json.dumps(
                    {"test_sample_id": row["test_sample_id"], "prediction": row["prediction"]},
                    ensure_ascii=False,
                )
                + "\n"
            )
            details.append(row["_detail"])
            detail_writer.write(json.dumps(row["_detail"], ensure_ascii=False) + "\n")
            if args.official_min_context:
                audit_rows.append(row["_audit"])
                audit_writer.write(json.dumps(row["_audit"], ensure_ascii=False) + "\n")

    summary: Dict[str, Any] = {
        "output_file": str(output_file),
        "details_file": str(details_file),
        "samples": len(indexed_rows),
        "prompt_style": args.prompt_style,
        "few_shot": args.few_shot,
        "official_min_context": bool(args.official_min_context),
        "cache_prefix_context_tokens": int(args.cache_prefix_context_tokens),
        "context_audit_file": str(audit_file) if args.official_min_context else "",
        "label_counts": {},
    }
    if audit_rows:
        summary["min_used_example_tokens"] = min(int(row["used_example_tokens"]) for row in audit_rows)
        summary["context_pass_rows"] = sum(1 for row in audit_rows if row["pass_min_context"])
    for item in details:
        label = item["prediction"]
        summary["label_counts"][label] = summary["label_counts"].get(label, 0) + 1
    summary_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
