from __future__ import annotations

import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Sequence

import requests
from sklearn.model_selection import train_test_split

from method import InferenceConfig, ResponseDetails, annotate_detailed, postprocess_prediction


TASK_ID = 7
DATASET_PATH = Path("data/raw/openseek-7_jeopardy_answer_generation_all.json")
STOP_SEQUENCES = ["\nQuestion:", "\nCategory:", "\nClue:"]
WORD_PATTERN = re.compile(r"[a-zA-Z0-9_]+")
NOISY_MARKERS = (
    "because",
    "supposed to be",
    "should be",
    "i think",
    "that doesn't make sense",
    "the answer",
    "alternatively",
    "which is",
    "but ",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task7 成功思路库构建与检索增强验证。")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build-bank", help="在训练集上构建成功思路库。")
    add_common_args(build_parser)
    build_parser.add_argument("--output-dir", type=str, required=True, help="输出目录。")
    build_parser.add_argument("--max-train-samples", type=int, default=0, help="最多处理多少条训练样本，0 表示全部。")
    build_parser.add_argument("--start-train-index", type=int, default=0, help="训练切片起始下标，便于分段续跑。")

    validate_parser = subparsers.add_parser("validate", help="使用成功思路库做验证集增强推理。")
    add_common_args(validate_parser)
    validate_parser.add_argument("--bank-path", type=str, required=True, help="成功思路库 JSONL 文件。")
    validate_parser.add_argument("--output-dir", type=str, required=True, help="输出目录。")
    validate_parser.add_argument("--max-val-samples", type=int, default=0, help="最多验证多少条样本，0 表示全部。")
    validate_parser.add_argument("--base-few-shot", type=int, default=4, help="普通 few-shot 示例数量。")
    validate_parser.add_argument("--retrieval-k", type=int, default=3, help="检索多少条成功思路。")
    validate_parser.add_argument("--reasoning-char-limit", type=int, default=800, help="每条思维链最多保留多少字符。")
    validate_parser.add_argument("--compress-answer", action="store_true", help="是否启用二阶段非思考模式答案压缩器。")
    validate_parser.add_argument("--compress-max-new-tokens", type=int, default=64, help="压缩器 token 上限。")
    validate_parser.add_argument("--compress-temperature", type=float, default=0.0, help="压缩器温度。")
    validate_parser.add_argument("--compress-top-p", type=float, default=0.8, help="压缩器 top-p。")
    return parser.parse_args()


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-path", type=str, default=str(DATASET_PATH), help="Task7 数据文件路径。")
    parser.add_argument("--seed", type=int, default=42, help="固定划分随机种子。")
    parser.add_argument("--val-size", type=int, default=500, help="验证集样本数。")
    parser.add_argument("--few-shot", type=int, default=12, help="构建训练题 prompt 时的 few-shot 数量。")
    parser.add_argument("--max-workers", type=int, default=4, help="并发请求数。")
    parser.add_argument("--api-base", type=str, default="http://127.0.0.1:2026", help="OpenAI 兼容接口地址。")
    parser.add_argument("--model-name", type=str, default="Qwen3-4B", help="模型名。")
    parser.add_argument("--timeout", type=float, default=240.0, help="单请求超时秒数。")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="生成 token 上限。")
    parser.add_argument("--temperature", type=float, default=0.6, help="采样温度。")
    parser.add_argument("--top-p", type=float, default=0.95, help="top-p。")
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="可选透传 Qwen3 thinking 开关；默认不传，保持服务端默认行为。",
    )


def load_dataset(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def split_dataset(examples: Sequence[Dict[str, Any]], val_size: int, seed: int) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    idx = list(range(len(examples)))
    train_idx, val_idx = train_test_split(idx, test_size=val_size, random_state=seed, shuffle=True)
    train_examples = [examples[i] for i in train_idx]
    val_examples = [examples[i] for i in val_idx]
    return train_examples, val_examples


def make_infer_cfg(args: argparse.Namespace) -> InferenceConfig:
    return InferenceConfig(
        api_base=args.api_base,
        model_name=args.model_name,
        timeout=args.timeout,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        stop=STOP_SEQUENCES,
        enable_thinking=getattr(args, "enable_thinking", None),
    )


def normalize_answer(text: str | None) -> str | None:
    value = postprocess_prediction(text or "", TASK_ID)
    if value is None:
        return None
    value = re.sub(r"\s+", " ", value.strip().lower())
    return value or None


def sample_gold(sample: Dict[str, Any]) -> str | None:
    output = sample.get("output", "")
    if isinstance(output, list):
        output = output[0] if output else ""
    return normalize_answer(str(output))


def word_overlap(query_text: str, candidate_text: str) -> float:
    query_tokens = set(token.lower() for token in WORD_PATTERN.findall(query_text))
    cand_tokens = set(token.lower() for token in WORD_PATTERN.findall(candidate_text))
    if not query_tokens or not cand_tokens:
        return 0.0
    return len(query_tokens & cand_tokens) / (len(query_tokens) ** 0.5 * len(cand_tokens) ** 0.5)


def select_examples(examples: Sequence[Dict[str, Any]], query_text: str, k: int) -> str:
    ranked = sorted(
        examples,
        key=lambda ex: word_overlap(query_text, str(ex.get("input", ""))),
        reverse=True,
    )
    blocks: List[str] = []
    for idx, example in enumerate(ranked[:k], start=1):
        example_input = str(example.get("input", "")).strip()
        example_output = example.get("output", "")
        if isinstance(example_output, list):
            example_output = example_output[0] if example_output else ""
        example_output = str(example_output).strip()
        if not example_input:
            continue
        blocks.append(
            f"[Example {idx}]\n"
            f"Input:\n{example_input}\n"
            f"Output:\n{example_output}\n"
        )
    return "\n".join(blocks).strip()


def build_optimized_prompt(task_id: int, dataset: Dict[str, Any], sample_input: str, few_shot_examples: int) -> str:
    if task_id != TASK_ID:
        raise ValueError(f"Task7 reasoning retrieval only supports task {TASK_ID}, got {task_id}")
    task_description = "\n".join(dataset.get("Definition", []))
    examples_text = select_examples(dataset.get("examples", []), sample_input, few_shot_examples)
    return (
        "You are answering a Jeopardy-style clue.\n\n"
        "[Task Description]\n"
        f"{task_description}\n\n"
        "[In-Context Examples]\n"
        f"{examples_text}\n\n"
        "[Sample To Annotate]\n"
        f"{sample_input}\n\n"
        "[Answer Rules]\n"
        "1. Return one lowercase answer only.\n"
        "2. Use the shortest unambiguous canonical answer.\n"
        "3. Omit leading articles like 'a', 'an', 'the'.\n"
        "4. Prefer the common canonical entity name; add a location qualifier only if needed to disambiguate.\n"
        "5. Do not output alternatives, explanation, or multiple candidates.\n\n"
        "Answer only:\n"
    )


def looks_like_clean_final_answer(text: str) -> bool:
    value = (text or "").strip()
    if not value:
        return False
    lowered = value.lower()
    if lowered.startswith("<think>"):
        return False
    if "\n" in value:
        return False
    if any(marker in lowered for marker in NOISY_MARKERS):
        return False
    if len(WORD_PATTERN.findall(value)) > 4:
        return False
    if value.count(",") > 0 or value.count(";") > 0:
        return False
    return True


def build_reasoning_prompt(
    dataset: Dict[str, Any],
    train_examples: Sequence[Dict[str, Any]],
    bank_rows: Sequence[Dict[str, Any]],
    sample_input: str,
    base_few_shot: int,
    retrieval_k: int,
    reasoning_char_limit: int,
) -> str:
    task_description = "\n".join(dataset.get("Definition", []))
    examples_text = select_examples(train_examples, sample_input, base_few_shot)
    ranked_bank = sorted(
        bank_rows,
        key=lambda row: word_overlap(sample_input, str(row.get("input", ""))),
        reverse=True,
    )
    reasoning_blocks: List[str] = []
    for idx, row in enumerate(ranked_bank[:retrieval_k], start=1):
        reasoning = str(row.get("thinking", "")).strip()
        if reasoning_char_limit > 0:
            reasoning = reasoning[:reasoning_char_limit].strip()
        if not reasoning:
            continue
        reasoning_blocks.append(
            f"[Successful Reasoning Case {idx}]\n"
            f"Question:\n{str(row.get('input', '')).strip()}\n"
            f"Reasoning:\n{reasoning}\n"
            f"Final Answer:\n{str(row.get('answer', '')).strip()}\n"
        )
    reasoning_text = "\n".join(reasoning_blocks).strip()
    return (
        "You are answering a Jeopardy-style clue.\n\n"
        "[Task Description]\n"
        f"{task_description}\n\n"
        "[In-Context Examples]\n"
        f"{examples_text}\n\n"
        "[Retrieved Successful Reasoning Cases]\n"
        f"{reasoning_text}\n\n"
        "[Sample To Annotate]\n"
        f"{sample_input}\n\n"
        "[Answer Rules]\n"
        "1. Study the retrieved successful reasoning patterns, but do not copy their final answers unless they truly fit the new clue.\n"
        "2. Return one lowercase answer only.\n"
        "3. Use the shortest unambiguous canonical answer.\n"
        "4. Omit leading articles like 'a', 'an', 'the'.\n"
        "5. Do not output explanation, alternatives, or multiple candidates.\n\n"
        "Answer only:\n"
    )


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _chat_url(api_base: str) -> str:
    return f"{api_base.rstrip('/')}/v1/chat/completions"


def build_compression_prompt(sample_input: str, draft_output: str, answer_rules: str = "shortest") -> str:
    if answer_rules == "preserve_articles":
        article_rule = (
            "5. Preserve leading articles like 'a', 'an', or 'the' when they are already present in the draft and are part of a title, named phrase, set phrase, or natural Jeopardy response.\n"
            "6. Do not add articles that are not already in the draft.\n"
        )
        expansion_rule_number = "7"
        reasoning_rule_number = "8"
    else:
        article_rule = (
            "5. If the draft already is a short standalone answer, return it unchanged.\n"
        )
        expansion_rule_number = "6"
        reasoning_rule_number = "7"
    return (
        "You are extracting the final answer from an existing Jeopardy-style draft response.\n\n"
        "[Original Question]\n"
        f"{sample_input}\n\n"
        "[Draft Model Output]\n"
        f"{draft_output}\n\n"
        "[Extraction Rules]\n"
        "1. Return one lowercase canonical answer only.\n"
        "2. You must copy the answer from the draft output; do not invent, paraphrase, or replace it with a synonym.\n"
        "3. Remove quotes, explanation, hesitation, and extra text.\n"
        "4. Keep only the shortest unambiguous final answer span that already appears in the draft output.\n"
        f"{article_rule}"
        f"{expansion_rule_number}. Do not expand a correct minimal answer into a fuller name, more common entity, or nearby category unless the draft requires that exact form.\n"
        f"{reasoning_rule_number}. Do not add new reasoning.\n\n"
        "Answer only:\n"
    )


def compress_answer_no_think(
    sample_input: str,
    draft_output: str,
    args: argparse.Namespace,
) -> str:
    answer_rules = getattr(args, "answer_rules", getattr(args, "task7_answer_rules", "shortest"))
    payload = {
        "model": args.model_name,
        "messages": [{"role": "user", "content": build_compression_prompt(sample_input, draft_output, answer_rules)}],
        "max_tokens": args.compress_max_new_tokens,
        "temperature": args.compress_temperature,
        "top_p": args.compress_top_p,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    resp = requests.post(
        _chat_url(args.api_base),
        json=payload,
        timeout=args.timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    choices = data.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message", {})
    content = message.get("content", "")
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts).strip()
    return str(content or "").strip()


def run_build_bank(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(Path(args.dataset_path))
    train_examples, _ = split_dataset(list(dataset["examples"]), args.val_size, args.seed)

    start_idx = max(0, args.start_train_index)
    selected_train = train_examples[start_idx:]
    if args.max_train_samples > 0:
        selected_train = selected_train[: args.max_train_samples]

    cfg = make_infer_cfg(args)
    start = time.time()
    rows: List[Dict[str, Any]] = []

    def run_one(sample: Dict[str, Any]) -> Dict[str, Any]:
        filtered_examples = [example for example in train_examples if example.get("id") != sample.get("id")]
        prompt_dataset = dict(dataset)
        prompt_dataset["examples"] = filtered_examples
        prompt = build_optimized_prompt(TASK_ID, prompt_dataset, str(sample["input"]), args.few_shot)
        details: ResponseDetails = annotate_detailed(prompt, cfg)
        prediction = normalize_answer(details.visible_text or details.raw_text)
        gold = sample_gold(sample)
        return {
            "id": sample["id"],
            "input": str(sample["input"]),
            "gold": gold,
            "prediction": prediction,
            "correct": prediction == gold,
            "thinking": details.reasoning_text.strip(),
            "visible_text": details.visible_text.strip(),
            "raw_text": details.raw_text.strip(),
        }

    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futures = {ex.submit(run_one, sample): sample["id"] for sample in selected_train}
        done = 0
        for future in as_completed(futures):
            rows.append(future.result())
            done += 1
            if done % 20 == 0 or done == len(selected_train):
                print(
                    json.dumps(
                        {
                            "event": "progress",
                            "mode": "build-bank",
                            "done": done,
                            "total": len(selected_train),
                            "elapsed_sec": round(time.time() - start, 2),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

    rows.sort(key=lambda item: item["id"])
    success_rows = [
        {
            "id": row["id"],
            "input": row["input"],
            "answer": row["gold"],
            "prediction": row["prediction"],
            "thinking": row["thinking"],
            "visible_text": row["visible_text"],
        }
        for row in rows
        if row["correct"] and row["thinking"]
    ]
    summary = {
        "mode": "build-bank",
        "seed": args.seed,
        "train_total_available": len(train_examples),
        "train_processed": len(selected_train),
        "start_train_index": start_idx,
        "max_train_samples": args.max_train_samples,
        "success_with_reasoning": len(success_rows),
        "success_ratio": round(len(success_rows) / len(selected_train), 6) if selected_train else 0.0,
        "elapsed_sec": round(time.time() - start, 2),
        "few_shot": args.few_shot,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }
    write_json(output_dir / "summary.json", summary)
    write_jsonl(output_dir / "train_generation_results.jsonl", rows)
    write_jsonl(output_dir / "successful_reasoning_bank.jsonl", success_rows)
    print(json.dumps({"event": "done", **summary}, ensure_ascii=False), flush=True)


def run_validate(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(Path(args.dataset_path))
    train_examples, val_examples = split_dataset(list(dataset["examples"]), args.val_size, args.seed)
    if args.max_val_samples > 0:
        val_examples = val_examples[: args.max_val_samples]

    bank_rows = [
        json.loads(line)
        for line in Path(args.bank_path).resolve().open("r", encoding="utf-8")
        if line.strip()
    ]
    cfg = make_infer_cfg(args)
    start = time.time()
    rows: List[Dict[str, Any]] = []

    def run_one(sample: Dict[str, Any]) -> Dict[str, Any]:
        prompt = build_reasoning_prompt(
            dataset=dataset,
            train_examples=train_examples,
            bank_rows=bank_rows,
            sample_input=str(sample["input"]),
            base_few_shot=args.base_few_shot,
            retrieval_k=args.retrieval_k,
            reasoning_char_limit=args.reasoning_char_limit,
        )
        details = annotate_detailed(prompt, cfg)
        draft_output = details.raw_text.strip() or details.visible_text.strip()
        compressed_output = ""
        base_visible = details.visible_text.strip()
        if args.compress_answer and not looks_like_clean_final_answer(base_visible):
            try:
                compressed_output = compress_answer_no_think(str(sample["input"]), draft_output, args)
            except Exception as exc:
                compressed_output = f"compression_error: {type(exc).__name__}: {exc}"
        final_text = compressed_output if compressed_output and not compressed_output.startswith("compression_error:") else (details.visible_text or details.raw_text)
        prediction = normalize_answer(final_text)
        gold = sample_gold(sample)
        retrieved = sorted(
            bank_rows,
            key=lambda row: word_overlap(str(sample["input"]), str(row.get("input", ""))),
            reverse=True,
        )[: args.retrieval_k]
        return {
            "id": sample["id"],
            "input": str(sample["input"]),
            "prediction": prediction,
            "gold": gold,
            "correct": prediction == gold,
            "visible_text": details.visible_text.strip(),
            "raw_text": details.raw_text.strip(),
            "thinking": details.reasoning_text.strip(),
            "compressed_output": compressed_output,
            "compress_applied": bool(compressed_output),
            "retrieved_ids": [item.get("id") for item in retrieved],
        }

    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futures = {ex.submit(run_one, sample): sample["id"] for sample in val_examples}
        done = 0
        for future in as_completed(futures):
            rows.append(future.result())
            done += 1
            if done % 20 == 0 or done == len(val_examples):
                print(
                    json.dumps(
                        {
                            "event": "progress",
                            "mode": "validate",
                            "done": done,
                            "total": len(val_examples),
                            "elapsed_sec": round(time.time() - start, 2),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

    rows.sort(key=lambda item: item["id"])
    correct = sum(1 for row in rows if row["correct"])
    non_null = sum(1 for row in rows if row["prediction"] is not None)
    summary = {
        "mode": "validate",
        "seed": args.seed,
        "val_size": len(rows),
        "bank_size": len(bank_rows),
        "base_few_shot": args.base_few_shot,
        "retrieval_k": args.retrieval_k,
        "reasoning_char_limit": args.reasoning_char_limit,
        "max_new_tokens": args.max_new_tokens,
        "compress_answer": bool(args.compress_answer),
        "compress_max_new_tokens": args.compress_max_new_tokens if args.compress_answer else 0,
        "compress_temperature": args.compress_temperature if args.compress_answer else 0.0,
        "compress_top_p": args.compress_top_p if args.compress_answer else 0.0,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "accuracy": round(correct / len(rows), 6) if rows else 0.0,
        "correct": correct,
        "total": len(rows),
        "coverage": round(non_null / len(rows), 6) if rows else 0.0,
        "non_null_predictions": non_null,
        "elapsed_sec": round(time.time() - start, 2),
    }
    write_json(output_dir / "summary.json", summary)
    write_jsonl(output_dir / "results.jsonl", rows)
    write_jsonl(output_dir / "mismatches.jsonl", [row for row in rows if not row["correct"]])
    print(json.dumps({"event": "done", **summary}, ensure_ascii=False), flush=True)


def main() -> None:
    args = parse_args()
    if args.command == "build-bank":
        run_build_bank(args)
        return
    if args.command == "validate":
        run_validate(args)
        return
    raise ValueError(f"未知命令：{args.command}")


if __name__ == "__main__":
    main()
