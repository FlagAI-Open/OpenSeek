from __future__ import annotations

import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Sequence

import requests
from tqdm import tqdm

from task7_reasoning_retrieval import (
    DATASET_PATH,
    load_dataset,
    looks_like_clean_final_answer,
    make_infer_cfg,
    normalize_answer,
    sample_gold,
    select_examples,
    split_dataset,
    word_overlap,
    compress_answer_no_think,
    write_json,
    write_jsonl,
)
from method import annotate_detailed, load_tokenizer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BANK_PATH = PROJECT_ROOT / "outputs" / "task7_reasoning_bank_scale1500_20260401" / "successful_reasoning_bank.jsonl"
DEFAULT_BANK_TAGS_PATH = PROJECT_ROOT / "outputs" / "task7_semantic_tag_retrieval_full500_20260401" / "bank_semantic_tags.jsonl"
TAG_KEYS = [
    "question_type",
    "answer_type",
    "knowledge_domain",
    "reasoning_style",
    "constraints",
    "risk_flags",
    "key_terms",
]
TAG_FIELD_WEIGHTS = {
    "question_type": 3.0,
    "answer_type": 2.5,
    "knowledge_domain": 2.0,
    "reasoning_style": 2.0,
    "constraints": 1.75,
    "risk_flags": 1.25,
    "key_terms": 1.0,
}
WORD_PATTERN = re.compile(r"[a-z0-9]+")
ANSWER_RULE_CHOICES = ("shortest", "preserve_articles")
RETRIEVAL_MODE_CHOICES = ("lexical", "semantic", "semantic_rerank", "semantic_mmr")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task7 Qwen 语义标签检索原型。")
    parser.add_argument("--dataset-path", type=str, default=str(DATASET_PATH), help="Task7 数据文件路径。")
    parser.add_argument("--bank-path", type=str, required=True, help="成功思路库 JSONL 文件。")
    parser.add_argument("--bank-tags-path", type=str, default="", help="可选，复用已有成功思路库语义标签缓存。")
    parser.add_argument("--val-tags-path", type=str, default="", help="可选，复用已有验证集语义标签缓存。")
    parser.add_argument("--output-dir", type=str, required=True, help="输出目录。")
    parser.add_argument("--seed", type=int, default=42, help="固定划分随机种子。")
    parser.add_argument("--val-size", type=int, default=500, help="验证集样本数。")
    parser.add_argument("--max-val-samples", type=int, default=100, help="最多验证多少条样本。")
    parser.add_argument("--base-few-shot", type=int, default=4, help="普通 few-shot 示例数量。")
    parser.add_argument("--official-min-context", action="store_true", help="主答案生成 prompt 用成功案例补足官方 30K ICL context。")
    parser.add_argument("--min-context-tokens", type=int, default=30000)
    parser.add_argument("--max-input-tokens", type=int, default=60000)
    parser.add_argument("--reserved-generation-tokens", type=int, default=2048)
    parser.add_argument("--tokenizer-path", type=str, default=str(PROJECT_ROOT / "models" / "Qwen3-4B"))
    parser.add_argument("--retrieval-k", type=int, default=3, help="检索多少条成功思路。")
    parser.add_argument("--reasoning-char-limit", type=int, default=800, help="每条思维链最多保留多少字符。")
    parser.add_argument("--compress-answer", action="store_true", help="是否启用二阶段答案压缩器。")
    parser.add_argument("--compress-max-new-tokens", type=int, default=64, help="压缩器 token 上限。")
    parser.add_argument("--compress-temperature", type=float, default=0.0, help="压缩器温度。")
    parser.add_argument("--compress-top-p", type=float, default=0.8, help="压缩器 top-p。")
    parser.add_argument("--max-workers", type=int, default=4, help="并发请求数。")
    parser.add_argument("--api-base", type=str, default="http://127.0.0.1:2026", help="OpenAI 兼容接口地址。")
    parser.add_argument("--model-name", type=str, default="Qwen3-4B", help="模型名。")
    parser.add_argument("--timeout", type=float, default=240.0, help="单请求超时秒数。")
    parser.add_argument("--max-new-tokens", type=int, default=1536, help="主阶段生成 token 上限。")
    parser.add_argument("--temperature", type=float, default=0.6, help="主阶段采样温度。")
    parser.add_argument("--top-p", type=float, default=0.95, help="主阶段 top-p。")
    parser.add_argument("--tag-max-new-tokens", type=int, default=256, help="语义标签生成 token 上限。")
    parser.add_argument("--tag-temperature", type=float, default=0.0, help="语义标签生成温度。")
    parser.add_argument("--tag-top-p", type=float, default=0.9, help="语义标签生成 top-p。")
    parser.add_argument(
        "--modes",
        nargs="*",
        default=["lexical", "semantic", "semantic_rerank", "semantic_mmr"],
        choices=["lexical", "semantic", "semantic_rerank", "semantic_mmr"],
        help="要评测的检索模式。",
    )
    parser.add_argument("--semantic-rerank-pool", type=int, default=24, help="语义重排时的词面召回候选池大小。")
    parser.add_argument("--semantic-mmr-pool", type=int, default=24, help="MMR 选择时的候选池大小。")
    parser.add_argument("--semantic-mmr-lambda", type=float, default=0.7, help="MMR 相似度与多样性平衡系数。")
    parser.add_argument(
        "--answer-rules",
        choices=ANSWER_RULE_CHOICES,
        default="shortest",
        help="Task7 最终答案规范化规则变体。",
    )
    return parser.parse_args()


def _chat_url(api_base: str) -> str:
    return f"{api_base.rstrip('/')}/v1/chat/completions"


def resolve_project_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    src_candidate = (Path(__file__).resolve().parent / path).resolve()
    if src_candidate.exists():
        return src_candidate
    project_candidate = (PROJECT_ROOT / path).resolve()
    if project_candidate.exists():
        return project_candidate
    return path.resolve()


def build_tag_prompt(sample_input: str) -> str:
    return (
        "You are analyzing a Jeopardy-style clue for retrieval only.\n"
        "Do not answer the clue.\n"
        "Return compact JSON only with these keys:\n"
        "question_type, answer_type, knowledge_domain, reasoning_style, constraints, risk_flags, key_terms.\n\n"
        "[Rules]\n"
        "1. Every value must be a JSON array of short lowercase strings.\n"
        "2. Use 1 to 4 labels per field.\n"
        "3. key_terms should be clue words or category words, not the answer.\n"
        "4. constraints should capture things like full_name, full_title, avoid_abbrev, short_answer, category_heavy, title_match.\n"
        "5. risk_flags should capture likely failure modes such as partial_answer_risk, entity_confusion, multiple_candidates, synonym_risk, title_confusion.\n"
        "6. Output valid JSON only.\n\n"
        "[Clue]\n"
        f"{sample_input}\n"
    )


def extract_json_object(text: str) -> Dict[str, Any]:
    content = (text or "").strip()
    if not content:
        return {}
    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}
    try:
        data = json.loads(content[start : end + 1])
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        return {}
    return data


def normalize_tag_values(values: Any) -> List[str]:
    if isinstance(values, list):
        raw_values = values
    elif isinstance(values, str):
        raw_values = [values]
    else:
        raw_values = []
    normalized: List[str] = []
    seen = set()
    for item in raw_values:
        value = re.sub(r"\s+", " ", str(item).strip().lower())
        value = value.strip(" \t\r\n\"'`")
        if not value:
            continue
        if value not in seen:
            normalized.append(value)
            seen.add(value)
    return normalized


def normalize_tag_object(data: Dict[str, Any]) -> Dict[str, List[str]]:
    normalized: Dict[str, List[str]] = {}
    for key in TAG_KEYS:
        normalized[key] = normalize_tag_values(data.get(key))
    return normalized


def request_semantic_tags(sample_input: str, args: argparse.Namespace) -> Dict[str, List[str]]:
    payload = {
        "model": args.model_name,
        "messages": [{"role": "user", "content": build_tag_prompt(sample_input)}],
        "max_tokens": args.tag_max_new_tokens,
        "temperature": args.tag_temperature,
        "top_p": args.tag_top_p,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    resp = requests.post(_chat_url(args.api_base), json=payload, timeout=args.timeout)
    resp.raise_for_status()
    data = resp.json()
    choices = data.get("choices") or []
    if not choices:
        return normalize_tag_object({})
    message = choices[0].get("message", {})
    content = message.get("content", "")
    if isinstance(content, list):
        text_parts: List[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(str(item.get("text", "")))
            elif isinstance(item, str):
                text_parts.append(item)
        content = "\n".join(text_parts)
    return normalize_tag_object(extract_json_object(str(content or "")))


def load_jsonl_by_id(path: Path) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[str(row["id"])] = row
    return rows


def ensure_tags(
    rows: Sequence[Dict[str, Any]],
    cache_path: Path,
    args: argparse.Namespace,
    label: str,
    seed_cache_path: Path | None = None,
) -> Dict[str, Dict[str, Any]]:
    existing: Dict[str, Dict[str, Any]] = {}
    if seed_cache_path is not None and seed_cache_path.exists():
        existing.update(load_jsonl_by_id(seed_cache_path))
    existing.update(load_jsonl_by_id(cache_path))
    pending = [row for row in rows if str(row["id"]) not in existing]
    if not pending:
        return existing

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    generated_rows: List[Dict[str, Any]] = []

    def run_one(row: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "id": str(row["id"]),
            "input": str(row["input"]),
            "semantic_tags": request_semantic_tags(str(row["input"]), args),
        }

    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futures = {ex.submit(run_one, row): str(row["id"]) for row in pending}
        done = 0
        total = len(pending)
        with tqdm(total=total, desc=f"Task7 tags {label}", unit="sample") as pbar:
            for future in as_completed(futures):
                generated_rows.append(future.result())
                done += 1
                pbar.update(1)
                if done % 20 == 0 or done == total:
                    pbar.set_postfix_str(f"done={done}/{total}")
                    print(
                        json.dumps(
                            {
                                "event": "tag_progress",
                                "label": label,
                                "done": done,
                                "total": total,
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )

    generated_rows.sort(key=lambda item: item["id"])
    with cache_path.open("a", encoding="utf-8") as f:
        for row in generated_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            existing[row["id"]] = row
    return existing


def tokenize_values(values: Sequence[str]) -> set[str]:
    tokens: set[str] = set()
    for value in values:
        tokens.update(WORD_PATTERN.findall(value.lower()))
    return tokens


def field_overlap_score(left: Sequence[str], right: Sequence[str]) -> float:
    left_tokens = tokenize_values(left)
    right_tokens = tokenize_values(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / ((len(left_tokens) * len(right_tokens)) ** 0.5)


def semantic_overlap_score(query_tags: Dict[str, List[str]], bank_tags: Dict[str, List[str]], query_input: str, bank_input: str) -> float:
    score = 0.0
    for key, weight in TAG_FIELD_WEIGHTS.items():
        score += weight * field_overlap_score(query_tags.get(key, []), bank_tags.get(key, []))
    score += 0.75 * word_overlap(query_input, bank_input)
    return score


def bank_to_bank_similarity(left: Dict[str, Any], right: Dict[str, Any], bank_tags_by_id: Dict[str, Dict[str, Any]]) -> float:
    left_id = str(left.get("id"))
    right_id = str(right.get("id"))
    left_tags = bank_tags_by_id.get(left_id, {}).get("semantic_tags", {})
    right_tags = bank_tags_by_id.get(right_id, {}).get("semantic_tags", {})
    score = 0.0
    for key, weight in TAG_FIELD_WEIGHTS.items():
        score += weight * field_overlap_score(left_tags.get(key, []), right_tags.get(key, []))
    score += 0.75 * word_overlap(str(left.get("input", "")), str(right.get("input", "")))
    return score


def mmr_select_rows(
    ranked_pool: Sequence[Dict[str, Any]],
    retrieval_k: int,
    bank_tags_by_id: Dict[str, Dict[str, Any]],
    mmr_lambda: float,
) -> List[Dict[str, Any]]:
    if len(ranked_pool) <= retrieval_k:
        return list(ranked_pool)
    selected: List[Dict[str, Any]] = []
    candidates = list(ranked_pool)
    while candidates and len(selected) < retrieval_k:
        best_row = None
        best_score = None
        for row in candidates:
            relevance = float(row.get("_query_score", 0.0))
            if not selected:
                mmr_score = relevance
            else:
                redundancy = max(bank_to_bank_similarity(row, chosen, bank_tags_by_id) for chosen in selected)
                mmr_score = mmr_lambda * relevance - (1.0 - mmr_lambda) * redundancy
            if best_score is None or mmr_score > best_score:
                best_score = mmr_score
                best_row = row
        if best_row is None:
            break
        selected.append(best_row)
        candidates = [row for row in candidates if str(row.get("id")) != str(best_row.get("id"))]
    selected_ids = {str(row.get("id")) for row in selected}
    tail = [row for row in ranked_pool if str(row.get("id")) not in selected_ids]
    return selected + tail


def rank_bank_rows(
    sample_input: str,
    bank_rows: Sequence[Dict[str, Any]],
    mode: str,
    query_tags: Dict[str, List[str]] | None,
    bank_tags_by_id: Dict[str, Dict[str, Any]],
    semantic_rerank_pool: int,
    semantic_mmr_pool: int,
    retrieval_k: int,
    semantic_mmr_lambda: float,
) -> List[Dict[str, Any]]:
    if mode == "lexical":
        return sorted(
            bank_rows,
            key=lambda row: word_overlap(sample_input, str(row.get("input", ""))),
            reverse=True,
        )
    if mode == "semantic_rerank":
        lexical_ranked = sorted(
            bank_rows,
            key=lambda row: word_overlap(sample_input, str(row.get("input", ""))),
            reverse=True,
        )
        rerank_pool = lexical_ranked[: max(1, min(len(lexical_ranked), semantic_rerank_pool))]
        reranked = sorted(
            rerank_pool,
            key=lambda row: semantic_overlap_score(
                query_tags or {},
                bank_tags_by_id.get(str(row.get("id")), {}).get("semantic_tags", {}),
                sample_input,
                str(row.get("input", "")),
            ),
            reverse=True,
        )
        return reranked + lexical_ranked[len(rerank_pool) :]
    semantic_ranked = sorted(
        (
            {
                **row,
                "_query_score": semantic_overlap_score(
                    query_tags or {},
                    bank_tags_by_id.get(str(row.get("id")), {}).get("semantic_tags", {}),
                    sample_input,
                    str(row.get("input", "")),
                ),
            }
            for row in bank_rows
        ),
        key=lambda row: float(row.get("_query_score", 0.0)),
        reverse=True,
    )
    if mode == "semantic_mmr":
        pool = semantic_ranked[: max(1, min(len(semantic_ranked), semantic_mmr_pool))]
        selected = mmr_select_rows(pool, retrieval_k, bank_tags_by_id, semantic_mmr_lambda)
        selected_ids = {str(row.get("id")) for row in selected}
        tail = [row for row in semantic_ranked if str(row.get("id")) not in selected_ids]
        return selected + tail
    return semantic_ranked


def build_answer_rules(answer_rules: str) -> str:
    if answer_rules == "preserve_articles":
        return (
            "[Answer Rules]\n"
            "1. Study the retrieved successful reasoning patterns, but do not copy their final answers unless they truly fit the new clue.\n"
            "2. Return one lowercase answer only.\n"
            "3. Prefer the shortest unambiguous canonical answer, but keep words needed to identify a title, term, phrase, group, or named work.\n"
            "4. Preserve leading articles like 'a', 'an', or 'the' when they are part of a title, named phrase, set phrase, or natural Jeopardy response.\n"
            "5. Do not expand a correct minimal answer into a fuller name, more common entity, or nearby category unless the clue requires that exact form.\n"
            "6. Do not add articles only to make the answer sound grammatical.\n"
            "7. Do not output explanation, alternatives, or multiple candidates.\n\n"
            "Answer only:\n"
        )
    return (
        "[Answer Rules]\n"
        "1. Study the retrieved successful reasoning patterns, but do not copy their final answers unless they truly fit the new clue.\n"
        "2. Return one lowercase answer only.\n"
        "3. Use the shortest unambiguous canonical answer.\n"
        "4. Omit leading articles like 'a', 'an', 'the'.\n"
        "5. Do not expand a correct minimal answer into a fuller name, more common entity, or nearby category unless the clue requires that exact form.\n"
        "6. Do not output explanation, alternatives, or multiple candidates.\n\n"
        "Answer only:\n"
    )


def estimate_tokens(text: str, tokenizer: Any) -> int:
    if tokenizer is None:
        return max(1, len(text) // 4)
    return len(tokenizer.encode(text, add_special_tokens=False))


def build_success_context_block(row: Dict[str, Any], idx: int, reasoning_char_limit: int) -> str:
    reasoning = str(row.get("thinking", "")).strip()
    if reasoning_char_limit > 0:
        reasoning = reasoning[:reasoning_char_limit].strip()
    return (
        f"[Long Successful Reasoning Case {idx}]\n"
        f"Question:\n{str(row.get('input', '')).strip()}\n"
        f"Reasoning:\n{reasoning}\n"
        f"Final Answer:\n{str(row.get('answer', '')).strip()}\n"
    )


def select_success_context(
    *,
    ranked_bank: Sequence[Dict[str, Any]],
    bank_rows: Sequence[Dict[str, Any]],
    tokenizer: Any,
    max_input_tokens: int,
    target_context_tokens: int,
    reserved_generation_tokens: int,
    reasoning_char_limit: int,
) -> tuple[str, int, int]:
    ordered: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for row in list(ranked_bank) + list(bank_rows):
        row_id = str(row.get("id", ""))
        key = row_id or str(len(seen))
        if key in seen:
            continue
        if not str(row.get("thinking", "")).strip():
            continue
        seen.add(key)
        ordered.append(row)
    if not ordered:
        return "", 0, 0

    budget_for_context = max(0, max_input_tokens - reserved_generation_tokens)
    selected_blocks: List[str] = []
    used_tokens = 0
    while used_tokens < target_context_tokens:
        added = False
        for row in ordered:
            block = build_success_context_block(row, len(selected_blocks) + 1, reasoning_char_limit)
            block_tokens = estimate_tokens(block, tokenizer)
            if used_tokens + block_tokens > budget_for_context:
                continue
            selected_blocks.append(block)
            used_tokens += block_tokens
            added = True
            if used_tokens >= target_context_tokens:
                break
        if not added:
            break
    return "\n".join(selected_blocks).strip(), len(selected_blocks), used_tokens


def build_prompt_with_ranked_rows(
    dataset: Dict[str, Any],
    train_examples: Sequence[Dict[str, Any]],
    ranked_bank: Sequence[Dict[str, Any]],
    sample_input: str,
    base_few_shot: int,
    retrieval_k: int,
    reasoning_char_limit: int,
    answer_rules: str = "shortest",
    official_examples_text: str = "",
) -> str:
    task_description = "\n".join(dataset.get("Definition", []))
    examples_text = select_examples(train_examples, sample_input, base_few_shot)
    if official_examples_text.strip():
        examples_text = (
            "[Long Successful Reasoning ICL Background]\n"
            f"{official_examples_text.strip()}\n\n"
            "[Nearest Short Calibration Examples]\n"
            f"{examples_text}"
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
        f"{build_answer_rules(answer_rules)}"
    )


def run_mode(
    mode: str,
    args: argparse.Namespace,
    dataset: Dict[str, Any],
    train_examples: Sequence[Dict[str, Any]],
    val_examples: Sequence[Dict[str, Any]],
    bank_rows: Sequence[Dict[str, Any]],
    bank_tags_by_id: Dict[str, Dict[str, Any]],
    val_tags_by_id: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    cfg = make_infer_cfg(args)
    rows: List[Dict[str, Any]] = []
    start = time.time()
    tokenizer = None
    if getattr(args, "official_min_context", False):
        tokenizer_path = resolve_project_path(getattr(args, "tokenizer_path", ""))
        tokenizer = load_tokenizer(str(tokenizer_path) if tokenizer_path.exists() else None)

    def run_one(sample: Dict[str, Any]) -> Dict[str, Any]:
        sample_id = str(sample["id"])
        sample_input = str(sample["input"])
        query_tags = val_tags_by_id.get(sample_id, {}).get("semantic_tags", {})
        ranked_bank = rank_bank_rows(
            sample_input=sample_input,
            bank_rows=bank_rows,
            mode=mode,
            query_tags=query_tags,
            bank_tags_by_id=bank_tags_by_id,
            semantic_rerank_pool=args.semantic_rerank_pool,
            semantic_mmr_pool=args.semantic_mmr_pool,
            retrieval_k=args.retrieval_k,
            semantic_mmr_lambda=args.semantic_mmr_lambda,
        )
        official_examples_text = ""
        official_selected_count = 0
        official_used_tokens = 0
        if getattr(args, "official_min_context", False):
            official_examples_text, official_selected_count, official_used_tokens = select_success_context(
                ranked_bank=ranked_bank,
                bank_rows=bank_rows,
                tokenizer=tokenizer,
                max_input_tokens=args.max_input_tokens,
                target_context_tokens=args.min_context_tokens,
                reserved_generation_tokens=args.reserved_generation_tokens,
                reasoning_char_limit=args.reasoning_char_limit,
            )
        prompt = build_prompt_with_ranked_rows(
            dataset=dataset,
            train_examples=train_examples,
            ranked_bank=ranked_bank,
            sample_input=sample_input,
            base_few_shot=args.base_few_shot,
            retrieval_k=args.retrieval_k,
            reasoning_char_limit=args.reasoning_char_limit,
            answer_rules=args.answer_rules,
            official_examples_text=official_examples_text,
        )
        details = annotate_detailed(prompt, cfg)
        draft_output = details.raw_text.strip() or details.visible_text.strip()
        compressed_output = ""
        base_visible = details.visible_text.strip()
        if args.compress_answer and not looks_like_clean_final_answer(base_visible):
            try:
                compressed_output = compress_answer_no_think(sample_input, draft_output, args)
            except Exception as exc:
                compressed_output = f"compression_error: {type(exc).__name__}: {exc}"
        final_text = (
            compressed_output
            if compressed_output and not compressed_output.startswith("compression_error:")
            else (details.visible_text or details.raw_text)
        )
        prediction = normalize_answer(final_text)
        gold = sample_gold(sample)
        return {
            "id": sample_id,
            "input": sample_input,
            "prediction": prediction,
            "gold": gold,
            "correct": prediction == gold,
            "visible_text": details.visible_text.strip(),
            "raw_text": details.raw_text.strip(),
            "thinking": details.reasoning_text.strip(),
            "compressed_output": compressed_output,
            "compress_applied": bool(compressed_output),
            "retrieved_ids": [item.get("id") for item in ranked_bank[: args.retrieval_k]],
            "query_semantic_tags": query_tags,
            "official_context_tokens": official_used_tokens,
            "_audit": {
                "task_id": 7,
                "sample_id": sample_id,
                "stage": "candidate_generation",
                "mode": mode,
                "target_min_context_tokens": getattr(args, "min_context_tokens", 30000),
                "used_example_tokens": official_used_tokens,
                "selected_example_count": official_selected_count,
                "pass_min_context": official_used_tokens >= getattr(args, "min_context_tokens", 30000),
            },
        }

    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futures = {ex.submit(run_one, sample): str(sample["id"]) for sample in val_examples}
        done = 0
        total = len(val_examples)
        with tqdm(total=total, desc=f"Task7 {mode}", unit="sample") as pbar:
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
                                "mode": mode,
                                "done": done,
                                "total": total,
                                "elapsed_sec": round(time.time() - start, 2),
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )

    rows.sort(key=lambda item: item["id"])
    return rows


def summarize_mode(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    correct = sum(1 for row in rows if row["correct"])
    non_null = sum(1 for row in rows if row["prediction"] is not None)
    return {
        "accuracy": round(correct / total, 6) if rows else 0.0,
        "correct": correct,
        "total": total,
        "coverage": round(non_null / total, 6) if rows else 0.0,
        "non_null_predictions": non_null,
    }


def main() -> None:
    args = parse_args()
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
    bank_cache_path = output_dir / "bank_semantic_tags.jsonl"
    val_cache_path = output_dir / "val_semantic_tags.jsonl"

    bank_seed_cache_path = resolve_project_path(args.bank_tags_path) if args.bank_tags_path else None
    val_seed_cache_path = resolve_project_path(args.val_tags_path) if args.val_tags_path else None
    bank_tags_by_id = ensure_tags(bank_rows, bank_cache_path, args, "bank", bank_seed_cache_path)
    val_tags_by_id = ensure_tags(val_examples, val_cache_path, args, "val", val_seed_cache_path)

    summary: Dict[str, Any] = {
        "tagging": {
            "bank_size": len(bank_rows),
            "val_size": len(val_examples),
        }
    }
    for mode in args.modes:
        rows = run_mode(
            mode=mode,
            args=args,
            dataset=dataset,
            train_examples=train_examples,
            val_examples=val_examples,
            bank_rows=bank_rows,
            bank_tags_by_id=bank_tags_by_id,
            val_tags_by_id=val_tags_by_id,
        )
        mode_summary = summarize_mode(rows)
        summary[mode] = mode_summary
        write_jsonl(output_dir / f"{mode}_results.jsonl", rows)
        write_jsonl(output_dir / f"{mode}_mismatches.jsonl", [row for row in rows if not row["correct"]])

    write_json(
        output_dir / "summary.json",
        {
            **summary,
            "config": {
                "seed": args.seed,
                "base_few_shot": args.base_few_shot,
                "retrieval_k": args.retrieval_k,
                "reasoning_char_limit": args.reasoning_char_limit,
                "max_new_tokens": args.max_new_tokens,
                "compress_answer": bool(args.compress_answer),
                "compress_max_new_tokens": args.compress_max_new_tokens if args.compress_answer else 0,
                "compress_temperature": args.compress_temperature if args.compress_answer else 0.0,
                "compress_top_p": args.compress_top_p if args.compress_answer else 0.0,
                "tag_max_new_tokens": args.tag_max_new_tokens,
                "tag_temperature": args.tag_temperature,
                "tag_top_p": args.tag_top_p,
                "modes": args.modes,
                "bank_tags_path": str(bank_seed_cache_path) if bank_seed_cache_path else "",
                "val_tags_path": str(val_seed_cache_path) if val_seed_cache_path else "",
                "semantic_rerank_pool": args.semantic_rerank_pool,
                "semantic_mmr_pool": args.semantic_mmr_pool,
                "semantic_mmr_lambda": args.semantic_mmr_lambda,
                "answer_rules": args.answer_rules,
            },
        },
    )
    print(json.dumps({"event": "done", "summary": summary}, ensure_ascii=False), flush=True)


def predict_task7_semantic_mmr(
    task_description: str,
    examples: Sequence[Dict[str, Any]],
    test_samples: Sequence[Dict[str, Any]],
    args: Any,
    cache_dir: Path,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    retrieval_mode = getattr(args, "task7_retrieval_mode", "semantic_mmr")
    if retrieval_mode not in RETRIEVAL_MODE_CHOICES:
        raise ValueError(f"Unsupported Task7 retrieval mode: {retrieval_mode}")
    runtime_args = SimpleNamespace(
        api_base=getattr(args, "api_base"),
        model_name=getattr(args, "model_name"),
        timeout=getattr(args, "timeout"),
        max_new_tokens=getattr(args, "max_new_tokens"),
        temperature=getattr(args, "temperature"),
        top_p=getattr(args, "top_p"),
        max_workers=max(4, getattr(args, "client_concurrency", getattr(args, "max_workers", 1))),
        compress_max_new_tokens=getattr(args, "task7_compress_max_new_tokens", 64),
        compress_temperature=getattr(args, "task7_compress_temperature", 0.0),
        compress_top_p=getattr(args, "task7_compress_top_p", 0.8),
        tag_max_new_tokens=getattr(args, "task7_tag_max_new_tokens", 256),
        tag_temperature=getattr(args, "task7_tag_temperature", 0.0),
        tag_top_p=getattr(args, "task7_tag_top_p", 0.9),
        enable_thinking=getattr(args, "enable_thinking", None),
        answer_rules=getattr(args, "task7_answer_rules", "shortest"),
    )

    bank_path = resolve_project_path(getattr(args, "task7_bank_path", DEFAULT_BANK_PATH))
    bank_tags_path = resolve_project_path(getattr(args, "task7_bank_tags_path", DEFAULT_BANK_TAGS_PATH))
    if not bank_path.exists():
        raise FileNotFoundError(f"Task7 success bank not found: {bank_path}")
    if not bank_tags_path.exists():
        raise FileNotFoundError(f"Task7 bank semantic tags not found: {bank_tags_path}")

    bank_rows = [json.loads(line) for line in bank_path.open("r", encoding="utf-8") if line.strip()]
    bank_tags_by_id = load_jsonl_by_id(bank_tags_path)
    if len(bank_tags_by_id) < len(bank_rows):
        raise RuntimeError(
            f"Task7 bank tags incomplete: tags={len(bank_tags_by_id)} bank_rows={len(bank_rows)} path={bank_tags_path}"
        )

    test_cache_path = cache_dir / "test_semantic_tags.jsonl"
    test_seed_cache_path = None
    if getattr(args, "task7_test_tags_path", ""):
        test_cache_path = resolve_project_path(getattr(args, "task7_test_tags_path"))
    test_tags_by_id = ensure_tags(test_samples, test_cache_path, runtime_args, "task7_test", test_seed_cache_path)
    cfg = make_infer_cfg(runtime_args)
    dataset = {"Definition": [task_description]}
    rows: List[Dict[str, Any]] = []
    start = time.time()
    client_concurrency = max(1, min(getattr(args, "client_concurrency", 1), len(test_samples) or 1))
    tokenizer = None
    if getattr(args, "official_min_context", False):
        tokenizer_path = resolve_project_path(getattr(args, "tokenizer_path", ""))
        tokenizer = load_tokenizer(str(tokenizer_path) if tokenizer_path.exists() else None)

    def run_one(sample: Dict[str, Any]) -> Dict[str, Any]:
        sample_id = str(sample["id"])
        sample_input = str(sample["input"])
        query_tags = test_tags_by_id.get(sample_id, {}).get("semantic_tags", {})
        ranked_bank = rank_bank_rows(
            sample_input=sample_input,
            bank_rows=bank_rows,
            mode=retrieval_mode,
            query_tags=query_tags,
            bank_tags_by_id=bank_tags_by_id,
            semantic_rerank_pool=getattr(args, "task7_semantic_rerank_pool", 24),
            semantic_mmr_pool=getattr(args, "task7_semantic_mmr_pool", 24),
            retrieval_k=getattr(args, "task7_retrieval_k", 3),
            semantic_mmr_lambda=getattr(args, "task7_semantic_mmr_lambda", 0.7),
        )
        official_examples_text = ""
        official_selected_count = 0
        official_used_tokens = 0
        if getattr(args, "official_min_context", False):
            official_examples_text, official_selected_count, official_used_tokens = select_success_context(
                ranked_bank=ranked_bank,
                bank_rows=bank_rows,
                tokenizer=tokenizer,
                max_input_tokens=getattr(args, "max_input_tokens", 60000),
                target_context_tokens=getattr(args, "min_context_tokens", 30000),
                reserved_generation_tokens=getattr(args, "reserved_generation_tokens", 2048),
                reasoning_char_limit=getattr(args, "task7_reasoning_char_limit", 800),
            )
        prompt = build_prompt_with_ranked_rows(
            dataset=dataset,
            train_examples=examples,
            ranked_bank=ranked_bank,
            sample_input=sample_input,
            base_few_shot=getattr(args, "task7_base_few_shot", 4),
            retrieval_k=getattr(args, "task7_retrieval_k", 3),
            reasoning_char_limit=getattr(args, "task7_reasoning_char_limit", 800),
            answer_rules=getattr(args, "task7_answer_rules", "shortest"),
            official_examples_text=official_examples_text,
        )
        details = annotate_detailed(prompt, cfg)
        draft_output = details.raw_text.strip() or details.visible_text.strip()
        compressed_output = ""
        base_visible = details.visible_text.strip()
        if getattr(args, "task7_compress_answer", True) and not looks_like_clean_final_answer(base_visible):
            try:
                compressed_output = compress_answer_no_think(sample_input, draft_output, runtime_args)
            except Exception as exc:
                compressed_output = f"compression_error: {type(exc).__name__}: {exc}"
        final_text = (
            compressed_output
            if compressed_output and not compressed_output.startswith("compression_error:")
            else (details.visible_text or details.raw_text)
        )
        prediction = normalize_answer(final_text)
        return {
            "test_sample_id": sample_id,
            "prediction": prediction if prediction is not None else "",
            "_debug": {
                "retrieved_ids": [item.get("id") for item in ranked_bank[: getattr(args, "task7_retrieval_k", 3)]],
                "query_semantic_tags": query_tags,
                "compressed_output": compressed_output,
                "visible_text": details.visible_text.strip(),
                "official_context_tokens": official_used_tokens,
            },
            "_audit": {
                "task_id": 7,
                "sample_id": sample_id,
                "stage": "candidate_generation",
                "mode": retrieval_mode,
                "target_min_context_tokens": getattr(args, "min_context_tokens", 30000),
                "used_example_tokens": official_used_tokens,
                "selected_example_count": official_selected_count,
                "pass_min_context": official_used_tokens >= getattr(args, "min_context_tokens", 30000),
            },
        }

    with ThreadPoolExecutor(max_workers=client_concurrency) as ex:
        futures = {ex.submit(run_one, sample): str(sample.get("id")) for sample in test_samples}
        done = 0
        total = len(test_samples)
        with tqdm(total=total, desc=f"Task7 {retrieval_mode}", unit="sample") as pbar:
            for future in as_completed(futures):
                rows.append(future.result())
                done += 1
                pbar.update(1)
                if done % 20 == 0 or done == total:
                    pbar.set_postfix_str(f"elapsed={round(time.time() - start, 1)}s")
                    print(
                        json.dumps(
                            {
                                "event": "task7_submission_progress",
                                "done": done,
                                "total": total,
                                "elapsed_sec": round(time.time() - start, 2),
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )

    rows.sort(key=lambda item: item["test_sample_id"])
    debug_rows = []
    audit_rows = []
    prediction_rows = []
    for row in rows:
        debug_rows.append({"test_sample_id": row["test_sample_id"], **row["_debug"]})
        if getattr(args, "official_min_context", False):
            audit_rows.append(row["_audit"])
        prediction_rows.append({"test_sample_id": row["test_sample_id"], "prediction": row["prediction"]})
    write_jsonl(cache_dir / "task7_debug.jsonl", debug_rows)
    if audit_rows:
        write_jsonl(cache_dir / "context_audit_task7.jsonl", audit_rows)
    metadata = {
        "mode": retrieval_mode,
        "bank_path": str(bank_path),
        "bank_tags_path": str(bank_tags_path),
        "test_tags_path": str(test_cache_path),
        "bank_size": len(bank_rows),
        "test_size": len(test_samples),
        "task7_base_few_shot": getattr(args, "task7_base_few_shot", 4),
        "task7_retrieval_k": getattr(args, "task7_retrieval_k", 3),
        "task7_reasoning_char_limit": getattr(args, "task7_reasoning_char_limit", 800),
        "task7_semantic_rerank_pool": getattr(args, "task7_semantic_rerank_pool", 24),
        "task7_semantic_mmr_pool": getattr(args, "task7_semantic_mmr_pool", 24),
        "task7_semantic_mmr_lambda": getattr(args, "task7_semantic_mmr_lambda", 0.7),
        "task7_answer_rules": getattr(args, "task7_answer_rules", "shortest"),
        "task7_compress_answer": bool(getattr(args, "task7_compress_answer", True)),
        "task7_compress_max_new_tokens": getattr(args, "task7_compress_max_new_tokens", 64),
        "task7_compress_temperature": getattr(args, "task7_compress_temperature", 0.0),
        "task7_compress_top_p": getattr(args, "task7_compress_top_p", 0.8),
        "official_min_context": bool(getattr(args, "official_min_context", False)),
        "context_audit_file": str(cache_dir / "context_audit_task7.jsonl") if audit_rows else "",
        "min_used_example_tokens": min((int(row["used_example_tokens"]) for row in audit_rows), default=0),
        "context_pass_rows": sum(1 for row in audit_rows if row["pass_min_context"]),
    }
    write_json(cache_dir / "task7_metadata.json", metadata)
    return prediction_rows, metadata


if __name__ == "__main__":
    main()
