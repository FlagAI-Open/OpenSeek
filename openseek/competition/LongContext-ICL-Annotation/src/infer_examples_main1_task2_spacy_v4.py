"""
main1 task2 推理：**v4** — 零样本；金标准对齐交给 spaCy 标注表，大模型只做 **计数统计**。

相对 ``infer_examples_main1_task2_spacy_v3.py``：

- 去掉长篇 benchmark 语言学规则与冗长 Legend。
- 题干仍为 OpenSeek-2；**Quoted sentence** 已由脚本用 ``en_core_web_sm`` 分词并标注 ``pos_`` / ``tag_``。
- 模型任务：**仅根据表中 ``pos_``** 按本条要求统计个数（名词 vs 动词分支规则见下文常量）。
- **评测**仍与数据集 ``output`` 字符串完全一致比对（OpenSeek 金标未必等于纯 spaCy 计数；本管道用于实验对比）。

计数约定（与 spaCy ``token.pos_`` 对齐）：

- **nouns**：``pos_`` 为 ``NOUN`` 或 ``PROPN`` 的行数。
- **verbs**：``pos_`` 为 ``VERB`` 的行数（**不计** ``AUX``，以免把 *is/was/have* 等与金标常见口径混为一谈——若你要连 ``AUX`` 一并计入，可改 ``_TASK_VERB_POS``）。

运行可加 ``--log`` / ``--log_prompt`` 查看每条 oracle / 金标 / 模型输出及本轮汇总。
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Callable

from tqdm import tqdm

from method_hyb import annotate_nvidia as annotate

# ---------------------------------------------------------------------------

PromptBuilder = Callable[[str, str], str]

TASK_PROMPT_BUILDERS: dict[int, PromptBuilder] = {}

_DEFAULT_ANNOTATION_GUIDELINES = (
    "### Annotation guidelines\n"
    "1. Optional brief reasoning **outside** `<label>...</label>`.\n"
    "2. Final answer: exactly one `<label>...</label>` pair.\n"
    "3. Inside `<label>`, output **only** one non-negative integer (decimal digits).\n"
)

_TASK_VERB_POS = frozenset({"VERB"})
_TASK_NOUN_POS = frozenset({"NOUN", "PROPN"})


def _task_prompt_shell_zero_shot(
    task_specific_guidelines: str,
    task_description: str,
    text2annotate: str,
    *,
    annotation_guidelines: str | None = None,
) -> str:
    guidelines = (
        annotation_guidelines if annotation_guidelines is not None else _DEFAULT_ANNOTATION_GUIDELINES
    )
    return (
        "### Role\n"
        "You aggregate **counts from the spaCy token table only**. Treat each non-header row as one token; use its "
        "**`pos_`** column for classification. Do not invent tokens or POS labels beyond this table.\n\n"
        f"{task_specific_guidelines}\n"
        "### Official task definition\n"
        f"{task_description}\n\n"
        "### Your input (answer for this instance only)\n"
        f"{text2annotate}\n\n"
        f"{guidelines}"
    )


def register_task_prompt(task_id: int, builder: PromptBuilder) -> None:
    TASK_PROMPT_BUILDERS[task_id] = builder


def _build_prompt_default(task_description: str, text2annotate: str) -> str:
    return (
        "### Core Task\n"
        f"{task_description}\n\n### Input\n{text2annotate}\n\n### Output\n<label>integer</label>\n"
    )


def build_prompt(
    task_description: str,
    text2annotate: str,
    *,
    task_id: int | None = None,
) -> str:
    if task_id is not None:
        custom = TASK_PROMPT_BUILDERS.get(task_id)
        if custom is not None:
            return custom(task_description, text2annotate)
    return _build_prompt_default(task_description, text2annotate)


# ---------------------------------------------------------------------------
# spaCy
# ---------------------------------------------------------------------------

_NLP = None


def _get_nlp():
    global _NLP  # noqa: PLW0603
    if _NLP is None:
        import spacy

        _NLP = spacy.load("en_core_web_sm")
    return _NLP


def _task2_count_target_from_input(text: str) -> str | None:
    m = re.search(
        r"Count\s+the\s+number\s+of\s+(nouns|verbs)\s+in\s+this\s+sentence",
        str(text),
        flags=re.IGNORECASE,
    )
    return None if not m else m.group(1).lower()


def _align_task2_official_definition(task_description: str, text2annotate: str) -> str:
    base = str(task_description).strip()
    target = _task2_count_target_from_input(text2annotate)
    if target == "verbs":
        return (
            f"{base}\n\n"
            "**For this instance:** report **how many verbs** per the **spaCy counting rule** below (not a noun count)."
        )
    if target == "nouns":
        return (
            f"{base}\n\n"
            "**For this instance:** report **how many nouns** per the **spaCy counting rule** below (not a verb count)."
        )
    return f"{base}\n\n**For this instance:** follow **Your input** for noun vs verb; counting rule uses **`pos_`** below."


def _extract_sentence_from_task2_input(text: str) -> str | None:
    t = str(text).strip()
    m = re.search(r"Sentence:\s*'(.*)'\.\s*Count\b", t, flags=re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r'Sentence:\s*"(.*)"\.\s*Count\b', t, flags=re.DOTALL)
    if m:
        return m.group(1).strip()
    return None


def _task_specific_stats_only(target: str | None) -> str:
    verb_rule = (
        "- **Verb count (spaCy-only):** Count **one** for each table row whose **`pos_`** equals **`VERB`** "
        f"(accepted verb POS set for this script: `{sorted(_TASK_VERB_POS)}`). "
        "**Do not** count **`AUX`** rows toward the verb total in this v4 pipeline.\n"
    )
    noun_rule = (
        "- **Noun count (spaCy-only):** Count **one** for each table row whose **`pos_`** is **`NOUN`** or **`PROPN`** "
        f"(set `{sorted(_TASK_NOUN_POS)}`). **Do not** count other `pos_` values.\n"
    )
    if target == "verbs":
        header = "### Task instruction (openseek-2 - spaCy statistics only, verbs)\n"
        return header + verb_rule + "- Ignore punctuation-only tokens unless they appear as rows with countable `pos_`.\n"
    if target == "nouns":
        header = "### Task instruction (openseek-2 - spaCy statistics only, nouns)\n"
        return header + noun_rule + "- Ignore rows that are not `NOUN`/`PROPN`.\n"
    return (
        "### Task instruction (openseek-2 - spaCy statistics only)\n"
        + noun_rule
        + verb_rule
        + "- Read **Your input** to see whether this item wants noun count or verb count; apply **exactly one** rule above.\n"
    )


def _spacy_table_block(sentence: str) -> str:
    if not sentence:
        return "### spaCy token table (`en_core_web_sm`)\n*(sentence extraction failed - empty table)*\n"
    nlp = _get_nlp()
    doc = nlp(sentence)
    lines = ["{:24} -> {:10} ({})".format(tok.text, tok.pos_, tok.tag_) for tok in doc]
    body = "\n".join(lines) if lines else "(empty)"
    return (
        "### spaCy token table (`en_core_web_sm`)\n"
        "Each row: **surface** -> **`pos_`** (Universal coarse POS) -> **`tag_`** (English fine tag).\n\n"
        "```text\n"
        f"{body}\n"
        "```\n"
    )


def _compose_guidelines(text2annotate: str) -> str:
    target = _task2_count_target_from_input(text2annotate)
    sent = _extract_sentence_from_task2_input(text2annotate) or ""
    return _task_specific_stats_only(target) + "\n" + _spacy_table_block(sent)


def _spacy_oracle_count(input_text: str) -> tuple[str | None, int]:
    """
    与提示词一致的确定性计数：``verbs`` → ``_TASK_VERB_POS``；``nouns`` → ``_TASK_NOUN_POS``。
    无法解析句子时返回 ``(..., -1)``。
    """
    target = _task2_count_target_from_input(input_text)
    sent = _extract_sentence_from_task2_input(input_text)
    if not sent:
        return target, -1
    nlp = _get_nlp()
    doc = nlp(sent)
    if target == "verbs":
        return target, sum(1 for t in doc if t.pos_ in _TASK_VERB_POS)
    if target == "nouns":
        return target, sum(1 for t in doc if t.pos_ in _TASK_NOUN_POS)
    return target, -1


def _prompt_openseek_2_spacy_v4(task_description: str, text2annotate: str) -> str:
    guidelines = _compose_guidelines(text2annotate)
    definition_for_prompt = _align_task2_official_definition(task_description, text2annotate)
    return _task_prompt_shell_zero_shot(guidelines, definition_for_prompt, text2annotate)


# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent

TASK_DATA_FILES: dict[int, str] = {
    1: "openseek-1_closest_integers.json",
    2: "openseek-2_count_nouns_verbs.json",
    3: "openseek-3_collatz_conjecture.json",
    4: "openseek-4_conala_concat_strings.json",
    5: "openseek-5_semeval_2018_task1_tweet_sadness_detection.json",
    6: "openseek-6_mnli_same_genre_classification.json",
    7: "openseek-7_jeopardy_answer_generation_all.json",
    8: "openseek-8_kernel_generation.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="main1 task2 spacy v4：零样本；仅按 spaCy pos_ 统计，LM 负责数数。"
    )
    parser.add_argument("--task_start", type=int, default=2)
    parser.add_argument("--task_end", type=int, default=2)
    parser.add_argument("--examples_limit", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="examples_main1")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry_wait_seconds", type=float, default=2.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--log",
        action="store_true",
        help="每条样本打印：target、spaCy-oracle、金标、模型输出、与金标/oracle 是否一致（经 tqdm.write，不与进度条打架）。",
    )
    parser.add_argument(
        "--log_prompt",
        action="store_true",
        help="在每条样本上额外打印完整 user prompt（很长；建议加 --examples_limit）.",
    )
    return parser.parse_args()


def _task_json_path(task_id: int) -> Path:
    if task_id not in TASK_DATA_FILES:
        raise ValueError(f"task_id should be in [1, 8], but got {task_id}.")
    return REPO_ROOT / "data" / TASK_DATA_FILES[task_id]


def _resolve_output_dir(output_dir: str) -> Path:
    p = Path(output_dir)
    return p.resolve() if p.is_absolute() else (REPO_ROOT / p).resolve()


def _normalize_text(text: str) -> str:
    return " ".join(str(text).strip().split())


def _extract_first_unsigned_int(text: str) -> int | None:
    """从模型输出中取首个非负整数（兼容 `<label>123</label>`）。"""
    s = str(text or "")
    m = re.search(r"<label>\s*(\d+)\s*</label>", s, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.search(r"\b(\d+)\b", s)
    return int(m.group(1)) if m else None


def _extract_output(output_value: object) -> str:
    if isinstance(output_value, list):
        return "" if not output_value else str(output_value[0]).strip()
    return str(output_value).strip()


def _load_done_ids(output_file: Path) -> set[str]:
    done: set[str] = set()
    if not output_file.exists():
        return done
    with output_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            eid = str(row.get("example_id", "")).strip()
            if eid:
                done.add(eid)
    return done


def _compute_metrics_from_jsonl(output_file: Path) -> tuple[int, int, float]:
    total = matched = 0
    if not output_file.exists():
        return total, matched, 0.0
    with output_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            total += 1
            if bool(row.get("is_match", False)):
                matched += 1
    acc = (matched / total) if total else 0.0
    return total, matched, acc


def run_task(
    task_id: int,
    output_dir: Path,
    examples_limit: int = 0,
    retries: int = 3,
    retry_wait_seconds: float = 2.0,
    resume: bool = False,
    *,
    log: bool = False,
    log_prompt: bool = False,
) -> dict:
    task_file = _task_json_path(task_id)
    with task_file.open("r", encoding="utf-8") as f:
        task_dict = json.load(f)

    task_name = task_dict["task_name"]
    task_description = task_dict["Definition"][0]
    all_examples = task_dict["examples"]
    if examples_limit > 0:
        all_examples = all_examples[:examples_limit]

    output_file = output_dir / f"openseek-{task_id}-examples-main1-compare_spacy_v4_zeroshot.jsonl"
    done_ids = _load_done_ids(output_file) if resume else set()
    mode = "a" if resume else "w"

    if done_ids:
        print(f"[断点续跑] task={task_id}, 已完成 {len(done_ids)} 条")

    processed = oracle_hits = gold_hits = 0

    with output_file.open(mode, encoding="utf-8") as wf:
        for example in tqdm(all_examples, desc=f"Task {task_id} task2 spacy v4 (stats-only): {task_name}"):
            example_id = str(example.get("id", "")).strip()
            if resume and example_id in done_ids:
                continue

            input_text = example.get("input", "")
            expected = _extract_output(example.get("output", ""))

            input_prompt = build_prompt(task_description, input_text, task_id=task_id)

            tgt, oracle_n = _spacy_oracle_count(input_text)

            prediction = ""
            for attempt in range(1, retries + 1):
                try:
                    raw_prediction = annotate(input_prompt)
                    prediction = "" if raw_prediction is None else str(raw_prediction).strip()
                    break
                except Exception as e:  # noqa: BLE001
                    if attempt >= retries:
                        print(f"[推理失败] task={task_id} id={example_id} attempt={attempt}/{retries} err={e}")
                    else:
                        print(f"[重试] task={task_id} id={example_id} attempt={attempt}/{retries} err={e}")
                        time.sleep(retry_wait_seconds)

            is_match = _normalize_text(prediction) == _normalize_text(expected)
            pred_int = _extract_first_unsigned_int(prediction)
            oracle_ok = oracle_n >= 0 and pred_int is not None and pred_int == oracle_n
            if log or log_prompt:
                sep = "-" * 72
                if log_prompt:
                    tqdm.write(f"{sep}\n[PROMPT] example_id={example_id}\n{input_prompt}\n{sep}")
                if log:
                    tqdm.write(
                        "[v4] "
                        f"id={example_id} "
                        f"target={tgt!s} "
                        f"oracle(spacy)={oracle_n} "
                        f"gold={expected!r} "
                        f"pred={prediction!r} "
                        f"match_gold={is_match} "
                        f"match_oracle={oracle_ok}"
                    )
            wf.write(
                json.dumps(
                    {
                        "example_id": example_id,
                        "input": input_text,
                        "expected_output": expected,
                        "model_output": prediction,
                        "is_match": is_match,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            wf.flush()
            done_ids.add(example_id)

            processed += 1
            if oracle_ok:
                oracle_hits += 1
            if is_match:
                gold_hits += 1

    total, match_count, accuracy = _compute_metrics_from_jsonl(output_file)
    if log and processed > 0:
        tqdm.write(
            f"[v4 batch] task={task_id} processed={processed} "
            f"match_gold={gold_hits}/{processed} ({gold_hits / processed:.2%}) "
            f"match_oracle={oracle_hits}/{processed} ({oracle_hits / processed:.2%})"
        )
    print(
        f"[保存完成] task2 spacy v4 zeroshot task={task_id} total={total} matched={match_count} "
        f"accuracy={accuracy:.2%} file={output_file}"
    )
    return {
        "task_id": task_id,
        "task_name": task_name,
        "total": total,
        "matched": match_count,
        "accuracy": accuracy,
        "file": str(output_file),
    }


def main() -> None:
    register_task_prompt(2, _prompt_openseek_2_spacy_v4)

    args = parse_args()
    task_start = max(1, args.task_start)
    task_end = min(8, args.task_end)
    if task_start > task_end:
        raise ValueError(f"task_start({task_start}) > task_end({task_end})")

    output_dir = _resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[输出目录] {output_dir}")
    print(
        f"[task2 v4] spaCy 统计约定: nouns={sorted(_TASK_NOUN_POS)} verbs={sorted(_TASK_VERB_POS)} "
        "(verbs 不含 AUX)"
    )

    summary: list[dict] = []
    for task_id in range(task_start, task_end + 1):
        summary.append(
            run_task(
                task_id=task_id,
                output_dir=output_dir,
                examples_limit=args.examples_limit,
                retries=args.retries,
                retry_wait_seconds=args.retry_wait_seconds,
                resume=args.resume,
                log=args.log,
                log_prompt=args.log_prompt,
            )
        )

    summary_file = output_dir / "summary_main1_task2_spacy_v4_zeroshot.json"
    with summary_file.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[汇总完成] {summary_file}")


if __name__ == "__main__":
    main()
