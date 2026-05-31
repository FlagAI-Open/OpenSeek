"""
main1 task2 推理：**零样本**（无 few-shot / 不插入 ``[[EXAMPLES]]``），提示词在本文件内自洽。

相对 ``infer_examples_main1_task2_spacy_v2.py`` 的变化：

- 不做 ``select_examples``，prompt 中不包含 Reference examples。
- 增加 **spaCy 词性与标签含义** 说明（``pos_`` 粗粒度 Universal POS，``tag_`` 为 ``en_core_web_sm`` 细粒度标签）。
- **Task-specific output** 按本条题干解析出的目标拆分：仅 **动词计数** 或仅 **名词计数** 时使用对应小节（互不混杂）。
- Official definition 仍做实例对齐（与 ``Your input`` 中的 nouns/verbs 要求一致）。

依赖：``pip install spacy`` 且 ``python -m spacy download en_core_web_sm``。
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
# 本地提示词（不从 method_hyb_prompts 导入）
# ---------------------------------------------------------------------------

PromptBuilder = Callable[[str, str], str]

TASK_PROMPT_BUILDERS: dict[int, PromptBuilder] = {}

_DEFAULT_ANNOTATION_GUIDELINES = (
    "### Annotation guidelines\n"
    "1. You may show step-by-step reasoning in plain text **before** the final answer; put all reasoning **outside** "
    "the `<label>...</label>` tags.\n"
    "2. The final answer MUST appear inside a single pair of tags: <label>...</label>.\n"
    "3. Inside the tags, output ONLY one non-negative integer in decimal digits (no extra words or punctuation).\n"
)


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
        "You solve OpenSeek benchmark items. Follow the official task definition below.\n\n"
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
        "### Role Definition\n"
        "You are a data annotation expert.\n\n"
        "### Core Task\n"
        f"{task_description}\n\n"
        "### Text to Annotate\n"
        f"{text2annotate}\n\n"
        "### Final output\n"
        "Wrap the answer in <label>...</label>.\n"
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
            "**For this instance:** count **verbs only** in the quoted sentence below. Do not report a noun count."
        )
    if target == "nouns":
        return (
            f"{base}\n\n"
            "**For this instance:** count **nouns only** in the quoted sentence below. Do not report a verb count."
        )
    return (
        f"{base}\n\n"
        "**For this instance:** the instruction below asks explicitly for either noun count or verb count—follow it."
    )


def _extract_sentence_from_task2_input(text: str) -> str | None:
    t = str(text).strip()
    m = re.search(r"Sentence:\s*'(.*)'\.\s*Count\b", t, flags=re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r'Sentence:\s*"(.*)"\.\s*Count\b', t, flags=re.DOTALL)
    if m:
        return m.group(1).strip()
    return None


_SPACY_LEGEND = (
    "### How to read spaCy token rows (`en_core_web_sm`)\n"
    "Each line is: **token** → **`pos_`** (coarse Universal POS) → **`tag_`** (fine-grained English tag from the model).\n"
    "- **`pos_` vs task counting**\n"
    "  - **`NOUN`** (`NN`, `NNS`, …): common nouns. **`PROPN`** (`NNP`, `NNPS`, …): proper nouns / names. "
    "For **noun-count** items, candidates usually appear as **NOUN** or **PROPN** (subject to task rules below—not determiners or pronouns).\n"
    "  - **`VERB`** (`VB`, `VBD`, `VBG`, `VBN`, `VBZ`, …): lexical verb forms. "
    "For **verb-count** items, **VERB** tokens are primary candidates **when they realize the benchmark’s lexical/action predicate**.\n"
    "  - **`AUX`** (`MD`, uses of *be*/*have*/*do* as auxiliary, …): supporting verbs. "
    "This benchmark typically **does not** count bare auxiliaries, light copulas, or modals **by themselves** as separate verbs.\n"
    "  - **`ADJ`**, **`DET`**, **`ADP`**, **`PRON`**, **`PART`**, **`PUNCT`**, **`NUM`**, etc.: generally **not** counted as nouns for noun tasks nor as verbs for verb tasks.\n"
    "- **Gerunds / participles (`VBG`, `VBN`, …)** may correspond to events in the scene; the benchmark often treats them as **verbs** when they denote an action/state, "
    "even inside noun phrases—follow the verb-count rules below when this instance asks for verbs.\n"
    "- **Disclaimer:** spaCy is automatic; if any tag conflicts with the **task-specific rules for this instance**, "
    "follow the task rules, not the tagger.\n"
)


_OPENSEEK_2_TASK_SPECIFIC_VERBS = (
    "### Task-specific output (openseek-2: count verbs — this instance)\n"
    "- This instance asks for **verbs only** in the quoted sentence.\n"
    "- Count **lexical / action predicates**, including many non-finite forms (*driving*, *wearing*, *displayed* in reduced clauses) when they denote the event/state.\n"
    "- **Do not** count **pure auxiliaries**, **light copulas alone** (*is/are/was/were* as mere linking), *have/has* as auxiliary, "
    "*do/does* as dummy auxiliary, or **modals** (*can/may/must/…*) **as standalone verbs** unless the benchmark example pattern treats them differently.\n"
    "- Resolve `-ing`/`-ed` ambiguity by syntax in context: scene predicates → usually **verbs** for this task.\n"
    "- Inside `<label></label>`, output **one non-negative integer** only (decimal digits); no other characters.\n"
)


_OPENSEEK_2_TASK_SPECIFIC_NOUNS = (
    "### Task-specific output (openseek-2: count nouns — this instance)\n"
    "- This instance asks for **nouns only** in the quoted sentence.\n"
    "- Count **common and proper nouns** (people, objects, places, named entities). **`PROPN`** in spaCy usually aligns with proper nouns.\n"
    "- **Do not** count **determiners** (*the/a*), **adjectives**, **pronouns**, **prepositions**, **verbs**, or **pure auxiliaries** as nouns.\n"
    "- Do **not** count an `-ing` verbal noun / gerund heading an action as a noun unless the benchmark convention for this task treats it as nominal—when unsure, "
    "prefer consistency with typical OpenSeek noun lists (objects/entities in the scene).\n"
    "- Inside `<label></label>`, output **one non-negative integer** only (decimal digits); no other characters.\n"
)


_OPENSEEK_2_TASK_SPECIFIC_FALLBACK = (
    "### Task-specific output (openseek-2: count nouns or verbs)\n"
    "- Read **Your input** below: it explicitly asks for either **nouns** or **verbs**—never both in one answer.\n"
    "- **Verb branch:** count lexical/action predicates; exclude bare auxiliaries/copulas/modals per usual OpenSeek verb rules.\n"
    "- **Noun branch:** count common/proper nouns; exclude determiners, adjectives, pronouns.\n"
    "- Inside `<label></label>`, output **one non-negative integer** only.\n"
)


def _spacy_pos_reference_block(sentence: str, *, instance_target: str | None) -> str:
    if not sentence:
        return "### Reference: spaCy POS (en_core_web_sm)\n(Could not isolate the quoted sentence; skipped tagging.)\n"

    nlp = _get_nlp()
    doc = nlp(sentence)
    lines = ["{:24} → {:10} ({})".format(tok.text, tok.pos_, tok.tag_) for tok in doc]
    table = "\n".join(lines) if lines else "(empty)"

    focus = ""
    if instance_target == "verbs":
        focus = (
            "**Using this table for this instance:** prioritize tokens whose **`pos_` is `VERB`** after applying the verb-count rules above; "
            "use **`AUX`** rows only when they realize the benchmark’s counted predicate (usually they do not count as extra verbs).\n"
        )
    elif instance_target == "nouns":
        focus = (
            "**Using this table for this instance:** prioritize **`NOUN`** and **`PROPN`** rows after applying the noun-count rules above.\n"
        )
    else:
        focus = "**Using this table:** match spaCy rows to whether this item asks for nouns or verbs (see Your input).\n"

    return (
        "### Reference: spaCy POS tags (model `en_core_web_sm`)\n"
        f"{focus}\n"
        "```text\n"
        f"{table}\n"
        "```\n"
    )


def _compose_task2_guidelines(text2annotate: str) -> str:
    target = _task2_count_target_from_input(text2annotate)
    sent = _extract_sentence_from_task2_input(text2annotate) or ""

    if target == "verbs":
        task_block = _OPENSEEK_2_TASK_SPECIFIC_VERBS
    elif target == "nouns":
        task_block = _OPENSEEK_2_TASK_SPECIFIC_NOUNS
    else:
        task_block = _OPENSEEK_2_TASK_SPECIFIC_FALLBACK

    pos_block = _spacy_pos_reference_block(sent, instance_target=target)
    return task_block + "\n" + _SPACY_LEGEND + "\n" + pos_block


def _prompt_openseek_2_spacy_v3(task_description: str, text2annotate: str) -> str:
    guidelines = _compose_task2_guidelines(text2annotate)
    definition_for_prompt = _align_task2_official_definition(task_description, text2annotate)
    return _task_prompt_shell_zero_shot(guidelines, definition_for_prompt, text2annotate)


# ---------------------------------------------------------------------------
# 推理
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
        description="main1 task2：零样本 + spaCy POS + 按名词/动词拆分的任务说明（infer_examples_main1_task2_spacy_v3）。"
    )
    parser.add_argument("--task_start", type=int, default=2)
    parser.add_argument("--task_end", type=int, default=2)
    parser.add_argument(
        "--examples_limit",
        type=int,
        default=0,
        help="每任务最多推理条数；<=0 为全部。",
    )
    parser.add_argument("--output_dir", type=str, default="examples_main1")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry_wait_seconds", type=float, default=2.0)
    parser.add_argument("--resume", action="store_true")
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
    output_compare_slug: str = "spacy_v3_zeroshot",
) -> dict:
    task_file = _task_json_path(task_id)
    with task_file.open("r", encoding="utf-8") as f:
        task_dict = json.load(f)

    task_name = task_dict["task_name"]
    task_description = task_dict["Definition"][0]
    all_examples = task_dict["examples"]
    if examples_limit > 0:
        all_examples = all_examples[:examples_limit]

    output_file = output_dir / f"openseek-{task_id}-examples-main1-compare_{output_compare_slug}.jsonl"
    done_ids = _load_done_ids(output_file) if resume else set()
    mode = "a" if resume else "w"

    if done_ids:
        print(f"[断点续跑] task={task_id}, 已完成 {len(done_ids)} 条")

    with output_file.open(mode, encoding="utf-8") as wf:
        for example in tqdm(all_examples, desc=f"Task {task_id} task2 spacy v3 zero-shot: {task_name}"):
            example_id = str(example.get("id", "")).strip()
            if resume and example_id in done_ids:
                continue

            input_text = example.get("input", "")
            expected = _extract_output(example.get("output", ""))

            input_prompt = build_prompt(task_description, input_text, task_id=task_id)

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

    total, match_count, accuracy = _compute_metrics_from_jsonl(output_file)
    print(
        f"[保存完成] task2 {output_compare_slug} task={task_id} total={total} matched={match_count} "
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
    register_task_prompt(2, _prompt_openseek_2_spacy_v3)

    args = parse_args()
    task_start = max(1, args.task_start)
    task_end = min(8, args.task_end)
    if task_start > task_end:
        raise ValueError(f"task_start({task_start}) > task_end({task_end})")

    output_dir = _resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[输出目录] {output_dir}")
    print("[task2] 已注册 spacy v3：零样本；名词/动词分项 Task-specific；spaCy 释义 + POS 表。")

    output_compare_slug = "spacy_v3_zeroshot"
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
                output_compare_slug=output_compare_slug,
            )
        )

    summary_file = output_dir / f"summary_main1_task2_{output_compare_slug}.json"
    with summary_file.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[汇总完成] {summary_file}")


if __name__ == "__main__":
    main()
