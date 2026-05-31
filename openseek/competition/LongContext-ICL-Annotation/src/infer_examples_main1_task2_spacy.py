"""
main1 task2 推理脚本：提示词模块在本文件中 **完全自洽**（不引用 ``method_hyb_prompts``）。

- 本地实现：``_task_prompt_shell``、``TASK_PROMPT_BUILDERS``、``register_task_prompt``、``build_prompt``。
- Task 2：在 **与 ``infer_examples_main1_task2_v2.py`` / ``method_hyb_prompts`` 相同的 v2 任务约束**
  基础上，追加 spaCy ``en_core_web_sm`` 的逐词 POS 参考块。
- **Official task definition** 在保留 JSON 原文的同时，会根据本条输入解析出的名词/动词计数目标追加一句实例范围说明，与 **Your input** 对齐。

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
from method_hyb import select_examples

# ---------------------------------------------------------------------------
# 本地提示词基础设施（不从 method_hyb_prompts 导入）
# ---------------------------------------------------------------------------

PromptBuilder = Callable[[str, str], str]

TASK_PROMPT_BUILDERS: dict[int, PromptBuilder] = {}

_DEFAULT_ANNOTATION_GUIDELINES = (
    "### Annotation guidelines\n"
    "1. You may show step-by-step reasoning in plain text, then give the final answer.\n"
    "2. The final answer MUST appear inside a single pair of tags: <label>...</label>.\n"
    "3. Inside the tags, output ONLY the answer string (no quotes around it unless the examples use quotes). "
    "Do not add commentary inside the tags.\n"
    "4. Match the examples: same capitalization, spacing inside list literals, and punctuation.\n"
)


def _task_prompt_shell(
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
        "You solve OpenSeek benchmark items. Follow the official task definition and the reference examples "
        "for answer type, formatting, and level of detail.\n\n"
        f"{task_specific_guidelines}\n"
        "### Official task definition\n"
        f"{task_description}\n\n"
        "### Reference examples (format and conventions)\n"
        "[[EXAMPLES]]\n\n"
        "### Your input (answer for this instance only)\n"
        f"{text2annotate}\n\n"
        f"{guidelines}"
    )


def register_task_prompt(task_id: int, builder: PromptBuilder) -> None:
    TASK_PROMPT_BUILDERS[task_id] = builder


def _build_prompt_default(task_description: str, text2annotate: str) -> str:
    return (
        "### Role Definition\n"
        "You are a professional data annotation expert specialized in long-context text labeling. "
        "Your work must strictly follow the task rules, fully learn from the provided examples, "
        "and ensure the final annotation result is 100% enclosed in <label> tags.\n\n"
        "### Core Task\n"
        f"{task_description}\n\n"
        "### Critical Annotation Guidelines\n"
        "1. **Example Learning Requirement**: Thoroughly analyze and fully learn from the annotation logic, "
        "format, and criteria in the Examples section.\n"
        "2. **Thinking Process**: You may explain your reasoning step by step.\n"
        "3. **Mandatory Output Rule**: Your final annotation result MUST be enclosed in <label> tags.\n"
        "4. **Length Adaptation**: For long texts, ensure the final <label> tags contain the accurate result.\n\n"
        "### Examples (Must Be Fully Followed)\n"
        "[[EXAMPLES]]\n\n"
        "### Text to Annotate\n"
        f"{text2annotate}\n\n"
        "### Final Requirement Summary\n"
        "Provide reasoning if helpful; the final result MUST be wrapped in <label> tags.\n"
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
# spaCy： Lazy load + 从题干中提取句子 + POS 表
# ---------------------------------------------------------------------------

_NLP = None


def _get_nlp():
    global _NLP  # noqa: PLW0603 — intentional singleton
    if _NLP is None:
        import spacy

        _NLP = spacy.load("en_core_web_sm")
    return _NLP


def _task2_count_target_from_input(text: str) -> str | None:
    """从题干解析本条要求数名词还是动词；返回小写 ``nouns`` 或 ``verbs``。"""
    m = re.search(
        r"Count\s+the\s+number\s+of\s+(nouns|verbs)\s+in\s+this\s+sentence",
        str(text),
        flags=re.IGNORECASE,
    )
    return None if not m else m.group(1).lower()


def _align_task2_official_definition(task_description: str, text2annotate: str) -> str:
    """保留官方 Definition 全文，并写明本条实例只计名词或只计动词，与 ``Your input`` 一致。"""
    base = str(task_description).strip()
    target = _task2_count_target_from_input(text2annotate)
    if target == "verbs":
        return (
            f"{base}\n\n"
            "**For this instance:** You must count **verbs only** in the sentence quoted in the instruction below. "
            "Do not report a noun count."
        )
    if target == "nouns":
        return (
            f"{base}\n\n"
            "**For this instance:** You must count **nouns only** in the sentence quoted in the instruction below. "
            "Do not report a verb count."
        )
    return (
        f"{base}\n\n"
        "**For this instance:** The instruction below explicitly asks for either a noun count or a verb count—follow "
        "that wording exactly."
    )


def _extract_sentence_from_task2_input(text: str) -> str | None:
    """OpenSeek-2 输入形如 ``Sentence: '...'. Count ...``。"""
    t = str(text).strip()
    m = re.search(r"Sentence:\s*'(.*)'\.\s*Count\b", t, flags=re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r'Sentence:\s*"(.*)"\.\s*Count\b', t, flags=re.DOTALL)
    if m:
        return m.group(1).strip()
    return None


def _spacy_pos_reference_block(sentence: str) -> str:
    if not sentence:
        return (
            "### Reference: spaCy POS (en_core_web_sm)\n"
            "(Could not isolate the quoted sentence from the prompt; skip automated tagging.)\n"
        )
    nlp = _get_nlp()
    doc = nlp(sentence)
    lines = ["{:24} → {:10} ({})".format(tok.text, tok.pos_, tok.tag_) for tok in doc]
    table = "\n".join(lines) if lines else "(empty)"
    return (
        "### Reference: spaCy POS tags (model `en_core_web_sm`)\n"
        "Automated token list below (`token → coarse POS (fine tag)`). **These are hints only**: align your "
        "noun/verb count with the **task definition** and examples (spaCy may differ on auxiliaries, "
        "copulas, gerunds/participles, MWE boundaries, etc.).\n\n"
        "```text\n"
        f"{table}\n"
        "```\n"
    )


# 与 ``method_hyb_prompts._OPENSEEK_2_COUNT_NOUNS_VERBS``（v2 / main1_task2_v2）逐字一致，并仅增加一条 spaCy 说明。
_OPENSEEK_2_COUNT_NOUNS_VERBS_SPACY = (
    "### Task-specific output (openseek-2: count nouns/verbs)\n"
    "- Read the instruction carefully and count exactly one target POS in the sentence: either nouns or verbs (never both).\n"
    "- For verb counting, prioritize lexical/action predicates (including non-finite forms like driving/wearing/displayed in reduced clauses); "
    "do not count pure auxiliaries or light copulas by themselves (e.g., is/are/was/were, have/has, do/does, modals).\n"
    "- For noun counting, count common/proper nouns (people, objects, places, entities); do not count determiners, adjectives, or pronouns.\n"
    "- Resolve ambiguity by local syntax: -ing/-ed forms that denote an event/state in the scene are usually counted as verbs, "
    "even when they modify a noun phrase.\n"
    "- Inside <label></label>, output a single non-negative integer in decimal digits only; no extra text inside the tags.\n"
    "- Optional: the spaCy POS reference printed below is machine-generated; if it disagrees with the criteria above or the examples, "
    "follow the task criteria and examples.\n"
)


def _prompt_openseek_2_spacy(task_description: str, text2annotate: str) -> str:
    sent = _extract_sentence_from_task2_input(text2annotate)
    pos_block = _spacy_pos_reference_block(sent or "")
    guidelines = _OPENSEEK_2_COUNT_NOUNS_VERBS_SPACY + "\n" + pos_block
    definition_for_prompt = _align_task2_official_definition(task_description, text2annotate)
    return _task_prompt_shell(guidelines, definition_for_prompt, text2annotate)


# ---------------------------------------------------------------------------
# 推理主流程（与 infer_examples_main1_task2_v2 对齐，仅用本地 build_prompt；输出文件名带 _spacy）
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
        description="main1 task2 推理；v2 任务提示 + spaCy POS 参考（本文件自带 register/build_prompt/shell）。"
    )
    parser.add_argument("--task_start", type=int, default=2, help="起始任务编号（含）。")
    parser.add_argument("--task_end", type=int, default=2, help="结束任务编号（含）。")
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="tokenizer 路径或 HuggingFace 模型 ID（同 main1.py）。",
    )
    parser.add_argument(
        "--examples_limit",
        type=int,
        default=0,
        help="每任务最多推理 examples 条数；<=0 表示全部。",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="examples_main1",
        help="输出目录（默认 examples_main1）。",
    )
    parser.add_argument("--retries", type=int, default=3, help="单条样本推理失败时最大重试次数。")
    parser.add_argument("--retry_wait_seconds", type=float, default=2.0, help="重试间隔秒数。")
    parser.add_argument("--resume", action="store_true", help="断点续跑：跳过已完成 example_id。")
    return parser.parse_args()


def _task_json_path(task_id: int) -> Path:
    if task_id not in TASK_DATA_FILES:
        raise ValueError(f"task_id should be in [1, 8], but got {task_id}.")
    return REPO_ROOT / "data" / TASK_DATA_FILES[task_id]


def _resolve_output_dir(output_dir: str) -> Path:
    p = Path(output_dir)
    if p.is_absolute():
        return p
    return (REPO_ROOT / p).resolve()


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
            example_id = str(row.get("example_id", "")).strip()
            if example_id:
                done.add(example_id)
    return done


def _compute_metrics_from_jsonl(output_file: Path) -> tuple[int, int, float]:
    total = 0
    matched = 0
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
    accuracy = (matched / total) if total else 0.0
    return total, matched, accuracy


def run_task(
    task_id: int,
    output_dir: Path,
    tokenizer_path: str | None = None,
    examples_limit: int = 0,
    retries: int = 3,
    retry_wait_seconds: float = 2.0,
    resume: bool = False,
) -> dict:
    task_file = _task_json_path(task_id)
    with task_file.open("r", encoding="utf-8") as f:
        task_dict = json.load(f)

    task_name = task_dict["task_name"]
    task_description = task_dict["Definition"][0]
    all_examples = task_dict["examples"]
    if examples_limit > 0:
        all_examples = all_examples[:examples_limit]

    icl_examples = task_dict["examples"][:100]

    output_file = output_dir / f"openseek-{task_id}-examples-main1-compare_v2_spacy.jsonl"
    done_ids = _load_done_ids(output_file) if resume else set()
    mode = "a" if resume else "w"

    if done_ids:
        print(f"[断点续跑] task={task_id}, 已完成 {len(done_ids)} 条，继续剩余样本")

    with output_file.open(mode, encoding="utf-8") as wf:
        for example in tqdm(all_examples, desc=f"Examples Inference Task {task_id} v2+spaCy: {task_name}"):
            example_id = str(example.get("id", "")).strip()
            if resume and example_id in done_ids:
                continue

            input_text = example.get("input", "")
            expected = _extract_output(example.get("output", ""))

            prompt = build_prompt(task_description, input_text, task_id=task_id)
            examples_str = select_examples(
                icl_examples,
                task_description,
                input_text,
                tokenizer_path=tokenizer_path,
                hybrid=True,
                top_k=3,
                use_explanation=True,
                use_bm25_semantic_rerank=True,
            )
            input_prompt = prompt.replace("[[EXAMPLES]]\n\n", f"{examples_str}\n\n")

            prediction = ""
            for attempt in range(1, retries + 1):
                try:
                    raw_prediction = annotate(input_prompt)
                    prediction = "" if raw_prediction is None else str(raw_prediction).strip()
                    break
                except Exception as e:  # noqa: BLE001
                    if attempt >= retries:
                        print(
                            f"[推理失败] task={task_id} example_id={example_id} "
                            f"attempt={attempt}/{retries} error={e}"
                        )
                    else:
                        print(
                            f"[重试] task={task_id} example_id={example_id} "
                            f"attempt={attempt}/{retries} error={e}"
                        )
                        time.sleep(retry_wait_seconds)

            is_match = _normalize_text(prediction) == _normalize_text(expected)
            row = {
                "example_id": example_id,
                "input": input_text,
                "expected_output": expected,
                "model_output": prediction,
                "is_match": is_match,
            }
            wf.write(json.dumps(row, ensure_ascii=False) + "\n")
            wf.flush()
            done_ids.add(example_id)

    total, match_count, accuracy = _compute_metrics_from_jsonl(output_file)
    print(
        f"[保存完成 v2+spaCy] task={task_id}, total={total}, matched={match_count}, "
        f"accuracy={accuracy:.2%}, file={output_file}"
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
    register_task_prompt(2, _prompt_openseek_2_spacy)

    args = parse_args()
    task_start = max(1, args.task_start)
    task_end = min(8, args.task_end)
    if task_start > task_end:
        raise ValueError(f"task_start({task_start}) 不能大于 task_end({task_end})")

    output_dir = _resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[输出目录] {output_dir}")
    print("[task2 提示] 已注册：method_hyb v2 计数约束 + spaCy en_core_web_sm POS 参考表。")

    summary: list[dict] = []
    for task_id in range(task_start, task_end + 1):
        summary_item = run_task(
            task_id=task_id,
            output_dir=output_dir,
            tokenizer_path=args.tokenizer_path,
            examples_limit=args.examples_limit,
            retries=args.retries,
            retry_wait_seconds=args.retry_wait_seconds,
            resume=args.resume,
        )
        summary.append(summary_item)

    summary_file = output_dir / "summary_main1_task2_v2_spacy.json"
    with summary_file.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[汇总完成] {summary_file}")


if __name__ == "__main__":
    main()
