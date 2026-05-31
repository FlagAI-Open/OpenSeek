"""
Task7 推理脚本（提示词 V2）。

在 ``method_hyb_prompts`` 默认 openseek-7 提示基础上，进一步强化「Jeopardy 式短答」约束
（类别优先、勿抄线索、地理/人设陷阱、专有名词优于泛化描述）。通过 ``register_task_prompt``
运行时覆盖任务 7 的模板，不改变其它任务。

输出文件与 ``infer_examples_compare_task7.py`` 区分，便于对比实验。
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
import re
import time
import unicodedata
from pathlib import Path

from tqdm import tqdm

from method_hyb import annotate_nvidia as annotate
from method_hyb import build_prompt, select_examples_hybrid
from method_hyb_prompts import _task_prompt_shell, register_task_prompt

REPO_ROOT = Path(__file__).resolve().parent.parent
TASK7_FILE = "openseek-7_jeopardy_answer_generation_all.json"
TASK7_CANONICAL_DESCRIPTION = (
    "You will be given a trivia clue, and the category it belongs to. "
    "You should answer with the best answer that belongs in the category "
    "and is described by the clue. For simplicity, answers should be in all lower cased letters."
)

_OUTPUT_JSONL_NAME = "openseek-7-examples-compare-task7opt-v2.jsonl"
_SUMMARY_JSON_NAME = "summary_task7opt_v2.json"

_OPENSEEK_7_JEOPARDY_V2 = (
    "### Task-specific output (openseek-7 v2: Jeopardy-style answer)\n"
    "- The input is Jeopardy!-style: a **Category** plus a **Clue**. Your `<label>` must contain **only** the "
    "short response that contestants would give—typically a **few words**, not a full sentence.\n"
    "- **Always read Category and Clue together.** The Category narrows **what kind of answer** is required "
    "(person vs place vs word vs movie title vs historical group, etc.); use it to veto wrong answer types "
    "(e.g. do not reply with only a landmark if the clue and category imply a city or country).\n"
    "- Prefer the **canonical, specific entity** the clue points to: proper names, established titles, standard "
    "phrases (person / place / team / book film or song title / disease name / numbered fact). "
    "**Do not substitute** vague references like \"the director of …\", \"this magazine\", "
    "\"the emperor\", \"the author's book title\"—name the concrete answer when one exists.\n"
    "- **Never use the clue as a copy-paste source** for unrelated entities: quoted lines, subtitles, lyric "
    "snippets, or titles mentioned in passing are clues, **not** the answer, unless the combined Category+Clue "
    "explicitly asks for **that exact** title or quotation string.\n"
    "- When a clue contrasts two halves (\"… & …\", \"not X but Y\"), answer the portion the clue **asks for**, "
    "not the distracting half.\n"
    "- For categories about **repairing/fixing quotations, rhymes, proverbs, or wordplay**, respond with **only** "
    "the repaired word or phrase; no preamble or explanation.\n"
    "- If the clue names a recognizable fact (dates, slogans translated, scientific terms), give the benchmark-style "
    "short wording (digits for counts when appropriate); avoid paraphrasing into long explanatory prose.\n"
    "- Inside <label></label>, **all lower case** only; single spaces between words. **No outer quotes** unless "
    "they appear exactly that way inside the golden reference examples.\n"
)

_OPENSEEK_7_ANNOTATION_GUIDELINES_V2 = (
    "### Annotation guidelines\n"
    "1. You may show step-by-step reasoning in plain text, then give the final answer.\n"
    "2. The final answer MUST appear inside a single pair of tags: <label>...</label>.\n"
    "3. Inside the tags, output ONLY the Jeopardy response string (no prefixes like \"Answer:\", no commentary).\n"
    "4. This task's official definition requires answers in **all lower case**. Match reference examples on "
    "internal punctuation and spacing inside the phrase; reference examples beat generic formatting rules "
    "from other OpenSeek tasks.\n"
    "5. Keep the labelled span **minimal**: omit leading articles if the shortest standard form drops them, "
    "unless the references consistently keep them.\n"
)


def _prompt_openseek_7_task7_v2(task_description: str, text2annotate: str) -> str:
    return _task_prompt_shell(
        _OPENSEEK_7_JEOPARDY_V2,
        task_description,
        text2annotate,
        annotation_guidelines=_OPENSEEK_7_ANNOTATION_GUIDELINES_V2,
    )


_register_done = False


def ensure_task7_v2_prompt_registered() -> None:
    """幂等注册，避免同一进程重复覆盖。"""
    global _register_done
    if not _register_done:
        register_task_prompt(7, _prompt_openseek_7_task7_v2)
        _register_done = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Task7 专用推理（V2 强化提示词 + examples 检索 + 准确率统计）。"
    )
    parser.add_argument("--examples_limit", type=int, default=0, help="最多推理多少条；<=0 表示全部。")
    parser.add_argument("--output_dir", type=str, default="examples", help="结果输出目录。")
    parser.add_argument("--retries", type=int, default=3, help="单条失败重试次数。")
    parser.add_argument("--retry_wait_seconds", type=float, default=2.0, help="重试等待秒数。")
    parser.add_argument("--resume", action="store_true", help="开启断点续跑。")
    parser.add_argument("--task7_shot_k", type=int, default=8, help="每条样本检索示例数，默认 8。")
    parser.add_argument(
        "--task7_retrieval_pool_size",
        type=int,
        default=200,
        help="候选检索池大小（按相似度排序后截断），默认 200。",
    )
    parser.add_argument(
        "--thinking",
        type=str,
        choices=["on", "off"],
        default="off",
        help="是否开启模型 thinking（映射到 DASHSCOPE_ENABLE_THINKING）。",
    )
    parser.add_argument(
        "--print_model_output",
        type=str,
        choices=["on", "off"],
        default="off",
        help="是否打印模型原始输出预览（通过 ANNOTATE_LOG_EVERY_RESPONSE）。",
    )
    parser.add_argument(
        "--print_empty_prediction",
        type=str,
        choices=["on", "off"],
        default="on",
        help="当解析结果为空时是否打印样本信息。",
    )
    parser.add_argument("--batch_size", type=int, default=8, help="并行调用批大小（线程数），默认 8。")
    return parser.parse_args()


def _task_json_path() -> Path:
    return REPO_ROOT / "data" / TASK7_FILE


def _normalize_text(text: str) -> str:
    return " ".join(str(text).strip().split())


def _extract_output(output_value) -> str:
    if isinstance(output_value, list):
        return "" if not output_value else str(output_value[0]).strip()
    return str(output_value).strip()


def _normalize_task7_answer(text: str) -> str:
    return _normalize_text(text).lower()


def _normalize_task7_match_text(text: str) -> str:
    s = _normalize_task7_answer(text)
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[\"'`“”‘’]", "", s)
    s = re.sub(r"[^a-z0-9\s/&-]", " ", s)
    s = re.sub(r"\b(the|a|an)\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _expand_expected_variants(expected: str) -> set[str]:
    base = _normalize_task7_match_text(expected)
    variants: set[str] = set()
    if base:
        variants.add(base)
    raw = _normalize_task7_answer(expected)
    if not raw:
        return variants
    normalized = (
        raw.replace("&/or", " or ").replace("and/or", " or ").replace("& or", " or ")
    )
    parts = [p.strip(" ,;/") for p in re.split(r"\bor\b|,|;", normalized) if p.strip(" ,;/")]
    for p in parts:
        v = _normalize_task7_match_text(p)
        if v:
            variants.add(v)
    return variants


def _is_task7_match(expected: str, prediction: str) -> bool:
    pred_norm = _normalize_task7_match_text(prediction)
    if not pred_norm:
        return False
    expected_variants = _expand_expected_variants(expected)
    if not expected_variants:
        return False
    if pred_norm in expected_variants:
        return True
    return any(
        pred_norm.endswith(f" {v}") or v.endswith(f" {pred_norm}") for v in expected_variants
    )


def _resolve_output_dir(output_dir: str) -> Path:
    p = Path(output_dir)
    if p.is_absolute():
        return p
    return (REPO_ROOT / p).resolve()


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


def _infer_task7_prediction(
    input_prompt: str,
    retries: int,
    retry_wait_seconds: float,
) -> str:
    prediction = ""
    for attempt in range(1, retries + 1):
        try:
            raw_prediction = annotate(input_prompt)
            prediction = "" if raw_prediction is None else str(raw_prediction).strip()
            prediction = _normalize_task7_answer(prediction)
            return prediction
        except Exception as e:  # noqa: BLE001
            if attempt >= retries:
                print(f"[推理失败] attempt={attempt}/{retries} error={e}")
            else:
                print(f"[重试] attempt={attempt}/{retries} error={e}")
                time.sleep(retry_wait_seconds)
    return prediction


def run_task7_v2(
    output_dir: Path,
    examples_limit: int = 0,
    retries: int = 3,
    retry_wait_seconds: float = 2.0,
    resume: bool = False,
    task7_shot_k: int = 8,
    task7_retrieval_pool_size: int = 200,
    print_empty_prediction: str = "on",
    batch_size: int = 8,
) -> dict:
    ensure_task7_v2_prompt_registered()

    with _task_json_path().open("r", encoding="utf-8") as f:
        task_dict = json.load(f)

    task_id = 7
    task_name = task_dict["task_name"]
    task_description = TASK7_CANONICAL_DESCRIPTION
    all_examples = task_dict["examples"]
    if examples_limit > 0:
        all_examples = all_examples[:examples_limit]

    output_file = output_dir / _OUTPUT_JSONL_NAME
    done_ids = _load_done_ids(output_file) if resume else set()
    mode = "a" if resume else "w"

    if done_ids:
        print(f"[断点续跑 v2] task=7, 已完成 {len(done_ids)} 条，继续剩余样本")

    cleaned_examples: list[dict] = []
    for ex in task_dict["examples"]:
        cleaned_examples.append(
            {
                "id": str(ex.get("id", "")).strip(),
                "input": str(ex.get("input", "")),
                "output": [_normalize_task7_answer(_extract_output(ex.get("output", "")))],
            }
        )

    pending_examples: list[dict] = []
    for example in all_examples:
        example_id = str(example.get("id", "")).strip()
        if resume and example_id in done_ids:
            continue
        pending_examples.append(example)

    with output_file.open(mode, encoding="utf-8") as wf:
        for start in tqdm(
            range(0, len(pending_examples), max(1, batch_size)),
            desc=f"Task7 V2 Inference: {task_name}",
        ):
            batch = pending_examples[start : start + max(1, batch_size)]
            prepared: list[dict] = []
            for example in batch:
                example_id = str(example.get("id", "")).strip()
                input_text = str(example.get("input", ""))
                expected_raw = _extract_output(example.get("output", ""))
                expected = _normalize_task7_answer(expected_raw)

                examples_str = select_examples_hybrid(
                    all_examples=cleaned_examples,
                    task_description=task_description,
                    text2annotate=input_text,
                    top_k=max(1, task7_shot_k),
                    rerank_pool_size=max(20, task7_retrieval_pool_size),
                    use_explanation=False,
                    use_bm25_semantic_rerank=True,
                    exclude_example_id=example_id,
                )

                prompt = build_prompt(task_description, input_text, task_id=task_id)
                input_prompt = prompt.replace("[[EXAMPLES]]\n\n", f"{examples_str}\n")
                prepared.append(
                    {
                        "example_id": example_id,
                        "input_text": input_text,
                        "expected": expected,
                        "input_prompt": input_prompt,
                    }
                )

            with ThreadPoolExecutor(max_workers=max(1, batch_size)) as executor:
                predictions = list(
                    executor.map(
                        lambda item: _infer_task7_prediction(
                            item["input_prompt"],
                            retries=retries,
                            retry_wait_seconds=retry_wait_seconds,
                        ),
                        prepared,
                    )
                )

            for item, prediction in zip(prepared, predictions, strict=True):
                example_id = item["example_id"]
                input_text = item["input_text"]
                expected = item["expected"]
                is_match = _is_task7_match(expected, prediction)
                if print_empty_prediction == "on" and not prediction:
                    print(
                        f"[空预测] example_id={example_id} expected={expected} "
                        f"input_preview={input_text[:160].replace(chr(10), ' ')}"
                    )
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
        f"[保存完成 v2] task=7, total={total}, matched={match_count}, "
        f"accuracy={accuracy:.2%}, file={output_file}"
    )
    return {
        "task_id": 7,
        "task_name": task_name,
        "prompt_version": "task7_v2",
        "total": total,
        "matched": match_count,
        "accuracy": accuracy,
        "file": str(output_file),
    }


def main() -> None:
    args = parse_args()
    ensure_task7_v2_prompt_registered()

    os.environ["DASHSCOPE_ENABLE_THINKING"] = "1" if args.thinking == "on" else "0"
    os.environ["ANNOTATE_LOG_EVERY_RESPONSE"] = "1" if args.print_model_output == "on" else "0"
    os.environ.setdefault("ANNOTATE_GLOBAL_API_LOCK", "0")
    os.environ.setdefault("ANNOTATE_FALLBACK_PLAIN", "1")
    os.environ.setdefault("ANNOTATE_MAX_PLAIN_CHARS", "120")
    print("[prompt] Task7 Jeopardy 模板已通过 register_task_prompt 覆盖为 v2")

    output_dir = _resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[输出目录] {output_dir} -> {_OUTPUT_JSONL_NAME}, {_SUMMARY_JSON_NAME}")

    summary_item = run_task7_v2(
        output_dir=output_dir,
        examples_limit=args.examples_limit,
        retries=args.retries,
        retry_wait_seconds=args.retry_wait_seconds,
        resume=args.resume,
        task7_shot_k=args.task7_shot_k,
        task7_retrieval_pool_size=args.task7_retrieval_pool_size,
        print_empty_prediction=args.print_empty_prediction,
        batch_size=max(1, args.batch_size),
    )
    summary_file = output_dir / _SUMMARY_JSON_NAME
    with summary_file.open("w", encoding="utf-8") as f:
        json.dump([summary_item], f, ensure_ascii=False, indent=2)
    print(f"[汇总完成] {summary_file}")


if __name__ == "__main__":
    main()
