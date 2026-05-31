"""
Task7 推理（examples）：在提示词首部嵌入 **[System Instructions]**（Jeopardy 专家 + 四步思维链），
沿用 ``infer_examples_compare_task7_v3`` 的混合检索与 ``_is_task7_match_v3`` 评测。

说明：当前 ``annotate_nvidia`` 仅向模型发送一条 user message，因此将 “System Instructions” 作为
Markdown 小节写在 user prompt 顶端，语义上等价于给定系统指令。
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm

from infer_examples_compare_task7_v3 import (
    TASK7_CANONICAL_DESCRIPTION,
    _compute_metrics_from_jsonl,
    _extract_output,
    _infer_task7_prediction,
    _is_task7_match_v3,
    _load_done_ids,
    _normalize_task7_answer,
    _resolve_output_dir,
    _task_json_path,
)
from method_hyb import build_prompt, select_examples_hybrid
from method_hyb_prompts import _task_prompt_shell, register_task_prompt

REPO_ROOT = Path(__file__).resolve().parent.parent

_OUTPUT_JSONL_NAME = "openseek-7-examples-compare-task7opt-stept-prompt.jsonl"
_SUMMARY_JSON_NAME = "summary_task7opt_stept_prompt.json"

_TASK7_STEPT_SYSTEM_BLOCK = """[System Instructions]
You are a Jeopardy! champion. Given a category and a clue, provide the
exact answer that belongs in the category and is described by the clue.

Critical rules for your answer:
- Use ALL LOWERCASE letters only
- Match Jeopardy convention: give the most common, concise form
- Preserve articles like 'the' or 'a' ONLY when part of the standard name
- Preserve apostrophes in standard spelling
- Use abbreviations if that is the standard form
- Give the EXACT answer, not a description or elaboration

[Few-shot Examples (6)]

--- Example 1 (shot) ---
Input: Category: WEIGHTS & MEASURES 
Clue: 1.5 ounces is equal to a jigger or one of these units--but Bartender, a little extra is always OK
Reasoning: 1. Parse the clue: The question asks for a unit of measure equal to 1.5 ounces, equivalent to a jigger, and referenced in a bartending context.
2. Check category fit: This unit falls under Weights & M
Answer: shot

--- Example 2 (ray allen) ---
Input: Category: FISHY SPORTSMEN 
Clue: This former UConn guard got a change of scenery in 2007 & won a championship ring with the Celtics
Reasoning: 1. The clue asks for a former UConn basketball guard who joined the Celtics in 2007 and won a championship ring, fitting the "Fishy Sportsmen" category.
2. Recall that Ray Allen, a UConn alum, was par
Answer: ray allen

--- Example 3 (eric clapton) ---
Input: Category: ROCK OF AGELESS 
Clue: He's lasted long enough to be inducted into the Rock & Roll Hall of Fame 3 times, the first with The Yardbirds
Reasoning: 1. Parse the clue: Identify a rock musician inducted into the Rock & Roll Hall of Fame three times, with the first induction tied to the band The Yardbirds.
2. Recall key details: Eric Clapton was ind
Answer: eric clapton

--- Example 4 (black sabbath) ---
Input: Category: ROCK'S FRONTMEN & WOMEN 
Clue: Ozzy Osbourne
Reasoning: 1. The category is Rock's Frontmen & Women, so we need to identify the key rock band associated with the named frontman, Ozzy Osbourne.
2. Recall that Ozzy Osbourne rose to mainstream fame as the lead
Answer: black sabbath

--- Example 5 (new zealand) ---
Input: Category: BIRDS 
Clue: Country which is the native habitat of the flightless bird seen here:
Reasoning: 1. The clue asks for the country that is the native habitat of a specific flightless bird.
2. Since no image is provided, we rely on the most iconic flightless bird-country pair commonly featured in J
Answer: new zealand

--- Example 6 (walter cronkite) ---
Input: Category: TELEVISION HISTORY 
Clue: He was named to the CBS board of directors after he gave up his news anchor position to Dan Rather
Reasoning: 1. Parse the clue: Identify a former CBS news anchor who was succeeded by Dan Rather and later joined the CBS board of directors.
2. Check category fit: The subject is a key figure in television news,
Answer: walter cronkite
"""

_TASK7_STEPT_TASK_OUTPUT = """### Task-specific output (openseek-7: Jeopardy-style answer)
- The input pairs a **Category** with a **Clue**. Your `<label>` must contain **only** the short
  contestant-style response—typically **a few words**, not a full sentence.
- **Always read Category and Clue together** to choose the correct entity type and specificity.
- Inside `<label>...</label>`: **all lower case**, spacing and punctuation consistent with reference examples."""

_ANNOTATION_GUIDELINES_STEPT = (
    "### Annotation guidelines\n"
    "1. You may show step-by-step reasoning in plain text before the final answer.\n"
    "2. The final answer MUST appear inside a single pair of tags: <label>...</label>.\n"
    "3. Inside the tags, output ONLY the Jeopardy response string (no prefixes like \"Answer:\").\n"
    "4. Match reference examples for internal punctuation and spacing; reference examples beat generic rules.\n"
)


def _prompt_openseek_7_task7_stept(task_description: str, text2annotate: str) -> str:
    task_specific = f"{_TASK7_STEPT_SYSTEM_BLOCK.strip()}\n\n{_TASK7_STEPT_TASK_OUTPUT.strip()}\n"
    return _task_prompt_shell(
        task_specific,
        task_description,
        text2annotate,
        annotation_guidelines=_ANNOTATION_GUIDELINES_STEPT,
    )


_register_done = False


def ensure_task7_stept_prompt_registered() -> None:
    global _register_done
    if not _register_done:
        register_task_prompt(7, _prompt_openseek_7_task7_stept)
        _register_done = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Task7 推理：Step-by-step [System Instructions] 提示 + V3 评测。"
            "支持 --rejudge_only 仅从已有 compare JSONL 重算 is_match。"
        )
    )
    parser.add_argument(
        "--rejudge_only",
        action="store_true",
        help="不调用模型：从 --rejudge_source_jsonl 重算 is_match（V3 规则）。",
    )
    parser.add_argument(
        "--rejudge_source_jsonl",
        type=str,
        default="",
        help="--rejudge_only 时的输入 JSONL（含 expected_output / model_output）。",
    )
    parser.add_argument("--examples_limit", type=int, default=0, help="<=0 表示全部。")
    parser.add_argument("--output_dir", type=str, default="examples", help="输出目录。")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry_wait_seconds", type=float, default=2.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--task7_shot_k", type=int, default=8)
    parser.add_argument("--task7_retrieval_pool_size", type=int, default=200)
    parser.add_argument(
        "--thinking",
        type=str,
        choices=["on", "off"],
        default="off",
        help="映射到 DASHSCOPE_ENABLE_THINKING。",
    )
    parser.add_argument(
        "--print_model_output",
        type=str,
        choices=["on", "off"],
        default="off",
        help="映射到 ANNOTATE_LOG_EVERY_RESPONSE。",
    )
    parser.add_argument(
        "--print_empty_prediction",
        type=str,
        choices=["on", "off"],
        default="on",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    return parser.parse_args()


def rejudge_compare_jsonl_stept(source: Path, dest: Path) -> dict:
    rows_out: list[dict] = []
    with source.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            exp = row.get("expected_output", "")
            pred = row.get("model_output", "")
            row["is_match"] = _is_task7_match_v3(str(exp), str(pred))
            rows_out.append(row)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as wf:
        for row in rows_out:
            wf.write(json.dumps(row, ensure_ascii=False) + "\n")
    total = len(rows_out)
    matched = sum(1 for r in rows_out if r.get("is_match"))
    accuracy = matched / total if total else 0.0
    print(
        f"[rejudge stept+V3] total={total}, matched={matched}, accuracy={accuracy:.2%}, "
        f"source={source}, dest={dest}"
    )
    return {
        "task_id": 7,
        "prompt_version": "task7_stept_prompt_match_v3_rejudge",
        "total": total,
        "matched": matched,
        "accuracy": accuracy,
        "file": str(dest),
        "rejudge_source": str(source),
    }


def run_task7_stept(
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
    ensure_task7_stept_prompt_registered()

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
        print(f"[断点续跑 stept] task=7, 已完成 {len(done_ids)} 条")

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
            desc=f"Task7 Stept-Prompt Inference: {task_name}",
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
                is_match = _is_task7_match_v3(expected, prediction)
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
        f"[保存完成 stept] task=7, total={total}, matched={match_count}, "
        f"accuracy={accuracy:.2%}, file={output_file}"
    )
    return {
        "task_id": 7,
        "task_name": task_name,
        "prompt_version": "task7_stept_system_style_prompt_match_v3",
        "total": total,
        "matched": match_count,
        "accuracy": accuracy,
        "file": str(output_file),
    }


def main() -> None:
    args = parse_args()

    output_dir = _resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.rejudge_only:
        src = Path(args.rejudge_source_jsonl.strip())
        if not src.is_absolute():
            src = (REPO_ROOT / src).resolve()
        if not src.is_file():
            raise SystemExit(f"--rejudge_only 需要有效文件: {src}")
        dest = output_dir / _OUTPUT_JSONL_NAME
        summary_item = rejudge_compare_jsonl_stept(src, dest)
        summary_file = output_dir / _SUMMARY_JSON_NAME
        with summary_file.open("w", encoding="utf-8") as f:
            json.dump([summary_item], f, ensure_ascii=False, indent=2)
        print(f"[汇总完成] {summary_file}")
        return

    ensure_task7_stept_prompt_registered()

    os.environ["DASHSCOPE_ENABLE_THINKING"] = "1" if args.thinking == "on" else "0"
    os.environ["ANNOTATE_LOG_EVERY_RESPONSE"] = "1" if args.print_model_output == "on" else "0"
    os.environ.setdefault("ANNOTATE_GLOBAL_API_LOCK", "0")
    os.environ.setdefault("ANNOTATE_FALLBACK_PLAIN", "1")
    os.environ.setdefault("ANNOTATE_MAX_PLAIN_CHARS", "120")

    print("[prompt] Task7 [System Instructions] + step-by-step + <label>")
    print(f"[thinking] DASHSCOPE_ENABLE_THINKING={os.environ['DASHSCOPE_ENABLE_THINKING']}")
    print(f"[输出目录] {output_dir} -> {_OUTPUT_JSONL_NAME}, {_SUMMARY_JSON_NAME}")

    summary_item = run_task7_stept(
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
