"""
Task7 推理脚本 V5（相对 V4）：

1. **同类目 few-shot**：仅在「与当前题 Category 字符串归一化后一致」的样本子集上做混合检索；
   若子集为空，则**不写**任何额外 ICL（仅保留提示词内自带的静态 Examples）。
2. **提示词**：在用户给定的 System（内化思维链）基础上，**最终答案与 V4 一致使用** ``<label>...</label>``；
   解析沿用 ``method_hyb.count_answer``（与 V4 ``_label_from_raw`` 同源）。
3. **可选 Verifier + Refine**（复用 V4 的 ``_refine_addon``）、**clue_echo**、评测 ``_is_task7_match_v3``。

说明：``select_examples_hybrid`` 内部仍要求 ``top_k >= 1``；当同类目池非空时才会调用。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm

from method_hyb import count_answer, select_examples_hybrid

from infer_examples_compare_task7_v3 import (
    TASK7_CANONICAL_DESCRIPTION,
    _compute_metrics_from_jsonl,
    _extract_output,
    _is_task7_match_v3,
    _load_done_ids,
    _normalize_task7_answer,
    _resolve_output_dir,
    _task_json_path,
    rejudge_compare_jsonl_v3,
)
from infer_examples_compare_task7_v4 import (
    _clue_echo_score,
    _parse_category_clue,
    _parse_verdict_block,
    _raw_call_with_retries,
    _refine_addon,
    _task7_retrieval_query_v4,
    _verifier_system_block,
    _verifier_user_block,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

_OUTPUT_JSONL_NAME = "openseek-7-examples-compare-task7opt-v5.jsonl"
_SUMMARY_JSON_NAME = "summary_task7opt_v5.json"

_TASK7_V5_SYSTEM_CORE = """### System Prompt（带思维链引导版）

You are a trivia question answering assistant.

You will be given:

* a category
* a trivia clue

Your task is to determine the single best answer that matches the clue and fits the category.

You MUST follow this reasoning process internally:

1. Identify what type of entity the clue is asking for (person, place, movie, book, organization, object, etc.).
2. Extract key hints (names, dates, locations, events, definitions).
3. Use the category as an additional constraint to eliminate wrong candidates.
4. Recall the most likely well-known answer that matches all constraints.
5. Double-check the final answer is consistent with the clue.

Important:

* Perform the reasoning silently (or at most a very short plain-text hint before the tags).
* Do NOT output long chain-of-thought or numbered reasoning steps.
* Do NOT output intermediate candidate answers outside the final tags.
* The **only** scored response is the string inside `<label>...</label>`.

### Output format (required; same convention as OpenSeek Task 7 / V4)

* After optional brief plain text, output **exactly one** pair: `<label>...</label>`.
* Inside `<label>`: **only** the short Jeopardy-style response (typically a few words), **all lower case**;
  single spaces; no outer quotation marks unless reference examples use them.
* No prefixes like `Answer:` or `Output:` **inside** `<label>`; no commentary inside the tags.
* If multiple answers are possible, choose the most famous and most directly matching one.
* If uncertain, still put your best guess inside `<label>` rather than `unknown`.

### Static examples

Category: COFFEE
Clue: Ladyfingers are a common ingredient of this coffee-flavored Italian dessert
<label>tiramisu</label>

Category: THE CINEMA
Clue: This Eddie Murphy remake of a Jerry Lewis film was the biggest-grossing comedy of the summer in 1996
<label>the nutty professor</label>

Remember: think step-by-step internally; the model output for grading is **only** the `<label>` span.
"""

_TASK7_V5_ANNOTATION_GUIDELINES = (
    "### Annotation guidelines\n"
    "1. Show brief plain-text reasoning **before** the final answer tags (solver pass), if helpful.\n"
    "2. The final answer MUST be in **one** `<label>...</label>` pair.\n"
    "3. No commentary inside `<label>`; Jeopardy string only, lower case unless refs demand otherwise.\n"
    "4. If later an automatic **verifier/refine pass** revises your work, obey it and still emit **exactly one** "
    "final `<label>...</label>` containing the repaired answer.\n"
)


def _normalize_category_key(category: str) -> str:
    return " ".join((category or "").strip().lower().split())


def _filter_examples_same_category(
    cleaned_examples: list[dict],
    category_key: str,
    *,
    exclude_id: str,
) -> list[dict]:
    if not category_key:
        return []
    out: list[dict] = []
    for ex in cleaned_examples:
        eid = str(ex.get("id", "")).strip()
        if eid == exclude_id:
            continue
        cat, _clue = _parse_category_clue(str(ex.get("input", "")))
        if _normalize_category_key(cat) == category_key:
            out.append(ex)
    return out


def _hybrid_blocks_to_v5_example_text(hybrid_out: str) -> str:
    """
    将 ``select_examples_hybrid`` 产出的 ``# input <label> ans </label>`` 转为与文中静态示例一致的
    Category/Clue/`<label>` 行文。
    """
    s = (hybrid_out or "").strip()
    if not s:
        return ""
    pairs = re.findall(
        r"#\s*([\s\S]*?)\s*<label>\s*([\s\S]*?)\s*</label>",
        s,
        flags=re.IGNORECASE,
    )
    if not pairs:
        return ""
    chunks: list[str] = []
    for inp_raw, ans in pairs:
        inp = inp_raw.strip()
        cat, clue = _parse_category_clue(inp)
        if not cat and not clue:
            continue
        a = ans.strip().replace("\n", " ")
        chunks.append(f"Category: {cat}\nClue: {clue}\n<label>{a}</label>")
    return "\n\n".join(chunks)


def _select_same_category_icl_v5(
    *,
    cleaned_examples: list[dict],
    task_description: str,
    current_input: str,
    category_key: str,
    exclude_example_id: str,
    task7_shot_k: int,
    task7_retrieval_pool_size: int,
    ic_query: bool,
) -> str:
    pool = _filter_examples_same_category(
        cleaned_examples,
        category_key,
        exclude_id=exclude_example_id,
    )
    if not pool:
        return ""
    k = max(1, min(int(task7_shot_k), len(pool)))
    rq = _task7_retrieval_query_v4(task_description, current_input, enabled=ic_query)
    hybrid = select_examples_hybrid(
        all_examples=pool,
        task_description=task_description,
        text2annotate=current_input,
        top_k=k,
        rerank_pool_size=max(20, min(task7_retrieval_pool_size, len(pool))),
        use_explanation=False,
        use_bm25_semantic_rerank=True,
        exclude_example_id=exclude_example_id,
        retrieval_query_override=rq,
    )
    return _hybrid_blocks_to_v5_example_text(hybrid)


def _build_task7_v5_solver_prompt(input_text: str, same_category_examples: str) -> str:
    parts: list[str] = [_TASK7_V5_SYSTEM_CORE.rstrip()]
    if same_category_examples.strip():
        parts.append(
            "\n### Additional reference (same Jeopardy Category as your question)\n"
            + same_category_examples.strip()
        )
    parts.append("\n### Your question\n" + input_text.strip() + "\n\n" + _TASK7_V5_ANNOTATION_GUIDELINES)
    return "\n".join(parts)


def _label_from_raw_v5(raw: str | None) -> str:
    """与 V4 ``_label_from_raw`` 一致：``count_answer`` + ``_normalize_task7_answer``。"""
    if raw is None:
        return ""
    parsed = count_answer(raw)
    if parsed is None:
        return ""
    return _normalize_task7_answer(str(parsed))


def _infer_task7_v5_one(
    solver_prompt: str,
    refine_prompt_base: str,
    category: str,
    clue: str,
    *,
    use_react: bool,
    echo_threshold_force_fail: float,
    retries: int,
    retry_wait_seconds: float,
    thinking_solver: str,
    thinking_verifier: str,
    log_model: str,
    max_refines: int,
) -> tuple[str, dict]:
    meta: dict = {"solver": {}, "verifier": None, "refine": None, "detector_echo": None}

    raw0 = _raw_call_with_retries(
        solver_prompt,
        retries=retries,
        retry_wait_seconds=retry_wait_seconds,
        thinking=thinking_solver,
        log_model=log_model,
    )
    draft = _label_from_raw_v5(raw0)
    meta["solver"] = {"raw_chars": len(raw0 or "")}

    echo = _clue_echo_score(draft, clue)
    meta["detector_echo"] = round(echo, 4)
    detector_note = ""
    if echo >= echo_threshold_force_fail:
        detector_note = (
            f"High token overlap between candidate and clue ({echo:.2f}); likely clue echo."
        )

    if not use_react:
        return draft, meta

    verifier_body = (
        _verifier_system_block()
        + "\n### Case\n"
        + _verifier_user_block(category, clue, draft, detector_note)
    )
    ver_raw = _raw_call_with_retries(
        verifier_body,
        retries=retries,
        retry_wait_seconds=retry_wait_seconds,
        thinking=thinking_verifier,
        log_model="0",
    )
    verdict, issues = _parse_verdict_block(ver_raw or "")
    meta["verifier"] = {"verdict": verdict, "issues_preview": issues[:240]}

    need_refine = verdict == "fail" or echo >= echo_threshold_force_fail
    refines_done = 0
    prediction = draft
    while need_refine and refines_done < max(0, max_refines):
        refines_done += 1
        addon = _refine_addon(
            issues if verdict == "fail" else detector_note or issues,
            prediction,
        )
        raw1 = _raw_call_with_retries(
            refine_prompt_base + addon,
            retries=retries,
            retry_wait_seconds=retry_wait_seconds,
            thinking=thinking_solver,
            log_model=log_model,
        )
        new_ans = _label_from_raw_v5(raw1)
        meta["refine"] = {"pass": refines_done, "got_answer": bool(new_ans)}
        if new_ans:
            prediction = new_ans
        break
    return prediction, meta


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Task7 V5：同类目 few-shot（无则不加）+ CoT 内化提示 + `<label>` 输出（与 V4 解析一致）+ 可选 Verifier/ReAct；"
            "评测同 V3（_is_task7_match_v3）。"
        )
    )
    p.add_argument(
        "--rejudge_only",
        action="store_true",
        help="仅重算 is_match（V3 规则）写回 V5 输出 JSONL。",
    )
    p.add_argument("--rejudge_source_jsonl", type=str, default="", help="--rejudge_only 输入 JSONL")
    p.add_argument("--examples_limit", type=int, default=0, help="最多推理多少条；<=0 全部")
    p.add_argument("--output_dir", type=str, default="examples", help="输出目录")
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--retry_wait_seconds", type=float, default=2.0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--task7_shot_k", type=int, default=8, help="同类目池内最多检索几条 ICL")
    p.add_argument("--task7_retrieval_pool_size", type=int, default=200)
    p.add_argument(
        "--ic_query_v4",
        type=str,
        choices=["on", "off"],
        default="on",
        help="同类目检索时是否使用 V4 的 Category/Clue 分解 query",
    )
    p.add_argument("--react", type=str, choices=["on", "off"], default="on")
    p.add_argument("--react_max_refines", type=int, default=1)
    p.add_argument("--echo_threshold", type=float, default=0.72)
    p.add_argument("--thinking_solver", type=str, choices=["on", "off"], default="off")
    p.add_argument("--thinking_verifier", type=str, choices=["on", "off"], default="off")
    p.add_argument("--print_model_output", type=str, choices=["on", "off"], default="off")
    p.add_argument("--print_empty_prediction", type=str, choices=["on", "off"], default="on")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument(
        "--save_react_trace",
        type=str,
        choices=["on", "off"],
        default="off",
        help="写入 task7_v5_react 调试字段",
    )
    return p.parse_args()


def run_task7_v5(
    output_dir: Path,
    *,
    examples_limit: int = 0,
    retries: int = 3,
    retry_wait_seconds: float = 2.0,
    resume: bool = False,
    task7_shot_k: int = 8,
    task7_retrieval_pool_size: int = 200,
    ic_query_v4: bool = True,
    use_react: bool = True,
    react_max_refines: int = 1,
    echo_threshold: float = 0.72,
    thinking_solver: str = "off",
    thinking_verifier: str = "off",
    print_empty_prediction: str = "on",
    batch_size: int = 8,
    save_react_trace: bool = False,
    log_model_output: str = "0",
) -> dict:
    with _task_json_path().open("r", encoding="utf-8") as f:
        task_dict = json.load(f)

    task_name = task_dict["task_name"]
    task_description = TASK7_CANONICAL_DESCRIPTION
    all_examples = task_dict["examples"]
    if examples_limit > 0:
        all_examples = all_examples[:examples_limit]

    output_file = output_dir / _OUTPUT_JSONL_NAME
    done_ids = _load_done_ids(output_file) if resume else set()
    mode = "a" if resume else "w"
    if done_ids:
        print(f"[断点续跑 V5] 已完成 {len(done_ids)} 条")

    cleaned_examples: list[dict] = []
    for ex in task_dict["examples"]:
        cleaned_examples.append(
            {
                "id": str(ex.get("id", "")).strip(),
                "input": str(ex.get("input", "")),
                "output": [_normalize_task7_answer(_extract_output(ex.get("output", "")))],
            }
        )

    pending: list[dict] = []
    for example in all_examples:
        eid = str(example.get("id", "")).strip()
        if resume and eid in done_ids:
            continue
        pending.append(example)

    with output_file.open(mode, encoding="utf-8") as wf:
        for start in tqdm(
            range(0, len(pending), max(1, batch_size)),
            desc=f"Task7 V5 Inference: {task_name}",
        ):
            batch = pending[start : start + max(1, batch_size)]
            prepared: list[dict] = []
            for example in batch:
                eid = str(example.get("id", "")).strip()
                input_text = str(example.get("input", ""))
                expected = _normalize_task7_answer(_extract_output(example.get("output", "")))
                cat, clue = _parse_category_clue(input_text)
                cat_key = _normalize_category_key(cat)
                icl_text = _select_same_category_icl_v5(
                    cleaned_examples=cleaned_examples,
                    task_description=task_description,
                    current_input=input_text,
                    category_key=cat_key,
                    exclude_example_id=eid,
                    task7_shot_k=task7_shot_k,
                    task7_retrieval_pool_size=task7_retrieval_pool_size,
                    ic_query=ic_query_v4,
                )
                solver_prompt = _build_task7_v5_solver_prompt(input_text, icl_text)
                prepared.append(
                    {
                        "example_id": eid,
                        "input_text": input_text,
                        "expected": expected,
                        "solver_prompt": solver_prompt,
                        "refine_prompt_base": solver_prompt,
                        "category": cat,
                        "clue": clue,
                        "task7_v5_same_category_icl_chars": len(icl_text),
                    }
                )

            workers = max(1, batch_size)

            def _work(item: dict) -> tuple[str, dict]:
                meta_base = {"icl_chars": item.get("task7_v5_same_category_icl_chars", 0)}
                pred, trace = _infer_task7_v5_one(
                    item["solver_prompt"],
                    item["refine_prompt_base"],
                    item["category"],
                    item["clue"],
                    use_react=use_react,
                    echo_threshold_force_fail=echo_threshold,
                    retries=retries,
                    retry_wait_seconds=retry_wait_seconds,
                    thinking_solver=thinking_solver,
                    thinking_verifier=thinking_verifier,
                    log_model=log_model_output,
                    max_refines=react_max_refines,
                )
                trace = {**meta_base, **trace}
                return pred, trace

            with ThreadPoolExecutor(max_workers=workers) as ex:
                results = list(ex.map(_work, prepared))

            for item, (prediction, react_meta) in zip(prepared, results):
                exp = item["expected"]
                is_ok = _is_task7_match_v3(exp, prediction)
                if print_empty_prediction == "on" and not prediction:
                    print(
                        f"[空预测 V5] id={item['example_id']} expected={exp} "
                        f"preview={item['input_text'][:160].replace(chr(10), ' ')}"
                    )
                row: dict = {
                    "example_id": item["example_id"],
                    "input": item["input_text"],
                    "expected_output": exp,
                    "model_output": prediction,
                    "is_match": is_ok,
                }
                if save_react_trace:
                    row["task7_v5_react"] = react_meta
                wf.write(json.dumps(row, ensure_ascii=False) + "\n")
                wf.flush()
                done_ids.add(item["example_id"])

    total, matched, acc = _compute_metrics_from_jsonl(output_file)
    print(f"[保存完成 V5] total={total}, matched={matched}, accuracy={acc:.2%}, file={output_file}")
    return {
        "task_id": 7,
        "task_name": task_name,
        "prompt_version": "task7_v5_same_category_icl_coT_label_match_v4",
        "total": total,
        "matched": matched,
        "accuracy": acc,
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
            raise SystemExit(f"--rejudge_only 需要有效输入: {src}")
        dest = output_dir / _OUTPUT_JSONL_NAME
        summary_rejudge = rejudge_compare_jsonl_v3(src, dest)
        summary_rejudge["dest_v5"] = str(dest)
        with (output_dir / _SUMMARY_JSON_NAME).open("w", encoding="utf-8") as f:
            json.dump([summary_rejudge], f, ensure_ascii=False, indent=2)
        return

    os.environ.setdefault("ANNOTATE_GLOBAL_API_LOCK", "0")
    os.environ.setdefault("ANNOTATE_FALLBACK_PLAIN", "1")
    os.environ.setdefault("ANNOTATE_MAX_PLAIN_CHARS", "120")
    os.environ.setdefault("DASHSCOPE_ENABLE_THINKING", "1" if args.thinking_solver == "on" else "0")

    log_model_out = "1" if args.print_model_output == "on" else "0"
    summary = run_task7_v5(
        output_dir=output_dir,
        examples_limit=args.examples_limit,
        retries=args.retries,
        retry_wait_seconds=args.retry_wait_seconds,
        resume=args.resume,
        task7_shot_k=args.task7_shot_k,
        task7_retrieval_pool_size=args.task7_retrieval_pool_size,
        ic_query_v4=args.ic_query_v4 == "on",
        use_react=args.react == "on",
        react_max_refines=max(0, args.react_max_refines),
        echo_threshold=float(args.echo_threshold),
        thinking_solver=args.thinking_solver,
        thinking_verifier=args.thinking_verifier,
        print_empty_prediction=args.print_empty_prediction,
        batch_size=max(1, args.batch_size),
        save_react_trace=args.save_react_trace == "on",
        log_model_output=log_model_out,
    )
    with (output_dir / _SUMMARY_JSON_NAME).open("w", encoding="utf-8") as f:
        json.dump([summary], f, ensure_ascii=False, indent=2)
    print(f"[汇总] {output_dir / _SUMMARY_JSON_NAME}")


if __name__ == "__main__":
    main()
