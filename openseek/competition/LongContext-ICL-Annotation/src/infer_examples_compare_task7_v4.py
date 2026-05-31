"""
Task7 推理脚本 V4：在 V3 评测（``_is_task7_match_v3``）与 Jeopardy 提示基线之上，强化

1. **所指实体判别**：明示反例范式 + 推理前先自检两句话术（内化在指引中）。
2. **检索 query**：弱化整段题干对 BM25/向量重排的碾压，双层强调 Category + Clue 焦点；
   对「歌名 / 歌词 / Prince 曲目」类目追加标题 vs 题干片段的备忘行。
3. **检测 + ReAct 多 Agent**：先做 **Solver** → 可选 **Verifier**（独立 API 调用，结构化 ``<verdict>``）
   → 若 ``fail`` 或本地 **clue_echo** 强触发 → **Refiner** 再生成一轮；最终仍仅 **一条** `<label>`
   进入 ``model_output``（与评测 JSONL 字段兼容）。

评测规则与 ``infer_examples_compare_task7_v3`` 一致（直接复用其匹配函数）。
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm

from method_hyb import annotate_nvidia_raw_text, build_prompt, count_answer, select_examples_hybrid
from method_hyb_prompts import _task_prompt_shell, register_task_prompt

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

REPO_ROOT = Path(__file__).resolve().parent.parent

_OUTPUT_JSONL_NAME = "openseek-7-examples-compare-task7opt-v4.jsonl"
_SUMMARY_JSON_NAME = "summary_task7opt_v4.json"

_REGISTER_DONE = False

# --- V4 独有：在 V3 题干约束上叠加「反例心态」与 ReAct 说明 ---
_OPENSEEK_7_JEOPARDY_V4 = (
    "### Task-specific output (openseek-7 v4: Jeopardy-style answer)\n"
    "- The input is Jeopardy!-style: a **Category** plus a **Clue**. Your `<label>` must contain **only** the "
    "short response that contestants would give—typically a **few words**, not a full sentence.\n"
    "- **Always read Category and Clue together.** Use Category as a veto on **wrong entity kinds** "
    "(person vs city vs landmark vs disease vs song TITLE vs rhyme repair, …).\n"
    "- Prefer the **canonical, specific entity** the clue intends: named people/places/work titles/phrases.\n"
    "- **Anti-copy rule:** lyric clauses, flashy quoted movie lines, subtitles, magazine names appearing as "
    "**scene-setting** usually cue a different Jeopardy response (author, title of the quiz item, era, …). "
    "Do **not** answer with long clue substrings unless Category+Clue explicitly demand **that exact** surface "
    "string.\n"
    "### Counter-pattern mini-examples (fiction; do NOT treat as benchmarks to memorize)\n"
    "**Example A (place vs plaque):** Category hints geography; clue spotlights **a landmark at an address**. "
    "Contestants give the city/region, not copying the billboard string from the clue as the whole answer "
    "(unless Category clearly pins the structure).\n"
    "**Example B (song TITLE vs lyric):** Category cues a **Prince / pop-song / \"…a la Trebek\"** style board. "
    "Clue may quote half a verse; the response is the **short canonical song title**, not a multi-word echo of "
    "the lyric line.\n"
    "- When a clue has **two juxtaposed halves**(\"… & …\", \"left Mork … and he flew …\"), answer the clue's "
    "**ask**, not whichever half mentions a flashy title unless that title is genuinely what Category demands.\n"
    "- Categories about **quotes/rhymes/proverbs repair**: `<label>` = fixed phrase only.\n"
    "- Inside `<label>`: **all lower case**; single spaces; no outer quotation marks unless refs show them.\n"
    "- Before you write `<label>`, silently answer in one line each: "
    "**(SelfQ1)** Is my candidate the **kind of thing Category names**?"
    "**(SelfQ2)** Am I grabbing a vivid phrase from Clue mainly because it's front-and-center—not because it's "
    "the sought Jeopardy response?\n"
)

_OPENSEEK_7_ANNOTATION_GUIDELINES_V4 = (
    "### Annotation guidelines\n"
    "1. Show brief plain-text reasoning **before** the final answer tags (solver pass).\n"
    "2. The final solver answer MUST be in **one** `<label>...</label>` pair.\n"
    "3. No commentary inside `<label>`; Jeopardy string only, lower case unless refs demand otherwise.\n"
    "4. If later an automatic **verifier/refine pass** revises your work, obey it and still emit **exactly one** "
    "final `<label>...</label>` containing the repaired Jeopardy response.\n"
)


def _prompt_openseek_7_task7_v4(task_description: str, text2annotate: str) -> str:
    return _task_prompt_shell(
        _OPENSEEK_7_JEOPARDY_V4,
        task_description,
        text2annotate,
        annotation_guidelines=_OPENSEEK_7_ANNOTATION_GUIDELINES_V4,
    )


def ensure_task7_v4_prompt_registered() -> None:
    """注册 Task7 V4 Solver 骨架（传给 ``build_prompt(..., task_id=7)``）。"""
    global _REGISTER_DONE
    if not _REGISTER_DONE:
        register_task_prompt(7, _prompt_openseek_7_task7_v4)
        _REGISTER_DONE = True


def _parse_category_clue(input_text: str) -> tuple[str, str]:
    s = (input_text or "").strip()
    m = re.match(r"Category:\s*(.+?)\s*\n\s*Clue:\s*(.*)$", s, re.DOTALL | re.IGNORECASE)
    if not m:
        return "", s
    return m.group(1).strip(), m.group(2).strip()


def _songish_category_marker(category: str) -> bool:
    if not category:
        return False
    return bool(
        re.search(
            r"song|songs|lyric|lyrics|prince\b|music\b|album|tune\b|quotes?\s+the\s+tune|\ba\s+la\s+trebek\b",
            category,
            re.IGNORECASE,
        )
    )


def _task7_retrieval_query_v4(task_description: str, input_text: str, *, enabled: bool) -> str | None:
    """
    BM25 / 向量 / 重排的 query；刻意重复 Category，Clue 独立成块以减少「整块题干抄写」对相关性的绑架。
    """
    if not enabled:
        return None
    cat, clue = _parse_category_clue(input_text)
    parts: list[str] = [
        task_description.strip(),
        "",
        "[task7_ic_query_v4]",
        "**Category anchors the answer TYPE; Clue is the semantic focal span.**",
    ]
    if cat:
        parts.extend([f"[category] {cat}", f"[category_repeat] {cat}"])
        if _songish_category_marker(cat):
            parts.append(
                "[song_focus] Retrieval hint: prioritize exemplars whose Category also concerns "
                "**song titles versus lyric fluff**."
            )
    parts.extend(["", "[clue]", clue if clue else input_text.strip()])
    return "\n".join(parts)


def _norm_alpha_tokens(s: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", (s or "").lower()) if len(t) > 1}


def _clue_echo_score(prediction: str, clue: str) -> float:
    """轻量检测：预测与 Clue 共享的「内容词」占比（越高越像照抄题干）。"""
    pt, ct = _norm_alpha_tokens(prediction), _norm_alpha_tokens(clue)
    if not pt or not ct:
        return 0.0
    inter = len(pt & ct)
    return inter / max(1, len(pt))


def _verifier_system_block() -> str:
    return (
        "You are a Jeopardy response auditor (second agent). "
        "You ONLY check type fit, clue-echo, and vagueness. You do NOT solve from general knowledge scratch.\n"
        "Respond in English.\n\n"
        "Output MUST contain ALL of the following lines exactly in order:\n"
        "<analysis>...</analysis>\n"
        "<verdict>pass</verdict> OR <verdict>fail</verdict>\n"
        "<issues>...</issues>\n"
        "Use `pass` if the candidate is plausibly the intended short Jeopardy response; else `fail`.\n"
    )


def _verifier_user_block(category: str, clue: str, candidate: str, detector_note: str) -> str:
    extra = "" if not detector_note else f"\n[detector_hint] {detector_note}\n"
    return (
        f"{extra}"
        f"Category:\n{category or '(missing)'}\n\n"
        f"Clue:\n{clue or '(missing)'}\n\n"
        f"Candidate answer (solver):\n{candidate or '(empty)'}\n\n"
        "Check:\n"
        "(1) Category — does this answer TYPE match what Category implies?\n"
        "(2) Clue-echo — is this mostly copying a flashy clue substring while Category wants a tighter entity?\n"
        "(3) Vague substitutes — rejects like «the emperor», «this poet» without naming the entity.\n"
    )


def _parse_verdict_block(text: str) -> tuple[str, str]:
    raw = str(text or "")
    vm = re.search(r"<verdict>\s*(pass|fail)\s*</verdict>", raw, re.IGNORECASE | re.DOTALL)
    verdict = (vm.group(1).lower() if vm else "fail").strip()
    im = re.search(r"<issues>\s*(.*?)\s*</issues>", raw, re.IGNORECASE | re.DOTALL)
    issues = (im.group(1).strip() if im else "").strip() or "(no issues text)"
    return verdict, issues


def _refine_addon(
    issues: str,
    draft: str,
) -> str:
    return (
        "\n\n### Automatic verifier / refine pass (mandatory)\n"
        "A separate auditor flagged problems. Fix the Jeopardy response **without** meta commentary in `<label>`.\n"
        f"<issues_from_verifier>\n{issues}\n</issues_from_verifier>\n"
        f"<previous_candidate>\n{draft}\n</previous_candidate>\n"
        "Re-reason briefly, then output **exactly one** corrected `<label>...</label>` with the best lower-case "
        "Jeopardy-style answer.\n"
    )


@contextlib.contextmanager
def _push_env(updates: dict[str, str]):
    saved: dict[str, str | None] = {}
    try:
        for k, v in updates.items():
            saved[k] = os.environ.get(k)
            os.environ[k] = v
        yield
    finally:
        for k, old in saved.items():
            if old is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = old


def _raw_call_with_retries(
    prompt: str,
    *,
    retries: int,
    retry_wait_seconds: float,
    thinking: str,
    log_model: str,
) -> str | None:
    for attempt in range(1, max(1, retries) + 1):
        try:
            with _push_env(
                {
                    "DASHSCOPE_ENABLE_THINKING": "1" if thinking == "on" else "0",
                    "ANNOTATE_LOG_EVERY_RESPONSE": log_model,
                }
            ):
                out = annotate_nvidia_raw_text(prompt)
            if out is not None:
                return out
        except Exception as e:  # noqa: BLE001
            if attempt >= retries:
                print(f"[V4 raw 调用失败] attempt={attempt}/{retries} error={e}")
            else:
                print(f"[V4 raw 重试] attempt={attempt}/{retries} error={e}")
                time.sleep(retry_wait_seconds)
    return None


def _label_from_raw(raw: str | None) -> str:
    if raw is None:
        return ""
    parsed = count_answer(raw)
    if parsed is None:
        return ""
    return _normalize_task7_answer(str(parsed))


def _infer_task7_v4_one(
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
    """返回 (prediction, react_meta)。"""
    meta: dict = {"solver": {}, "verifier": None, "refine": None, "detector_echo": None}

    raw0 = _raw_call_with_retries(
        solver_prompt,
        retries=retries,
        retry_wait_seconds=retry_wait_seconds,
        thinking=thinking_solver,
        log_model=log_model,
    )
    draft = _label_from_raw(raw0)
    meta["solver"] = {"raw_chars": len(raw0 or "")}

    echo = _clue_echo_score(draft, clue)
    meta["detector_echo"] = round(echo, 4)
    detector_note = ""
    if echo >= echo_threshold_force_fail:
        detector_note = (
            f"High token overlap between candidate and clue ({echo:.2f}); likely clue echo—prefer "
            "`fail` unless Category genuinely demands that surface wording."
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
        addon = _refine_addon(issues if verdict == "fail" else detector_note or issues, prediction)
        raw1 = _raw_call_with_retries(
            refine_prompt_base + addon,
            retries=retries,
            retry_wait_seconds=retry_wait_seconds,
            thinking=thinking_solver,
            log_model=log_model,
        )
        new_label = _label_from_raw(raw1)
        meta["refine"] = {"pass": refines_done, "got_label": bool(new_label)}
        if new_label:
            prediction = new_label
        break
    return prediction, meta


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Task7 V4：Category/Clue 聚焦检索 + 可选 Verifier/ReAct 修正；评测同 V3（_is_task7_match_v3）。"
        )
    )
    p.add_argument(
        "--rejudge_only",
        action="store_true",
        help="仅重算 is_match（V3 规则）写回 V4 输出 JSONL。",
    )
    p.add_argument("--rejudge_source_jsonl", type=str, default="", help="--rejudge_only 输入 JSONL")
    p.add_argument("--examples_limit", type=int, default=0, help="最多推理多少条；<=0 全部")
    p.add_argument("--output_dir", type=str, default="examples", help="输出目录")
    p.add_argument("--retries", type=int, default=3, help="单次 API 重试上限")
    p.add_argument("--retry_wait_seconds", type=float, default=2.0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--task7_shot_k", type=int, default=8)
    p.add_argument("--task7_retrieval_pool_size", type=int, default=200)
    p.add_argument(
        "--ic_query_v4",
        type=str,
        choices=["on", "off"],
        default="on",
        help="是否启用 Category/Clue 分解后的检索 query（默认 on）",
    )
    p.add_argument(
        "--react",
        type=str,
        choices=["on", "off"],
        default="on",
        help="是否启用 Verifier + 条件性 Refine（默认 on）",
    )
    p.add_argument(
        "--react_max_refines",
        type=int,
        default=1,
        help="Verifier 触发后最多追加几轮 Refine（默认 1）",
    )
    p.add_argument(
        "--echo_threshold",
        type=float,
        default=0.72,
        help="预测与 Clue 词重叠超过该值时，强制走 Refine（0~1，默认 0.72）",
    )
    p.add_argument("--thinking_solver", type=str, choices=["on", "off"], default="off")
    p.add_argument("--thinking_verifier", type=str, choices=["on", "off"], default="off")
    p.add_argument("--print_model_output", type=str, choices=["on", "off"], default="off")
    p.add_argument("--print_empty_prediction", type=str, choices=["on", "off"], default="on")
    p.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="并行处理条数（V4 每条约 2~3 次 API；默认 8，可按配额调低）",
    )
    p.add_argument(
        "--save_react_trace",
        type=str,
        choices=["on", "off"],
        default="off",
        help="是否在 JSONL 中写入 task7_v4_react 调试字段（默认 off）",
    )
    return p.parse_args()


def run_task7_v4(
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
    ensure_task7_v4_prompt_registered()
    task_id = 7

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
        print(f"[断点续跑 V4] 已完成 {len(done_ids)} 条")

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
            desc=f"Task7 V4 Inference: {task_name}",
        ):
            batch = pending[start : start + max(1, batch_size)]
            prepared: list[dict] = []
            for example in batch:
                eid = str(example.get("id", "")).strip()
                input_text = str(example.get("input", ""))
                expected = _normalize_task7_answer(_extract_output(example.get("output", "")))
                cat, clue = _parse_category_clue(input_text)
                rq = _task7_retrieval_query_v4(task_description, input_text, enabled=ic_query_v4)
                examples_str = select_examples_hybrid(
                    all_examples=cleaned_examples,
                    task_description=task_description,
                    text2annotate=input_text,
                    top_k=max(1, task7_shot_k),
                    rerank_pool_size=max(20, task7_retrieval_pool_size),
                    use_explanation=False,
                    use_bm25_semantic_rerank=True,
                    exclude_example_id=eid,
                    retrieval_query_override=rq,
                )
                base_prompt = build_prompt(task_description, input_text, task_id=task_id)
                solver_prompt = base_prompt.replace("[[EXAMPLES]]\n\n", f"{examples_str}\n")
                prepared.append(
                    {
                        "example_id": eid,
                        "input_text": input_text,
                        "expected": expected,
                        "solver_prompt": solver_prompt,
                        "refine_prompt_base": solver_prompt,
                        "category": cat,
                        "clue": clue,
                    }
                )

            workers = max(1, batch_size)

            def _work(item: dict) -> tuple[str, dict]:
                return _infer_task7_v4_one(
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

            with ThreadPoolExecutor(max_workers=workers) as ex:
                results = list(ex.map(_work, prepared))

            for item, (prediction, react_meta) in zip(prepared, results, strict=True):
                exp = item["expected"]
                is_ok = _is_task7_match_v3(exp, prediction)
                if print_empty_prediction == "on" and not prediction:
                    print(
                        f"[空预测 V4] id={item['example_id']} expected={exp} "
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
                    row["task7_v4_react"] = react_meta
                wf.write(json.dumps(row, ensure_ascii=False) + "\n")
                wf.flush()
                done_ids.add(item["example_id"])

    total, matched, acc = _compute_metrics_from_jsonl(output_file)
    print(f"[保存完成 V4] total={total}, matched={matched}, accuracy={acc:.2%}, file={output_file}")
    return {
        "task_id": 7,
        "task_name": task_name,
        "prompt_version": "task7_v4_solver_verifier_ic_query",
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
        summary_rejudge["dest_v4"] = str(dest)
        with (output_dir / _SUMMARY_JSON_NAME).open("w", encoding="utf-8") as f:
            json.dump([summary_rejudge], f, ensure_ascii=False, indent=2)
        return

    ensure_task7_v4_prompt_registered()
    os.environ.setdefault("ANNOTATE_GLOBAL_API_LOCK", "0")
    os.environ.setdefault("ANNOTATE_FALLBACK_PLAIN", "1")
    os.environ.setdefault("ANNOTATE_MAX_PLAIN_CHARS", "120")
    os.environ.setdefault("DASHSCOPE_ENABLE_THINKING", "1" if args.thinking_solver == "on" else "0")

    log_model_out = "1" if args.print_model_output == "on" else "0"
    summary = run_task7_v4(
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
