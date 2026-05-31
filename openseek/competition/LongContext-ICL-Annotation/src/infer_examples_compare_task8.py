import argparse
import ast
import concurrent.futures
import json
import os
import tempfile
import time
import traceback
import types
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from method_hyb import annotate_nvidia as annotate
from method_hyb import build_prompt, select_examples_hybrid


REPO_ROOT = Path(__file__).resolve().parent.parent
# executor 判别：success（可 exec 且无致命报错）与其它失败类别的对比，见 Task8ExecOutcome.kind
TASK8_FILE = "openseek-8_kernel_generation.json"
TASK8_CANONICAL_DESCRIPTION = (
    "In this task, you are asked to generate a Triton kernel implementation based on the given instruction. "
    "The output should be valid, executable Python code containing Triton kernel definitions and necessary wrappers."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task8 专用推理（examples 检索 + 准确率统计）。")
    parser.add_argument(
        "--infer_split",
        type=str,
        choices=["examples", "test_samples"],
        default="test_samples",
        help="推理数据划分：examples 或 test_samples，默认 test_samples。",
    )
    parser.add_argument("--examples_limit", type=int, default=0, help="最多推理多少条；<=0 表示全部。")
    parser.add_argument("--output_dir", type=str, default="examples", help="结果输出目录。")
    parser.add_argument("--retries", type=int, default=3, help="单条失败重试次数。")
    parser.add_argument("--retry_wait_seconds", type=float, default=2.0, help="重试等待秒数。")
    parser.add_argument("--resume", action="store_true", help="开启断点续跑。")
    parser.add_argument("--task8_shot_k", type=int, default=6, help="每条样本检索示例数，默认 6。")
    parser.add_argument(
        "--task8_retrieval_pool_size",
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
    parser.add_argument(
        "--retrieval_batch_size",
        type=int,
        default=8,
        help="并行推理批大小（按批并发调用模型），默认 8。",
    )
    parser.add_argument(
        "--accuracy_criterion",
        type=str,
        choices=["exec_dual_pass", "text", "text_and_exec"],
        default="exec_dual_pass",
        help="准确率：exec_dual_pass=参考答案与预测均通过执行器；text=清洗后代码全文逐字等价（LLM 常与参考实现不同故易为 0）；text_and_exec=二者同时满足。",
    )
    parser.add_argument(
        "--single_example_id",
        type=str,
        default="",
        help="若非空：仅推理该 ID（在当前 --infer_split 对应列表内）。与 --single_index 互斥时可优先使用该参数。",
    )
    parser.add_argument(
        "--single_index",
        type=int,
        default=-1,
        help="若 >=0：仅推理当前划分内经 resume/done 筛选后的第 N 条（0-based）。不与 --single_example_id 同时为真。",
    )
    return parser.parse_args()


def _narrow_pending_to_single_row(
    pending_items: list[dict],
    *,
    single_example_id: str = "",
    single_index: int = -1,
) -> list[dict]:
    """在当前待推理列表上收紧为一条。"""
    ex_id = str(single_example_id or "").strip()
    if ex_id:
        if not pending_items:
            raise ValueError(
                "单条模式：待推理队列为空（可能已全部被 --resume 跳过，或划分内无样本）。"
            )
        out = [
            row
            for row in pending_items
            if str(row.get("example_id", "") or "").strip() == ex_id
        ]
        if not out:
            raise ValueError(
                f"单条模式：在待推理队列中找不到 example_id={ex_id!r}（请先检查 --infer_split / --resume）。"
            )
        return out
    if single_index >= 0:
        if not pending_items:
            raise ValueError(
                "单条模式：待推理队列为空（可能已全部被 --resume 跳过，或划分内无样本）。"
            )
        if single_index >= len(pending_items):
            raise ValueError(
                f"单条模式：single_index={single_index} 越界（待推理仅 {len(pending_items)} 条）。"
            )
        return [pending_items[single_index]]
    return pending_items


def _task_json_path() -> Path:
    return REPO_ROOT / "data" / TASK8_FILE


def _normalize_text(text: str) -> str:
    return " ".join(str(text).strip().split())


def _extract_output(output_value) -> str:
    if isinstance(output_value, list):
        return "" if not output_value else str(output_value[0]).strip()
    return str(output_value).strip()


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
            example_id = str(
                row.get("example_id", "") or row.get("test_sample_id", "")
            ).strip()
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

@dataclass(frozen=True)
class Task8ExecOutcome:
    """单份代码在执行器中的一致判别结果。"""
    ok: bool
    kind: str
    message: str


class Task8CodeExecutor:
    """对 Triton 参考/预测代码做 compile+exec + 可调对象检查的统一执行器。"""
    def execute(self, code: Optional[str], *, verbose: bool = True) -> Task8ExecOutcome:
        return _task8_executor_run(code, verbose=verbose)


def _task8_executor_run(code: Optional[str], *, verbose: bool = True) -> Task8ExecOutcome:

    """
    compile + exec 轻量执行；用于与参考答案对比「是否在相同执行语义下可走通」。
    kind: ok | optional_dep | empty | syntax | structure_no_def | import | runtime | callable_missing
    """
    prefix = "[执行器]"
    log = lambda msg: print(msg) if verbose else None

    def fail(kind: str, msg: str) -> Task8ExecOutcome:
        log(f"{prefix} {kind}: {msg[:200]}" if verbose and len(msg) > 200 else f"{prefix} {kind}: {msg}")
        return Task8ExecOutcome(ok=False, kind=kind, message=msg)

    def win(kind: str, msg: str) -> Task8ExecOutcome:
        if kind == "ok":
            log(f"{prefix} 通过")
        else:
            log(f"{prefix} {kind}: {msg}")
        return Task8ExecOutcome(ok=True, kind=kind, message=msg)

    if not code or not isinstance(code, str):
        log(f"{prefix} 失败: empty code")
        return fail("empty", "empty code")

    cleaned = _extract_code_candidate(code)
    stripped = cleaned.strip()
    if not stripped:
        log(f"{prefix} 失败: empty code after cleaning")
        return fail("empty", "empty code after cleaning")

    try:
        tree = ast.parse(cleaned)
    except SyntaxError as e:
        log(f"{prefix} 失败: syntax error: {e}")
        return fail("syntax", f"syntax error: {e}")

    has_def = any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        for node in ast.walk(tree)
    )
    if not has_def:
        log(f"{prefix} 失败: no function/class definition found")
        return fail("structure_no_def", "no function/class definition found")

    exec_ns: dict = {"__builtins__": __builtins__}
    tmp_path = ""
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".py", encoding="utf-8", delete=False) as tf:
            tf.write(cleaned)
            tf.flush()
            tmp_path = tf.name
        compiled = compile(cleaned, tmp_path, "exec")
        exec(compiled, exec_ns, exec_ns)
    except ModuleNotFoundError as e:
        missing_name = str(getattr(e, "name", "") or "")
        err = traceback.format_exc(limit=4)
        if missing_name in {"torch", "triton"}:
            log(f"{prefix} 提示: 缺少依赖 {missing_name}，语法与结构已通过")
            return win("optional_dep", f"missing optional dependency: {missing_name}")
        log(f"{prefix} 编译/执行失败:\n{err}")
        return fail("import", err)
    except Exception:
        err = traceback.format_exc(limit=4)
        log(f"{prefix} 编译/执行失败:\n{err}")
        return fail("runtime", err)
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    has_callable = any(
        callable(obj)
        and isinstance(obj, (types.FunctionType, types.BuiltinFunctionType))
        for obj in exec_ns.values()
    )
    if not has_callable:
        log(f"{prefix} 失败: no callable function found in namespace")
        return fail("callable_missing", "no callable function found in generated code namespace")

    return win("ok", "")


def _run_task8_code_sanity_check(code: str) -> tuple[bool, str]:
    """兼容：与 Task8CodeExecutor.execute 等价，返回 (ok, message)。"""
    r = _task8_executor_run(code, verbose=True)
    return r.ok, r.message


def _extract_code_candidate(text: str) -> str:
    if not text:
        return ""
    cleaned = str(text).strip()
    if "<label>" in cleaned and "</label>" in cleaned:
        start = cleaned.find("<label>") + len("<label>")
        end = cleaned.rfind("</label>")
        if end > start:
            cleaned = cleaned[start:end].strip()
    if "```" in cleaned:
        blocks: list[str] = []
        chunks = cleaned.split("```")
        for i in range(1, len(chunks), 2):
            block = chunks[i]
            if block.startswith("python"):
                block = block[len("python") :]
            blocks.append(block.strip())
        if blocks:
            cleaned = max(blocks, key=len)
        else:
            cleaned = cleaned.replace("```python", "").replace("```", "")
    return textwrap.dedent(cleaned).strip()


def _task8_code_text_equal(lhs: str, rhs: str) -> bool:
    """text 判据：双方经 extract + 空白折叠后完全相同。"""
    return _normalize_text(_extract_code_candidate(lhs)) == _normalize_text(_extract_code_candidate(rhs))


def _infer_compute_task8_accuracy(
    *,
    criterion: str,
    expected_non_empty: bool,
    expected_text: str,
    final_output: str,
    ref_outcome: Optional[Task8ExecOutcome],
    pred_outcome: Task8ExecOutcome,
) -> tuple[bool, bool, bool]:
    """
    criterion: exec_dual_pass | text | text_and_exec。
    返回 (is_match, is_text_match, exec_dual_pass)。无参考答案时 is_match/exec_dual_pass 为 False。
    """
    has_ref = expected_non_empty
    is_text = has_ref and _task8_code_text_equal(final_output, expected_text)
    exec_dual = bool(ref_outcome is not None and ref_outcome.ok and pred_outcome.ok)

    if not has_ref:
        return False, False, False

    if criterion == "text":
        return is_text, is_text, exec_dual
    if criterion == "exec_dual_pass":
        return exec_dual, is_text, exec_dual
    if criterion == "text_and_exec":
        merged = is_text and exec_dual
        return merged, is_text, exec_dual
    return exec_dual, is_text, exec_dual


def _truncate_for_log(text: str, limit: int = 800) -> str:
    t = str(text or "")
    if len(t) <= limit:
        return t
    return t[:limit] + f"...(+{len(t) - limit} chars)"


def _task8_outcome_as_dict(o: Task8ExecOutcome) -> dict:
    return {
        "ok": o.ok,
        "kind": o.kind,
        "message_truncated": _truncate_for_log(o.message, 1200),
    }


def _task8_build_exec_comparison(
    ref_outcome: Optional[Task8ExecOutcome],
    pred_outcome: Task8ExecOutcome,
    *,
    has_reference: bool,
) -> dict:
    """
    参考答案与预测在执行器下的对比摘要。
    outcome_categories_match: ok 与 kind 一致（含同为某类失败）。
    both_pass_executor: 二者均被判定 ok（参见 Task8ExecOutcome.ok）。
    """
    pred_dict = _task8_outcome_as_dict(pred_outcome)
    if not has_reference or ref_outcome is None:
        return {
            "has_reference": False,
            "reference_exec": None,
            "prediction_exec": pred_dict,
            "outcome_categories_match": None,
            "both_pass_executor": False,
            "notes": "无参考答案代码，仅能记录预测侧执行结果（常见于 test_samples 划分）。",
        }
    ref_dict = _task8_outcome_as_dict(ref_outcome)
    categories_match = (
        ref_outcome.ok == pred_outcome.ok and ref_outcome.kind == pred_outcome.kind
    )
    return {
        "has_reference": True,
        "reference_exec": ref_dict,
        "prediction_exec": pred_dict,
        "outcome_categories_match": categories_match,
        "both_pass_executor": bool(ref_outcome.ok and pred_outcome.ok),
    }


def _print_task8_exec_comparison_report(cmp: dict) -> None:
    """单条模式下在控制台打印参考答案与预测的执行器对比。"""
    print("")
    print("========== [执行对比] 参考答案代码 vs 预测代码 ==========")
    print("（compile + exec + 可调对象检查；非内核数值单测）")
    if not cmp.get("has_reference"):
        pe = cmp.get("prediction_exec") or {}
        print("参考答案: （无）")
        print(f"预测执行结果: ok={pe.get('ok')} kind={pe.get('kind')}")
        print(f"message 预览:\n{_truncate_for_log(str(pe.get('message_truncated', '')), 600)}")
        print(f"备注: {cmp.get('notes', '')}")
        print("=========================================================\n")
        return
    r = cmp.get("reference_exec") or {}
    p = cmp.get("prediction_exec") or {}
    print(f"参考答案执行结果: ok={r.get('ok')} kind={r.get('kind')}")
    print(f"预测代码执行结果: ok={p.get('ok')} kind={p.get('kind')}")
    print(f"两类结果完全一致 (ok + kind): {cmp.get('outcome_categories_match')}")
    print(f"两侧均被判为通过 (both_pass): {cmp.get('both_pass_executor')}")
    print("--- 参考答案 message ---")
    print(_truncate_for_log(str(r.get("message_truncated", "")), 2000))
    print("--- 预测 message ---")
    print(_truncate_for_log(str(p.get("message_truncated", "")), 2000))
    print("=========================================================\n")


def _build_task8_zero_shot_prompt(task_description: str, input_text: str, task_id: int) -> str:
    """与首次推理相同的任务模板，但不插入 [[EXAMPLES]]（ReAct 纠错轮用）。"""
    p = build_prompt(task_description, input_text, task_id=task_id)
    return p.replace("[[EXAMPLES]]\n\n", "").replace("[[EXAMPLES]]\n", "").replace("[[EXAMPLES]]", "")


def _build_task8_react_repair_prompt(
    *,
    zero_shot_prompt: str,
    input_text: str,
    last_code: str,
    outcome_kind: str,
    outcome_message: str,
    round_idx: int,
    max_rounds: int,
) -> str:
    """ReAct：零样本 + 上一轮全文 + 按执行器 kind 的硬约束（首轮仍用 few-shot，见 base_prompt）。"""
    last_code = str(last_code or "")
    msg = str(outcome_message or "")
    kind = str(outcome_kind or "")

    kind_hints: list[str] = []
    if kind == "structure_no_def":
        kind_hints.extend(
            [
                "- The validator ran ``ast.parse`` on your extracted code and found ZERO top-level ``def`` / ``class``.",
                "- You MUST emit real Python with at least one module-level ``def`` (typical pattern: one or more ``@triton.jit`` kernels plus a host wrapper ``def``).",
                "- Do NOT answer with only ``pass``, comments, imports-only, or prose. Minimum substantive implementation is required.",
            ]
        )
    elif kind == "empty":
        kind_hints.extend(
            [
                "- The extracted code was empty. Put the full program ONLY inside a single ``<label>...</label>`` pair.",
            ]
        )
    elif kind == "syntax":
        kind_hints.extend(
            [
                "- Fix syntax errors so ``ast.parse`` succeeds. Ensure brackets/quotes/colons are balanced.",
            ]
        )
    elif kind == "callable_missing":
        kind_hints.extend(
            [
                "- After execution, no callable function was visible. Define at least one normal ``def`` wrapper (not only nested defs that never bind to a name).",
            ]
        )
    else:
        kind_hints.append("- Fix the failure below while keeping a complete Triton solution.")

    hints_block = "\n".join(kind_hints)
    return (
        f"{zero_shot_prompt}\n\n"
        "### Repair round (no few-shot)\n"
        f"You are in correction round {round_idx}/{max_rounds}. Your previous attempt failed an automatic code check.\n\n"
        "[Previous model output — exact text you must improve on]\n"
        "```text\n"
        f"{last_code[:12000]}\n"
        "```\n\n"
        "[Validator]\n"
        f"- outcome_kind: {kind}\n"
        f"- message: {msg[:4000]}\n\n"
        "[Hard requirements]\n"
        f"{hints_block}\n"
        "- Output format: one pair ``<label>`` ... ``</label>`` wrapping **only** executable Python "
        "(no markdown fences, no commentary inside the tags — consistent with task 8 rules).\n"
        "- Return the **full** corrected program, not a unified diff.\n\n"
        f"[Instruction / input to satisfy]\n{input_text}\n"
    )


def _infer_task8_item(
    item: dict,
    task_description: str,
    task_id: int,
    cleaned_examples: list[dict],
    task8_shot_k: int,
    task8_retrieval_pool_size: int,
    retries: int,
    retry_wait_seconds: float,
    *,
    enable_react: bool = False,
    max_react_rounds: int = 2,
    accuracy_criterion: str = "exec_dual_pass",
) -> dict:
    example_id = item["example_id"]
    input_text = item["input_text"]
    expected = item["expected"]

    examples_str = select_examples_hybrid(
        all_examples=cleaned_examples,
        task_description=task_description,
        text2annotate=input_text,
        top_k=max(1, task8_shot_k),
        rerank_pool_size=max(20, task8_retrieval_pool_size),
        use_explanation=False,
        use_bm25_semantic_rerank=False,
        use_bm25_keyword_only=True,
        exclude_example_id=example_id,
    )

    prompt = build_prompt(task_description, input_text, task_id=task_id)
    base_prompt = prompt.replace("[[EXAMPLES]]\n\n", f"{examples_str}\n")

    print(f"[开始推理] task=8 example_id={example_id}")
    prediction = ""
    last_model_raw = ""
    for attempt in range(1, retries + 1):
        try:
            raw_prediction = annotate(base_prompt)
            last_model_raw = "" if raw_prediction is None else str(raw_prediction).strip()
            prediction = _extract_code_candidate(last_model_raw)
            print(
                f"[首次回答] task=8 example_id={example_id} "
                f"attempt={attempt}/{retries} output_len={len(prediction)}"
            )
            break
        except Exception as e:  # noqa: BLE001
            if attempt >= retries:
                print(f"[推理失败] task=8 example_id={example_id} attempt={attempt}/{retries} error={e}")
            else:
                print(f"[重试] task=8 example_id={example_id} attempt={attempt}/{retries} error={e}")
                time.sleep(retry_wait_seconds)
    code_executor = Task8CodeExecutor()

    passed, err_msg = False, ""
    pred_outcome = Task8ExecOutcome(ok=False, kind="empty", message="empty prediction")
    if prediction.strip():
        pred_outcome = code_executor.execute(prediction)
        passed, err_msg = pred_outcome.ok, pred_outcome.message
    if passed:
        print(f"[执行成功] task=8 example_id={example_id} 首次执行器检查通过")
    else:
        print(
            f"[执行报错] task=8 example_id={example_id} 首次执行器检查失败 "
            f"error_preview={str(err_msg)[:300].replace(chr(10), ' ')}"
        )
    react_round = 0
    react_prediction = prediction

    zero_shot_prompt = _build_task8_zero_shot_prompt(task_description, input_text, task_id)

    while enable_react and not passed and react_round < max_react_rounds:
        react_round += 1
        print(f"[ReAct开始] task=8 example_id={example_id} round={react_round}/{max_react_rounds}")
        repair_body = last_model_raw if last_model_raw.strip() else prediction
        react_prompt = _build_task8_react_repair_prompt(
            zero_shot_prompt=zero_shot_prompt,
            input_text=input_text,
            last_code=repair_body,
            outcome_kind=pred_outcome.kind,
            outcome_message=pred_outcome.message,
            round_idx=react_round,
            max_rounds=max_react_rounds,
        )
        try:
            raw_prediction = annotate(react_prompt)
            last_model_raw = "" if raw_prediction is None else str(raw_prediction).strip()
            react_prediction = _extract_code_candidate(last_model_raw)
            print(
                f"[ReAct回答] task=8 example_id={example_id} "
                f"round={react_round} output_len={len(react_prediction)}"
            )
        except Exception as e:  # noqa: BLE001
            print(f"[ReAct 失败] task=8 example_id={example_id} round={react_round} error={e}")
            break

        pred_outcome = code_executor.execute(react_prediction)
        passed, err_msg = pred_outcome.ok, pred_outcome.message
        if passed:
            print(f"[ReAct执行成功] task=8 example_id={example_id} round={react_round}")
        else:
            print(
                f"[ReAct执行报错] task=8 example_id={example_id} round={react_round} "
                f"error_preview={str(err_msg)[:300].replace(chr(10), ' ')}"
            )

    final_output = react_prediction if enable_react else prediction

    has_ref_answer = isinstance(expected, str) and expected.strip()
    ref_outcome: Optional[Task8ExecOutcome] = (
        code_executor.execute(expected, verbose=False) if has_ref_answer else None
    )

    prediction_outcome = (
        code_executor.execute(final_output, verbose=False)
        if final_output.strip()
        else pred_outcome
    )
    passed = prediction_outcome.ok
    err_msg = prediction_outcome.message

    is_match, is_text_match, exec_dual_pass = _infer_compute_task8_accuracy(
        criterion=accuracy_criterion,
        expected_non_empty=has_ref_answer,
        expected_text=expected if isinstance(expected, str) else "",
        final_output=final_output,
        ref_outcome=ref_outcome,
        pred_outcome=prediction_outcome,
    )

    exec_cmp = _task8_build_exec_comparison(
        ref_outcome,
        prediction_outcome,
        has_reference=has_ref_answer,
    )

    print(
        f"[样本完成] task=8 example_id={example_id} "
        f"criterion={accuracy_criterion} is_match={is_match} exec_dual_pass={exec_dual_pass} "
        f"is_text_match={is_text_match} passed_sanity_check={passed} react_rounds={react_round}"
    )
    return {
        "example_id": example_id,
        "input": input_text,
        "expected_output": expected,
        "model_output": final_output,
        "accuracy_criterion": accuracy_criterion,
        "is_match": is_match,
        "is_text_match": is_text_match,
        "exec_dual_pass": exec_dual_pass,
        "exec_comparison": exec_cmp,
        "executor_reference_ok": None if ref_outcome is None else ref_outcome.ok,
        "executor_reference_kind": None if ref_outcome is None else ref_outcome.kind,
        "executor_prediction_ok": prediction_outcome.ok,
        "executor_prediction_kind": prediction_outcome.kind,
        "passed_sanity_check": prediction_outcome.ok,
        "react_rounds": react_round,
        "last_error": prediction_outcome.message if not prediction_outcome.ok else "",
    }


def run_task8(
    output_dir: Path,
    examples_limit: int = 0,
    retries: int = 3,
    retry_wait_seconds: float = 2.0,
    resume: bool = False,
    task8_shot_k: int = 6,
    task8_retrieval_pool_size: int = 200,
    print_empty_prediction: str = "on",
    retrieval_batch_size: int = 8,
    accuracy_criterion: str = "exec_dual_pass",
    single_example_id: str = "",
    single_index: int = -1,
) -> dict:
    with _task_json_path().open("r", encoding="utf-8") as f:
        task_dict = json.load(f)

    task_id = 8
    task_name = task_dict["task_name"]
    task_description = TASK8_CANONICAL_DESCRIPTION
    all_examples = task_dict["examples"]
    if examples_limit > 0:
        all_examples = all_examples[:examples_limit]

    output_file = output_dir / f"openseek-{task_id}-examples-compare-task8opt.jsonl"
    done_ids = _load_done_ids(output_file) if resume else set()
    mode = "a" if resume else "w"

    if done_ids:
        print(f"[断点续跑] task=8, 已完成 {len(done_ids)} 条，继续剩余样本")

    cleaned_examples: list[dict] = []
    for ex in task_dict["examples"]:
        cleaned_examples.append(
            {
                "id": str(ex.get("id", "")).strip(),
                "input": str(ex.get("input", "")),
                "output": [_extract_output(ex.get("output", ""))],
            }
        )

    pending_items: list[dict] = []
    for example in all_examples:
        example_id = str(example.get("id", "")).strip()
        if resume and example_id in done_ids:
            continue
        pending_items.append(
            {
                "example_id": example_id,
                "input_text": str(example.get("input", "")),
                "expected": _extract_output(example.get("output", "")),
            }
        )

    pending_items = _narrow_pending_to_single_row(
        pending_items,
        single_example_id=single_example_id,
        single_index=single_index,
    )
    if str(single_example_id or "").strip() or single_index >= 0:
        only_id = str(pending_items[0].get("example_id", "")).strip()
        print(f"[单条推理] examples 划分，待跑 1 条 example_id={only_id}")

    bs = max(1, retrieval_batch_size)
    with output_file.open(mode, encoding="utf-8") as wf:
        for i in tqdm(range(0, len(pending_items), bs), desc=f"Task8 Optimized Inference: {task_name}"):
            chunk = pending_items[i : i + bs]
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(bs, len(chunk))) as tp_executor:
                futures = [
                    tp_executor.submit(
                        _infer_task8_item,
                        item=item,
                        task_description=task_description,
                        task_id=task_id,
                        cleaned_examples=cleaned_examples,
                        task8_shot_k=task8_shot_k,
                        task8_retrieval_pool_size=task8_retrieval_pool_size,
                        retries=retries,
                        retry_wait_seconds=retry_wait_seconds,
                        enable_react=True,
                        max_react_rounds=2,
                        accuracy_criterion=accuracy_criterion,
                    )
                    for item in chunk
                ]
                infer_results = [f.result() for f in futures]

            for row in infer_results:
                if print_empty_prediction == "on" and not row["model_output"]:
                    expected = str(row["expected_output"])
                    input_text = str(row["input"])
                    print(
                        f"[空预测] example_id={row['example_id']} expected_preview={expected[:80].replace(chr(10), ' ')} "
                        f"input_preview={input_text[:160].replace(chr(10), ' ')}"
                    )
                wf.write(json.dumps(row, ensure_ascii=False) + "\n")
                wf.flush()
                done_ids.add(row["example_id"])
                if str(single_example_id or "").strip() or single_index >= 0:
                    _print_task8_exec_comparison_report(row.get("exec_comparison") or {})

    total, match_count, accuracy = _compute_metrics_from_jsonl(output_file)
    print(
        f"[保存完成] task=8, criterion={accuracy_criterion}, total={total}, "
        f"matched={match_count}, accuracy={accuracy:.2%}, file={output_file}"
    )
    return {
        "task_id": 8,
        "task_name": task_name,
        "accuracy_criterion": accuracy_criterion,
        "total": total,
        "matched": match_count,
        "accuracy": accuracy,
        "file": str(output_file),
    }


def run_task8_test_samples(
    output_dir: Path,
    retries: int = 3,
    retry_wait_seconds: float = 2.0,
    resume: bool = False,
    task8_shot_k: int = 6,
    task8_retrieval_pool_size: int = 200,
    retrieval_batch_size: int = 8,
    accuracy_criterion: str = "exec_dual_pass",
    single_example_id: str = "",
    single_index: int = -1,
) -> dict:
    with _task_json_path().open("r", encoding="utf-8") as f:
        task_dict = json.load(f)

    task_id = 8
    task_name = task_dict["task_name"]
    task_description = TASK8_CANONICAL_DESCRIPTION
    test_samples = list(task_dict.get("test_samples", []))

    output_file = output_dir / f"openseek-{task_id}-test_samples-predictions.jsonl"
    done_ids = _load_done_ids(output_file) if resume else set()
    mode = "a" if resume else "w"

    if done_ids:
        print(f"[断点续跑] task=8 test_samples, 已完成 {len(done_ids)} 条，继续剩余样本")

    cleaned_examples: list[dict] = []
    for ex in task_dict["examples"]:
        cleaned_examples.append(
            {
                "id": str(ex.get("id", "")).strip(),
                "input": str(ex.get("input", "")),
                "output": [_extract_output(ex.get("output", ""))],
            }
        )

    pending_items: list[dict] = []
    for sample in test_samples:
        sample_id = str(sample.get("id", "")).strip()
        if resume and sample_id in done_ids:
            continue
        pending_items.append(
            {
                "example_id": sample_id,
                "input_text": str(sample.get("input", "")),
                "expected": "",
            }
        )

    pending_items = _narrow_pending_to_single_row(
        pending_items,
        single_example_id=single_example_id,
        single_index=single_index,
    )
    if str(single_example_id or "").strip() or single_index >= 0:
        only_id = str(pending_items[0].get("example_id", "")).strip()
        print(f"[单条推理] test_samples 划分，待跑 1 条 test_sample_id={only_id}")

    bs = max(1, retrieval_batch_size)
    total = 0
    with output_file.open(mode, encoding="utf-8") as wf:
        for i in tqdm(range(0, len(pending_items), bs), desc=f"Task8 Test Samples Inference: {task_name}"):
            chunk = pending_items[i : i + bs]
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(bs, len(chunk))) as tp_executor:
                futures = [
                    tp_executor.submit(
                        _infer_task8_item,
                        item=item,
                        task_description=task_description,
                        task_id=task_id,
                        cleaned_examples=cleaned_examples,
                        task8_shot_k=task8_shot_k,
                        task8_retrieval_pool_size=task8_retrieval_pool_size,
                        retries=retries,
                        retry_wait_seconds=retry_wait_seconds,
                        enable_react=True,
                        max_react_rounds=2,
                        accuracy_criterion=accuracy_criterion,
                    )
                    for item in chunk
                ]
                infer_results = [f.result() for f in futures]

            for row in infer_results:
                out_row = {
                    "test_sample_id": row["example_id"],
                    "prediction": row["model_output"],
                    "exec_comparison": row.get("exec_comparison"),
                }
                wf.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                wf.flush()
                done_ids.add(row["example_id"])
                total += 1
                if str(single_example_id or "").strip() or single_index >= 0:
                    _print_task8_exec_comparison_report(row.get("exec_comparison") or {})

    print(f"[保存完成] task=8 test_samples, total={total}, file={output_file}")
    return {
        "task_id": 8,
        "task_name": task_name,
        "total": total,
        "file": str(output_file),
    }


def main() -> None:
    args = parse_args()
    os.environ["DASHSCOPE_ENABLE_THINKING"] = "1" if args.thinking == "on" else "0"
    os.environ["ANNOTATE_LOG_EVERY_RESPONSE"] = "1" if args.print_model_output == "on" else "0"
    print(f"[thinking] DASHSCOPE_ENABLE_THINKING={os.environ['DASHSCOPE_ENABLE_THINKING']}")
    print(f"[print_model_output] ANNOTATE_LOG_EVERY_RESPONSE={os.environ['ANNOTATE_LOG_EVERY_RESPONSE']}")

    output_dir = _resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[准确率判据] accuracy_criterion={args.accuracy_criterion}")

    single_id = str(args.single_example_id or "").strip()
    if single_id and args.single_index >= 0:
        raise SystemExit("请只指定 --single_example_id 或 --single_index 之一，不要同时使用。")

    single_kw = {
        "single_example_id": single_id,
        "single_index": int(args.single_index),
    }

    if args.infer_split == "test_samples":
        summary_item = run_task8_test_samples(
            output_dir=output_dir,
            retries=args.retries,
            retry_wait_seconds=args.retry_wait_seconds,
            resume=args.resume,
            task8_shot_k=args.task8_shot_k,
            task8_retrieval_pool_size=args.task8_retrieval_pool_size,
            retrieval_batch_size=args.retrieval_batch_size,
            accuracy_criterion=args.accuracy_criterion,
            **single_kw,
        )
        summary_file = output_dir / "summary_task8_test_samples.json"
    else:
        summary_item = run_task8(
            output_dir=output_dir,
            examples_limit=args.examples_limit,
            retries=args.retries,
            retry_wait_seconds=args.retry_wait_seconds,
            resume=args.resume,
            task8_shot_k=args.task8_shot_k,
            task8_retrieval_pool_size=args.task8_retrieval_pool_size,
            print_empty_prediction=args.print_empty_prediction,
            retrieval_batch_size=args.retrieval_batch_size,
            accuracy_criterion=args.accuracy_criterion,
            **single_kw,
        )
        summary_file = output_dir / "summary_task8opt.json"

    with summary_file.open("w", encoding="utf-8") as f:
        json.dump([summary_item], f, ensure_ascii=False, indent=2)
    print(f"[汇总完成] {summary_file}")


if __name__ == "__main__":
    main()
