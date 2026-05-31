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
from pathlib import Path

from tqdm import tqdm

from method_hyb import annotate_nvidia as annotate
from method_hyb import build_prompt, select_examples_hybrid


REPO_ROOT = Path(__file__).resolve().parent.parent
TASK8_FILE = "openseek-8_kernel_generation.json"
TASK8_CANONICAL_DESCRIPTION = (
    "In this task, you are asked to generate a Triton kernel implementation based on the given instruction. "
    "The output should be valid, executable Python code containing Triton kernel definitions and necessary wrappers."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task8 test_samples 专用推理脚本。")
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
        "--retrieval_batch_size",
        type=int,
        default=8,
        help="并行推理批大小（按批并发调用模型），默认 8。",
    )
    return parser.parse_args()


def _task_json_path() -> Path:
    return REPO_ROOT / "data" / TASK8_FILE


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


def _run_task8_code_sanity_check(code: str) -> tuple[bool, str]:
    if not code or not isinstance(code, str):
        print("[执行器] sanity_check 失败: empty code")
        return False, "empty code"

    cleaned = _extract_code_candidate(code)

    stripped = cleaned.strip()
    if not stripped:
        print("[执行器] sanity_check 失败: empty code after cleaning")
        return False, "empty code after cleaning"

    try:
        tree = ast.parse(cleaned)
    except SyntaxError as e:
        print(f"[执行器] sanity_check 失败: syntax error: {e}")
        return False, f"syntax error: {e}"

    has_def = any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        for node in ast.walk(tree)
    )
    if not has_def:
        print("[执行器] sanity_check 失败: no function/class definition found")
        return False, "no function/class definition found"

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
        if missing_name in {"torch", "triton"}:
            print(f"[执行器] sanity_check 提示: 缺少依赖 {missing_name}，语法与结构已通过")
            return True, f"missing optional dependency: {missing_name}"
        err = traceback.format_exc(limit=4)
        print(f"[执行器] sanity_check 编译/执行失败:\n{err}")
        return False, err
    except Exception:
        err = traceback.format_exc(limit=4)
        print(f"[执行器] sanity_check 编译/执行失败:\n{err}")
        return False, err
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
        print("[执行器] sanity_check 失败: no callable function found in namespace")
        return False, "no callable function found in generated code namespace"

    print("[执行器] sanity_check 通过")
    return True, ""


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
) -> dict:
    example_id = item["example_id"]
    input_text = item["input_text"]

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
    for attempt in range(1, retries + 1):
        try:
            raw_prediction = annotate(base_prompt)
            prediction = _extract_code_candidate("" if raw_prediction is None else str(raw_prediction))
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

    passed, err_msg = _run_task8_code_sanity_check(prediction)
    if passed:
        print(f"[执行成功] task=8 example_id={example_id} 首次执行器检查通过")
    else:
        print(
            f"[执行报错] task=8 example_id={example_id} 首次执行器检查失败 "
            f"error_preview={str(err_msg)[:300].replace(chr(10), ' ')}"
        )

    react_round = 0
    react_prediction = prediction
    while enable_react and not passed and react_round < max_react_rounds:
        react_round += 1
        print(f"[ReAct开始] task=8 example_id={example_id} round={react_round}/{max_react_rounds}")
        react_prompt = (
            f"{base_prompt}\n\n"
            "The previous answer tried to provide Triton kernel code, but it FAILED basic checks.\n"
            "Here is the code you previously gave:\n"
            "```python\n"
            f"{prediction[:4000]}\n"
            "```\n"
            "The error message was:\n"
            f"{err_msg[:2000]}\n\n"
            "Please think step by step, fix the issues, and return a COMPLETE, self-contained "
            "Triton kernel + Python wrapper implementation wrapped in <label>...</label>."
        )
        try:
            raw_prediction = annotate(react_prompt)
            react_prediction = _extract_code_candidate(
                "" if raw_prediction is None else str(raw_prediction)
            )
            print(
                f"[ReAct回答] task=8 example_id={example_id} "
                f"round={react_round} output_len={len(react_prediction)}"
            )
        except Exception as e:  # noqa: BLE001
            print(f"[ReAct 失败] task=8 example_id={example_id} round={react_round} error={e}")
            break

        passed, err_msg = _run_task8_code_sanity_check(react_prediction)
        if passed:
            print(f"[ReAct执行成功] task=8 example_id={example_id} round={react_round}")
        else:
            print(
                f"[ReAct执行报错] task=8 example_id={example_id} round={react_round} "
                f"error_preview={str(err_msg)[:300].replace(chr(10), ' ')}"
            )

    final_output = react_prediction if enable_react else prediction
    return {
        "test_sample_id": example_id,
        "prediction": final_output,
    }


def run_task8_test_samples(
    output_dir: Path,
    retries: int = 3,
    retry_wait_seconds: float = 2.0,
    resume: bool = False,
    task8_shot_k: int = 6,
    task8_retrieval_pool_size: int = 200,
    retrieval_batch_size: int = 8,
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
            }
        )

    bs = max(1, retrieval_batch_size)
    total = 0
    with output_file.open(mode, encoding="utf-8") as wf:
        for i in tqdm(range(0, len(pending_items), bs), desc=f"Task8 Test Samples Inference: {task_name}"):
            chunk = pending_items[i : i + bs]
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(bs, len(chunk))) as executor:
                futures = [
                    executor.submit(
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
                    )
                    for item in chunk
                ]
                infer_results = [f.result() for f in futures]

            for row in infer_results:
                wf.write(json.dumps(row, ensure_ascii=False) + "\n")
                wf.flush()
                done_ids.add(row["test_sample_id"])
                total += 1

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
    print(f"[输出目录] {output_dir}")

    summary_item = run_task8_test_samples(
        output_dir=output_dir,
        retries=args.retries,
        retry_wait_seconds=args.retry_wait_seconds,
        resume=args.resume,
        task8_shot_k=args.task8_shot_k,
        task8_retrieval_pool_size=args.task8_retrieval_pool_size,
        retrieval_batch_size=args.retrieval_batch_size,
    )
    summary_file = output_dir / "summary_task8_test_samples.json"

    with summary_file.open("w", encoding="utf-8") as f:
        json.dump([summary_item], f, ensure_ascii=False, indent=2)
    print(f"[汇总完成] {summary_file}")


if __name__ == "__main__":
    main()
