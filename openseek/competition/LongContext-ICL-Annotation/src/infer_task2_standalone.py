import argparse
import concurrent.futures
import json
import os
import re
import time
from pathlib import Path

from tqdm import tqdm

from method_hyb_thinking_default import annotate_nvidia as annotate
from method_hyb_thinking_default import build_prompt, select_examples


REPO_ROOT = Path(__file__).resolve().parent.parent
TASK2_FILE = "openseek-2_count_nouns_verbs.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task2 独立推理脚本（count nouns/verbs）。")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="tokenizer 路径或模型名。")
    parser.add_argument("--examples_limit", type=int, default=0, help="最多推理多少条 examples；<=0 全部。")
    parser.add_argument("--output_dir", type=str, default="examples_main1_task2_v2", help="输出目录。")
    parser.add_argument("--retries", type=int, default=3, help="单条失败重试次数。")
    parser.add_argument("--retry_wait_seconds", type=float, default=2.0, help="重试等待秒数。")
    parser.add_argument("--resume", action="store_true", help="开启断点续跑。")
    parser.add_argument("--icl_pool_size", type=int, default=100, help="ICL 候选池大小（取前 N 条）。")
    parser.add_argument("--top_k", type=int, default=5, help="每条样本召回示例数量。")
    parser.add_argument(
        "--same_pos_only",
        type=str,
        choices=["on", "off"],
        default="on",
        help="是否仅从同任务子类型（nouns/verbs）示例中召回。",
    )
    parser.add_argument(
        "--use_explanation",
        type=str,
        choices=["on", "off"],
        default="on",
        help="是否在 ICL 示例中保留 explanation 字段。",
    )
    parser.add_argument(
        "--use_bm25_semantic_rerank",
        type=str,
        choices=["on", "off"],
        default="on",
        help="是否启用 BM25+语义+重排混合召回。",
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
        help="是否打印模型原始输出预览（ANNOTATE_LOG_EVERY_RESPONSE）。",
    )
    parser.add_argument("--bs", type=int, default=8, help="并行批大小（同时推理请求数）。")
    parser.add_argument(
        "--main1_compatible",
        type=str,
        choices=["on", "off"],
        default="off",
        help="是否使用与 infer_examples_main1.py 对齐的配置。",
    )
    return parser.parse_args()


def _task_json_path() -> Path:
    return REPO_ROOT / "data" / TASK2_FILE


def _resolve_output_dir(output_dir: str) -> Path:
    p = Path(output_dir)
    if p.is_absolute():
        return p
    return (REPO_ROOT / p).resolve()


def _normalize_text(text: str) -> str:
    return " ".join(str(text).strip().split())


def _extract_output(output_value) -> str:
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
    return total, matched, (matched / total if total else 0.0)


def _target_pos(input_text: str) -> str:
    text = (input_text or "").lower()
    if re.search(r"count the number of nouns", text):
        return "nouns"
    if re.search(r"count the number of verbs", text):
        return "verbs"
    return "unknown"


def _build_icl_pool(
    all_examples: list[dict],
    current_example_id: str,
    current_input: str,
    icl_pool_size: int,
    same_pos_only: bool,
) -> list[dict]:
    pool = [e for e in all_examples[:icl_pool_size] if str(e.get("id", "")).strip() != current_example_id]
    if not same_pos_only:
        return pool

    cur_pos = _target_pos(current_input)
    if cur_pos == "unknown":
        return pool

    filtered = [e for e in pool if _target_pos(str(e.get("input", ""))) == cur_pos]
    return filtered if filtered else pool


def _compose_task2_prompt(
    base_prompt: str,
    examples_str: str,
    *,
    append_task2_constraint: bool = True,
) -> str:
    """
    统一组装 task2 输入提示，避免示例块换行/占位符替换细节导致的格式漂移。
    """
    ex_block = (examples_str or "").strip()
    if ex_block:
        ex_block = f"{ex_block}\n\n"

    # 与 main1 对齐：优先替换标准占位符；若占位符缺失则安全追加到末尾。
    marker = "[[EXAMPLES]]\n\n"
    if marker in base_prompt:
        prompt = base_prompt.replace(marker, ex_block)
    else:
        prompt = f"{base_prompt.rstrip()}\n\n{ex_block}"

    # task2 强约束：最终输出只保留一个数字标签，减少 Label: 3 / \"label\" 3 等漂格式。
    if append_task2_constraint:
        prompt += (
            "\n### Output constraint for task2\n"
            "Final line MUST be exactly one pair of tags with one non-negative integer only:\n"
            "<label>NUMBER</label>\n"
            "Do not output Label:, quoted label, or any text outside tags.\n"
        )
    return prompt


def _task2_prediction_fallback(prediction: str) -> str:
    """
    task2 本地兜底：若返回文本不是纯数字，尝试提取首个非负整数。
    """
    s = (prediction or "").strip()
    if not s:
        return s
    if re.fullmatch(r"\d+", s):
        return s
    m = re.search(r"\b(\d+)\b", s)
    if m:
        return m.group(1)
    return s


def _infer_one(
    example: dict,
    task_dict: dict,
    task_description: str,
    tokenizer_path: str | None,
    icl_pool_size: int,
    same_pos_only: bool,
    top_k: int,
    use_explanation: bool,
    use_bm25_semantic_rerank: bool,
    retries: int,
    retry_wait_seconds: float,
    append_task2_constraint: bool,
) -> dict:
    task_id = 2
    example_id = str(example.get("id", "")).strip()
    input_text = str(example.get("input", ""))
    expected = _extract_output(example.get("output", ""))

    icl_pool = _build_icl_pool(
        all_examples=task_dict["examples"],
        current_example_id=example_id,
        current_input=input_text,
        icl_pool_size=max(1, icl_pool_size),
        same_pos_only=same_pos_only,
    )
    prompt = build_prompt(task_description, input_text, task_id=task_id)
    examples_str = select_examples(
        icl_pool,
        task_description,
        input_text,
        tokenizer_path=tokenizer_path,
        hybrid=True,
        top_k=max(1, top_k),
        use_explanation=use_explanation,
        use_bm25_semantic_rerank=use_bm25_semantic_rerank,
    )
    input_prompt = _compose_task2_prompt(
        prompt,
        examples_str,
        append_task2_constraint=append_task2_constraint,
    )

    prediction = ""
    for attempt in range(1, retries + 1):
        try:
            raw_prediction = annotate(input_prompt)
            prediction = "" if raw_prediction is None else str(raw_prediction).strip()
            prediction = _task2_prediction_fallback(prediction)
            break
        except Exception as e:  # noqa: BLE001
            if attempt >= retries:
                print(
                    f"[推理失败] task=2 example_id={example_id} "
                    f"attempt={attempt}/{retries} error={e}"
                )
            else:
                print(
                    f"[重试] task=2 example_id={example_id} "
                    f"attempt={attempt}/{retries} error={e}"
                )
                time.sleep(retry_wait_seconds)

    is_match = _normalize_text(prediction) == _normalize_text(expected)
    return {
        "example_id": example_id,
        "target_pos": _target_pos(input_text),
        "input": input_text,
        "expected_output": expected,
        "model_output": prediction,
        "is_match": is_match,
    }


def run_task2(
    output_dir: Path,
    tokenizer_path: str | None = None,
    examples_limit: int = 0,
    retries: int = 3,
    retry_wait_seconds: float = 2.0,
    resume: bool = False,
    icl_pool_size: int = 100,
    top_k: int = 5,
    same_pos_only: bool = True,
    use_explanation: bool = True,
    use_bm25_semantic_rerank: bool = True,
    bs: int = 8,
    main1_compatible: bool = False,
) -> dict:
    with _task_json_path().open("r", encoding="utf-8") as f:
        task_dict = json.load(f)

    task_id = 2
    task_name = task_dict["task_name"]
    task_description = task_dict["Definition"][0]
    all_examples = list(task_dict["examples"])
    if examples_limit > 0:
        all_examples = all_examples[:examples_limit]

    output_file = output_dir / "openseek-2-examples-task2-standalone-compare.jsonl"
    done_ids = _load_done_ids(output_file) if resume else set()
    mode = "a" if resume else "w"
    if done_ids:
        print(f"[断点续跑] task=2, 已完成 {len(done_ids)} 条，继续剩余样本")

    if main1_compatible:
        icl_pool_size = 100
        top_k = 3
        same_pos_only = False
        use_explanation = True
        use_bm25_semantic_rerank = True
        bs = 1
        print(
            "[main1_compatible] enabled: icl_pool_size=100, top_k=3, "
            "same_pos_only=off, use_explanation=on, "
            "use_bm25_semantic_rerank=on, bs=1"
        )

    pending_examples = []
    for example in all_examples:
        example_id = str(example.get("id", "")).strip()
        if resume and example_id in done_ids:
            continue
        pending_examples.append(example)

    batch_size = max(1, int(bs))
    print(f"[并行推理] bs={batch_size}")
    with output_file.open(mode, encoding="utf-8") as wf:
        for i in tqdm(
            range(0, len(pending_examples), batch_size),
            desc=f"Task2 Inference: {task_name}",
        ):
            chunk = pending_examples[i : i + batch_size]
            with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
                futures = [
                    executor.submit(
                        _infer_one,
                        example=example,
                        task_dict=task_dict,
                        task_description=task_description,
                        tokenizer_path=tokenizer_path,
                        icl_pool_size=icl_pool_size,
                        same_pos_only=same_pos_only,
                        top_k=top_k,
                        use_explanation=use_explanation,
                        use_bm25_semantic_rerank=use_bm25_semantic_rerank,
                        retries=retries,
                        retry_wait_seconds=retry_wait_seconds,
                        append_task2_constraint=not main1_compatible,
                    )
                    for example in chunk
                ]
                infer_rows = [f.result() for f in futures]

            for row in infer_rows:
                wf.write(json.dumps(row, ensure_ascii=False) + "\n")
                wf.flush()
                done_ids.add(str(row.get("example_id", "")).strip())

    total, matched, accuracy = _compute_metrics_from_jsonl(output_file)
    print(
        f"[保存完成] task=2 total={total}, matched={matched}, "
        f"accuracy={accuracy:.2%}, file={output_file}"
    )
    return {
        "task_id": 2,
        "task_name": task_name,
        "total": total,
        "matched": matched,
        "accuracy": accuracy,
        "file": str(output_file),
    }


def main() -> None:
    args = parse_args()
    os.environ["DASHSCOPE_ENABLE_THINKING"] = "1" if args.thinking == "on" else "0"
    os.environ["ANNOTATE_LOG_EVERY_RESPONSE"] = "1" if args.print_model_output == "on" else "0"
    # task2 可开数值兜底：无 <label> 时尝试从文本中提取首个整数。
    os.environ["ANNOTATE_FALLBACK_NUMBER"] = "1"
    print(f"[thinking] DASHSCOPE_ENABLE_THINKING={os.environ['DASHSCOPE_ENABLE_THINKING']}")
    print(f"[print_model_output] ANNOTATE_LOG_EVERY_RESPONSE={os.environ['ANNOTATE_LOG_EVERY_RESPONSE']}")
    print(f"[numeric_fallback] ANNOTATE_FALLBACK_NUMBER={os.environ['ANNOTATE_FALLBACK_NUMBER']}")

    output_dir = _resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[输出目录] {output_dir}")

    summary = run_task2(
        output_dir=output_dir,
        tokenizer_path=args.tokenizer_path,
        examples_limit=args.examples_limit,
        retries=args.retries,
        retry_wait_seconds=args.retry_wait_seconds,
        resume=args.resume,
        icl_pool_size=args.icl_pool_size,
        top_k=args.top_k,
        same_pos_only=args.same_pos_only == "on",
        use_explanation=args.use_explanation == "on",
        use_bm25_semantic_rerank=args.use_bm25_semantic_rerank == "on",
        bs=args.bs,
        main1_compatible=args.main1_compatible == "on",
    )

    summary_file = output_dir / "summary_task2_standalone.json"
    with summary_file.open("w", encoding="utf-8") as f:
        json.dump([summary], f, ensure_ascii=False, indent=2)
    print(f"[汇总完成] {summary_file}")


if __name__ == "__main__":
    main()
