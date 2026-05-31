"""
Task 2（openseek-2）：在数据集 examples 上做推理，流程对齐 main1.py / infer_examples_main1.py。

要点：
- method_hyb 的 build_prompt、select_examples、annotate_nvidia
- ICL 池固定为 data 内该任务 examples 的前 100 条（与 main1 evaluate 一致，不做剔除）
- 每条样本从这 100 条中 hybrid 检索 top_k=3、use_explanation=True、BM25+语义重排
"""

import argparse
import json
import os
import time
from pathlib import Path

from tqdm import tqdm

from method_hyb import build_prompt, select_examples
from method_hyb import annotate_nvidia as annotate


REPO_ROOT = Path(__file__).resolve().parent.parent
TASK2_FILE = "openseek-2_count_nouns_verbs.json"
TASK_ID = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Task2 examples 推理，流程与 main1.py / infer_examples_main1 对齐（仅任务 2）。"
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="Qwen tokenizer 目录或 HuggingFace 模型 ID；默认 Qwen3-4B（若不存在则 Qwen/Qwen3-4B）。",
    )
    parser.add_argument(
        "--examples_limit",
        type=int,
        default=0,
        help="最多推理多少条 examples；<=0 表示全部。",
    )
    parser.add_argument("--output_dir", type=str, default="examples_main1_task2", help="输出目录。")
    parser.add_argument(
        "--output_file_name",
        type=str,
        default="openseek-2-examples-main1-compare.jsonl",
        help="输出 jsonl 文件名。",
    )
    parser.add_argument("--retries", type=int, default=3, help="单条推理失败时的最大重试次数。")
    parser.add_argument(
        "--retry_wait_seconds",
        type=float,
        default=2.0,
        help="重试间隔秒数。",
    )
    parser.add_argument("--resume", action="store_true", help="断点续跑：跳过已写入的 example_id。")
    parser.add_argument(
        "--thinking",
        type=str,
        choices=["on", "off"],
        default="off",
        help="是否开启 thinking（DASHSCOPE_ENABLE_THINKING）。",
    )
    parser.add_argument(
        "--print_model_output",
        type=str,
        choices=["on", "off"],
        default="off",
        help="是否打印每条模型响应预览（ANNOTATE_LOG_EVERY_RESPONSE）。",
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
    accuracy = (matched / total) if total else 0.0
    return total, matched, accuracy


def run_task2_examples_main1_aligned(
    output_dir: Path,
    tokenizer_path: str | None = None,
    examples_limit: int = 0,
    output_file_name: str = "openseek-2-examples-main1-compare.jsonl",
    retries: int = 3,
    retry_wait_seconds: float = 2.0,
    resume: bool = False,
) -> dict:
    with _task_json_path().open("r", encoding="utf-8") as f:
        task_dict = json.load(f)

    task_name = task_dict["task_name"]
    task_description = task_dict["Definition"][0]

    icl_examples = task_dict["examples"][:100]

    all_examples = task_dict["examples"]
    if examples_limit > 0:
        all_examples = all_examples[:examples_limit]

    output_file = output_dir / output_file_name
    done_ids = _load_done_ids(output_file) if resume else set()
    mode = "a" if resume else "w"

    if resume and done_ids:
        print(f"[断点续跑] 已完成 {len(done_ids)} 条，跳过已写入 example_id")

    with output_file.open(mode, encoding="utf-8") as wf:
        for example in tqdm(all_examples, desc=f"Task2 Examples (main1 对齐): {task_name}"):
            example_id = str(example.get("id", "")).strip()
            if resume and example_id in done_ids:
                continue

            input_text = example.get("input", "")
            expected = _extract_output(example.get("output", ""))

            prompt = build_prompt(task_description, input_text, task_id=TASK_ID)
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
                            f"[推理失败] example_id={example_id} "
                            f"attempt={attempt}/{retries} error={e}"
                        )
                    else:
                        print(
                            f"[重试] example_id={example_id} "
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
        f"[保存完成] task=2, total={total}, matched={match_count}, "
        f"accuracy={accuracy:.2%}, file={output_file}"
    )
    return {
        "task_id": TASK_ID,
        "task_name": task_name,
        "total": total,
        "matched": match_count,
        "accuracy": accuracy,
        "icl_pool_size_fixed": len(icl_examples),
        "file": str(output_file),
    }


def main() -> None:
    args = parse_args()
    os.environ["DASHSCOPE_ENABLE_THINKING"] = "1" if args.thinking == "on" else "0"
    os.environ["ANNOTATE_LOG_EVERY_RESPONSE"] = "1" if args.print_model_output == "on" else "0"

    _default_tok = REPO_ROOT / "Qwen3-4B"
    tokenizer_path = args.tokenizer_path
    if tokenizer_path is None:
        tokenizer_path = str(_default_tok) if _default_tok.is_dir() else "Qwen/Qwen3-4B"

    output_dir = _resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[输出目录] {output_dir}")
    print(f"[tokenizer_path] {tokenizer_path}")

    summary = run_task2_examples_main1_aligned(
        output_dir=output_dir,
        tokenizer_path=tokenizer_path,
        examples_limit=args.examples_limit,
        output_file_name=args.output_file_name,
        retries=args.retries,
        retry_wait_seconds=args.retry_wait_seconds,
        resume=args.resume,
    )

    summary_file = output_dir / "summary_task2_examples_main1.json"
    with summary_file.open("w", encoding="utf-8") as f:
        json.dump([summary], f, ensure_ascii=False, indent=2)
    print(f"[汇总完成] {summary_file}")


if __name__ == "__main__":
    main()
