import argparse
import json
import os
import re
import time
from pathlib import Path

from tqdm import tqdm

from method_task6 import (
    annotate_nvidia as annotate,
    build_prompt,
    select_task6_examples_balanced,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
TASK6_FILE = "openseek-6_mnli_same_genre_classification.json"
TASK6_CANONICAL_DESCRIPTION = (
    "In this task, you are given Sentence 1, Sentence 2, and a Genre. "
    "Your job is to decide whether both sentences belong to that same genre, "
    'and output "Y" for yes or "N" for no.\n'
    "Candidate genres: face-to-face, government, letters, 9/11, slate, telephone, "
    "travel, verbatim, oup, fiction."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task6 专用推理（类平衡检索 + 准确率统计）。")
    parser.add_argument("--examples_limit", type=int, default=0, help="最多推理多少条；<=0 表示全部。")
    parser.add_argument("--output_dir", type=str, default="examples", help="结果输出目录。")
    parser.add_argument("--retries", type=int, default=3, help="单条失败重试次数。")
    parser.add_argument("--retry_wait_seconds", type=float, default=2.0, help="重试等待秒数。")
    parser.add_argument("--resume", action="store_true", help="开启断点续跑。")
    parser.add_argument("--task6_shot_k", type=int, default=8, help="每条样本检索示例数，默认 8。")
    parser.add_argument(
        "--task6_balanced_icl",
        type=str,
        choices=["on", "off"],
        default="on",
        help="是否对 Y/N 做类平衡检索（推荐 on）。",
    )
    parser.add_argument(
        "--task6_retrieval_pool_size",
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
    return parser.parse_args()


def _task_json_path() -> Path:
    return REPO_ROOT / "data" / TASK6_FILE


def _normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


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


def _genre_from_input(input_text: str) -> str:
    m = re.search(r"Genre:\s*([^\.\n]+)\.?", input_text, flags=re.IGNORECASE)
    return m.group(1).strip().lower() if m else "unknown"


def _normalize_pred_to_label(pred: str) -> str:
    p = _normalize_text(pred).upper()
    if p in {"Y", "N"}:
        return p
    if "<LABEL>" in p and "</LABEL>" in p:
        m = re.search(r"<LABEL>\s*([YN])\s*</LABEL>", p)
        if m:
            return m.group(1)
    return pred.strip()


def run_task6(
    output_dir: Path,
    examples_limit: int = 0,
    retries: int = 3,
    retry_wait_seconds: float = 2.0,
    resume: bool = False,
    task6_shot_k: int = 8,
    task6_balanced_icl: str = "on",
    task6_retrieval_pool_size: int = 200,
    print_empty_prediction: str = "on",
) -> dict:
    with _task_json_path().open("r", encoding="utf-8") as f:
        task_dict = json.load(f)

    task_id = 6
    task_name = task_dict["task_name"]
    task_description = TASK6_CANONICAL_DESCRIPTION
    all_examples = task_dict["examples"]
    if examples_limit > 0:
        all_examples = all_examples[:examples_limit]

    mode_name = "balanced" if task6_balanced_icl == "on" else "plain"
    output_file = output_dir / f"openseek-{task_id}-examples-compare-task6opt-{mode_name}.jsonl"
    done_ids = _load_done_ids(output_file) if resume else set()
    mode = "a" if resume else "w"

    if done_ids:
        print(f"[断点续跑] task=6, 已完成 {len(done_ids)} 条，继续剩余样本")

    with output_file.open(mode, encoding="utf-8") as wf:
        for example in tqdm(all_examples, desc=f"Task6 Optimized Inference: {task_name}"):
            example_id = str(example.get("id", "")).strip()
            if resume and example_id in done_ids:
                continue

            input_text = str(example.get("input", ""))
            expected = _extract_output(example.get("output", ""))
            genre = _genre_from_input(input_text)
            examples_str = select_task6_examples_balanced(
                all_examples=task_dict["examples"],
                task_description=task_description,
                text2annotate=input_text,
                top_k=task6_shot_k,
                rerank_pool_size=task6_retrieval_pool_size,
                exclude_example_id=example_id,
                balanced=(task6_balanced_icl == "on"),
            )

            prompt = build_prompt(task_description, input_text, task_id=task_id)
            input_prompt = prompt.replace("[[EXAMPLES]]\n\n", f"{examples_str}\n")

            prediction = ""
            for attempt in range(1, retries + 1):
                try:
                    raw_prediction = annotate(input_prompt)
                    prediction = "" if raw_prediction is None else str(raw_prediction).strip()
                    prediction = _normalize_pred_to_label(prediction)
                    break
                except Exception as e:  # noqa: BLE001
                    if attempt >= retries:
                        print(f"[推理失败] task=6 example_id={example_id} attempt={attempt}/{retries} error={e}")
                    else:
                        print(f"[重试] task=6 example_id={example_id} attempt={attempt}/{retries} error={e}")
                        time.sleep(retry_wait_seconds)

            is_match = _normalize_text(prediction) == _normalize_text(expected)
            if print_empty_prediction == "on" and not prediction:
                print(
                    f"[空预测] example_id={example_id} expected={expected} genre={genre} "
                    f"input_preview={input_text[:160].replace(chr(10), ' ')}"
                )
            row = {
                "example_id": example_id,
                "input": input_text,
                "genre": genre,
                "expected_output": expected,
                "model_output": prediction,
                "is_match": is_match,
            }
            wf.write(json.dumps(row, ensure_ascii=False) + "\n")
            wf.flush()
            done_ids.add(example_id)

    total, match_count, accuracy = _compute_metrics_from_jsonl(output_file)
    print(f"[保存完成] task=6, total={total}, matched={match_count}, accuracy={accuracy:.2%}, file={output_file}")
    return {
        "task_id": 6,
        "task_name": task_name,
        "total": total,
        "matched": match_count,
        "accuracy": accuracy,
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

    summary_item = run_task6(
        output_dir=output_dir,
        examples_limit=args.examples_limit,
        retries=args.retries,
        retry_wait_seconds=args.retry_wait_seconds,
        resume=args.resume,
        task6_shot_k=args.task6_shot_k,
        task6_balanced_icl=args.task6_balanced_icl,
        task6_retrieval_pool_size=args.task6_retrieval_pool_size,
        print_empty_prediction=args.print_empty_prediction,
    )
    summary_file = output_dir / "summary_task6opt.json"
    with summary_file.open("w", encoding="utf-8") as f:
        json.dump([summary_item], f, ensure_ascii=False, indent=2)
    print(f"[汇总完成] {summary_file}")


if __name__ == "__main__":
    main()
