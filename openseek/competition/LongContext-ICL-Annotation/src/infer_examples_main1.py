import argparse
import json
import time
from pathlib import Path

from tqdm import tqdm

from method_hyb import build_prompt, select_examples
from method_hyb import annotate_nvidia as annotate
from postprocess_task1_outputs import postprocess_jsonl_inplace


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
        description="按 main1.py 推理流程，仅在 examples 上推理并统计准确率。"
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
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="单条样本推理失败时最大重试次数。",
    )
    parser.add_argument(
        "--retry_wait_seconds",
        type=float,
        default=2.0,
        help="重试间隔秒数。",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="开启断点续跑：已存在文件时跳过已完成 example_id。",
    )
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

    # 完全对齐 main1.py：固定 examples 池为前 100 条，不做泄漏剔除。
    icl_examples = task_dict["examples"][:100]

    output_file = output_dir / f"openseek-{task_id}-examples-main1-compare-test.jsonl"
    done_ids = _load_done_ids(output_file) if resume else set()
    mode = "a" if resume else "w"

    if done_ids:
        print(f"[断点续跑] task={task_id}, 已完成 {len(done_ids)} 条，继续剩余样本")

    with output_file.open(mode, encoding="utf-8") as wf:
        for example in tqdm(all_examples, desc=f"Examples Inference Task {task_id}: {task_name}"):
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

    if task_id == 1 and output_file.exists():
        pp_stats = postprocess_jsonl_inplace(output_file)
        print(
            f"[task1 后处理] changed={pp_stats['changed']}/{pp_stats['total']}, "
            f"accuracy={pp_stats['accuracy']:.2%}, file={output_file}"
        )

    total, match_count, accuracy = _compute_metrics_from_jsonl(output_file)
    print(
        f"[保存完成] task={task_id}, total={total}, matched={match_count}, "
        f"accuracy={accuracy:.2%}, file={output_file}"
    )
    return {
        "task_id": task_id,
        "task_name": task_name,
        "total": total,
        "matched": match_count,
        "accuracy": accuracy,
        "file": str(output_file),
        "postprocessed": task_id == 1,
    }


def main() -> None:
    args = parse_args()
    task_start = max(1, args.task_start)
    task_end = min(8, args.task_end)
    if task_start > task_end:
        raise ValueError(f"task_start({task_start}) 不能大于 task_end({task_end})")

    output_dir = _resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[输出目录] {output_dir}")

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

    summary_file = output_dir / "summary.json"
    with summary_file.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[汇总完成] {summary_file}")


if __name__ == "__main__":
    main()
