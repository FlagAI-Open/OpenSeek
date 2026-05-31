import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

from tqdm import tqdm

from method_task6_v2 import extract_task6_parts, judge_sentence_genre
from postprocess_task6_v2 import apply_task6_v2_postprocess


REPO_ROOT = Path(__file__).resolve().parent.parent
TASK6_FILE = "openseek-6_mnli_same_genre_classification.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Task6-v3：句子级 AND 合成后，立即执行与 postprocess_task6_v2 相同的后处理"
            "（N+N 同领域、Y+N / N+Y 带上下文重判另一句），"
            "写出带 domain_same_pred / sentence2_context_pred / model_output_base 的 JSONL。"
        )
    )
    parser.add_argument("--examples_limit", type=int, default=0, help="最多推理多少条；<=0 表示全部。")
    parser.add_argument("--output_dir", type=str, default="examples", help="结果输出目录。")
    parser.add_argument("--retries", type=int, default=3, help="单条句子判定 / 后处理失败重试次数。")
    parser.add_argument("--retry_wait_seconds", type=float, default=2.0, help="重试等待秒数。")
    parser.add_argument("--resume", action="store_true", help="开启断点续跑（按最终输出 JSONL 的 example_id）。")
    parser.add_argument(
        "--thinking",
        type=str,
        choices=["on", "off"],
        default="off",
        help="是否开启模型 thinking（映射到 DASHSCOPE_ENABLE_THINKING）。",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="并行推理批大小：每批最多同时处理多少条样本（线程池 max_workers）。默认 16。",
    )
    return parser.parse_args()


def _task_json_path() -> Path:
    return REPO_ROOT / "data" / TASK6_FILE


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


def _compute_base_accuracy_from_jsonl(output_file: Path) -> tuple[int, int, float]:
    """按 model_output_base 与 expected 统计 AND 阶段准确率。"""
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
            exp = _extract_output(row.get("expected_output", ""))
            base = _extract_output(row.get("model_output_base", ""))
            if base == exp:
                matched += 1
    accuracy = (matched / total) if total else 0.0
    return total, matched, accuracy


def _predict_with_retry(sentence: str, genre: str, retries: int, retry_wait_seconds: float) -> str:
    pred = ""
    for attempt in range(1, retries + 1):
        try:
            pred = judge_sentence_genre(sentence=sentence, genre=genre)
            return pred
        except Exception as e:  # noqa: BLE001
            if attempt >= retries:
                print(f"[句子判定失败] attempt={attempt}/{retries} error={e}")
            else:
                print(f"[句子判定重试] attempt={attempt}/{retries} error={e}")
                time.sleep(retry_wait_seconds)
    return pred


def _process_single_v3_example(
    example: dict,
    *,
    retries: int,
    retry_wait_seconds: float,
) -> dict:
    """单条样本：两句判定 + AND + 后处理，供线程池调用。"""
    example_id = str(example.get("id", "")).strip()
    input_text = str(example.get("input", ""))
    expected = _extract_output(example.get("output", ""))
    sentence1, sentence2, genre = extract_task6_parts(input_text)

    s1_pred = _predict_with_retry(sentence1, genre, retries, retry_wait_seconds)
    s2_pred = _predict_with_retry(sentence2, genre, retries, retry_wait_seconds)

    if s1_pred == "Y" and s2_pred == "Y":
        base_pred = "Y"
    elif s1_pred in {"Y", "N"} and s2_pred in {"Y", "N"}:
        base_pred = "N"
    else:
        base_pred = "N"

    post = apply_task6_v2_postprocess(
        sentence1=sentence1,
        sentence2=sentence2,
        genre=genre,
        s1_pred=s1_pred,
        s2_pred=s2_pred,
        base_pred=base_pred,
        retries=retries,
        retry_wait_seconds=retry_wait_seconds,
    )

    is_match = post["final_pred"] == expected
    return {
        "example_id": example_id,
        "input": input_text,
        "genre": genre,
        "sentence1": sentence1,
        "sentence2": sentence2,
        "sentence1_pred": s1_pred,
        "sentence2_pred": s2_pred,
        "expected_output": expected,
        "model_output_base": base_pred,
        "domain_same_pred": post["domain_same_pred"],
        "sentence2_context_pred": post["sentence2_context_pred"],
        "sentence1_context_pred": post["sentence1_context_pred"],
        "model_output": post["final_pred"],
        "is_match": is_match,
    }


def run_task6_v3(
    output_dir: Path,
    examples_limit: int = 0,
    retries: int = 3,
    retry_wait_seconds: float = 2.0,
    resume: bool = False,
    batch_size: int = 16,
) -> dict:
    with _task_json_path().open("r", encoding="utf-8") as f:
        task_dict = json.load(f)

    task_name = task_dict["task_name"]
    all_examples = task_dict["examples"]
    if examples_limit > 0:
        all_examples = all_examples[:examples_limit]

    output_file = output_dir / "openseek-6-examples-compare-task6-v3-sentence-and-postprocess.jsonl"
    done_ids = _load_done_ids(output_file) if resume else set()
    mode = "a" if resume else "w"

    if done_ids:
        print(f"[断点续跑] task6-v3 已完成 {len(done_ids)} 条，继续剩余样本")

    to_run = [
        ex
        for ex in all_examples
        if not (resume and str(ex.get("id", "")).strip() in done_ids)
    ]
    bs = max(1, int(batch_size))
    worker = partial(_process_single_v3_example, retries=retries, retry_wait_seconds=retry_wait_seconds)

    with output_file.open(mode, encoding="utf-8") as wf:
        bar = tqdm(total=len(to_run), desc=f"Task6-v3 Inference (bs={bs}): {task_name}")
        for i in range(0, len(to_run), bs):
            batch = to_run[i : i + bs]
            workers = min(bs, len(batch))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                rows = list(pool.map(worker, batch))
            for row in rows:
                wf.write(json.dumps(row, ensure_ascii=False) + "\n")
                done_ids.add(str(row.get("example_id", "")).strip())
            wf.flush()
            bar.update(len(batch))
        bar.close()

    total, match_count, accuracy = _compute_metrics_from_jsonl(output_file)
    _, base_match, base_acc = _compute_base_accuracy_from_jsonl(output_file)
    print(
        f"[保存完成] task=6-v3, total={total}, matched={match_count}, "
        f"accuracy={accuracy:.2%} (AND-only base_acc={base_acc:.2%}), file={output_file}"
    )
    return {
        "task_id": 6,
        "task_name": task_name,
        "pipeline": "task6-v3-sentence-and-postprocess",
        "batch_size": bs,
        "total": total,
        "matched": match_count,
        "accuracy": accuracy,
        "base_matched": base_match,
        "base_accuracy": base_acc,
        "file": str(output_file),
    }


def main() -> None:
    args = parse_args()
    os.environ["DASHSCOPE_ENABLE_THINKING"] = "1" if args.thinking == "on" else "0"
    print(f"[thinking] DASHSCOPE_ENABLE_THINKING={os.environ['DASHSCOPE_ENABLE_THINKING']}")

    output_dir = _resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[输出目录] {output_dir}")
    print(f"[并行] batch_size={args.batch_size}")

    summary_item = run_task6_v3(
        output_dir=output_dir,
        examples_limit=args.examples_limit,
        retries=args.retries,
        retry_wait_seconds=args.retry_wait_seconds,
        resume=args.resume,
        batch_size=args.batch_size,
    )
    summary_file = output_dir / "summary_task6_v3.json"
    with summary_file.open("w", encoding="utf-8") as f:
        json.dump([summary_item], f, ensure_ascii=False, indent=2)
    print(f"[汇总完成] {summary_file}")


if __name__ == "__main__":
    main()
