import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any

from tqdm import tqdm

from method_task6_v2 import extract_task6_parts, judge_sentence_genre
from task6_infer_pipeline import (
    DEFAULT_BATCH_SIZE,
    postprocess_v2_path,
    postprocess_v3_path,
    run_standard_v2_postprocess,
    run_standard_v3_postprocess,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TASK6_FILE = "openseek-6_mnli_same_genre_classification.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Task6-v2 数据推理：句子判定（并行）+ 标准 v2/v3 后处理。"
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default=DEFAULT_TASK6_FILE,
        help="`data/` 下的 task6 数据文件名。",
    )
    parser.add_argument("--limit", type=int, default=0, help="最多推理多少条；<=0 表示全部。")
    parser.add_argument("--output_dir", type=str, default="outputs", help="结果输出目录。")
    parser.add_argument("--retries", type=int, default=3, help="单条失败重试次数。")
    parser.add_argument("--retry_wait_seconds", type=float, default=2.0, help="重试等待秒数。")
    parser.add_argument("--resume", action="store_true", help="开启断点续跑。")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="基础推理并行批大小（线程池），默认 16。",
    )
    parser.add_argument(
        "--thinking",
        type=str,
        choices=["on", "off"],
        default="off",
        help="是否开启模型 thinking（映射到 DASHSCOPE_ENABLE_THINKING）。",
    )
    parser.add_argument(
        "--flip_gate",
        type=str,
        choices=["none", "not_fiction", "block_nn_domain_n", "combined"],
        default="not_fiction",
        help="v3 语篇后处理翻转门控，默认 not_fiction。",
    )
    parser.add_argument(
        "--skip_v3",
        action="store_true",
        help="仅跑 v2 标准后处理，跳过 v3 语篇后处理。",
    )
    return parser.parse_args()


def _task_json_path(data_file: str) -> Path:
    return REPO_ROOT / "data" / data_file


def _extract_output(output_value: Any) -> str:
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
            example_id = str(row.get("test_sample_id", "") or row.get("example_id", "")).strip()
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


def _process_base_item(
    item: dict,
    *,
    retries: int,
    retry_wait_seconds: float,
) -> dict:
    example_id = str(item.get("id", "")).strip()
    input_text = str(item.get("input", ""))
    expected = _extract_output(item.get("output", ""))
    sentence1, sentence2, genre = extract_task6_parts(input_text)

    s1_pred = _predict_with_retry(sentence1, genre, retries, retry_wait_seconds)
    s2_pred = _predict_with_retry(sentence2, genre, retries, retry_wait_seconds)

    if s1_pred == "Y" and s2_pred == "Y":
        final_pred = "Y"
    elif s1_pred in {"Y", "N"} and s2_pred in {"Y", "N"}:
        final_pred = "N"
    else:
        final_pred = "N"

    return {
        "test_sample_id": example_id,
        "example_id": example_id,
        "input": input_text,
        "genre": genre,
        "sentence1": sentence1,
        "sentence2": sentence2,
        "sentence1_pred": s1_pred,
        "sentence2_pred": s2_pred,
        "model_output": final_pred,
        "prediction": final_pred,
        "expected_output": expected,
        "is_match": final_pred == expected if expected else False,
    }


def _run_base_inference(
    data_file: str,
    output_dir: Path,
    limit: int,
    retries: int,
    retry_wait_seconds: float,
    resume: bool,
    batch_size: int,
) -> tuple[dict, Path]:
    data_path = _task_json_path(data_file)
    with data_path.open("r", encoding="utf-8") as f:
        task_dict = json.load(f)

    task_name = task_dict["task_name"]
    all_items = list(task_dict.get("test_samples", []))
    if limit > 0:
        all_items = all_items[:limit]

    output_file = output_dir / "openseek-6-v1.jsonl"
    done_ids = _load_done_ids(output_file) if resume else set()
    mode = "a" if resume else "w"

    to_run = [item for item in all_items if not (resume and str(item.get("id", "")).strip() in done_ids)]
    bs = max(1, int(batch_size))
    worker = partial(_process_base_item, retries=retries, retry_wait_seconds=retry_wait_seconds)

    if done_ids:
        print(f"[断点续跑] task6-v2 已完成 {len(done_ids)} 条，继续剩余 {len(to_run)} 条")
    print(f"[并行] batch_size={bs}")

    with output_file.open(mode, encoding="utf-8") as wf:
        bar = tqdm(total=len(to_run), desc=f"Task6-v2 Inference (bs={bs}): {task_name}")
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

    total, matched, accuracy = _compute_metrics_from_jsonl(output_file)
    print(
        f"[基础推理完成] total={total}, matched={matched}, "
        f"accuracy={accuracy:.2%}, file={output_file}"
    )
    return (
        {
            "task_id": 6,
            "task_name": task_name,
            "stage": "base",
            "batch_size": bs,
            "total": total,
            "matched": matched,
            "accuracy": accuracy,
            "file": str(output_file),
        },
        output_file,
    )


def _export_submission(input_path: Path, output_path: Path) -> dict:
    total = 0
    with input_path.open("r", encoding="utf-8") as rf, output_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as wf:
        for line_no, raw_line in enumerate(rf, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"提交导出时第 {line_no} 行 JSON 解析失败: {exc}") from exc

            example_id = str(row.get("test_sample_id", "") or row.get("example_id", "")).strip()
            pred = _extract_output(row.get("model_output", "") or row.get("prediction", ""))
            if pred not in {"Y", "N"}:
                pred = "N"

            wf.write(
                json.dumps({"test_sample_id": example_id, "prediction": pred}, ensure_ascii=False) + "\n"
            )
            total += 1

    print(f"[提交文件导出] total={total}, file={output_path}")
    return {
        "task_id": 6,
        "stage": "submit",
        "total": total,
        "file": str(output_path),
    }


def main() -> None:
    args = parse_args()
    os.environ["DASHSCOPE_ENABLE_THINKING"] = "1" if args.thinking == "on" else "0"
    print(f"[thinking] DASHSCOPE_ENABLE_THINKING={os.environ['DASHSCOPE_ENABLE_THINKING']}")

    output_dir = _resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[输出目录] {output_dir}")

    base_summary, base_file = _run_base_inference(
        data_file=args.data_file,
        output_dir=output_dir,
        limit=args.limit,
        retries=args.retries,
        retry_wait_seconds=args.retry_wait_seconds,
        resume=args.resume,
        batch_size=args.batch_size,
    )

    v2_file = postprocess_v2_path(base_file)
    v2_summary = run_standard_v2_postprocess(
        base_file,
        v2_file,
        retries=args.retries,
        retry_wait_seconds=args.retry_wait_seconds,
    )
    print(
        f"[v2后处理] post_acc={v2_summary['post_acc']:.2%} changed={v2_summary['changed']} "
        f"file={v2_summary['file']}"
    )

    final_file = Path(v2_summary["file"])
    summaries = [base_summary, v2_summary]

    if not args.skip_v3:
        v3_file = postprocess_v3_path(v2_file)
        v3_summary = run_standard_v3_postprocess(
            v2_file,
            v3_file,
            retries=args.retries,
            retry_wait_seconds=args.retry_wait_seconds,
            flip_gate=args.flip_gate,
        )
        print(
            f"[v3后处理] post_v3_acc={v3_summary['post_v3_accuracy']:.2%} "
            f"delta={v3_summary['delta_accuracy']:+.2%} file={v3_summary['file']}"
        )
        final_file = Path(v3_summary["file"])
        summaries.append(v3_summary)

    submit_file = output_dir / f"{base_file.stem}-submit{base_file.suffix}"
    submit_summary = _export_submission(input_path=final_file, output_path=submit_file)
    summaries.append(submit_summary)

    summary_file = output_dir / "summary_task6_v2.json"
    with summary_file.open("w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)
    print(f"[汇总完成] {summary_file}")


if __name__ == "__main__":
    main()
