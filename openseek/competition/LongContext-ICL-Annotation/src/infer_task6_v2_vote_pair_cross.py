"""
Task6-v2 变体 A：检索式「交叉句上下文」——每条句子的判定 prompt 中显式检索并注入另一句作为对照，
后处理阶段对 N/N、Y/N 分支使用与 infer_task6_v2.py 不同的 pair-level / 复核 prompt，
便于与原版或其它变体做投票融合。
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any

from tqdm import tqdm

import infer_task6_v2 as t6
from method_hyb import annotate_nvidia
from method_task6_v2 import extract_task6_parts, normalize_binary_label
from task6_infer_pipeline import (
    DEFAULT_BATCH_SIZE,
    postprocess_v2_path,
    postprocess_v3_path,
    run_standard_v2_postprocess,
    run_standard_v3_postprocess,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TASK6_FILE = "openseek-6_mnli_same_genre_classification.json"
BASE_JSONL_NAME = "openseek-6-v2-pair-cross.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Task6-v2 变体：交叉句上下文检索 + pair 风格后处理（用于投票融合）。"
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
        "--thinking",
        type=str,
        choices=["on", "off"],
        default="off",
        help="是否开启模型 thinking（映射到 DASHSCOPE_ENABLE_THINKING）。",
    )
    parser.add_argument(
        "--cross_context_max_chars",
        type=int,
        default=900,
        help="注入到对句中的另一句最大字符数（超出截断）。",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="基础推理并行批大小，默认 16。",
    )
    parser.add_argument(
        "--flip_gate",
        type=str,
        choices=["none", "not_fiction", "block_nn_domain_n", "combined"],
        default="not_fiction",
        help="标准 v3 语篇后处理门控。",
    )
    parser.add_argument("--skip_v3", action="store_true", help="跳过标准 v3 后处理。")
    return parser.parse_args()


def _task_json_path(data_file: str) -> Path:
    return REPO_ROOT / "data" / data_file


def _trunc(text: str, max_chars: int) -> str:
    s = (text or "").strip()
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 3] + "..."


def _build_cross_context_sentence_prompt(
    sentence: str, other_sentence: str, genre: str, max_other: int
) -> str:
    g = (genre or "").strip()
    other = _trunc(other_sentence, max_other)
    return (
        "### Role\n"
        "You are a strict genre validator for OpenSeek task6.\n\n"
        "### Task\n"
        "Decide whether the TARGET sentence fits the candidate genre.\n"
        "Use the OTHER sentence only as a register/style contrast cue (not as a second label target).\n\n"
        "### Candidate Genre\n"
        f"- genre: {g}\n\n"
        "### OTHER sentence (context / contrast; do not output a separate label for it)\n"
        f"{other}\n\n"
        "### TARGET sentence (your label is only for this one)\n"
        f"{sentence}\n\n"
        "### Decision Rules\n"
        "1. Prefer discourse markers, syntax, and typical sources over shared keywords.\n"
        "2. If the target could plausibly be another genre, output N.\n"
        "3. Output only one label: Y or N.\n\n"
        "### Output Format\n"
        "Return exactly one label wrapped in tags: <label>Y</label> or <label>N</label>.\n"
    )


def _predict_cross_with_retry(
    sentence: str,
    other_sentence: str,
    genre: str,
    retries: int,
    retry_wait_seconds: float,
    max_other: int,
) -> str:
    prompt = _build_cross_context_sentence_prompt(sentence, other_sentence, genre, max_other)
    pred = ""
    for attempt in range(1, retries + 1):
        try:
            raw = annotate_nvidia(prompt)
            pred = normalize_binary_label("" if raw is None else str(raw).strip())
            if pred in {"Y", "N"}:
                return pred
        except Exception as e:  # noqa: BLE001
            if attempt >= retries:
                print(f"[交叉句判定失败] attempt={attempt}/{retries} error={e}")
            else:
                print(f"[交叉句判定重试] attempt={attempt}/{retries} error={e}")
                time.sleep(retry_wait_seconds)
    return pred


def _build_pair_same_register_prompt(sentence1: str, sentence2: str, genre: str) -> str:
    g = (genre or "").strip()
    return (
        "### Role\n"
        "You are a strict pair-level genre consistency checker.\n\n"
        "### Task\n"
        "Both sentences were initially judged as NOT matching the candidate genre in isolation.\n"
        "Now decide whether they TOGETHER still plausibly both belong to the stated genre "
        "(same register/source style as required by OpenSeek task6).\n\n"
        "### Candidate Genre\n"
        f"- genre: {g}\n\n"
        "### Sentence 1\n"
        f"{sentence1}\n\n"
        "### Sentence 2\n"
        f"{sentence2}\n\n"
        "### Decision Rules\n"
        "1. If either sentence is clearly from a different genre/register, output N.\n"
        "2. If both fit the genre when considered as a pair, output Y.\n"
        "3. If uncertain, output N.\n\n"
        "### Output Format\n"
        "Return exactly one label wrapped in tags: <label>Y</label> or <label>N</label>.\n"
    )


def _build_ny_mismatch_review_prompt(sentence1: str, sentence2: str, genre: str) -> str:
    g = (genre or "").strip()
    return (
        "### Role\n"
        "You are revising a borderline OpenSeek task6 genre decision.\n\n"
        "### Setup\n"
        "Sentence 2 was judged as matching the genre; Sentence 1 was judged as not matching.\n"
        "Decide whether Sentence 1 should ALSO be treated as matching the genre, "
        "using Sentence 2 only as a style anchor.\n\n"
        "### Candidate Genre\n"
        f"- genre: {g}\n\n"
        "### Sentence 2 (style anchor)\n"
        f"{sentence2}\n\n"
        "### Sentence 1 (recheck target)\n"
        f"{sentence1}\n\n"
        "### Decision Rules\n"
        "1. If Sentence 1 matches the genre cues as well as Sentence 2 does, output Y.\n"
        "2. If Sentence 1 is a different register/source, output N.\n"
        "3. If uncertain, output N.\n\n"
        "### Output Format\n"
        "Return exactly one label wrapped in tags: <label>Y</label> or <label>N</label>.\n"
    )


def _build_yn_mismatch_review_prompt(sentence1: str, sentence2: str, genre: str) -> str:
    g = (genre or "").strip()
    return (
        "### Role\n"
        "You are revising a borderline OpenSeek task6 genre decision.\n\n"
        "### Setup\n"
        "Sentence 1 was judged as matching the genre; Sentence 2 was judged as not matching.\n"
        "Decide whether Sentence 2 should ALSO be treated as matching the genre, "
        "using Sentence 1 only as a style anchor.\n\n"
        "### Candidate Genre\n"
        f"- genre: {g}\n\n"
        "### Sentence 1 (style anchor)\n"
        f"{sentence1}\n\n"
        "### Sentence 2 (recheck target)\n"
        f"{sentence2}\n\n"
        "### Decision Rules\n"
        "1. If Sentence 2 matches the genre cues as well as Sentence 1 does, output Y.\n"
        "2. If Sentence 2 is a different register/source, output N.\n"
        "3. If uncertain, output N.\n\n"
        "### Output Format\n"
        "Return exactly one label wrapped in tags: <label>Y</label> or <label>N</label>.\n"
    )


def _judge_with_prompt_builder(
    build_prompt_fn,
    retries: int,
    retry_wait_seconds: float,
    **kwargs: Any,
) -> str:
    prompt = build_prompt_fn(**kwargs)
    pred = ""
    for attempt in range(1, retries + 1):
        try:
            raw = annotate_nvidia(prompt)
            pred = normalize_binary_label("" if raw is None else str(raw).strip())
            if pred in {"Y", "N"}:
                return pred
        except Exception as e:  # noqa: BLE001
            if attempt < retries:
                print(f"[后处理辅助判定重试] attempt={attempt}/{retries} error={e}")
                time.sleep(retry_wait_seconds)
            else:
                print(f"[后处理辅助判定失败] attempt={attempt}/{retries} error={e}")
    return pred


def _process_base_item(
    item: dict,
    *,
    retries: int,
    retry_wait_seconds: float,
    cross_context_max_chars: int,
) -> dict:
    example_id = str(item.get("id", "")).strip()
    input_text = str(item.get("input", ""))
    expected = t6._extract_output(item.get("output", ""))
    sentence1, sentence2, genre = extract_task6_parts(input_text)

    s1_pred = _predict_cross_with_retry(
        sentence1,
        sentence2,
        genre,
        retries,
        retry_wait_seconds,
        cross_context_max_chars,
    )
    s2_pred = _predict_cross_with_retry(
        sentence2,
        sentence1,
        genre,
        retries,
        retry_wait_seconds,
        cross_context_max_chars,
    )

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
        "variant": "pair_cross",
    }


def _run_base_inference(
    data_file: str,
    output_dir: Path,
    limit: int,
    retries: int,
    retry_wait_seconds: float,
    resume: bool,
    cross_context_max_chars: int,
    batch_size: int,
) -> tuple[dict, Path]:
    data_path = _task_json_path(data_file)
    with data_path.open("r", encoding="utf-8") as f:
        task_dict = json.load(f)

    task_name = task_dict["task_name"]
    all_items = list(task_dict.get("test_samples", []))
    if limit > 0:
        all_items = all_items[:limit]

    output_file = output_dir / BASE_JSONL_NAME
    done_ids = t6._load_done_ids(output_file) if resume else set()
    mode = "a" if resume else "w"

    to_run = [
        item
        for item in all_items
        if not (resume and str(item.get("id", "")).strip() in done_ids)
    ]
    bs = max(1, int(batch_size))
    worker = partial(
        _process_base_item,
        retries=retries,
        retry_wait_seconds=retry_wait_seconds,
        cross_context_max_chars=cross_context_max_chars,
    )

    if done_ids:
        print(f"[断点续跑] {BASE_JSONL_NAME} 已完成 {len(done_ids)} 条，继续 {len(to_run)} 条")
    print(f"[并行] batch_size={bs}")

    with output_file.open(mode, encoding="utf-8") as wf:
        bar = tqdm(total=len(to_run), desc=f"Task6-v2-pair-cross (bs={bs}): {task_name}")
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

    total, matched, accuracy = t6._compute_metrics_from_jsonl(output_file)
    print(
        f"[基础推理完成] total={total}, matched={matched}, "
        f"accuracy={accuracy:.2%}, file={output_file}"
    )
    return (
        {
            "task_id": 6,
            "task_name": task_name,
            "stage": "base",
            "variant": "pair_cross",
            "batch_size": bs,
            "total": total,
            "matched": matched,
            "accuracy": accuracy,
            "file": str(output_file),
        },
        output_file,
    )


def _run_postprocess(
    input_path: Path,
    output_path: Path,
    retries: int,
    retry_wait_seconds: float,
) -> dict:
    total = 0
    base_matched = 0
    post_matched = 0
    triggered_pair_nn = 0
    triggered_review_yn = 0
    triggered_review_ny = 0
    changed = 0

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
                raise ValueError(f"第 {line_no} 行 JSON 解析失败: {exc}") from exc

            total += 1
            s1_pred = t6._extract_output(row.get("sentence1_pred", ""))
            s2_pred = t6._extract_output(row.get("sentence2_pred", ""))
            expected = t6._extract_output(row.get("expected_output", ""))
            base_pred = t6._extract_output(row.get("model_output", "") or row.get("prediction", ""))
            genre = t6._extract_output(row.get("genre", ""))
            sentence1 = t6._extract_output(row.get("sentence1", ""))
            sentence2 = t6._extract_output(row.get("sentence2", ""))

            if base_pred == expected:
                base_matched += 1

            final_pred = base_pred
            pair_same_register_pred = ""
            yn_review_pred = ""
            ny_review_pred = ""

            if s1_pred == "N" and s2_pred == "N":
                triggered_pair_nn += 1
                pair_same_register_pred = _judge_with_prompt_builder(
                    _build_pair_same_register_prompt,
                    retries,
                    retry_wait_seconds,
                    sentence1=sentence1,
                    sentence2=sentence2,
                    genre=genre,
                )
                final_pred = "Y" if pair_same_register_pred == "Y" else "N"
            elif s1_pred == "Y" and s2_pred == "N":
                triggered_review_yn += 1
                yn_review_pred = _judge_with_prompt_builder(
                    _build_yn_mismatch_review_prompt,
                    retries,
                    retry_wait_seconds,
                    sentence1=sentence1,
                    sentence2=sentence2,
                    genre=genre,
                )
                if yn_review_pred == "Y":
                    final_pred = "Y"
            elif s1_pred == "N" and s2_pred == "Y":
                triggered_review_ny += 1
                ny_review_pred = _judge_with_prompt_builder(
                    _build_ny_mismatch_review_prompt,
                    retries,
                    retry_wait_seconds,
                    sentence1=sentence1,
                    sentence2=sentence2,
                    genre=genre,
                )
                if ny_review_pred == "Y":
                    final_pred = "Y"

            if final_pred != base_pred:
                changed += 1

            row["pair_same_register_pred"] = pair_same_register_pred
            row["yn_mismatch_review_pred"] = yn_review_pred
            row["ny_mismatch_review_pred"] = ny_review_pred
            row["model_output_base"] = base_pred
            row["model_output"] = final_pred
            row["is_match"] = final_pred == expected
            if row["is_match"]:
                post_matched += 1
            wf.write(json.dumps(row, ensure_ascii=False) + "\n")

    base_acc = (base_matched / total) if total else 0.0
    post_acc = (post_matched / total) if total else 0.0
    print(
        f"[后处理完成] total={total} base_matched={base_matched} post_matched={post_matched} "
        f"base_acc={base_acc:.2%} post_acc={post_acc:.2%} output={output_path}"
    )
    return {
        "task_id": 6,
        "stage": "postprocess",
        "variant": "pair_cross",
        "total": total,
        "base_matched": base_matched,
        "post_matched": post_matched,
        "base_acc": base_acc,
        "post_acc": post_acc,
        "triggered_pair_nn": triggered_pair_nn,
        "triggered_review_yn": triggered_review_yn,
        "triggered_review_ny": triggered_review_ny,
        "changed": changed,
        "file": str(output_path),
    }


def main() -> None:
    args = parse_args()
    os.environ["DASHSCOPE_ENABLE_THINKING"] = "1" if args.thinking == "on" else "0"
    print(f"[thinking] DASHSCOPE_ENABLE_THINKING={os.environ['DASHSCOPE_ENABLE_THINKING']}")

    output_dir = t6._resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[输出目录] {output_dir}")

    base_summary, base_file = _run_base_inference(
        data_file=args.data_file,
        output_dir=output_dir,
        limit=args.limit,
        retries=args.retries,
        retry_wait_seconds=args.retry_wait_seconds,
        resume=args.resume,
        cross_context_max_chars=args.cross_context_max_chars,
        batch_size=args.batch_size,
    )

    variant_post_file = output_dir / f"{base_file.stem}-postprocess-variant{base_file.suffix}"
    variant_summary = _run_postprocess(
        input_path=base_file,
        output_path=variant_post_file,
        retries=args.retries,
        retry_wait_seconds=args.retry_wait_seconds,
    )

    v2_file = postprocess_v2_path(variant_post_file)
    v2_summary = run_standard_v2_postprocess(
        variant_post_file,
        v2_file,
        retries=args.retries,
        retry_wait_seconds=args.retry_wait_seconds,
    )
    print(f"[标准v2后处理] post_acc={v2_summary['post_acc']:.2%} file={v2_summary['file']}")

    summaries = [base_summary, variant_summary, v2_summary]
    final_file = Path(v2_summary["file"])

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
            f"[标准v3后处理] post_v3_acc={v3_summary['post_v3_accuracy']:.2%} "
            f"file={v3_summary['file']}"
        )
        final_file = Path(v3_summary["file"])
        summaries.append(v3_summary)

    submit_file = output_dir / f"{base_file.stem}-submit{base_file.suffix}"
    submit_summary = t6._export_submission(input_path=final_file, output_path=submit_file)
    summaries.append(submit_summary)

    summary_file = output_dir / "summary_task6_v2_pair_cross.json"
    with summary_file.open("w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)
    print(f"[汇总完成] {summary_file}")


if __name__ == "__main__":
    main()
