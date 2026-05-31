"""
Task6-v2 变体 B：检索式「任务定义片段 + 句对联合判定」——从数据 JSON 读取 Definition 作为
可检索规格注入各 prompt；基础阶段对单句 1、单句 2 与「双句联合」三种信号做多数表决，
后处理分支使用与原版不同的、仍锚定在任务定义上的复核 prompt，便于投票融合。
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
BASE_JSONL_NAME = "openseek-6-v2-taskdef-joint.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Task6-v2 变体：任务定义检索 + 联合句对多数表决（用于投票融合）。"
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
        "--taskdef_max_chars",
        type=int,
        default=1200,
        help="注入 prompt 的任务 Definition 最大字符数（超出截断）。",
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


def _get_task_definition(task_dict: dict[str, Any]) -> str:
    defs = task_dict.get("Definition") or []
    if not defs:
        return ""
    return str(defs[0]).strip()


def _build_taskdef_sentence_prompt(sentence: str, genre: str, task_definition: str) -> str:
    g = (genre or "").strip()
    spec = _trunc(task_definition, 2000)
    return (
        "### Retrieved Task Specification (do not quote; use as constraints)\n"
        f"{spec}\n\n"
        "### Role\n"
        "You are a strict genre validator for OpenSeek task6.\n\n"
        "### Task\n"
        "Given ONE sentence and ONE candidate genre, decide whether the sentence fits that genre.\n\n"
        "### Candidate Genre\n"
        f"- genre: {g}\n\n"
        "### Input Sentence\n"
        f"{sentence}\n\n"
        "### Decision Rules\n"
        "1. Follow the retrieved specification above for what each genre means.\n"
        "2. Prefer style/register/source cues over topic overlap.\n"
        "3. If evidence is weak or ambiguous, output N.\n\n"
        "### Output Format\n"
        "Return exactly one label wrapped in tags: <label>Y</label> or <label>N</label>.\n"
    )


def _build_joint_pair_prompt(sentence1: str, sentence2: str, genre: str, task_definition: str) -> str:
    g = (genre or "").strip()
    spec = _trunc(task_definition, 2000)
    return (
        "### Retrieved Task Specification (do not quote; use as constraints)\n"
        f"{spec}\n\n"
        "### Role\n"
        "You are a strict judge for OpenSeek task6.\n\n"
        "### Task\n"
        "You are given Sentence 1, Sentence 2, and a candidate genre.\n"
        "Decide whether BOTH sentences belong to that SAME stated genre (i.e., answer Y only if "
        "both fit the genre under the specification).\n\n"
        "### Candidate Genre\n"
        f"- genre: {g}\n\n"
        "### Sentence 1\n"
        f"{sentence1}\n\n"
        "### Sentence 2\n"
        f"{sentence2}\n\n"
        "### Decision Rules\n"
        "1. If either sentence fails the genre test, output N.\n"
        "2. If uncertain, output N.\n\n"
        "### Output Format\n"
        "Return exactly one label wrapped in tags: <label>Y</label> or <label>N</label>.\n"
    )


def _annotate_normalized(prompt: str, retries: int, retry_wait_seconds: float) -> str:
    pred = ""
    for attempt in range(1, retries + 1):
        try:
            raw = annotate_nvidia(prompt)
            pred = normalize_binary_label("" if raw is None else str(raw).strip())
            if pred in {"Y", "N"}:
                return pred
        except Exception as e:  # noqa: BLE001
            if attempt >= retries:
                print(f"[判定失败] attempt={attempt}/{retries} error={e}")
            else:
                print(f"[判定重试] attempt={attempt}/{retries} error={e}")
                time.sleep(retry_wait_seconds)
    return pred


def _majority_three(a: str, b: str, c: str) -> str:
    votes = [v for v in (a, b, c) if v in {"Y", "N"}]
    if not votes:
        return "N"
    y = sum(1 for v in votes if v == "Y")
    n = len(votes) - y
    if y > n:
        return "Y"
    if n > y:
        return "N"
    return "N"


def _build_def_anchored_nn_prompt(
    sentence1: str, sentence2: str, genre: str, task_definition: str
) -> str:
    g = (genre or "").strip()
    spec = _trunc(task_definition, 2000)
    return (
        "### Retrieved Task Specification\n"
        f"{spec}\n\n"
        "### Role\n"
        "You are resolving a hard case for OpenSeek task6.\n\n"
        "### Setup\n"
        "Both sentences were judged N for the candidate genre in isolation.\n"
        "Re-evaluate whether BOTH sentences still jointly satisfy the genre according to the specification.\n\n"
        "### Candidate Genre\n"
        f"- genre: {g}\n\n"
        "### Sentence 1\n"
        f"{sentence1}\n\n"
        "### Sentence 2\n"
        f"{sentence2}\n\n"
        "### Output Format\n"
        "Return exactly one label wrapped in tags: <label>Y</label> or <label>N</label>.\n"
    )


def _build_def_anchored_ny_prompt(sentence1: str, sentence2: str, genre: str, task_definition: str) -> str:
    g = (genre or "").strip()
    spec = _trunc(task_definition, 2000)
    return (
        "### Retrieved Task Specification\n"
        f"{spec}\n\n"
        "### Role\n"
        "You are resolving a split verdict for OpenSeek task6.\n\n"
        "### Setup\n"
        "Sentence 2 matched the genre; Sentence 1 did not.\n"
        "Using the specification, decide whether Sentence 1 should be upgraded to also match the genre "
        "(treat Sentence 2 as a positive style example).\n\n"
        "### Candidate Genre\n"
        f"- genre: {g}\n\n"
        "### Sentence 2 (style anchor)\n"
        f"{sentence2}\n\n"
        "### Sentence 1 (recheck target)\n"
        f"{sentence1}\n\n"
        "### Output Format\n"
        "Return exactly one label wrapped in tags: <label>Y</label> or <label>N</label>.\n"
    )


def _build_def_anchored_yn_prompt(sentence1: str, sentence2: str, genre: str, task_definition: str) -> str:
    g = (genre or "").strip()
    spec = _trunc(task_definition, 2000)
    return (
        "### Retrieved Task Specification\n"
        f"{spec}\n\n"
        "### Role\n"
        "You are resolving a split verdict for OpenSeek task6.\n\n"
        "### Setup\n"
        "Sentence 1 matched the genre; Sentence 2 did not.\n"
        "Using the specification, decide whether Sentence 2 should be upgraded to also match the genre "
        "(treat Sentence 1 as a positive style example).\n\n"
        "### Candidate Genre\n"
        f"- genre: {g}\n\n"
        "### Sentence 1\n"
        f"{sentence1}\n\n"
        "### Sentence 2\n"
        f"{sentence2}\n\n"
        "### Output Format\n"
        "Return exactly one label wrapped in tags: <label>Y</label> or <label>N</label>.\n"
    )


def _process_base_item(
    item: dict,
    *,
    task_definition: str,
    retries: int,
    retry_wait_seconds: float,
) -> dict:
    example_id = str(item.get("id", "")).strip()
    input_text = str(item.get("input", ""))
    expected = t6._extract_output(item.get("output", ""))
    sentence1, sentence2, genre = extract_task6_parts(input_text)

    p1 = _annotate_normalized(
        _build_taskdef_sentence_prompt(sentence1, genre, task_definition),
        retries,
        retry_wait_seconds,
    )
    p2 = _annotate_normalized(
        _build_taskdef_sentence_prompt(sentence2, genre, task_definition),
        retries,
        retry_wait_seconds,
    )
    joint = _annotate_normalized(
        _build_joint_pair_prompt(sentence1, sentence2, genre, task_definition),
        retries,
        retry_wait_seconds,
    )
    final_pred = _majority_three(p1, p2, joint)

    return {
        "test_sample_id": example_id,
        "example_id": example_id,
        "input": input_text,
        "genre": genre,
        "sentence1": sentence1,
        "sentence2": sentence2,
        "sentence1_pred": p1,
        "sentence2_pred": p2,
        "joint_pair_pred": joint,
        "model_output": final_pred,
        "prediction": final_pred,
        "expected_output": expected,
        "is_match": final_pred == expected if expected else False,
        "variant": "taskdef_joint",
    }


def _run_base_inference(
    data_file: str,
    output_dir: Path,
    limit: int,
    retries: int,
    retry_wait_seconds: float,
    resume: bool,
    taskdef_max_chars: int,
    batch_size: int,
) -> tuple[dict, Path]:
    data_path = _task_json_path(data_file)
    with data_path.open("r", encoding="utf-8") as f:
        task_dict = json.load(f)

    task_name = task_dict["task_name"]
    task_definition = _get_task_definition(task_dict)
    task_definition = _trunc(task_definition, taskdef_max_chars)
    if not task_definition:
        print("[警告] 数据 JSON 中 Definition 为空，将退化为无定义检索的联合 prompt。")

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
        task_definition=task_definition,
        retries=retries,
        retry_wait_seconds=retry_wait_seconds,
    )

    if done_ids:
        print(f"[断点续跑] {BASE_JSONL_NAME} 已完成 {len(done_ids)} 条，继续 {len(to_run)} 条")
    print(f"[并行] batch_size={bs}")

    with output_file.open(mode, encoding="utf-8") as wf:
        bar = tqdm(total=len(to_run), desc=f"Task6-v2-taskdef-joint (bs={bs}): {task_name}")
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
            "variant": "taskdef_joint",
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
    task_definition: str,
    retries: int,
    retry_wait_seconds: float,
) -> dict:
    total = 0
    base_matched = 0
    post_matched = 0
    triggered_nn = 0
    triggered_yn = 0
    triggered_ny = 0
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
            nn_aux = ""
            yn_aux = ""
            ny_aux = ""

            if s1_pred == "N" and s2_pred == "N":
                triggered_nn += 1
                nn_aux = _annotate_normalized(
                    _build_def_anchored_nn_prompt(
                        sentence1, sentence2, genre, task_definition
                    ),
                    retries,
                    retry_wait_seconds,
                )
                final_pred = "Y" if nn_aux == "Y" else "N"
            elif s1_pred == "Y" and s2_pred == "N":
                triggered_yn += 1
                yn_aux = _annotate_normalized(
                    _build_def_anchored_yn_prompt(
                        sentence1, sentence2, genre, task_definition
                    ),
                    retries,
                    retry_wait_seconds,
                )
                if yn_aux == "Y":
                    final_pred = "Y"
            elif s1_pred == "N" and s2_pred == "Y":
                triggered_ny += 1
                ny_aux = _annotate_normalized(
                    _build_def_anchored_ny_prompt(
                        sentence1, sentence2, genre, task_definition
                    ),
                    retries,
                    retry_wait_seconds,
                )
                if ny_aux == "Y":
                    final_pred = "Y"

            if final_pred != base_pred:
                changed += 1

            row["def_anchored_nn_pred"] = nn_aux
            row["def_anchored_yn_pred"] = yn_aux
            row["def_anchored_ny_pred"] = ny_aux
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
        "variant": "taskdef_joint",
        "total": total,
        "base_matched": base_matched,
        "post_matched": post_matched,
        "base_acc": base_acc,
        "post_acc": post_acc,
        "triggered_nn": triggered_nn,
        "triggered_yn": triggered_yn,
        "triggered_ny": triggered_ny,
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

    data_path = _task_json_path(args.data_file)
    with data_path.open("r", encoding="utf-8") as f:
        task_dict = json.load(f)
    task_definition = _trunc(_get_task_definition(task_dict), args.taskdef_max_chars)

    base_summary, base_file = _run_base_inference(
        data_file=args.data_file,
        output_dir=output_dir,
        limit=args.limit,
        retries=args.retries,
        retry_wait_seconds=args.retry_wait_seconds,
        resume=args.resume,
        taskdef_max_chars=args.taskdef_max_chars,
        batch_size=args.batch_size,
    )

    variant_post_file = output_dir / f"{base_file.stem}-postprocess-variant{base_file.suffix}"
    variant_summary = _run_postprocess(
        input_path=base_file,
        output_path=variant_post_file,
        task_definition=task_definition,
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

    summary_file = output_dir / "summary_task6_v2_taskdef_joint.json"
    with summary_file.open("w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)
    print(f"[汇总完成] {summary_file}")


if __name__ == "__main__":
    main()
