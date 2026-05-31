"""
根据 examples 的 input / 标准 output，调用 DeepSeek API 生成 explanation，
并写出带 explanation 的数据与训练用 jsonl。

API：OpenAI 兼容接口。密钥请设置环境变量 DEEPSEEK_API_KEY，勿写入代码。
默认 base_url: https://api.deepseek.com/v1
"""

from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

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

DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_MODEL = "deepseek-chat"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="调用 DeepSeek 为 examples 生成 explanation，并导出训练集。"
    )
    parser.add_argument(
        "--output_data_dir",
        type=str,
        default="data/examples_with_explanation",
        help="带 explanation 的任务数据输出目录。",
    )
    parser.add_argument(
        "--output_train_file",
        type=str,
        default="data/examples_cot_train.jsonl",
        help="汇总训练集输出（jsonl，逐条追加）。",
    )
    parser.add_argument(
        "--progress_dir",
        type=str,
        default="data/examples_with_explanation/_progress",
        help="断点续跑：每个任务一条 jsonl，每行完整 example+explanation。",
    )
    parser.add_argument(
        "--task_start",
        type=int,
        default=7,
        help="起始任务编号（含），默认 7。",
    )
    parser.add_argument(
        "--task_end",
        type=int,
        default=8,
        help="结束任务编号（含），默认 8。",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"DeepSeek 模型名，默认 {DEFAULT_MODEL}。",
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default=os.environ.get("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL),
        help=f"API Base URL，默认 {DEFAULT_BASE_URL}。",
    )
    parser.add_argument(
        "--max_input_chars",
        type=int,
        default=32000,
        help="单条 input 传入模型的最大字符数（过长则截断），默认 32000。",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=5,
        help="单条请求失败时的最大重试次数。",
    )
    parser.add_argument(
        "--retry_wait",
        type=float,
        default=2.0,
        help="重试基础等待秒数（指数退避）。",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="每条成功请求后的额外休眠秒数，用于限流。",
    )
    parser.add_argument(
        "--examples_limit",
        type=int,
        default=0,
        help="每个任务最多处理多少条 examples；0 表示全部。",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="不调 API，仅写占位 explanation（用于联调路径）。",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="并行调用 API 的线程数，默认 16。",
    )
    return parser.parse_args()


def _extract_label(output_value) -> str:
    if isinstance(output_value, list):
        return "" if not output_value else str(output_value[0]).strip()
    return str(output_value).strip()


def _strip_explanation_tags(text: str) -> str:
    """去掉模型误加的 <explanation>...</explanation> 包裹。"""
    t = text.strip()
    m = re.match(
        r"^\s*<explanation>\s*(.*?)\s*</explanation>\s*$",
        t,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if m:
        return m.group(1).strip()
    return t


def _load_progress_by_id(progress_file: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not progress_file.exists():
        return out
    with progress_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            eid = str(row.get("id", "")).strip()
            if eid:
                out[eid] = row
    return out


def _append_progress(progress_file: Path, row: dict) -> None:
    progress_file.parent.mkdir(parents=True, exist_ok=True)
    with progress_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


_FALLBACK_CONTENT_RISK = (
    "This sample was rejected by the model provider's content filter (Content Exists Risk). "
    "The gold label is kept as provided; no chain-of-thought was generated for this row."
)


def _is_content_exists_risk_error(exc: BaseException) -> bool:
    """DeepSeek 等对敏感/风控输入返回 400 Content Exists Risk。"""
    text = str(exc).lower()
    if "content exists risk" in text:
        return True
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict):
            msg = str(err.get("message", "")).lower()
            if "content exists risk" in msg:
                return True
    return False


def _deepseek_explanation_safe_mode(
    *,
    client,
    model: str,
    task_description: str,
    task_name: str,
    label: str,
) -> str | None:
    """
    不附带原始 input，仅根据任务说明与 gold label 写泛化推理；用于规避 Content Exists Risk。
    若仍失败返回 None。
    """
    desc = task_description[:4000]
    user_msg = (
        f"Task description:\n{desc}\n\n"
        f"Task name:\n{task_name}\n\n"
        "The raw input was omitted because a previous request was blocked by a content filter.\n"
        f"Gold label (answer) to justify:\n{label}\n\n"
        "Write 2–8 sentences in English: describe the general reasoning pattern someone would use "
        "on this task type to arrive at this label. Do not invent specific facts about unseen text; "
        "stay generic. No Markdown headings, no <explanation> tags."
    )
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You write safe, generic task reasoning in English only. "
                    "You never reproduce or quote disallowed content."
                ),
            },
            {"role": "user", "content": user_msg},
        ],
        temperature=0.3,
        max_tokens=1024,
    )
    content = completion.choices[0].message.content
    text = (content or "").strip()
    return _strip_explanation_tags(text) if text else None


def _deepseek_generate_explanation(
    *,
    client,
    model: str,
    task_description: str,
    task_name: str,
    input_text: str,
    label: str,
    max_input_chars: int,
    retries: int,
    retry_wait: float,
) -> str:
    trimmed = (
        input_text
        if len(input_text) <= max_input_chars
        else input_text[:max_input_chars] + "\n\n[... input truncated due to length ...]"
    )
    user_msg = (
        f"Task description:\n{task_description}\n\n"
        f"Task name:\n{task_name}\n\n"
        f"Input:\n{trimmed}\n\n"
        f"Gold label (answer):\n{label}\n\n"
        "Write a concise chain-of-thought explanation in English that shows how to derive the "
        "gold label from the input. Requirements: 2–10 sentences; do not contradict the gold label; "
        "do not revise or argue against it; output in English only (no Chinese); do not merely "
        "repeat the prompt; output plain reasoning text only—no Markdown headings and no "
        "<explanation> tags."
    )
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a careful data annotation assistant. Given the task, input, "
                            "and gold label, write the reasoning in English only."
                        ),
                    },
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.3,
                max_tokens=2048,
            )
            content = completion.choices[0].message.content
            text = (content or "").strip()
            return _strip_explanation_tags(text)
        except Exception as e:  # noqa: BLE001
            if _is_content_exists_risk_error(e):
                try:
                    alt = _deepseek_explanation_safe_mode(
                        client=client,
                        model=model,
                        task_description=task_description,
                        task_name=task_name,
                        label=label,
                    )
                    if alt:
                        return alt
                except Exception:  # noqa: BLE001
                    pass
                return _FALLBACK_CONTENT_RISK
            last_err = e
            if attempt < retries:
                time.sleep(retry_wait * attempt)
            else:
                raise last_err from None
    raise RuntimeError("unreachable")


_tls = threading.local()


def _thread_local_client(api_key: str, base_url: str):
    """每个 worker 线程各自一个 OpenAI 客户端，避免多线程共享连接不安全。"""
    c = getattr(_tls, "client", None)
    if c is None:
        from openai import OpenAI

        _tls.client = OpenAI(api_key=api_key, base_url=base_url.rstrip("/"))
    return _tls.client


def _worker_one_example(
    idx: int,
    ex: dict,
    *,
    dry_run: bool,
    api_key: str,
    base_url: str,
    model: str,
    task_description: str,
    task_name: str,
    max_input_chars: int,
    retries: int,
    retry_wait: float,
) -> tuple[int, dict]:
    """在线程池中生成单条 explanation，返回 (原索引, 补全后的 example)。"""
    input_text = str(ex.get("input", ""))
    label = _extract_label(ex.get("output", ""))
    explanation = ""
    try:
        if dry_run:
            explanation = f"[dry_run] Placeholder reasoning; gold label: {label}"
        else:
            client = _thread_local_client(api_key, base_url)
            explanation = _deepseek_generate_explanation(
                client=client,
                model=model,
                task_description=task_description,
                task_name=task_name,
                input_text=input_text,
                label=label,
                max_input_chars=max_input_chars,
                retries=retries,
                retry_wait=retry_wait,
            )
    except Exception as e:  # noqa: BLE001
        explanation = (
            "[generation_failed] "
            f"{type(e).__name__}: {e}. "
            "No model chain-of-thought is available for this row; the gold label remains unchanged."
        )
    target = f"{label}<explanation>{explanation}</explanation>"
    ex_new = dict(ex)
    ex_new["explanation"] = explanation
    ex_new["target_with_explanation"] = target
    return idx, ex_new


def main() -> None:
    args = parse_args()
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not args.dry_run and not api_key:
        api_key = 'sk-340c73f8209e4ba3ae44c7e6cf570506'
        

    out_data_dir = (REPO_ROOT / args.output_data_dir).resolve()
    out_train_file = (REPO_ROOT / args.output_train_file).resolve()
    progress_root = (REPO_ROOT / args.progress_dir).resolve()
    out_data_dir.mkdir(parents=True, exist_ok=True)
    out_train_file.parent.mkdir(parents=True, exist_ok=True)
    progress_root.mkdir(parents=True, exist_ok=True)

    train_append_f = out_train_file.open("a", encoding="utf-8")
    write_lock = threading.Lock()
    existing_train_ids: set[str] = set()
    if out_train_file.exists() and out_train_file.stat().st_size > 0:
        with out_train_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    eid = str(row.get("id", "")).strip()
                    if eid:
                        existing_train_ids.add(eid)
                except json.JSONDecodeError:
                    continue

    try:
        for task_id in range(args.task_start, args.task_end + 1):
            file_name = TASK_DATA_FILES[task_id]
            src_file = REPO_ROOT / "data" / file_name
            with src_file.open("r", encoding="utf-8") as f:
                task_dict = json.load(f)

            task_description = task_dict["Definition"][0]
            task_name = task_dict.get("task_name", "")
            tid = task_dict.get("task_id", f"openseek-{task_id}")

            progress_file = progress_root / f"{Path(file_name).stem}.jsonl"
            done_by_id = _load_progress_by_id(progress_file)

            examples = task_dict.get("examples", [])
            if args.examples_limit > 0:
                examples = examples[: args.examples_limit]

            new_examples: list[dict | None] = [None] * len(examples)
            pending: list[tuple[int, dict]] = []

            for idx, ex in enumerate(examples):
                eid = str(ex.get("id", "")).strip()

                if eid in done_by_id:
                    ex_new = done_by_id[eid]
                    new_examples[idx] = ex_new
                    if eid not in existing_train_ids:
                        explanation = str(ex_new.get("explanation", ""))
                        lab = _extract_label(ex_new.get("output", ""))
                        target = ex_new.get("target_with_explanation")
                        if not target:
                            target = f"{lab}<explanation>{explanation}</explanation>"
                        row = {
                            "task_id": tid,
                            "task_name": task_name,
                            "id": eid,
                            "input": str(ex_new.get("input", "")),
                            "label": lab,
                            "explanation": explanation,
                            "target": target,
                        }
                        with write_lock:
                            train_append_f.write(
                                json.dumps(row, ensure_ascii=False) + "\n"
                            )
                            train_append_f.flush()
                        existing_train_ids.add(eid)
                    continue

                pending.append((idx, ex))

            if pending:
                workers = max(1, int(args.workers))
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    futures = {
                        pool.submit(
                            _worker_one_example,
                            idx,
                            ex,
                            dry_run=args.dry_run,
                            api_key=api_key,
                            base_url=args.base_url,
                            model=args.model,
                            task_description=task_description,
                            task_name=task_name,
                            max_input_chars=args.max_input_chars,
                            retries=args.retries,
                            retry_wait=args.retry_wait,
                        ): (idx, ex)
                        for idx, ex in pending
                    }
                    for fut in tqdm(
                        as_completed(futures),
                        total=len(futures),
                        desc=f"task {task_id} {file_name}",
                    ):
                        try:
                            idx, ex_new = fut.result()
                        except Exception as e:  # noqa: BLE001
                            idx, ex_ref = futures[fut]
                            lab = _extract_label(ex_ref.get("output", ""))
                            explanation = (
                                "[worker_failed] "
                                f"{type(e).__name__}: {e}. "
                                "No model chain-of-thought is available; "
                                "the gold label remains unchanged."
                            )
                            target = f"{lab}<explanation>{explanation}</explanation>"
                            ex_new = dict(ex_ref)
                            ex_new["explanation"] = explanation
                            ex_new["target_with_explanation"] = target
                        eid = str(ex_new.get("id", "")).strip()
                        input_text = str(ex_new.get("input", ""))
                        label = _extract_label(ex_new.get("output", ""))
                        explanation = str(ex_new.get("explanation", ""))
                        target = ex_new.get("target_with_explanation", "")
                        new_examples[idx] = ex_new

                        with write_lock:
                            _append_progress(progress_file, ex_new)
                            if eid not in existing_train_ids:
                                row = {
                                    "task_id": tid,
                                    "task_name": task_name,
                                    "id": eid,
                                    "input": input_text,
                                    "label": label,
                                    "explanation": explanation,
                                    "target": target,
                                }
                                train_append_f.write(
                                    json.dumps(row, ensure_ascii=False) + "\n"
                                )
                                train_append_f.flush()
                                existing_train_ids.add(eid)

                        if args.sleep > 0:
                            time.sleep(args.sleep)

            merged: list[dict] = [e for e in new_examples if e is not None]
            if len(merged) != len(examples):
                raise RuntimeError(
                    f"internal: examples length mismatch task={task_id}"
                )

            task_out = dict(task_dict)
            task_out["examples"] = merged
            out_json = out_data_dir / file_name
            with out_json.open("w", encoding="utf-8") as f:
                json.dump(task_out, f, ensure_ascii=False, indent=2)
            print(f"[任务完成] {out_json} examples={len(merged)}")

    finally:
        train_append_f.close()
    print(f"[训练集已追加至] {out_train_file}")


if __name__ == "__main__":
    main()
