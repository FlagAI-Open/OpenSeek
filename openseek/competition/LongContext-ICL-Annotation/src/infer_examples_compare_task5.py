import argparse
import concurrent.futures
import json
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")
import re
import time
import unicodedata
from pathlib import Path
from typing import Any

from tqdm import tqdm

from method_task5 import (
    annotate_nvidia as annotate,
    build_prompt,
)
from method_hyb import select_examples_hybrid
from postprocess_task5_outputs import postprocess_recall_jsonl_inplace


REPO_ROOT = Path(__file__).resolve().parent.parent
TASK5_FILE = "openseek-5_semeval_2018_task1_tweet_sadness_detection.json"
TASK5_CANONICAL_DESCRIPTION = (
    'In this task you are given a tweet. You must judge whether the author of the tweet is sad or not. '
    'Label the instances as "Sad" or "Not sad" based on your judgment. '
    "You can get help from hashtags and emojis, but you should not judge only based on them, "
    "and should pay attention to tweet's text as well."
)

EMOJI_TEXT_MAP_ZH: dict[str, str] = {
    "😭": "大哭",
    "😢": "难过",
    "😔": "失落",
    "😞": "沮丧",
    "😡": "愤怒",
    "😠": "生气",
    "😤": "不满、气恼",
    "😣": "痛苦",
    "😩": "疲惫痛苦",
    "😫": "崩溃",
    "💔": "心碎",
}
EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001F5FF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FAFF"
    "\u2600-\u26FF"
    "\u2700-\u27BF"
    "]+"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task5 优化版推理（去偏置 prompt + 相似检索类平衡示例）。")
    parser.add_argument("--examples_limit", type=int, default=0, help="最多推理多少条；<=0 表示全部。")
    parser.add_argument("--output_dir", type=str, default="examples", help="结果输出目录。")
    parser.add_argument("--retries", type=int, default=3, help="单条失败重试次数。")
    parser.add_argument("--retry_wait_seconds", type=float, default=2.0, help="重试等待秒数。")
    parser.add_argument("--resume", action="store_true", help="开启断点续跑。")
    parser.add_argument(
        "--task5_emoji_mode",
        type=str,
        choices=["off", "alias", "zh"],
        default="off",
        help="emoji 转换模式：off/alias/zh。",
    )
    parser.add_argument(
        "--task5_strip_hashtag",
        type=str,
        choices=["on", "off"],
        default="on",
        help="是否去掉 hashtag 的 #。",
    )
    parser.add_argument(
        "--task5_postemoji",
        type=str,
        choices=["on", "off"],
        default="off",
        help=(
            "是否在模型输出后做 emoji 先验校准（与 infer_task5_test_samples / task5_eval_emoji_postprocess_compare 同源）；"
            "开启后文件名会含 postemoji-on- 前缀。"
        ),
    )
    parser.add_argument(
        "--task5_postemoji_prob",
        type=float,
        default=90.0,
        help="emoji 先验阈值：金标共现比例最大侧需 >（默认）或 ≥ 该百分数（见 --task5_postemoji_prob_ge）。",
    )
    parser.add_argument(
        "--task5_postemoji_prob_ge",
        action="store_true",
        help="概率条件改为 ≥ 阈值（默认为严格大于阈值）。",
    )
    parser.add_argument(
        "--task5_postemoji_min_distinct",
        type=int,
        default=2,
        help="推文去重 emoji 种类数下限（默认 2，即「>1」种才考虑后处理）。",
    )
    parser.add_argument(
        "--task5_output_frame",
        type=str,
        choices=["task5opt", "task5vote"],
        default="task5opt",
        help="输出 JSONL 文件名中段：task5opt（默认）或 task5vote（投票融合多路跑批）。",
    )
    parser.add_argument("--task5_shot_k", type=int, default=6, help="每条样本检索示例数，默认 6。")
    parser.add_argument(
        "--task5_retrieval_pool_size",
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
    parser.add_argument(
        "--task5_fp_postprocess",
        type=str,
        choices=["on", "off"],
        default="on",
        help="推理结束后执行 FP 压制后处理（emoji+text_calib，推荐），默认 on。",
    )
    parser.add_argument(
        "--task5_postprocess_fn_boost",
        action="store_true",
        help="后处理同时开启 FN 提升（默认仅压 FP）。",
    )
    parser.add_argument(
        "--retrieval_batch_size",
        type=int,
        default=8,
        help="并行推理批大小（按批并发调用模型），默认 8。",
    )
    return parser.parse_args()


def _task_json_path() -> Path:
    return REPO_ROOT / "data" / TASK5_FILE


def _normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


def _safe_console_text(text: str) -> str:
    return str(text).encode("ascii", "backslashreplace").decode("ascii")


def _strip_hashtag_symbol(text: str) -> str:
    if not text:
        return text
    return re.sub(r"#([A-Za-z0-9_]+)", r"\1", text)


def _finalize_task5_prediction(
    prediction_raw: str,
    input_for_post: str,
    *,
    postemoji_enabled: bool,
    prior: dict[str, dict[str, Any]] | None,
    prob_threshold: float,
    strict_prob_gt: bool,
    min_distinct_emojis: int,
) -> tuple[str, bool]:
    """返回 (写入 JSON 的 model_output, 是否应用 emoji 表决覆盖)。"""
    raw = prediction_raw.strip() if prediction_raw else ""
    if not postemoji_enabled or prior is None:
        return raw, False

    from task5_eval_emoji_postprocess_compare import (  # noqa: PLC0415
        _emoji_vote_postprocess,
        _normalize_label,
    )

    base_norm = _normalize_label(prediction_raw)
    maj, _ = _emoji_vote_postprocess(
        input_for_post,
        prior,
        prob_threshold=prob_threshold,
        strict_prob_gt=strict_prob_gt,
        min_distinct_emojis=min_distinct_emojis,
    )
    if maj is not None:
        return maj, True
    if base_norm is not None:
        return base_norm, False
    return raw, False


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


def _build_task5_icl_pool(all_examples: list[dict], current_example_id: str) -> list[dict]:
    """只用 task5 的 examples 做召回，并严格排除当前样本。"""
    cur = str(current_example_id).strip()
    pool: list[dict] = []
    for ex in all_examples:
        ex_id = str(ex.get("id", "")).strip()
        if ex_id and ex_id == cur:
            continue
        pool.append(ex)
    return pool


def _emoji_to_alias(ch: str) -> str:
    try:
        name = unicodedata.name(ch).lower().replace(" ", "_")
        return f":{name}:"
    except ValueError:
        return ch


def _convert_task5_emoji(text: str, mode: str) -> str:
    if mode == "off":
        return text
    if not text:
        return text

    def _replace(match: re.Match) -> str:
        chunk = match.group(0)
        converted: list[str] = []
        for ch in chunk:
            if mode == "zh":
                converted.append(EMOJI_TEXT_MAP_ZH.get(ch, _emoji_to_alias(ch)))
            else:
                converted.append(_emoji_to_alias(ch))
        return " " + " ".join(converted) + " "

    return EMOJI_RE.sub(_replace, text)


def _prepare_task5_item(
    example: dict,
    processed_all_examples: list[dict],
    task_description: str,
    task5_emoji_mode: str,
    strip_hashtag_enabled: bool,
    task5_shot_k: int,
    task5_retrieval_pool_size: int,
) -> dict:
    """准备单条样本的检索上下文（可并行）。"""
    example_id = str(example.get("id", "")).strip()
    input_text = str(example.get("input", ""))
    if strip_hashtag_enabled:
        input_text = _strip_hashtag_symbol(input_text)
    prompt_input_text = _convert_task5_emoji(str(input_text), task5_emoji_mode)
    expected = _extract_output(example.get("output", ""))

    # 复用预处理后的全量 task5 examples，每条仅在召回阶段排除当前样本。
    examples_str = select_examples_hybrid(
        all_examples=processed_all_examples,
        task_description=task_description,
        text2annotate=prompt_input_text,
        top_k=max(1, task5_shot_k),
        rerank_pool_size=max(20, task5_retrieval_pool_size),
        use_explanation=False,
        use_bm25_semantic_rerank=True,
        exclude_example_id=example_id,
    )
    return {
        "example_id": example_id,
        "input_text": input_text,
        "prompt_input_text": prompt_input_text,
        "expected": expected,
        "examples_str": examples_str,
    }


def _infer_task5_item(
    item: dict,
    task_description: str,
    retries: int,
    retry_wait_seconds: float,
) -> dict:
    example_id = item["example_id"]
    prompt_input_text = item["prompt_input_text"]
    expected = item["expected"]
    examples_str = item["examples_str"]

    prompt = build_prompt(task_description, prompt_input_text, task_id=5)
    input_prompt = prompt.replace("[[EXAMPLES]]\n\n", f"{examples_str}\n\n")

    prediction = ""
    for attempt in range(1, retries + 1):
        try:
            raw_prediction = annotate(input_prompt)
            prediction = "" if raw_prediction is None else str(raw_prediction).strip()
            break
        except Exception as e:  # noqa: BLE001
            if attempt >= retries:
                print(f"[推理失败] task=5 example_id={example_id} attempt={attempt}/{retries} error={e}")
            else:
                print(f"[重试] task=5 example_id={example_id} attempt={attempt}/{retries} error={e}")
                time.sleep(retry_wait_seconds)

    return {
        "example_id": example_id,
        "input_text": item["input_text"],
        "expected": expected,
        "prediction_raw": prediction,
    }


def run_task5(
    output_dir: Path,
    examples_limit: int = 0,
    retries: int = 3,
    retry_wait_seconds: float = 2.0,
    resume: bool = False,
    task5_emoji_mode: str = "off",
    task5_strip_hashtag: str = "on",
    task5_postemoji: str = "off",
    task5_postemoji_prob: float = 90.0,
    task5_postemoji_prob_ge: bool = False,
    task5_postemoji_min_distinct: int = 2,
    task5_output_frame: str = "task5opt",
    task5_shot_k: int = 6,
    task5_retrieval_pool_size: int = 200,
    print_empty_prediction: str = "on",
    retrieval_batch_size: int = 1,
    task5_fp_postprocess: str = "on",
    task5_postprocess_fn_boost: bool = False,
) -> dict:
    with _task_json_path().open("r", encoding="utf-8") as f:
        task_dict = json.load(f)

    task_id = 5
    task_name = task_dict["task_name"]
    task_description = TASK5_CANONICAL_DESCRIPTION
    all_examples = task_dict["examples"]
    if examples_limit > 0:
        all_examples = all_examples[:examples_limit]

    strip_hashtag_enabled = task5_strip_hashtag == "on"
    hashtag_mode = "striphash-on" if strip_hashtag_enabled else "striphash-off"
    postemoji_enabled = task5_postemoji == "on"
    prior: dict[str, dict[str, Any]] | None = None
    if postemoji_enabled:
        from task5_eval_emoji_postprocess_compare import (  # noqa: PLC0415
            _build_emoji_prior_from_examples,
        )

        prior = _build_emoji_prior_from_examples(task_dict)
        print(
            f"[emoji 后处理] 已根据 task5 JSON 的 examples 构建先验，"
            f"emoji 种类={len(prior)}；prob {'>=' if task5_postemoji_prob_ge else '>'} {task5_postemoji_prob}%，"
            f"去重 emoji 下限={task5_postemoji_min_distinct}"
        )

    pp = "postemoji-on-" if postemoji_enabled else ""
    frame = task5_output_frame if task5_output_frame in ("task5opt", "task5vote") else "task5opt"
    output_file = output_dir / (
        f"openseek-{task_id}-examples-compare-{frame}-{pp}emoji-{task5_emoji_mode}-{hashtag_mode}.jsonl"
    )
    done_ids = _load_done_ids(output_file) if resume else set()
    mode = "a" if resume else "w"

    if done_ids:
        print(f"[断点续跑] task=5, 已完成 {len(done_ids)} 条，继续剩余样本")

    # task5 全量样本仅预处理一次（strip hashtag + emoji convert）。
    processed_all_examples: list[dict] = []
    for ex in task_dict["examples"]:
        ex_input = str(ex.get("input", "")).strip()
        if strip_hashtag_enabled:
            ex_input = _strip_hashtag_symbol(ex_input)
        ex_input = _convert_task5_emoji(ex_input, task5_emoji_mode)
        processed_all_examples.append(
            {
                "id": ex.get("id", ""),
                "input": ex_input,
                "output": [_extract_output(ex.get("output", ""))],
            }
        )

    pending_examples = []
    for example in all_examples:
        example_id = str(example.get("id", "")).strip()
        if resume and example_id in done_ids:
            continue
        pending_examples.append(example)

    bs = max(1, retrieval_batch_size)
    strict_prob_gt = not task5_postemoji_prob_ge
    with output_file.open(mode, encoding="utf-8") as wf:
        for i in tqdm(range(0, len(pending_examples), bs), desc=f"Task5 Optimized Inference: {task_name}"):
            chunk = pending_examples[i : i + bs]
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(bs, len(chunk))) as executor:
                prepare_futures = [
                    executor.submit(
                        _prepare_task5_item,
                        example=ex,
                        processed_all_examples=processed_all_examples,
                        task_description=task_description,
                        task5_emoji_mode=task5_emoji_mode,
                        strip_hashtag_enabled=strip_hashtag_enabled,
                        task5_shot_k=task5_shot_k,
                        task5_retrieval_pool_size=task5_retrieval_pool_size,
                    )
                    for ex in chunk
                ]
                prepared_items = [f.result() for f in prepare_futures]

            with concurrent.futures.ThreadPoolExecutor(max_workers=min(bs, len(prepared_items))) as executor:
                infer_futures = [
                    executor.submit(
                        _infer_task5_item,
                        item=item,
                        task_description=task_description,
                        retries=retries,
                        retry_wait_seconds=retry_wait_seconds,
                    )
                    for item in prepared_items
                ]
                infer_results = [f.result() for f in infer_futures]

            for result in infer_results:
                example_id = result["example_id"]
                input_text = result["input_text"]
                expected = result["expected"]
                prediction_raw = result["prediction_raw"]
                final_pred, emoji_applied = _finalize_task5_prediction(
                    prediction_raw,
                    input_text,
                    postemoji_enabled=postemoji_enabled,
                    prior=prior,
                    prob_threshold=task5_postemoji_prob,
                    strict_prob_gt=strict_prob_gt,
                    min_distinct_emojis=task5_postemoji_min_distinct,
                )
                is_match = _normalize_text(final_pred) == _normalize_text(expected)
                if print_empty_prediction == "on" and not final_pred:
                    input_preview = _safe_console_text(input_text[:160].replace(chr(10), " "))
                    print(
                        f"[空预测] example_id={example_id} expected={expected} "
                        f"input_preview={input_preview}"
                    )
                row: dict[str, Any] = {
                    "example_id": example_id,
                    "input": input_text,
                    "expected_output": expected,
                    "model_output": final_pred,
                    "is_match": is_match,
                }
                if postemoji_enabled:
                    row["model_output_base"] = prediction_raw
                    row["emoji_postprocess_applied"] = emoji_applied
                wf.write(json.dumps(row, ensure_ascii=False) + "\n")
                wf.flush()
                done_ids.add(example_id)

    postprocessed = False
    if task5_fp_postprocess == "on" and output_file.exists():
        pp_stats = postprocess_recall_jsonl_inplace(
            output_file,
            strip_hashtag=strip_hashtag_enabled,
            fn_boost=task5_postprocess_fn_boost,
        )
        postprocessed = True
        b, r = pp_stats.get("baseline"), pp_stats.get("result")
        if b and r:
            print(
                f"[task5 后处理] changed={pp_stats['changed']}/{pp_stats['total']}, "
                f"acc {b['accuracy']:.2%}->{r['accuracy']:.2%} "
                f"FP {b['fp']}->{r['fp']} FN {b['fn']}->{r['fn']}, file={output_file}"
            )
        else:
            print(
                f"[task5 后处理] changed={pp_stats['changed']}/{pp_stats['total']}, "
                f"file={output_file}"
            )

    total, match_count, accuracy = _compute_metrics_from_jsonl(output_file)
    print(f"[保存完成] task=5, total={total}, matched={match_count}, accuracy={accuracy:.2%}, file={output_file}")
    return {
        "task_id": 5,
        "task_name": task_name,
        "total": total,
        "matched": match_count,
        "accuracy": accuracy,
        "file": str(output_file),
        "task5_postemoji": task5_postemoji,
        "task5_output_frame": frame,
        "postprocessed": postprocessed,
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

    summary_item = run_task5(
        output_dir=output_dir,
        examples_limit=args.examples_limit,
        retries=args.retries,
        retry_wait_seconds=args.retry_wait_seconds,
        resume=args.resume,
        task5_emoji_mode=args.task5_emoji_mode,
        task5_strip_hashtag=args.task5_strip_hashtag,
        task5_postemoji=args.task5_postemoji,
        task5_postemoji_prob=args.task5_postemoji_prob,
        task5_postemoji_prob_ge=args.task5_postemoji_prob_ge,
        task5_postemoji_min_distinct=args.task5_postemoji_min_distinct,
        task5_output_frame=args.task5_output_frame,
        task5_shot_k=args.task5_shot_k,
        task5_retrieval_pool_size=args.task5_retrieval_pool_size,
        print_empty_prediction=args.print_empty_prediction,
        retrieval_batch_size=args.retrieval_batch_size,
        task5_fp_postprocess=args.task5_fp_postprocess,
        task5_postprocess_fn_boost=args.task5_postprocess_fn_boost,
    )
    summary_file = output_dir / f"summary_{Path(summary_item['file']).stem}.json"
    with summary_file.open("w", encoding="utf-8") as f:
        json.dump([summary_item], f, ensure_ascii=False, indent=2)
    print(f"[汇总完成] {summary_file}")


if __name__ == "__main__":
    main()
