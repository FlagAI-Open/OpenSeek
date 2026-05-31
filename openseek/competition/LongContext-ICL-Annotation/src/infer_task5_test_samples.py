import argparse
import concurrent.futures
import json
import os
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
    "😤": "愤怒",
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
    parser = argparse.ArgumentParser(description="Task5 test_samples 推理（零样本；可选 emoji 先验后处理）。")
    parser.add_argument("--samples_limit", type=int, default=0, help="最多推理多少条；<=0 表示全部。")
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
            "是否在写出前做 emoji 先验后处理（与 task5_eval_emoji_postprocess_compare 同源规则）；"
            "开启后输出文件名会增加 postemoji-on 片段。"
        ),
    )
    parser.add_argument(
        "--task5_postemoji_prob",
        type=float,
        default=90.0,
        help="后处理阈值：emoji 在金标上出现比例的最大侧需 >（默认）或 ≥ 该百分数（见 --task5_postemoji_prob_ge）。",
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
        "--task5_fp_postprocess",
        type=str,
        choices=["on", "off"],
        default="on",
        help="推理结束后执行 FP 压制后处理（emoji+text_calib），默认 on。",
    )
    parser.add_argument(
        "--task5_postprocess_fn_boost",
        action="store_true",
        help="后处理同时开启 FN 提升（默认仅压 FP）。",
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
    parser.add_argument("--infer_parallelism", type=int, default=8, help="并发推理线程数。")
    return parser.parse_args()


def _task_json_path() -> Path:
    return REPO_ROOT / "data" / TASK5_FILE


def _strip_hashtag_symbol(text: str) -> str:
    if not text:
        return text
    return re.sub(r"#([A-Za-z0-9_]+)", r"\1", text)


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


def _taskdata_infer_output_filename(
    task_id: int,
    task5_emoji_mode: str,
    strip_hashtag_enabled: bool,
    postemoji_enabled: bool,
) -> str:
    """与 examples 命名对齐：...-emoji-{mode}-striphash-{on|off}.jsonl；后处理开启时嵌入 postemoji-on。"""
    hashtag_mode = "striphash-on" if strip_hashtag_enabled else "striphash-off"
    pp = "postemoji-on-" if postemoji_enabled else ""
    return f"openseek-{task_id}-taskdata-infer-{pp}emoji-{task5_emoji_mode}-{hashtag_mode}.jsonl"


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
            sample_id = str(row.get("sample_id", "")).strip()
            if sample_id:
                done.add(sample_id)
    return done


def _prepare_task5_item(
    sample: dict,
    task5_emoji_mode: str,
    strip_hashtag_enabled: bool,
) -> dict:
    sample_id = str(sample.get("id", "")).strip()
    input_text = str(sample.get("input", ""))
    if strip_hashtag_enabled:
        input_text = _strip_hashtag_symbol(input_text)
    prompt_input_text = _convert_task5_emoji(str(input_text), task5_emoji_mode)
    return {
        "sample_id": sample_id,
        "input_text": input_text,
        "prompt_input_text": prompt_input_text,
    }


def _infer_task5_item(
    item: dict,
    task_description: str,
    retries: int,
    retry_wait_seconds: float,
) -> dict:
    sample_id = item["sample_id"]
    prompt_input_text = item["prompt_input_text"]

    prompt = build_prompt(task_description, prompt_input_text, task_id=5)
    input_prompt = prompt.replace("[[EXAMPLES]]\n\n", "\n\n")

    prediction = ""
    for attempt in range(1, retries + 1):
        try:
            raw_prediction = annotate(input_prompt)
            prediction = "" if raw_prediction is None else str(raw_prediction).strip()
            break
        except Exception as e:  # noqa: BLE001
            if attempt >= retries:
                print(f"[推理失败] task=5 sample_id={sample_id} attempt={attempt}/{retries} error={e}")
            else:
                print(f"[重试] task=5 sample_id={sample_id} attempt={attempt}/{retries} error={e}")
                time.sleep(retry_wait_seconds)

    return {
        "sample_id": sample_id,
        "input_text": item["input_text"],
        "prediction": prediction,
    }


def run_task5_test_samples(
    output_dir: Path,
    samples_limit: int = 0,
    retries: int = 3,
    retry_wait_seconds: float = 2.0,
    resume: bool = False,
    task5_emoji_mode: str = "off",
    task5_strip_hashtag: str = "on",
    infer_parallelism: int = 8,
    task5_postemoji: str = "off",
    task5_postemoji_prob: float = 90.0,
    task5_postemoji_prob_ge: bool = False,
    task5_postemoji_min_distinct: int = 2,
    task5_fp_postprocess: str = "on",
    task5_postprocess_fn_boost: bool = False,
) -> dict:
    with _task_json_path().open("r", encoding="utf-8") as f:
        task_dict = json.load(f)

    task_id = 5
    task_name = task_dict["task_name"]
    task_description = TASK5_CANONICAL_DESCRIPTION
    all_samples = list(task_dict.get("test_samples", []))
    if samples_limit > 0:
        all_samples = all_samples[:samples_limit]

    strip_hashtag_enabled = task5_strip_hashtag == "on"
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

    output_file = output_dir / _taskdata_infer_output_filename(
        task_id,
        task5_emoji_mode,
        strip_hashtag_enabled,
        postemoji_enabled,
    )
    done_ids = _load_done_ids(output_file) if resume else set()
    mode = "a" if resume else "w"

    if done_ids:
        print(f"[断点续跑] task=5, 已完成 {len(done_ids)} 条，继续剩余样本")

    pending_samples = []
    for sample in all_samples:
        sample_id = str(sample.get("id", "")).strip()
        if resume and sample_id in done_ids:
            continue
        pending_samples.append(sample)

    infer_parallelism = max(1, int(infer_parallelism))
    strict_prob_gt = not task5_postemoji_prob_ge
    with output_file.open(mode, encoding="utf-8") as wf:
        for i in tqdm(
            range(0, len(pending_samples), infer_parallelism),
            desc=f"Task5 TaskData Inference: {task_name}",
        ):
            chunk = pending_samples[i : i + infer_parallelism]
            prepared_items = [
                _prepare_task5_item(
                    sample=sample,
                    task5_emoji_mode=task5_emoji_mode,
                    strip_hashtag_enabled=strip_hashtag_enabled,
                )
                for sample in chunk
            ]
            with concurrent.futures.ThreadPoolExecutor(max_workers=infer_parallelism) as executor:
                futures = [
                    executor.submit(
                        _infer_task5_item,
                        item=item,
                        task_description=task_description,
                        retries=retries,
                        retry_wait_seconds=retry_wait_seconds,
                    )
                    for item in prepared_items
                ]
                infer_results = [f.result() for f in futures]

            for result in infer_results:
                final, applied = _finalize_task5_prediction(
                    result["prediction"],
                    result["input_text"],
                    postemoji_enabled=postemoji_enabled,
                    prior=prior,
                    prob_threshold=task5_postemoji_prob,
                    strict_prob_gt=strict_prob_gt,
                    min_distinct_emojis=task5_postemoji_min_distinct,
                )
                row: dict[str, Any] = {
                    "sample_id": result["sample_id"],
                    "input": result["input_text"],
                    "model_output": final,
                }
                if postemoji_enabled:
                    row["model_output_base"] = result["prediction"]
                    row["emoji_postprocess_applied"] = applied
                wf.write(json.dumps(row, ensure_ascii=False) + "\n")
                wf.flush()
                done_ids.add(result["sample_id"])

    postprocessed = False
    if task5_fp_postprocess == "on" and output_file.exists():
        pp_stats = postprocess_recall_jsonl_inplace(
            output_file,
            strip_hashtag=strip_hashtag_enabled,
            fn_boost=task5_postprocess_fn_boost,
        )
        postprocessed = True
        print(
            f"[task5 后处理] changed={pp_stats['changed']}/{pp_stats['total']}, "
            f"file={output_file}"
        )

    print(f"[保存完成] task=5, total={len(done_ids)}, file={output_file}")
    return {
        "task_id": 5,
        "task_name": task_name,
        "total": len(done_ids),
        "file": str(output_file),
        "task5_postemoji": task5_postemoji,
        "postprocessed": postprocessed,
    }


def main() -> None:
    args = parse_args()
    # task5 约束：关闭检索相关阶段，仅做零样本判别。
    os.environ["ICL_DISABLE_SEMANTIC"] = "1"
    os.environ["ICL_DISABLE_RERANK"] = "1"
    os.environ["DASHSCOPE_ENABLE_THINKING"] = "1" if args.thinking == "on" else "0"
    os.environ["ANNOTATE_LOG_EVERY_RESPONSE"] = "1" if args.print_model_output == "on" else "0"
    print(f"[retrieval] ICL_DISABLE_SEMANTIC={os.environ['ICL_DISABLE_SEMANTIC']}")
    print(f"[retrieval] ICL_DISABLE_RERANK={os.environ['ICL_DISABLE_RERANK']}")
    print(f"[thinking] DASHSCOPE_ENABLE_THINKING={os.environ['DASHSCOPE_ENABLE_THINKING']}")
    print(f"[print_model_output] ANNOTATE_LOG_EVERY_RESPONSE={os.environ['ANNOTATE_LOG_EVERY_RESPONSE']}")

    output_dir = _resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[输出目录] {output_dir}")

    summary_item = run_task5_test_samples(
        output_dir=output_dir,
        samples_limit=args.samples_limit,
        retries=args.retries,
        retry_wait_seconds=args.retry_wait_seconds,
        resume=args.resume,
        task5_emoji_mode=args.task5_emoji_mode,
        task5_strip_hashtag=args.task5_strip_hashtag,
        infer_parallelism=args.infer_parallelism,
        task5_postemoji=args.task5_postemoji,
        task5_postemoji_prob=args.task5_postemoji_prob,
        task5_postemoji_prob_ge=args.task5_postemoji_prob_ge,
        task5_postemoji_min_distinct=args.task5_postemoji_min_distinct,
        task5_fp_postprocess=args.task5_fp_postprocess,
        task5_postprocess_fn_boost=args.task5_postprocess_fn_boost,
    )
    summary_file = output_dir / "summary_task5_taskdata.json"
    with summary_file.open("w", encoding="utf-8") as f:
        json.dump([summary_item], f, ensure_ascii=False, indent=2)
    print(f"[汇总完成] {summary_file}")


if __name__ == "__main__":
    main()
