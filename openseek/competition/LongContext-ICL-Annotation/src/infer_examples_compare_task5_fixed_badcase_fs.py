"""
Task5 推理：在 ``infer_examples_compare_task5.py`` 基础上，前缀固定 **8** 条 curated few-shot（``FIXED_TASK5_FS_SHOTS``），
覆盖常见假阳性（感激+😭、歌词式措辞、愤怒吐槽、「hurt tweet」推广、噪声英文）与假阴性
（丢钥匙、被吵醒、「want … so bad」口语）；Not sad 中含一条偏正面的感激推文作朴素对照，Sad 中含平淡倒霉类推 SemEval 边界。

检索示例仍通过 ``select_examples_hybrid`` 追加在固定示例之后（``--task5_shot_k 0`` 则仅用固定 8 条）。
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")
import re
import time
import unicodedata
from pathlib import Path

from tqdm import tqdm

from method_hyb import select_examples_hybrid
from method_task5 import annotate_nvidia as annotate, build_prompt

REPO_ROOT = Path(__file__).resolve().parent.parent
TASK5_FILE = "openseek-5_semeval_2018_task1_tweet_sadness_detection.json"
TASK5_CANONICAL_DESCRIPTION = (
    'In this task you are given a tweet. You must judge whether the author of the tweet is sad or not. '
    'Label the instances as "Sad" or "Not sad" based on your judgment. '
    "You can get help from hashtags and emojis, but you should not judge only based on them, "
    "and should pay attention to tweet's text as well."
)

# 固定 few-shot：8 条 = 5 类常见 FP（Not sad）+ 3 类常见 FN（Sad）。
FIXED_TASK5_FS_SHOTS: list[tuple[str, str]] = [
    # --- 假阳性取向（易被模型打成 Sad，金标多为 Not sad）---
    (
        "@giveaway_bot Thanks so much for picking me I literally never win anything 😭💕 you made my week",
        "Not sad",
    ),
    (
        "Baby I'm dancing in the dark with you between my arms barefoot on the grass listening to our song",
        "Not sad",
    ),
    (
        "Bastard squirrels ate every tomato on my plants again I'm so done 😡",
        "Not sad",
    ),
    (
        "Hiya everyone if you want please retweet my pin rt help romance wattpad hurt fic promo twitter thanks",
        "Not sad",
    ),
    (
        "All and boy play n0 no play dull and makes.",
        "Not sad",
    ),
    # --- 假阴性取向（语气平淡 / 口语双关，金标常为 Sad）---
    (
        "Damn I lost my keys and I forgot to get the garage opener",
        "Sad",
    ),
    (
        "Got woken up by a road sweeper I was trying to sleep",
        "Sad",
    ),
    (
        "I want to do digital art so bad but my dad won't let me use my iPad till exams are over 😂",
        "Sad",
    ),
]

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
    parser = argparse.ArgumentParser(
        description="Task5：固定 bad-case few-shot（8 条）+ 可选 hybrid 检索示例。",
    )
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
        "--task5_shot_k",
        type=int,
        default=4,
        help="hybrid 检索追加的示例数；0 表示仅使用固定 8 条 few-shot。",
    )
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


def _build_fixed_fewshot_block(
    strip_hashtag_enabled: bool,
    task5_emoji_mode: str,
) -> str:
    parts: list[str] = []
    for raw_inp, label in FIXED_TASK5_FS_SHOTS:
        t = raw_inp
        if strip_hashtag_enabled:
            t = _strip_hashtag_symbol(t)
        t = _convert_task5_emoji(str(t), task5_emoji_mode)
        parts.append(f"# {t} <label> {label} </label>\n")
    return "".join(parts)


def _prepare_task5_item(
    example: dict,
    processed_all_examples: list[dict],
    task_description: str,
    fixed_examples_prefix: str,
    task5_emoji_mode: str,
    strip_hashtag_enabled: bool,
    task5_shot_k: int,
    task5_retrieval_pool_size: int,
) -> dict:
    example_id = str(example.get("id", "")).strip()
    input_text = str(example.get("input", ""))
    if strip_hashtag_enabled:
        input_text = _strip_hashtag_symbol(input_text)
    prompt_input_text = _convert_task5_emoji(str(input_text), task5_emoji_mode)
    expected = _extract_output(example.get("output", ""))

    if task5_shot_k > 0:
        retrieved = select_examples_hybrid(
            all_examples=processed_all_examples,
            task_description=task_description,
            text2annotate=prompt_input_text,
            top_k=max(1, task5_shot_k),
            rerank_pool_size=max(20, task5_retrieval_pool_size),
            use_explanation=False,
            use_bm25_semantic_rerank=True,
            exclude_example_id=example_id,
        )
        examples_str = fixed_examples_prefix + retrieved
    else:
        examples_str = fixed_examples_prefix

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

    is_match = _normalize_text(prediction) == _normalize_text(expected)
    return {
        "example_id": example_id,
        "input_text": item["input_text"],
        "expected": expected,
        "prediction": prediction,
        "is_match": is_match,
    }


def run_task5_fixed_fs(
    output_dir: Path,
    examples_limit: int = 0,
    retries: int = 3,
    retry_wait_seconds: float = 2.0,
    resume: bool = False,
    task5_emoji_mode: str = "off",
    task5_strip_hashtag: str = "on",
    task5_shot_k: int = 4,
    task5_retrieval_pool_size: int = 200,
    print_empty_prediction: str = "on",
    retrieval_batch_size: int = 1,
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
    n_fixed = len(FIXED_TASK5_FS_SHOTS)
    output_file = output_dir / (
        f"openseek-{task_id}-examples-compare-task5opt-fixedbadcase{n_fixed}fs-ret{task5_shot_k}-emoji-"
        f"{task5_emoji_mode}-{hashtag_mode}.jsonl"
    )
    done_ids = _load_done_ids(output_file) if resume else set()
    mode = "a" if resume else "w"

    if done_ids:
        print(f"[断点续跑] task=5 fixed-fs, 已完成 {len(done_ids)} 条，继续剩余样本")

    fixed_examples_prefix = _build_fixed_fewshot_block(strip_hashtag_enabled, task5_emoji_mode)

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
    with output_file.open(mode, encoding="utf-8") as wf:
        for i in tqdm(range(0, len(pending_examples), bs), desc=f"Task5 fixed badcase FS: {task_name}"):
            chunk = pending_examples[i : i + bs]
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(bs, len(chunk))) as executor:
                prepare_futures = [
                    executor.submit(
                        _prepare_task5_item,
                        example=ex,
                        processed_all_examples=processed_all_examples,
                        task_description=task_description,
                        fixed_examples_prefix=fixed_examples_prefix,
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
                prediction = result["prediction"]
                is_match = result["is_match"]
                if print_empty_prediction == "on" and not prediction:
                    input_preview = _safe_console_text(input_text[:160].replace(chr(10), " "))
                    print(
                        f"[空预测] example_id={example_id} expected={expected} "
                        f"input_preview={input_preview}"
                    )
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
    print(f"[保存完成] task=5 fixed-fs, total={total}, matched={match_count}, accuracy={accuracy:.2%}, file={output_file}")
    return {
        "task_id": 5,
        "task_name": task_name,
        "mode": "fixed_badcase_fewshot",
        "fixed_fs_count": n_fixed,
        "retrieval_shot_k": task5_shot_k,
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

    summary_item = run_task5_fixed_fs(
        output_dir=output_dir,
        examples_limit=args.examples_limit,
        retries=args.retries,
        retry_wait_seconds=args.retry_wait_seconds,
        resume=args.resume,
        task5_emoji_mode=args.task5_emoji_mode,
        task5_strip_hashtag=args.task5_strip_hashtag,
        task5_shot_k=args.task5_shot_k,
        task5_retrieval_pool_size=args.task5_retrieval_pool_size,
        print_empty_prediction=args.print_empty_prediction,
        retrieval_batch_size=args.retrieval_batch_size,
    )
    summary_file = output_dir / "summary_task5opt_fixed_badcase_fs.json"
    with summary_file.open("w", encoding="utf-8") as f:
        json.dump([summary_item], f, ensure_ascii=False, indent=2)
    print(f"[汇总完成] {summary_file}")


if __name__ == "__main__":
    main()
