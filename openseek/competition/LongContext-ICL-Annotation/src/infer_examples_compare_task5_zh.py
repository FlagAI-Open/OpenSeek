"""
Task5 推理（中文版）：先将预处理后的英文推文译成简体中文，再用中文任务说明打 Sad / Not sad。

- ICL 检索仍基于英文 ``TASK5_CANONICAL_DESCRIPTION`` + ``prompt_input_text``（与 ``infer_examples_compare_task5.py`` 一致）。
- 分类阶段参考示例仍为英文推文；待标注正文为译文。
- 翻译解析沿用 ``count_answer`` 的 ``<label>`` 抽取逻辑。
- 若翻译多次失败，则退化为与原脚本相同的英文分类路径（记录 ``translation_failed``）。
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
from method_hyb_prompts import _task_prompt_shell
from method_task5 import annotate_nvidia as annotate, build_prompt

REPO_ROOT = Path(__file__).resolve().parent.parent
TASK5_FILE = "openseek-5_semeval_2018_task1_tweet_sadness_detection.json"
TASK5_CANONICAL_DESCRIPTION = (
    'In this task you are given a tweet. You must judge whether the author of the tweet is sad or not. '
    'Label the instances as "Sad" or "Not sad" based on your judgment. '
    "You can get help from hashtags and emojis, but you should not judge only based on them, "
    "and should pay attention to tweet's text as well."
)

TASK5_CANONICAL_DESCRIPTION_ZH = (
    "在本任务中，给定一条推文。你必须判断推文作者是否悲伤。"
    '请将每条样本标注为 \"Sad\" 或 \"Not sad\"（必须与示例完全一致的大小写与空格）。'
    "可参考话题标签与表情符号，但不要仅凭它们下结论，应结合正文语义。"
)

TASK5_SPECIFIC_ZH = (
    "### Task-specific output（OpenSeek task5：推文悲伤识别）\n"
    "- 待标注推文正文已为简体中文；参考资料示例仍为英文，但其标注取值必须与示例完全一致。\n"
    "- 根据推文整体语义，判断作者是否表达悲伤。\n"
    "- 以正文为主；标签话题与标点为辅。\n"
    "- 当存在明确的哀伤、悲痛、孤独、无助、情感受伤等情感证据时标 Sad。\n"
    "- 对中性陈述、玩笑、纯讽刺，或以愤怒/厌烦/惊讶为主且不伴随悲伤的情况，标 Not sad。\n"
    "- 在 <label></label> 中仅输出恰好其一：Sad 或 Not sad（与英文 gold 完全一致）。\n"
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
    parser = argparse.ArgumentParser(
        description="Task5：英→中翻译后再判断 Sad / Not sad（检索阶段仍为英文）。",
    )
    parser.add_argument("--examples_limit", type=int, default=0, help="最多推理多少条；<=0 表示全部。")
    parser.add_argument("--output_dir", type=str, default="examples", help="结果输出目录。")
    parser.add_argument("--retries", type=int, default=3, help="分类阶段单条失败重试次数。")
    parser.add_argument("--translate_retries", type=int, default=0, help="翻译阶段重试次数；0 表示与 --retries 相同。")
    parser.add_argument("--retry_wait_seconds", type=float, default=2.0, help="重试等待秒数。")
    parser.add_argument("--resume", action="store_true", help="开启断点续跑。")
    parser.add_argument(
        "--task5_emoji_mode",
        type=str,
        choices=["off", "alias", "zh"],
        default="off",
        help="emoji 转换模式：off/alias/zh（作用于检索与翻译输入，与生成的英文预处理一致）。",
    )
    parser.add_argument(
        "--task5_strip_hashtag",
        type=str,
        choices=["on", "off"],
        default="on",
        help="是否去掉 hashtag 的 #。",
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


def _build_translate_prompt(en_text: str) -> str:
    return (
        "### Translation\n"
        "Translate the English tweet below into natural Simplified Chinese.\n"
        "- Keep @mentions and URLs unchanged.\n"
        "- If emoji carry sentiment, you may insert a short bracket note like 【大哭】 next to them.\n"
        "- Output ONLY your translation wrapped exactly once as: <label>…</label>\n\n"
        "Tweet:\n"
        f"{en_text}\n"
    )


def _prepare_task5_item(
    example: dict,
    processed_all_examples: list[dict],
    task_description: str,
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


def _infer_task5_zh_item(
    item: dict,
    retries: int,
    translate_retries: int,
    retry_wait_seconds: float,
) -> dict:
    example_id = item["example_id"]
    prompt_input_text = item["prompt_input_text"]
    expected = item["expected"]
    examples_str = item["examples_str"]

    tr_retries = translate_retries if translate_retries > 0 else retries
    translated_zh = ""
    for attempt in range(1, tr_retries + 1):
        try:
            tr_raw = annotate(_build_translate_prompt(prompt_input_text))
            chunk = "" if tr_raw is None else str(tr_raw).strip()
            if chunk:
                translated_zh = chunk
                break
        except Exception as e:  # noqa: BLE001
            print(f"[翻译异常] task=5 example_id={example_id} attempt={attempt}/{tr_retries} error={e}")
        if attempt < tr_retries:
            time.sleep(retry_wait_seconds)

    translation_failed = not translated_zh
    if translation_failed:
        classify_prompt = build_prompt(TASK5_CANONICAL_DESCRIPTION, prompt_input_text, task_id=5)
    else:
        classify_prompt = _task_prompt_shell(
            TASK5_SPECIFIC_ZH,
            TASK5_CANONICAL_DESCRIPTION_ZH,
            translated_zh,
        )
    input_prompt = classify_prompt.replace("[[EXAMPLES]]\n\n", f"{examples_str}\n\n")

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
        "translated_zh": translated_zh,
        "translation_failed": translation_failed,
        "expected": expected,
        "prediction": prediction,
        "is_match": is_match,
    }


def run_task5_zh(
    output_dir: Path,
    examples_limit: int = 0,
    retries: int = 3,
    translate_retries: int = 0,
    retry_wait_seconds: float = 2.0,
    resume: bool = False,
    task5_emoji_mode: str = "off",
    task5_strip_hashtag: str = "on",
    task5_shot_k: int = 6,
    task5_retrieval_pool_size: int = 200,
    print_empty_prediction: str = "on",
    retrieval_batch_size: int = 1,
) -> dict:
    with _task_json_path().open("r", encoding="utf-8") as f:
        task_dict = json.load(f)

    task_id = 5
    task_name = task_dict["task_name"]
    task_description_en = TASK5_CANONICAL_DESCRIPTION
    all_examples = task_dict["examples"]
    if examples_limit > 0:
        all_examples = all_examples[:examples_limit]

    strip_hashtag_enabled = task5_strip_hashtag == "on"
    hashtag_mode = "striphash-on" if strip_hashtag_enabled else "striphash-off"
    output_file = output_dir / (
        f"openseek-{task_id}-examples-compare-task5opt-zh-translate-emoji-"
        f"{task5_emoji_mode}-{hashtag_mode}.jsonl"
    )
    done_ids = _load_done_ids(output_file) if resume else set()
    mode = "a" if resume else "w"

    if done_ids:
        print(f"[断点续跑] task=5 zh-translate, 已完成 {len(done_ids)} 条，继续剩余样本")

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
    tr_retries_eff = translate_retries if translate_retries > 0 else retries
    with output_file.open(mode, encoding="utf-8") as wf:
        for i in tqdm(range(0, len(pending_examples), bs), desc=f"Task5 ZH-translate: {task_name}"):
            chunk = pending_examples[i : i + bs]
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(bs, len(chunk))) as executor:
                prepare_futures = [
                    executor.submit(
                        _prepare_task5_item,
                        example=ex,
                        processed_all_examples=processed_all_examples,
                        task_description=task_description_en,
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
                        _infer_task5_zh_item,
                        item=item,
                        retries=retries,
                        translate_retries=tr_retries_eff,
                        retry_wait_seconds=retry_wait_seconds,
                    )
                    for item in prepared_items
                ]
                infer_results = [f.result() for f in infer_futures]

            for result in infer_results:
                example_id = result["example_id"]
                input_text = result["input_text"]
                translated_zh = result["translated_zh"]
                translation_failed = result["translation_failed"]
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
                    "translated_zh": translated_zh,
                    "translation_failed": translation_failed,
                    "expected_output": expected,
                    "model_output": prediction,
                    "is_match": is_match,
                }
                wf.write(json.dumps(row, ensure_ascii=False) + "\n")
                wf.flush()
                done_ids.add(example_id)

    total, match_count, accuracy = _compute_metrics_from_jsonl(output_file)
    print(f"[保存完成] task=5 zh-translate, total={total}, matched={match_count}, accuracy={accuracy:.2%}, file={output_file}")
    return {
        "task_id": 5,
        "task_name": task_name,
        "mode": "zh_translate_then_classify",
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

    summary_item = run_task5_zh(
        output_dir=output_dir,
        examples_limit=args.examples_limit,
        retries=args.retries,
        translate_retries=args.translate_retries,
        retry_wait_seconds=args.retry_wait_seconds,
        resume=args.resume,
        task5_emoji_mode=args.task5_emoji_mode,
        task5_strip_hashtag=args.task5_strip_hashtag,
        task5_shot_k=args.task5_shot_k,
        task5_retrieval_pool_size=args.task5_retrieval_pool_size,
        print_empty_prediction=args.print_empty_prediction,
        retrieval_batch_size=args.retrieval_batch_size,
    )
    summary_file = output_dir / "summary_task5opt_zh_translate.json"
    with summary_file.open("w", encoding="utf-8") as f:
        json.dump([summary_item], f, ensure_ascii=False, indent=2)
    print(f"[汇总完成] {summary_file}")


if __name__ == "__main__":
    main()
