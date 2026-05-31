import argparse
import concurrent.futures
import json
import os
import random
import re
import time
import unicodedata
from pathlib import Path

from tqdm import tqdm

from method import annotate_nvidia as annotate
from method import build_prompt, select_examples
from postprocess_task3_outputs import postprocess_jsonl_inplace
from postprocess_task5_outputs import postprocess_recall_jsonl_inplace


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
        description="对 8 个任务数据的 examples 做推理，并对比模型输出与标准 output。"
    )
    parser.add_argument(
        "--task_start",
        type=int,
        default=5,
        help="起始任务编号（含），默认 1。",
    )
    parser.add_argument(
        "--task_end",
        type=int,
        default=5,
        help="结束任务编号（含），默认 8。",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="tokenizer 路径或 HuggingFace 模型 ID，默认沿用现有逻辑。",
    )
    parser.add_argument(
        "--examples_limit",
        type=int,
        default=0,
        help="每个任务最多推理多少条 examples；<=0 表示全部。",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="examples",
        help="结果输出目录（相对路径按仓库根目录解析），默认 examples。",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="单条样本推理失败时最大重试次数，默认 3。",
    )
    parser.add_argument(
        "--retry_wait_seconds",
        type=float,
        default=2.0,
        help="重试前等待秒数，默认 2.0。",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="开启断点续跑：若输出文件已存在，则跳过已完成的 example_id。",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=6,
        help="并行任务数；<=0 表示自动（不超过任务数与 4）。",
    )
    parser.add_argument(
        "--random_k",
        type=int,
        default=3,
        help="每条样本随机召回的示例数，默认 3。",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="随机召回种子，默认 42。",
    )
    parser.add_argument(
        "--task5_emoji_mode",
        type=str,
        choices=["off", "alias", "zh"],
        default="off",
        help=(
            "task5 的 emoji 转换模式：off(不转换)、alias(:crying_face:)、"
            "zh(中文词)。默认 alias。"
        ),
    )
    parser.add_argument(
        "--task5_strip_hashtag",
        type=str,
        choices=["on", "off"],
        default="on",
        help="task5 是否去掉 hashtag 的 #（保留词本身），默认 on。",
    )
    parser.add_argument(
        "--thinking",
        type=str,
        choices=["on", "off"],
        default="off",
        help="是否开启模型 thinking（映射到 DASHSCOPE_ENABLE_THINKING），默认 off。",
    )
    parser.add_argument(
        "--task5_postprocess",
        type=str,
        choices=["on", "off"],
        default="on",
        help="task5 推理结束后执行 FP 压制后处理（emoji+text_calib），默认 on。",
    )
    parser.add_argument(
        "--task5_postprocess_fn_boost",
        action="store_true",
        help="task5 后处理同时开启 FN 提升（默认仅压 FP）。",
    )
    return parser.parse_args()


def _task_json_path(task_id: int) -> Path:
    if task_id not in TASK_DATA_FILES:
        raise ValueError(f"task_id should be in [1, 8], but got {task_id}.")
    return REPO_ROOT / "data" / TASK_DATA_FILES[task_id]


def _normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


def _strip_hashtag_symbol(text: str) -> str:
    """去掉 hashtag 的 #，保留词本身。"""
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
        print('converted:', converted)
        return " " + " ".join(converted) + " "

    return EMOJI_RE.sub(_replace, text)


def run_task(
    task_id: int,
    output_dir: Path,
    tokenizer_path: str | None = None,
    examples_limit: int = 0,
    retries: int = 3,
    retry_wait_seconds: float = 2.0,
    resume: bool = False,
    random_k: int = 3,
    random_seed: int = 42,
    task5_emoji_mode: str = "alias",
    task5_strip_hashtag: str = "on",
    task5_postprocess: str = "on",
    task5_postprocess_fn_boost: bool = False,
) -> dict:
    task_file = _task_json_path(task_id)
    with task_file.open("r", encoding="utf-8") as f:
        task_dict = json.load(f)

    task_name = task_dict["task_name"]
    task_description = task_dict["Definition"][0]
    all_examples = task_dict["examples"]
    if examples_limit > 0:
        all_examples = all_examples[:examples_limit]

    strip_hashtag_enabled = task5_strip_hashtag == "on"
    hashtag_mode = "striphash-on" if strip_hashtag_enabled else "striphash-off"
    output_file = output_dir / (
        f"openseek-{task_id}-examples-compare-emoji-{task5_emoji_mode}-{hashtag_mode}.jsonl"
    )
    done_ids = _load_done_ids(output_file) if resume else set()
    mode = "a" if resume else "w"

    if done_ids:
        print(f"[断点续跑] task={task_id}, 已完成 {len(done_ids)} 条，继续剩余样本")

    with output_file.open(mode, encoding="utf-8") as wf:
        for example in tqdm(all_examples, desc=f"Examples Inference Task {task_id}: {task_name}"):
            example_id = str(example.get("id", "")).strip()
            if resume and example_id in done_ids:
                continue

            input_text = str(example.get("input", ""))
            if task_id == 5 and strip_hashtag_enabled:
                input_text = _strip_hashtag_symbol(input_text)
            prompt_input_text = (
                _convert_task5_emoji(str(input_text), task5_emoji_mode)
                if task_id == 5
                else input_text
            )
            expected = _extract_output(example.get("output", ""))

            # 避免将当前样本泄漏到上下文示例里。
            icl_pool = [e for e in task_dict["examples"][:100] if e.get("id") != example_id]
            k = max(1, random_k)
            if len(icl_pool) <= k:
                selected = icl_pool
            else:
                # 为同一任务的同一样本生成稳定随机结果，避免续跑时漂移
                sample_seed = random_seed + task_id * 1_000_003 + hash(example_id)
                rnd = random.Random(sample_seed)
                selected = rnd.sample(icl_pool, k=k)
            examples_str = ""
            for ex in selected:
                ex_input = str(ex.get("input", "")).strip()
                if task_id == 5:
                    if strip_hashtag_enabled:
                        ex_input = _strip_hashtag_symbol(ex_input)
                    ex_input = _convert_task5_emoji(ex_input, task5_emoji_mode)
                ex_output = _extract_output(ex.get("output", ""))
                examples_str += f"# {ex_input} <label> {ex_output} </label>\n"
            prompt = build_prompt(task_description, prompt_input_text, task_id=task_id)
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

    if task_id == 3 and output_file.exists():
        pp_stats = postprocess_jsonl_inplace(output_file)
        print(
            f"[task3 后处理] changed={pp_stats['changed']}/{pp_stats['total']}, "
            f"accuracy={pp_stats['accuracy']:.2%}, file={output_file}"
        )

    postprocessed = task_id == 3
    if (
        task_id == 5
        and task5_postprocess == "on"
        and output_file.exists()
    ):
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
        "postprocessed": postprocessed,
    }


def main() -> None:
    args = parse_args()
    os.environ["DASHSCOPE_ENABLE_THINKING"] = "1" if args.thinking == "on" else "0"
    print(f"[thinking] DASHSCOPE_ENABLE_THINKING={os.environ['DASHSCOPE_ENABLE_THINKING']}")
    task_start = max(1, args.task_start)
    task_end = min(8, args.task_end)
    if task_start > task_end:
        raise ValueError(f"task_start({task_start}) 不能大于 task_end({task_end})")

    output_dir = _resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[输出目录] {output_dir}")

    task_ids = list(range(task_start, task_end + 1))
    default_workers = min(4, len(task_ids)) if task_ids else 1
    max_workers = args.max_workers if args.max_workers > 0 else default_workers
    max_workers = max(1, min(max_workers, len(task_ids) if task_ids else 1))
    print(f"[并行执行] max_workers={max_workers}, tasks={task_ids}")

    summary: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(
                run_task,
                task_id=task_id,
                output_dir=output_dir,
                tokenizer_path=args.tokenizer_path,
                examples_limit=args.examples_limit,
                retries=args.retries,
                retry_wait_seconds=args.retry_wait_seconds,
                resume=args.resume,
                random_k=args.random_k,
                random_seed=args.random_seed,
                task5_emoji_mode=args.task5_emoji_mode,
                task5_strip_hashtag=args.task5_strip_hashtag,
                task5_postprocess=args.task5_postprocess,
                task5_postprocess_fn_boost=args.task5_postprocess_fn_boost,
            ): task_id
            for task_id in task_ids
        }
        for future in concurrent.futures.as_completed(future_to_task):
            task_id = future_to_task[future]
            try:
                summary_item = future.result()
                summary.append(summary_item)
            except Exception as exc:  # noqa: BLE001
                print(f"[任务失败] task={task_id}, error={exc}")

    summary.sort(key=lambda x: x.get("task_id", 0))

    summary_file = output_dir / "summary.json"
    with summary_file.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[汇总完成] {summary_file}")


if __name__ == "__main__":
    main()
