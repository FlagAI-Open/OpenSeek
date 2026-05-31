import argparse
import json
import os
import re
import time
from pathlib import Path

from tqdm import tqdm

from method_hyb import annotate_nvidia as annotate
from method_hyb import build_prompt, select_examples


REPO_ROOT = Path(__file__).resolve().parent.parent
TASK7_FILE = "openseek-7_jeopardy_answer_generation_all.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Task7 test_samples 推理脚本（对齐 infer_examples_main1.py 的检索与提示配置）。"
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="tokenizer 路径或 HuggingFace 模型 ID（同 infer_examples_main1.py）。",
    )
    parser.add_argument("--samples_limit", type=int, default=0, help="最多推理多少条 test_samples；<=0 表示全部。")
    parser.add_argument("--output_dir", type=str, default="examples_main1", help="输出目录。")
    parser.add_argument("--retries", type=int, default=3, help="单条样本推理失败时最大重试次数。")
    parser.add_argument("--retry_wait_seconds", type=float, default=2.0, help="重试间隔秒数。")
    parser.add_argument("--resume", action="store_true", help="开启断点续跑：已存在输出时跳过已完成样本。")
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
        help="是否打印模型原始输出预览（映射到 ANNOTATE_LOG_EVERY_RESPONSE）。",
    )
    return parser.parse_args()


def _task_json_path() -> Path:
    return REPO_ROOT / "data" / TASK7_FILE


def _resolve_output_dir(output_dir: str) -> Path:
    p = Path(output_dir)
    if p.is_absolute():
        return p
    return (REPO_ROOT / p).resolve()


def _normalize_text(text: str) -> str:
    return " ".join(str(text).strip().split())


def _safe_parse_prediction(raw_prediction) -> str:
    """
    对模型输出做容错解析，尽量减少空预测：
    1) 直接字符串化并清洗空白；
    2) 优先提取常见标签包裹内容；
    3) 回退到非空行、去引号文本；
    4) 最终再做一次兜底清洗。
    """
    if raw_prediction is None:
        return ""

    text = _normalize_text(str(raw_prediction))
    if not text:
        return ""

    # 常见标签解析：<label>xxx</label> / <answer>xxx</answer>
    m = re.search(r"<\s*(?:label|answer)\s*>\s*(.*?)\s*<\s*/\s*(?:label|answer)\s*>", text, flags=re.IGNORECASE)
    if m:
        parsed = _normalize_text(m.group(1))
        if parsed:
            return parsed.strip("\"'` ")

    # 去掉 markdown 代码块标记后再尝试。
    text_wo_fence = text.replace("```", " ")
    text_wo_fence = _normalize_text(text_wo_fence)
    if text_wo_fence:
        # 例如 "Answer: xxx" / "Prediction: xxx"
        m2 = re.search(r"(?:answer|prediction|output)\s*:\s*(.+)$", text_wo_fence, flags=re.IGNORECASE)
        if m2:
            parsed = _normalize_text(m2.group(1))
            if parsed:
                return parsed.strip("\"'` ")

        # 最后兜底：返回清洗后的原文本（去首尾引号）。
        return text_wo_fence.strip("\"'` ")

    return ""


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


def run_task7_test_samples(
    output_dir: Path,
    tokenizer_path: str | None = None,
    samples_limit: int = 0,
    retries: int = 3,
    retry_wait_seconds: float = 2.0,
    resume: bool = False,
) -> dict:
    with _task_json_path().open("r", encoding="utf-8") as f:
        task_dict = json.load(f)

    task_id = 7
    task_name = task_dict["task_name"]
    task_description = task_dict["Definition"][0]
    test_samples = list(task_dict.get("test_samples", []))
    if samples_limit > 0:
        test_samples = test_samples[:samples_limit]

    # 完全对齐 infer_examples_main1.py：固定 examples 池为前 100 条。
    icl_examples = task_dict["examples"][:100]

    output_file = output_dir / f"openseek-{task_id}-test_samples-main1-predictions.jsonl"
    done_ids = _load_done_ids(output_file) if resume else set()
    mode = "a" if resume else "w"

    if done_ids:
        print(f"[断点续跑] task=7, 已完成 {len(done_ids)} 条，继续剩余样本")

    with output_file.open(mode, encoding="utf-8") as wf:
        for sample in tqdm(test_samples, desc=f"Task7 Test Samples Inference: {task_name}"):
            example_id = str(sample.get("id", "")).strip()
            if resume and example_id in done_ids:
                continue

            input_text = str(sample.get("input", ""))
            prompt = build_prompt(task_description, input_text, task_id=task_id)
            examples_str = select_examples(
                icl_examples,
                task_description,
                input_text,
                tokenizer_path=tokenizer_path,
                hybrid=True,
                top_k=3,
                use_explanation=True,
                use_bm25_semantic_rerank=True,
            )
            input_prompt = prompt.replace("[[EXAMPLES]]\n\n", f"{examples_str}\n\n")

            prediction = ""
            for attempt in range(1, retries + 1):
                try:
                    raw_prediction = annotate(input_prompt)
                    prediction = _safe_parse_prediction(raw_prediction)
                    if os.environ.get("ANNOTATE_LOG_EVERY_RESPONSE", "0") == "1":
                        raw_preview = _normalize_text("" if raw_prediction is None else str(raw_prediction))
                        print(
                            f"[模型输出] task=7 example_id={example_id} "
                            f"attempt={attempt}/{retries} raw={raw_preview[:300]}"
                        )
                        print(
                            f"[解析结果] task=7 example_id={example_id} "
                            f"attempt={attempt}/{retries} prediction={prediction[:300]}"
                        )
                    if prediction:
                        break
                    if attempt < retries:
                        print(f"[空预测重试] task=7 example_id={example_id} attempt={attempt}/{retries}")
                        time.sleep(retry_wait_seconds)
                except Exception as e:  # noqa: BLE001
                    if attempt >= retries:
                        print(
                            f"[推理失败] task=7 example_id={example_id} "
                            f"attempt={attempt}/{retries} error={e}"
                        )
                    else:
                        print(
                            f"[重试] task=7 example_id={example_id} "
                            f"attempt={attempt}/{retries} error={e}"
                        )
                        time.sleep(retry_wait_seconds)

            row = {"test_sample_id": example_id, "prediction": prediction}
            wf.write(json.dumps(row, ensure_ascii=False) + "\n")
            wf.flush()
            done_ids.add(example_id)

    total = 0
    if output_file.exists():
        with output_file.open("r", encoding="utf-8") as rf:
            total = sum(1 for line in rf if line.strip())
    print(f"[保存完成] task=7, total={total}, file={output_file}")
    return {
        "task_id": 7,
        "task_name": task_name,
        "total": total,
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

    summary = run_task7_test_samples(
        output_dir=output_dir,
        tokenizer_path=args.tokenizer_path,
        samples_limit=args.samples_limit,
        retries=args.retries,
        retry_wait_seconds=args.retry_wait_seconds,
        resume=args.resume,
    )
    summary_file = output_dir / "summary_task7_test_samples.json"
    with summary_file.open("w", encoding="utf-8") as f:
        json.dump([summary], f, ensure_ascii=False, indent=2)
    print(f"[汇总完成] {summary_file}")


if __name__ == "__main__":
    main()
