import argparse
import json
import os
from pathlib import Path

from tqdm import tqdm

from method_hyb import annotate_nvidia as annotate
from method_hyb import build_prompt, select_examples


REPO_ROOT = Path(__file__).resolve().parent.parent
TASK2_FILE = "openseek-2_count_nouns_verbs.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Task2 examples 推理（main1 风格，前100召回去泄露）。"
    )
    parser.add_argument("--tokenizer_path", type=str, default=None, help="tokenizer 路径或模型名。")
    parser.add_argument("--examples_limit", type=int, default=0, help="最多推理多少条 examples；<=0 表示全部。")
    parser.add_argument("--output_dir", type=str, default="examples_main1_task2_noleak", help="输出目录。")
    parser.add_argument("--output_file_name", type=str, default="openseek-2-examples-main1-noleak-compare.jsonl")
    parser.add_argument("--icl_pool_size", type=int, default=100, help="召回候选池大小（默认前100）。")
    parser.add_argument("--top_k", type=int, default=3, help="每条样本召回示例数量。")
    parser.add_argument(
        "--use_bm25_semantic_rerank",
        type=str,
        choices=["on", "off"],
        default="on",
        help="是否启用 BM25+语义+重排混合召回。",
    )
    parser.add_argument(
        "--thinking",
        type=str,
        choices=["on", "off"],
        default="off",
        help="是否开启 thinking（映射到 DASHSCOPE_ENABLE_THINKING）。",
    )
    parser.add_argument(
        "--print_model_output",
        type=str,
        choices=["on", "off"],
        default="off",
        help="是否打印模型原始输出预览（ANNOTATE_LOG_EVERY_RESPONSE）。",
    )
    return parser.parse_args()


def _task_json_path() -> Path:
    return REPO_ROOT / "data" / TASK2_FILE


def _resolve_output_dir(output_dir: str) -> Path:
    p = Path(output_dir)
    if p.is_absolute():
        return p
    return (REPO_ROOT / p).resolve()


def _normalize_text(text: str) -> str:
    return " ".join(str(text or "").strip().split()).lower()


def _extract_output(output_value) -> str:
    if isinstance(output_value, list):
        return "" if not output_value else str(output_value[0]).strip()
    return str(output_value).strip()


def _task2_prediction_fallback(prediction: str) -> str:
    import re

    s = (prediction or "").strip()
    if not s:
        return s
    if re.fullmatch(r"\d+", s):
        return s
    m = re.search(r"\b(\d+)\b", s)
    return m.group(1) if m else s


def _build_noleak_icl_pool(
    all_examples: list[dict],
    current_example_id: str,
    current_input: str,
    icl_pool_size: int,
) -> list[dict]:
    """
    去泄露策略（前N候选）：
    1) 排除当前样本 id；
    2) 排除与当前 input 归一化后完全相同的样本（即使 id 不同）。
    """
    normalized_current_input = _normalize_text(current_input)
    pool: list[dict] = []
    for ex in all_examples[: max(1, icl_pool_size)]:
        ex_id = str(ex.get("id", "")).strip()
        ex_input = str(ex.get("input", ""))
        if ex_id == current_example_id:
            continue
        if _normalize_text(ex_input) == normalized_current_input:
            continue
        pool.append(ex)
    return pool


def run_task2_examples(
    output_dir: Path,
    tokenizer_path: str | None = None,
    examples_limit: int = 0,
    output_file_name: str = "openseek-2-examples-main1-noleak-compare.jsonl",
    icl_pool_size: int = 100,
    top_k: int = 3,
    use_bm25_semantic_rerank: bool = True,
) -> dict:
    with _task_json_path().open("r", encoding="utf-8") as f:
        task_dict = json.load(f)

    task_id = 2
    task_name = task_dict["task_name"]
    task_description = task_dict["Definition"][0]
    all_examples = list(task_dict["examples"])
    eval_examples = all_examples[:examples_limit] if examples_limit > 0 else all_examples

    output_file = output_dir / output_file_name
    rows: list[dict] = []
    leaked_out_count = 0
    # 先清空文件，再逐条追加写入，保证运行期间可实时查看结果。
    with output_file.open("w", encoding="utf-8"):
        pass

    with output_file.open("a", encoding="utf-8") as wf:
        for example in tqdm(eval_examples, desc=f"Task2 Examples Inference: {task_name}"):
            example_id = str(example.get("id", "")).strip()
            input_text = str(example.get("input", ""))
            expected = _extract_output(example.get("output", ""))

            icl_pool = _build_noleak_icl_pool(
                all_examples=all_examples,
                current_example_id=example_id,
                current_input=input_text,
                icl_pool_size=icl_pool_size,
            )
            if not icl_pool:
                leaked_out_count += 1

            prompt = build_prompt(task_description, input_text, task_id=task_id)
            examples_str = select_examples(
                icl_pool,
                task_description,
                input_text,
                tokenizer_path=tokenizer_path,
                hybrid=True,
                top_k=max(1, top_k),
                use_explanation=True,
                use_bm25_semantic_rerank=use_bm25_semantic_rerank,
            )
            input_prompt = prompt.replace("[[EXAMPLES]]\n\n", f"{examples_str}\n\n")

            raw_prediction = annotate(input_prompt)
            prediction = "" if raw_prediction is None else str(raw_prediction).strip()
            prediction = _task2_prediction_fallback(prediction)

            is_match = _normalize_text(prediction) == _normalize_text(expected)
            row = {
                "example_id": example_id,
                "input": input_text,
                "expected_output": expected,
                "model_output": prediction,
                "is_match": is_match,
                "icl_pool_size_after_noleak": len(icl_pool),
            }
            rows.append(row)
            wf.write(json.dumps(row, ensure_ascii=False) + "\n")
            wf.flush()

    total = len(rows)
    matched = sum(1 for r in rows if bool(r.get("is_match", False)))
    accuracy = matched / total if total else 0.0
    summary = {
        "task_id": 2,
        "task_name": task_name,
        "total": total,
        "matched": matched,
        "accuracy": accuracy,
        "icl_pool_size": icl_pool_size,
        "top_k": top_k,
        "noleak_excluded_all_candidates_count": leaked_out_count,
        "file": str(output_file),
    }
    print(
        f"[保存完成] task=2 total={total}, matched={matched}, accuracy={accuracy:.2%}, "
        f"noleak_empty_pool={leaked_out_count}, file={output_file}"
    )
    return summary


def main() -> None:
    args = parse_args()
    os.environ["DASHSCOPE_ENABLE_THINKING"] = "1" if args.thinking == "on" else "0"
    os.environ["ANNOTATE_LOG_EVERY_RESPONSE"] = "1" if args.print_model_output == "on" else "0"
    os.environ["ANNOTATE_FALLBACK_NUMBER"] = "1"

    output_dir = _resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[输出目录] {output_dir}")

    summary = run_task2_examples(
        output_dir=output_dir,
        tokenizer_path=args.tokenizer_path,
        examples_limit=args.examples_limit,
        output_file_name=args.output_file_name,
        icl_pool_size=args.icl_pool_size,
        top_k=args.top_k,
        use_bm25_semantic_rerank=args.use_bm25_semantic_rerank == "on",
    )

    summary_file = output_dir / "summary_task2_examples_main1_noleak.json"
    with summary_file.open("w", encoding="utf-8") as f:
        json.dump([summary], f, ensure_ascii=False, indent=2)
    print(f"[汇总完成] {summary_file}")


if __name__ == "__main__":
    main()
