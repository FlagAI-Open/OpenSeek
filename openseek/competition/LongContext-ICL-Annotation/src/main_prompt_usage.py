import argparse
import json
import os
import re
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")

from method_hyb import select_examples
from method_hyb import annotate_nvidia as annotate
from method_hyb_prompts import build_prompt


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


def _task_json_path(task_id: int) -> Path:
    return REPO_ROOT / "data" / TASK_DATA_FILES[task_id]


def _resolve_output_dir(log_path_prefix: str | None) -> str:
    """输出目录：默认项目根下 outputs/；相对路径相对项目根解析。"""
    if not log_path_prefix:
        return str((REPO_ROOT / "outputs").resolve())
    p = Path(log_path_prefix)
    if p.is_absolute():
        return str(p.resolve())
    return str((REPO_ROOT / log_path_prefix).resolve())


def _pick_output_jsonl(log_path_prefix: str, task_id: int) -> str:
    """生成 main1 同款命名：openseek-{id}-v1.jsonl；若存在则递增版本号。"""
    os.makedirs(log_path_prefix, exist_ok=True)
    primary = os.path.join(log_path_prefix, f"openseek-{task_id}-v1.jsonl")
    if not os.path.exists(primary):
        return primary
    v = 2
    while True:
        alt = os.path.join(log_path_prefix, f"openseek-{task_id}-v{v}.jsonl")
        if not os.path.exists(alt):
            return alt
        v += 1


def _tokenize_for_prefilter(text: str) -> set[str]:
    return set(re.findall(r"\w+", (text or "").lower()))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _prefilter_examples_by_jaccard(
    all_examples: list[dict],
    query_text: str,
    limit: int,
) -> list[dict]:
    """先用词级 Jaccard 做轻量初筛，再交给后续混合检索精排。"""
    if limit <= 0 or limit >= len(all_examples):
        return all_examples

    q = _tokenize_for_prefilter(query_text)
    scored: list[tuple[float, int, dict]] = []
    for idx, ex in enumerate(all_examples):
        inp = str(ex.get("input", ""))
        s = _jaccard(q, _tokenize_for_prefilter(inp))
        scored.append((s, idx, ex))

    scored.sort(key=lambda x: (-x[0], x[1]))
    return [ex for _, _, ex in scored[:limit]]


def parse_args():
    parser = argparse.ArgumentParser(
        description="演示如何使用 method_hyb_prompts + 混合检索(top3) 构造最终 input_prompt。"
    )
    parser.add_argument("--task_id", type=int, default=1, choices=range(1, 9))
    parser.add_argument(
        "--task_start",
        type=int,
        default=3,
        help="起始任务编号（含），与 README 中 openseek-[id] 一致，默认 1。",
    )
    parser.add_argument(
        "--task_end",
        type=int,
        default=8,
        help="结束任务编号（含），提交需 8 个 jsonl 时设为 8，默认 8。",
    )
    parser.add_argument("--sample_idx", type=int, default=0, help="test_samples 中第几个样本")
    parser.add_argument(
        "--all_samples",
        action="store_true",
        help="开启后每个 task 跑完整 test_samples；否则只跑 --sample_idx 指定的单条。",
    )
    parser.add_argument("--icl_size", type=int, default=100, help="参与检索的示例池大小")
    parser.add_argument("--top_k", type=int, default=3, help="混合检索召回条数，推荐 3")
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="Qwen tokenizer 目录或 HF 模型 ID；不传时自动回退",
    )
    parser.add_argument(
        "--save_prompt",
        type=str,
        default=None,
        help="可选：将最终 input_prompt 保存到指定文件路径",
    )
    parser.add_argument(
        "--run_infer",
        action="store_true",
        help="开启后会在构造 prompt 后直接调用模型推理。",
    )
    parser.add_argument(
        "--log_model_output",
        action="store_true",
        help="开启后打印模型接口返回日志（依赖 ANNOTATE_LOG_EVERY_RESPONSE=1）。",
    )
    parser.add_argument(
        "--save_pred_jsonl",
        type=str,
        default=None,
        help="可选：保存推理结果，格式对齐 main1.py（按任务保存 openseek-{id}-v*.jsonl）。",
    )
    return parser.parse_args()


def build_input_prompt_for_sample(
    task_id: int,
    sample_idx: int = 0,
    icl_size: int = 100,
    top_k: int = 3,
    tokenizer_path: str | None = None,
) -> tuple[str, dict]:
    task_file = _task_json_path(task_id)
    with open(task_file, "r", encoding="utf-8") as f:
        task_dict = json.load(f)

    task_name = task_dict["task_name"]
    task_description = task_dict["Definition"][0]
    all_examples = task_dict["examples"]
    test_samples = task_dict["test_samples"]
    if not test_samples:
        raise ValueError(f"task {task_id} 没有 test_samples")
    if sample_idx < 0 or sample_idx >= len(test_samples):
        raise IndexError(
            f"sample_idx 越界: {sample_idx}, 合法范围 [0, {len(test_samples) - 1}]"
        )

    sample = test_samples[sample_idx]
    text2annotate = sample["input"]
    query_text = f"{task_description}\n{text2annotate}".strip()

    # 先用 Jaccard 对全量 examples 做词法初筛，避免直接按顺序截断。
    icl_examples = _prefilter_examples_by_jaccard(
        all_examples=all_examples,
        query_text=query_text,
        limit=icl_size,
    )

    # 1) 使用 method_hyb_prompts 的按任务 prompt（task_id -> 专用模板）
    prompt = build_prompt(task_description, text2annotate, task_id=task_id)

    # 2) 使用 method_hyb 的混合检索（BM25 + 语义 + 重排）召回 few-shot，固定 top_k
    examples_str = select_examples(
        icl_examples,
        task_description,
        text2annotate,
        tokenizer_path=tokenizer_path,
        hybrid=True,
        top_k=top_k,
        use_explanation=True,
        use_bm25_semantic_rerank=True,
    )

    # 3) 将 few-shot 填充进 [[EXAMPLES]]
    input_prompt = prompt.replace("[[EXAMPLES]]\n\n", examples_str + "\n\n")
    meta = {
        "task_id": task_id,
        "task_name": task_name,
        "test_sample_id": sample.get("id", ""),
        "examples_total": len(all_examples),
        "icl_size": icl_size,
        "prefiltered_size": len(icl_examples),
        "top_k": top_k,
    }
    return input_prompt, meta


def _task_dict(task_id: int) -> dict:
    task_file = _task_json_path(task_id)
    with open(task_file, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    args = parse_args()
    if args.log_model_output:
        os.environ["ANNOTATE_LOG_EVERY_RESPONSE"] = "1"
        print("[日志设置] 已开启模型输出日志：ANNOTATE_LOG_EVERY_RESPONSE=1")
    task_start = max(1, args.task_start)
    task_end = min(8, args.task_end)
    if task_start > task_end:
        raise ValueError(f"task_start({task_start}) 不能大于 task_end({task_end})")
    out_dir = _resolve_output_dir(args.save_pred_jsonl) if args.save_pred_jsonl else None
    task_output_files: dict[int, str] = {}

    for task_id in range(task_start, task_end + 1):
        if args.all_samples:
            td = _task_dict(task_id)
            total = len(td.get("test_samples", []))
            sample_indices = range(total)
        else:
            sample_indices = [args.sample_idx]

        for sample_idx in sample_indices:
            input_prompt, meta = build_input_prompt_for_sample(
                task_id=task_id,
                sample_idx=sample_idx,
                icl_size=args.icl_size,
                top_k=args.top_k,
                tokenizer_path=args.tokenizer_path,
            )

            print(
                f"[prompt构造完成] task={meta['task_id']}({meta['task_name']}), "
                f"sample={sample_idx}, test_sample_id={meta['test_sample_id']}, top_k={meta['top_k']}"
            )
            if (not args.all_samples) or sample_idx == 0:
                print("\n===== Prompt Preview (first 1200 chars) =====\n")
                print(input_prompt[:1200])
                print("\n===== Prompt Preview End =====\n")

            if args.save_prompt:
                out_path = Path(args.save_prompt)
                if not out_path.is_absolute():
                    out_path = (REPO_ROOT / out_path).resolve()
                suffix = f"_task{task_id}_sample{sample_idx}" if args.all_samples else f"_task{task_id}" if task_start != task_end else ""
                if suffix:
                    out_path = out_path.with_name(f"{out_path.stem}{suffix}{out_path.suffix}")
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(input_prompt, encoding="utf-8")
                print(f"[已保存] {out_path}")

            if args.run_infer:
                pred = annotate(input_prompt)
                prediction = "" if pred is None else pred
                print(
                    f"[推理结果] task={meta['task_id']} sample={sample_idx} "
                    f"test_sample_id={meta['test_sample_id']} prediction={prediction!r}"
                )
                if args.save_pred_jsonl:
                    assert out_dir is not None
                    if meta["task_id"] not in task_output_files:
                        task_output_files[meta["task_id"]] = _pick_output_jsonl(out_dir, meta["task_id"])
                    pred_path = task_output_files[meta["task_id"]]
                    row = {"test_sample_id": meta["test_sample_id"], "prediction": prediction}
                    with open(pred_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    print(f"[预测已追加] {pred_path}")


if __name__ == "__main__":
    main()

