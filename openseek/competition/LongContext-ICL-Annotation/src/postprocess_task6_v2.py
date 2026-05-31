#!/usr/bin/env python3
import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from method_hyb import annotate_nvidia


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "task6-v2 后处理："
            "1) sentence1_pred=N 且 sentence2_pred=N 时做同领域判定；"
            "2) sentence1_pred=Y 且 sentence2_pred=N 时，以 sentence1 为背景对 sentence2 补充判定；"
            "3) sentence1_pred=N 且 sentence2_pred=Y 时，以 sentence2 为背景对 sentence1 补充判定。"
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("examples/openseek-6-examples-compare-task6-v2-sentence-and.jsonl"),
        help="输入 JSONL（默认使用 task6-v2 主流程输出文件）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="输出 JSONL；不传则默认在输入文件名后追加 -postprocess-domain",
    )
    parser.add_argument("--retries", type=int, default=3, help="单条后处理失败重试次数")
    parser.add_argument("--retry_wait_seconds", type=float, default=2.0, help="重试等待秒数")
    parser.add_argument(
        "--thinking",
        type=str,
        choices=["on", "off"],
        default="off",
        help="是否开启模型 thinking（映射到 DASHSCOPE_ENABLE_THINKING）",
    )
    return parser.parse_args()


def normalize_binary_label(text: str) -> str:
    t = " ".join((text or "").strip().split()).upper()
    if t in {"Y", "N"}:
        return t
    m = re.search(r"<LABEL>\s*([YN])\s*</LABEL>", t)
    if m:
        return m.group(1)
    m = re.search(r"\b([YN])\b", t)
    if m:
        return m.group(1)
    return ""


def _extract_output(output_value: Any) -> str:
    if isinstance(output_value, list):
        return "" if not output_value else str(output_value[0]).strip()
    return str(output_value).strip()


def _resolve_output_path(input_path: Path, output_path: Path | None) -> Path:
    if output_path is not None:
        return output_path
    return input_path.with_name(f"{input_path.stem}-postprocess-domain{input_path.suffix}")


def _build_same_domain_prompt(sentence1: str, sentence2: str) -> str:
    return (
        "### Role\n"
        "You are a strict domain-consistency judge.\n\n"
        "### Task\n"
        "Given two sentences, decide whether they likely come from the same domain/register/source style.\n\n"
        "### Decision Rules\n"
        "1. Focus on style/register/source cues, not only surface keywords.\n"
        "2. Topic overlap alone is not enough.\n"
        "3. If uncertain, output N.\n"
        "4. Output only one label: Y or N.\n\n"
        "### Sentence 1\n"
        f"{sentence1}\n\n"
        "### Sentence 2\n"
        f"{sentence2}\n\n"
        "### Output Format\n"
        "Return exactly one label wrapped in tags: <label>Y</label> or <label>N</label>.\n"
    )


def _build_sentence_with_context_prompt(
    context_sentence: str,
    target_sentence: str,
    genre: str,
    *,
    context_label: str,
    target_label: str,
) -> str:
    g = (genre or "").strip()
    return (
        "### Role\n"
        "You are a strict genre validator for OpenSeek task6.\n\n"
        "### Task\n"
        f"Use {context_label} as context, then decide whether {target_label} fits the candidate genre.\n\n"
        "### Candidate Genre\n"
        f"- genre: {g}\n\n"
        "### Decision Rules\n"
        f"1. {context_label} is only background context; final label is for {target_label}.\n"
        "2. Focus on style/register/source cues, not only topic overlap.\n"
        "3. If evidence is weak or ambiguous, output N.\n"
        "4. Output only one label: Y or N.\n\n"
        f"### Background Context ({context_label})\n"
        f"{context_sentence}\n\n"
        f"### Target Sentence ({target_label})\n"
        f"{target_sentence}\n\n"
        "### Output Format\n"
        "Return exactly one label wrapped in tags: <label>Y</label> or <label>N</label>.\n"
    )


def _build_sentence2_with_context_prompt(sentence1: str, sentence2: str, genre: str) -> str:
    return _build_sentence_with_context_prompt(
        sentence1,
        sentence2,
        genre,
        context_label="Sentence 1",
        target_label="Sentence 2",
    )


def _build_sentence1_with_context_prompt(sentence1: str, sentence2: str, genre: str) -> str:
    return _build_sentence_with_context_prompt(
        sentence2,
        sentence1,
        genre,
        context_label="Sentence 2",
        target_label="Sentence 1",
    )


def _judge_same_domain(sentence1: str, sentence2: str, retries: int, retry_wait_seconds: float) -> str:
    prompt = _build_same_domain_prompt(sentence1=sentence1, sentence2=sentence2)
    pred = ""
    for attempt in range(1, retries + 1):
        try:
            raw = annotate_nvidia(prompt)
            pred = normalize_binary_label("" if raw is None else str(raw).strip())
            if pred in {"Y", "N"}:
                return pred
        except Exception as e:  # noqa: BLE001
            if attempt < retries:
                print(f"[同领域判定重试] attempt={attempt}/{retries} error={e}")
                time.sleep(retry_wait_seconds)
            else:
                print(f"[同领域判定失败] attempt={attempt}/{retries} error={e}")
    return pred


def _judge_with_context(
    prompt: str,
    retries: int,
    retry_wait_seconds: float,
    log_tag: str,
) -> str:
    pred = ""
    for attempt in range(1, retries + 1):
        try:
            raw = annotate_nvidia(prompt)
            pred = normalize_binary_label("" if raw is None else str(raw).strip())
            if pred in {"Y", "N"}:
                return pred
        except Exception as e:  # noqa: BLE001
            if attempt < retries:
                print(f"[{log_tag}重试] attempt={attempt}/{retries} error={e}")
                time.sleep(retry_wait_seconds)
            else:
                print(f"[{log_tag}失败] attempt={attempt}/{retries} error={e}")
    return pred


def _judge_sentence2_with_context(
    sentence1: str, sentence2: str, genre: str, retries: int, retry_wait_seconds: float
) -> str:
    prompt = _build_sentence2_with_context_prompt(sentence1=sentence1, sentence2=sentence2, genre=genre)
    return _judge_with_context(prompt, retries, retry_wait_seconds, "S2上下文判定")


def _judge_sentence1_with_context(
    sentence1: str, sentence2: str, genre: str, retries: int, retry_wait_seconds: float
) -> str:
    prompt = _build_sentence1_with_context_prompt(sentence1=sentence1, sentence2=sentence2, genre=genre)
    return _judge_with_context(prompt, retries, retry_wait_seconds, "S1上下文判定")


def apply_task6_v2_postprocess(
    *,
    sentence1: str,
    sentence2: str,
    genre: str,
    s1_pred: str,
    s2_pred: str,
    base_pred: str,
    retries: int,
    retry_wait_seconds: float,
) -> dict[str, str]:
    """
    v2 后处理统一入口。在 AND 基线 base_pred 上按句对模式复核。
    返回 final_pred 与各中间判定字段。
    """
    final_pred = base_pred
    same_domain_pred = ""
    sentence2_context_pred = ""
    sentence1_context_pred = ""

    if s1_pred == "N" and s2_pred == "N":
        same_domain_pred = _judge_same_domain(
            sentence1=sentence1,
            sentence2=sentence2,
            retries=retries,
            retry_wait_seconds=retry_wait_seconds,
        )
        final_pred = "Y" if same_domain_pred == "Y" else "N"
    elif s1_pred == "Y" and s2_pred == "N":
        sentence2_context_pred = _judge_sentence2_with_context(
            sentence1=sentence1,
            sentence2=sentence2,
            genre=genre,
            retries=retries,
            retry_wait_seconds=retry_wait_seconds,
        )
        if sentence2_context_pred == "Y":
            final_pred = "Y"
    elif s1_pred == "N" and s2_pred == "Y":
        sentence1_context_pred = _judge_sentence1_with_context(
            sentence1=sentence1,
            sentence2=sentence2,
            genre=genre,
            retries=retries,
            retry_wait_seconds=retry_wait_seconds,
        )
        if sentence1_context_pred == "Y":
            final_pred = "Y"

    return {
        "final_pred": final_pred,
        "domain_same_pred": same_domain_pred,
        "sentence2_context_pred": sentence2_context_pred,
        "sentence1_context_pred": sentence1_context_pred,
    }


def process_file(
    input_path: Path,
    output_path: Path,
    retries: int,
    retry_wait_seconds: float,
) -> tuple[int, int, int, int, int, int, int, int, int, int, float, float]:
    total = 0
    base_matched = 0
    post_matched = 0
    triggered_domain_nn = 0
    triggered_context_yn = 0
    triggered_context_ny = 0
    changed = 0
    changed_to_y = 0
    changed_from_context_yn = 0
    changed_from_context_ny = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)

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

            s1_pred = _extract_output(row.get("sentence1_pred", ""))
            s2_pred = _extract_output(row.get("sentence2_pred", ""))
            expected = _extract_output(row.get("expected_output", ""))
            base_pred = _extract_output(row.get("model_output", ""))
            genre = _extract_output(row.get("genre", ""))
            sentence1 = _extract_output(row.get("sentence1", ""))
            sentence2 = _extract_output(row.get("sentence2", ""))

            if base_pred == expected:
                base_matched += 1

            post = apply_task6_v2_postprocess(
                sentence1=sentence1,
                sentence2=sentence2,
                genre=genre,
                s1_pred=s1_pred,
                s2_pred=s2_pred,
                base_pred=base_pred,
                retries=retries,
                retry_wait_seconds=retry_wait_seconds,
            )
            final_pred = post["final_pred"]
            same_domain_pred = post["domain_same_pred"]
            sentence2_context_pred = post["sentence2_context_pred"]
            sentence1_context_pred = post["sentence1_context_pred"]

            if s1_pred == "N" and s2_pred == "N":
                triggered_domain_nn += 1
            elif s1_pred == "Y" and s2_pred == "N":
                triggered_context_yn += 1
            elif s1_pred == "N" and s2_pred == "Y":
                triggered_context_ny += 1

            if final_pred != base_pred:
                changed += 1
                if final_pred == "Y":
                    changed_to_y += 1
                if s1_pred == "Y" and s2_pred == "N":
                    changed_from_context_yn += 1
                if s1_pred == "N" and s2_pred == "Y":
                    changed_from_context_ny += 1

            row["domain_same_pred"] = same_domain_pred
            row["sentence2_context_pred"] = sentence2_context_pred
            row["sentence1_context_pred"] = sentence1_context_pred
            row["model_output_base"] = base_pred
            row["model_output"] = final_pred
            row["is_match"] = final_pred == expected

            if row["is_match"]:
                post_matched += 1

            wf.write(json.dumps(row, ensure_ascii=False) + "\n")

    base_acc = (base_matched / total) if total else 0.0
    post_acc = (post_matched / total) if total else 0.0
    return (
        total,
        base_matched,
        post_matched,
        triggered_domain_nn,
        triggered_context_yn,
        triggered_context_ny,
        changed,
        changed_to_y,
        changed_from_context_yn,
        changed_from_context_ny,
        base_acc,
        post_acc,
    )


def main() -> None:
    args = parse_args()
    os.environ["DASHSCOPE_ENABLE_THINKING"] = "1" if args.thinking == "on" else "0"
    print(f"[thinking] DASHSCOPE_ENABLE_THINKING={os.environ['DASHSCOPE_ENABLE_THINKING']}")

    input_path = args.input.resolve()
    output_path = _resolve_output_path(input_path, args.output.resolve() if args.output else None)

    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    (
        total,
        base_matched,
        post_matched,
        triggered_domain_nn,
        triggered_context_yn,
        triggered_context_ny,
        changed,
        changed_to_y,
        changed_from_context_yn,
        changed_from_context_ny,
        base_acc,
        post_acc,
    ) = process_file(
        input_path=input_path,
        output_path=output_path,
        retries=args.retries,
        retry_wait_seconds=args.retry_wait_seconds,
    )

    print(f"[完成] input={input_path}")
    print(f"[完成] output={output_path}")
    print(
        f"[统计] total={total} triggered_domain_nn={triggered_domain_nn} "
        f"triggered_context_yn={triggered_context_yn} triggered_context_ny={triggered_context_ny} "
        f"changed={changed} changed_to_y={changed_to_y} "
        f"changed_from_context_yn={changed_from_context_yn} changed_from_context_ny={changed_from_context_ny} "
        f"base_matched={base_matched} post_matched={post_matched} base_acc={base_acc:.2%} post_acc={post_acc:.2%}"
    )


if __name__ == "__main__":
    main()
