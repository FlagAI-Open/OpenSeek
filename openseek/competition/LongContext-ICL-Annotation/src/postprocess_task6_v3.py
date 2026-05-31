#!/usr/bin/env python3
"""
Task6 后处理 v3：在 v2 结果基础上，对仍为 N 的预测做「语篇上下文连贯」复核。

与 postprocess_task6_v2 的区别：
- v2：N+N 同领域、Y+N 带 genre 对 sentence2 复核；
- v3：仅当 model_output（v2 后）为 N 时，用 sentence1 作背景判断两句是否同一语篇/对话/叙事上下文；
  若为 Y 则升为 Y，**不考虑 genre/领域**。

默认输入：examples/openseek-6-examples-compare-task6-v2-sentence-and-postprocess-v2.jsonl
默认输出：...-post-v3.jsonl
"""

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
            "task6 后处理 v3：对 v2 结果中 model_output=N 的样本，"
            "判断 sentence1/sentence2 是否同一语篇上下文；是则升为 Y（不看 genre）。"
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "examples/openseek-6-examples-compare-task6-v2-sentence-and-postprocess-v2.jsonl"
        ),
        help="v2 后处理结果 JSONL",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="输出 JSONL；默认在输入 stem 后追加 -post-v3",
    )
    parser.add_argument("--retries", type=int, default=3, help="单条判定失败重试次数")
    parser.add_argument("--retry_wait_seconds", type=float, default=2.0, help="重试等待秒数")
    parser.add_argument(
        "--thinking",
        type=str,
        choices=["on", "off"],
        default="off",
        help="是否开启模型 thinking（映射到 DASHSCOPE_ENABLE_THINKING）",
    )
    parser.add_argument(
        "--flip_gate",
        type=str,
        choices=["none", "not_fiction", "block_nn_domain_n", "combined"],
        default="none",
        help=(
            "discourse=Y 时是否允许翻转为 Y："
            "none=全量；not_fiction=排除 fiction；"
            "block_nn_domain_n=禁止 N+N 且 domain_same=N；"
            "combined=禁 fiction∧N+N∧domain N，否则需 v2 信号∨非 NN∨任句 Y。"
        ),
    )
    return parser.parse_args()


def normalize_binary_label(text: str) -> str:
    t = " ".join((text or "").strip().split()).upper()
    if t in {"Y", "N"}:
        return t
    m = re.search(r"<LABEL>\s*([YN])\s*</LABEL>", t, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = re.search(r"\b([YN])\b", t)
    if m:
        return m.group(1).upper()
    return ""


def _extract_output(output_value: Any) -> str:
    if isinstance(output_value, list):
        return "" if not output_value else str(output_value[0]).strip()
    return str(output_value).strip()


def _resolve_output_path(input_path: Path, output_path: Path | None) -> Path:
    if output_path is not None:
        return output_path
    return input_path.with_name(f"{input_path.stem}-post-v3{input_path.suffix}")


def _build_discourse_context_prompt(sentence1: str, sentence2: str) -> str:
    """判断两句是否处于同一语篇/对话/叙事上下文（不涉及 genre）。"""
    return (
        "### Role\n"
        "You are a strict discourse-coherence judge.\n\n"
        "### Task\n"
        "Use Sentence 1 as prior context. Decide whether Sentence 2 belongs to the "
        "same discourse, conversation, or narrative continuation as Sentence 1.\n\n"
        "### Important\n"
        "- Do NOT judge genre, domain, or register.\n"
        "- Only judge whether Sentence 2 can naturally continue, paraphrase, or "
        "coherently follow Sentence 1 in the same situational context.\n"
        "- Topic overlap alone is NOT enough; there must be plausible discourse continuity.\n"
        "- If uncertain, output N.\n\n"
        "### Decision Rules\n"
        "Output Y if Sentence 2 is a plausible continuation, reformulation, or same-scene "
        "follow-up of Sentence 1.\n"
        "Output N if Sentence 2 switches scene/speaker/topic, is unrelated, or cannot be "
        "read as continuing Sentence 1.\n\n"
        "### Prior Context (Sentence 1)\n"
        f"{sentence1}\n\n"
        "### Candidate Continuation (Sentence 2)\n"
        f"{sentence2}\n\n"
        "### Output Format\n"
        "Return exactly one label wrapped in tags: <label>Y</label> or <label>N</label>.\n"
    )


def _judge_discourse_context(
    sentence1: str,
    sentence2: str,
    retries: int,
    retry_wait_seconds: float,
) -> str:
    prompt = _build_discourse_context_prompt(sentence1=sentence1, sentence2=sentence2)
    pred = ""
    for attempt in range(1, retries + 1):
        try:
            raw = annotate_nvidia(prompt)
            pred = normalize_binary_label("" if raw is None else str(raw).strip())
            if pred in {"Y", "N"}:
                return pred
        except Exception as e:  # noqa: BLE001
            if attempt < retries:
                print(f"[语篇上下文判定重试] attempt={attempt}/{retries} error={e}")
                time.sleep(retry_wait_seconds)
            else:
                print(f"[语篇上下文判定失败] attempt={attempt}/{retries} error={e}")
    return pred


def _label(value: Any) -> str:
    return _extract_output(value).upper()


def _is_nn_row(row: dict) -> bool:
    return _label(row.get("sentence1_pred")) == "N" and _label(row.get("sentence2_pred")) == "N"


def _allow_discourse_flip(row: dict, flip_gate: str) -> bool:
    """在 discourse_context_pred=Y 时，是否允许将最终标签升为 Y。"""
    if flip_gate == "none":
        return True

    genre = _label(row.get("genre"))
    domain = _label(row.get("domain_same_pred"))
    s1ctx = _label(row.get("sentence1_context_pred"))
    s2ctx = _label(row.get("sentence2_context_pred"))
    s1 = _label(row.get("sentence1_pred"))
    s2 = _label(row.get("sentence2_pred"))

    if flip_gate == "not_fiction":
        return genre != "FICTION"

    if flip_gate == "block_nn_domain_n":
        return not (_is_nn_row(row) and domain == "N")

    if flip_gate == "combined":
        if genre == "FICTION" and _is_nn_row(row) and domain == "N":
            return False
        v2_signal = domain == "Y" or s1ctx == "Y" or s2ctx == "Y"
        any_sentence_y = s1 == "Y" or s2 == "Y"
        return v2_signal or not _is_nn_row(row) or any_sentence_y

    return True


def process_file(
    input_path: Path,
    output_path: Path,
    retries: int,
    retry_wait_seconds: float,
    flip_gate: str = "none",
) -> dict[str, Any]:
    total = 0
    v2_matched = 0
    post_v3_matched = 0
    pred_n_total = 0
    triggered_v3 = 0
    changed = 0
    changed_to_y = 0
    changed_correct = 0
    changed_wrong = 0
    blocked_flip = 0

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
            expected = _extract_output(row.get("expected_output", ""))
            v2_pred = _extract_output(row.get("model_output", ""))
            sentence1 = _extract_output(row.get("sentence1", ""))
            sentence2 = _extract_output(row.get("sentence2", ""))

            if v2_pred == expected:
                v2_matched += 1

            row["model_output_post_v2"] = v2_pred
            discourse_context_pred = ""
            discourse_flip_allowed = ""
            final_pred = v2_pred

            if v2_pred == "N":
                pred_n_total += 1
                triggered_v3 += 1
                discourse_context_pred = _judge_discourse_context(
                    sentence1=sentence1,
                    sentence2=sentence2,
                    retries=retries,
                    retry_wait_seconds=retry_wait_seconds,
                )
                if discourse_context_pred == "Y":
                    allowed = _allow_discourse_flip(row, flip_gate)
                    discourse_flip_allowed = "Y" if allowed else "N"
                    if allowed:
                        final_pred = "Y"
                    else:
                        blocked_flip += 1

            if final_pred != v2_pred:
                changed += 1
                if final_pred == "Y":
                    changed_to_y += 1
                    if final_pred == expected:
                        changed_correct += 1
                    else:
                        changed_wrong += 1

            row["discourse_context_pred"] = discourse_context_pred
            row["discourse_flip_allowed"] = discourse_flip_allowed
            row["flip_gate"] = flip_gate
            row["model_output"] = final_pred
            row["is_match"] = final_pred == expected

            if row["is_match"]:
                post_v3_matched += 1

            wf.write(json.dumps(row, ensure_ascii=False) + "\n")

    v2_acc = (v2_matched / total) if total else 0.0
    post_v3_acc = (post_v3_matched / total) if total else 0.0
    return {
        "total": total,
        "pred_n_total": pred_n_total,
        "triggered_v3": triggered_v3,
        "changed": changed,
        "changed_to_y": changed_to_y,
        "changed_correct": changed_correct,
        "changed_wrong": changed_wrong,
        "v2_matched": v2_matched,
        "post_v3_matched": post_v3_matched,
        "v2_accuracy": v2_acc,
        "post_v3_accuracy": post_v3_acc,
        "delta_accuracy": post_v3_acc - v2_acc,
        "flip_gate": flip_gate,
        "blocked_flip": blocked_flip,
    }


def main() -> None:
    args = parse_args()
    os.environ["DASHSCOPE_ENABLE_THINKING"] = "1" if args.thinking == "on" else "0"
    print(f"[thinking] DASHSCOPE_ENABLE_THINKING={os.environ['DASHSCOPE_ENABLE_THINKING']}")

    input_path = args.input.resolve()
    output_path = _resolve_output_path(input_path, args.output.resolve() if args.output else None)

    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    print(f"[flip_gate] {args.flip_gate}")

    stats = process_file(
        input_path=input_path,
        output_path=output_path,
        retries=args.retries,
        retry_wait_seconds=args.retry_wait_seconds,
        flip_gate=args.flip_gate,
    )

    print(f"[完成] input={input_path}")
    print(f"[完成] output={output_path}")
    print(
        f"[统计] total={stats['total']} pred_N={stats['pred_n_total']} "
        f"triggered_v3={stats['triggered_v3']} changed={stats['changed']} "
        f"changed_to_Y={stats['changed_to_y']} "
        f"flip_correct={stats['changed_correct']} flip_wrong={stats['changed_wrong']} "
        f"blocked_flip={stats['blocked_flip']}"
    )
    print(
        f"[准确率] v2={stats['v2_accuracy']:.2%} post_v3={stats['post_v3_accuracy']:.2%} "
        f"delta={stats['delta_accuracy']:+.2%}"
    )


if __name__ == "__main__":
    main()
