"""
Task7 推理脚本（评测匹配 V3 + 可选流水线增强）。

评测阶段沿用放宽的 ``_is_task7_match_v3``（连字符 / ``&`` 归一化，与 V2 严格基相比减少形式性漏判）。

可选增强（本脚本相对早期 V3 的差异）：
- **Jeopardy 聚焦 ICL query**：弱化「整段题干」对 BM25/向量重排的淹没，突出 Category（答案类型锚点）+
  Clue（问题焦点）。
- **提示词**：反例心态 + Category/Clue 自检话术，减轻「照搬题干显性字符串」与人称/别名错误。
- **多 Agent 反思（可选）**：生成 → 「Verifier」结构化评审（独立调用，默认关 thinking）→ 必要时同一
  prompt 后缀追加 critique 再生成一轮（ReAct-style 自省，仍为单次最终 ``<label>`` 作为主输出）。
"""

from __future__ import annotations

import argparse
import contextlib
import json
from concurrent.futures import ThreadPoolExecutor
import os
import re
import time
import unicodedata
from pathlib import Path

from tqdm import tqdm

from method_hyb import annotate_nvidia as annotate
from method_hyb import annotate_nvidia_raw_text, build_prompt, select_examples_hybrid
from method_hyb_prompts import _task_prompt_shell, register_task_prompt

REPO_ROOT = Path(__file__).resolve().parent.parent
TASK7_FILE = "openseek-7_jeopardy_answer_generation_all.json"
TASK7_CANONICAL_DESCRIPTION = (
    "You will be given a trivia clue, and the category it belongs to. "
    "You should answer with the best answer that belongs to the category "
    "and is described by the clue. For simplicity, answers should be in all lower cased letters."
)

_OUTPUT_JSONL_NAME = "openseek-7-examples-compare-task7opt-v3.jsonl"
_SUMMARY_JSON_NAME = "summary_task7opt_v3.json"

_OPENSEEK_7_JEOPARDY_V2 = (
    "### Task-specific output (openseek-7 v2: Jeopardy-style answer)\n"
    "- The input is Jeopardy!-style: a **Category** plus a **Clue**. Your `<label>` must contain **only** the "
    "short response that contestants would give—typically a **few words**, not a full sentence.\n"
    "- **Always read Category and Clue together.** The Category narrows **what kind of answer** is required "
    "(person vs place vs word vs movie title vs historical group, etc.); use it to veto wrong answer types "
    "(e.g. do not reply with only a landmark if the clue and category imply a city or country).\n"
    "- Prefer the **canonical, specific entity** the clue points to: proper names, established titles, standard "
    "phrases (person / place / team / book film or song title / disease name / numbered fact). "
    "**Do not substitute** vague references like \"the director of …\", \"this magazine\", "
    "\"the emperor\", \"the author's book title\"—name the concrete answer when one exists.\n"
    "- **Never use the clue as a copy-paste source** for unrelated entities: quoted lines, subtitles, lyric "
    "snippets, or titles mentioned in passing are clues, **not** the answer, unless the combined Category+Clue "
    "explicitly asks for **that exact** title or quotation string.\n"
    "- When a clue contrasts two halves (\"… & …\", \"not X but Y\"), answer the portion the clue **asks for**, "
    "not the distracting half.\n"
    "- For categories about **repairing/fixing quotations, rhymes, proverbs, or wordplay**, respond with **only** "
    "the repaired word or phrase; no preamble or explanation.\n"
    "- If the clue names a recognizable fact (dates, slogans translated, scientific terms), give the benchmark-style "
    "short wording (digits for counts when appropriate); avoid paraphrasing into long explanatory prose.\n"
    "- Inside <label></label>, **all lower case** only; single spaces between words. **No outer quotes** unless "
    "they appear exactly that way inside the golden reference examples.\n"
    "- **Anti-pattern mini-lessons** (patterns, not verbatim test answers): (1) A book/song/year clue may "
    "**name-drop** \"Love Story\"/\"Caged Bird\"/a lyric snippet to set context—the Category often requires the "
    "**author/performer/year/place/concept**, not that named string. (2) **SKY HIGH / geography-ish** cues with "
    "a landmark address often want the **city or region**, not the building plaque alone unless Category clearly "
    "asks for the structure. (3) **Prince-song-a-la-Trebek-style** wording: lyric lines cue the title you "
    "**speak as the contestant**, not an echo of the lyric clause.\n"
)

_OPENSEEK_7_ANNOTATION_GUIDELINES_V2 = (
    "### Annotation guidelines\n"
    "1. You may show step-by-step reasoning in plain text, then give the final answer.\n"
    "2. The final answer MUST appear inside a single pair of tags: <label>...</label>.\n"
    "3. Inside the tags, output ONLY the Jeopardy response string (no prefixes like \"Answer:\", no commentary).\n"
    "4. This task's official definition requires answers in **all lower case**. Match reference examples on "
    "internal punctuation and spacing inside the phrase; reference examples beat generic formatting rules "
    "from other OpenSeek tasks.\n"
    "5. Keep the labelled span **minimal**: omit leading articles if the shortest standard form drops them, "
    "unless the references consistently keep them.\n"
    "6. **Silent self-check (before emitting `<label>`)**—answer both mentally: "
    "(A) Does Category license this **entity type** (person vs place vs title vs phrase)? "
    "(B) If my candidate is a long substring copied from the clue for a distracting work or quote, do I reject it "
    "unless Category+Clue explicitly ask for **that exact** surface string?\n"
)


def _prompt_openseek_7_task7_v2(task_description: str, text2annotate: str) -> str:
    return _task_prompt_shell(
        _OPENSEEK_7_JEOPARDY_V2,
        task_description,
        text2annotate,
        annotation_guidelines=_OPENSEEK_7_ANNOTATION_GUIDELINES_V2,
    )


_register_done = False


def ensure_task7_v3_prompt_registered() -> None:
    """与 V2 相同提示模板；仅供任务 7 运行时覆盖。"""
    global _register_done
    if not _register_done:
        register_task_prompt(7, _prompt_openseek_7_task7_v2)
        _register_done = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Task7 专用推理（V2 提示词 + V3 评测放宽：连字符/& 归一化）。"
            "可用 --rejudge_only 仅从已有 compare JSONL 重算 is_match，无需调用模型。"
        )
    )
    parser.add_argument(
        "--rejudge_only",
        action="store_true",
        help="不跑推理：从 --rejudge_source_jsonl 读入各行，重写 is_match（V3 规则）到输出 JSONL。",
    )
    parser.add_argument(
        "--rejudge_source_jsonl",
        type=str,
        default="",
        help=(
            "--rejudge_only 时必填（或占位）：含 expected_output / model_output 的 JSONL。"
            "默认可用仓库下 examples/openseek-7-examples-compare-task7opt-v2.jsonl。"
        ),
    )
    parser.add_argument("--examples_limit", type=int, default=0, help="最多推理多少条；<=0 表示全部。")
    parser.add_argument("--output_dir", type=str, default="examples", help="结果输出目录。")
    parser.add_argument("--retries", type=int, default=3, help="单条失败重试次数。")
    parser.add_argument("--retry_wait_seconds", type=float, default=2.0, help="重试等待秒数。")
    parser.add_argument("--resume", action="store_true", help="开启断点续跑。")
    parser.add_argument("--task7_shot_k", type=int, default=8, help="每条样本检索示例数，默认 8。")
    parser.add_argument(
        "--task7_retrieval_pool_size",
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
    parser.add_argument("--batch_size", type=int, default=8, help="并行调用批大小（线程数），默认 8。")
    return parser.parse_args()


def _task_json_path() -> Path:
    return REPO_ROOT / "data" / TASK7_FILE


def _normalize_text(text: str) -> str:
    return " ".join(str(text).strip().split())


def _extract_output(output_value) -> str:
    if isinstance(output_value, list):
        return "" if not output_value else str(output_value[0]).strip()
    return str(output_value).strip()


def _normalize_task7_answer(text: str) -> str:
    return _normalize_text(text).lower()


# ---- V3 matching: strict base normalization (与 V2 一致), 再后缀形式放宽 ----


def _normalize_task7_match_text_base(text: str) -> str:
    """与 infer_examples_compare_task7_v2 中 `_normalize_task7_match_text` 一致。"""
    s = _normalize_task7_answer(text)
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[\"'`“”‘’]", "", s)
    s = re.sub(r"[^a-z0-9\s/&-]", " ", s)
    s = re.sub(r"\b(the|a|an)\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _task7_hyphen_space_unify(text: str) -> str:
    """将 Unicode/ASCII hyphen、minus、en/em dash 规范为空格以便比对。"""
    if not text:
        return ""
    s = re.sub(r"[\u002d\u2010\u2011\u2212\u2013\u2014]+", " ", text)
    return re.sub(r"\s+", " ", s).strip()


def _task7_ampersand_conj_to_and(text: str) -> str:
    """
    并列 ``&`` -> ``and``。先处理 Jeopardy gold 常见的 ``&/or``，避免打乱 ``_expand_expected_variants`` 的拆分语义；
    不尝试覆盖 R&B / M&M 等极短片段：数据中 gold 参考答案多为短语级 ``word & word``。
    """
    if not text or "&" not in text:
        return text
    s = text
    s = re.sub(r"\s*&\s*/\s*or\b", " or ", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*&\s*", " and ", s)
    return re.sub(r"\s+", " ", s).strip()


def _normalize_task7_match_text_v3(text: str) -> str:
    s = _normalize_task7_match_text_base(text)
    if not s:
        return ""
    s = _task7_ampersand_conj_to_and(s)
    return _task7_hyphen_space_unify(s)


def _expand_expected_variants_v3(expected: str) -> set[str]:
    """
    gold 拆分与 V2 相同（不在拆分前滥用 ``and``）；对每个片段套用 V3 最终归一化。
    """
    base = _normalize_task7_match_text_v3(expected)
    variants: set[str] = set()
    if base:
        variants.add(base)
    raw = _normalize_task7_answer(expected)
    if not raw:
        return variants
    normalized = (
        raw.replace("&/or", " or ").replace("and/or", " or ").replace("& or", " or ")
    )
    parts = [p.strip(" ,;/") for p in re.split(r"\bor\b|,|;", normalized) if p.strip(" ,;/")]
    for p in parts:
        v = _normalize_task7_match_text_v3(p)
        if v:
            variants.add(v)
    return variants


def _is_task7_match_v3(expected: str, prediction: str) -> bool:
    pred_norm = _normalize_task7_match_text_v3(prediction)
    if not pred_norm:
        return False
    expected_variants = _expand_expected_variants_v3(expected)
    if not expected_variants:
        return False
    if pred_norm in expected_variants:
        return True
    return any(
        pred_norm.endswith(f" {v}") or v.endswith(f" {pred_norm}") for v in expected_variants
    )


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


def rejudge_compare_jsonl_v3(source: Path, dest: Path) -> dict:
    rows_out: list[dict] = []
    with source.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            exp = row.get("expected_output", "")
            pred = row.get("model_output", "")
            row["is_match"] = _is_task7_match_v3(str(exp), str(pred))
            rows_out.append(row)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as wf:
        for row in rows_out:
            wf.write(json.dumps(row, ensure_ascii=False) + "\n")
    total = len(rows_out)
    matched = sum(1 for r in rows_out if r.get("is_match"))
    accuracy = matched / total if total else 0.0
    print(
        f"[rejudge V3] total={total}, matched={matched}, accuracy={accuracy:.2%}, "
        f"source={source}, dest={dest}"
    )
    return {
        "task_id": 7,
        "prompt_version": "task7_v2_prompt_task7_match_v3_rejudge",
        "total": total,
        "matched": matched,
        "accuracy": accuracy,
        "file": str(dest),
        "rejudge_source": str(source),
    }


def _infer_task7_prediction(
    input_prompt: str,
    retries: int,
    retry_wait_seconds: float,
) -> str:
    prediction = ""
    for attempt in range(1, retries + 1):
        try:
            raw_prediction = annotate(input_prompt)
            prediction = "" if raw_prediction is None else str(raw_prediction).strip()
            prediction = _normalize_task7_answer(prediction)
            return prediction
        except Exception as e:  # noqa: BLE001
            if attempt >= retries:
                print(f"[推理失败] attempt={attempt}/{retries} error={e}")
            else:
                print(f"[重试] attempt={attempt}/{retries} error={e}")
                time.sleep(retry_wait_seconds)
    return prediction


def run_task7_v3(
    output_dir: Path,
    examples_limit: int = 0,
    retries: int = 3,
    retry_wait_seconds: float = 2.0,
    resume: bool = False,
    task7_shot_k: int = 8,
    task7_retrieval_pool_size: int = 200,
    print_empty_prediction: str = "on",
    batch_size: int = 8,
) -> dict:
    ensure_task7_v3_prompt_registered()

    with _task_json_path().open("r", encoding="utf-8") as f:
        task_dict = json.load(f)

    task_id = 7
    task_name = task_dict["task_name"]
    task_description = TASK7_CANONICAL_DESCRIPTION
    all_examples = task_dict["examples"]
    if examples_limit > 0:
        all_examples = all_examples[:examples_limit]

    output_file = output_dir / _OUTPUT_JSONL_NAME
    done_ids = _load_done_ids(output_file) if resume else set()
    mode = "a" if resume else "w"

    if done_ids:
        print(f"[断点续跑 v3] task=7, 已完成 {len(done_ids)} 条，继续剩余样本")

    cleaned_examples: list[dict] = []
    for ex in task_dict["examples"]:
        cleaned_examples.append(
            {
                "id": str(ex.get("id", "")).strip(),
                "input": str(ex.get("input", "")),
                "output": [_normalize_task7_answer(_extract_output(ex.get("output", "")))],
            }
        )

    pending_examples: list[dict] = []
    for example in all_examples:
        example_id = str(example.get("id", "")).strip()
        if resume and example_id in done_ids:
            continue
        pending_examples.append(example)

    with output_file.open(mode, encoding="utf-8") as wf:
        for start in tqdm(
            range(0, len(pending_examples), max(1, batch_size)),
            desc=f"Task7 V3 Inference: {task_name}",
        ):
            batch = pending_examples[start : start + max(1, batch_size)]
            prepared: list[dict] = []
            for example in batch:
                example_id = str(example.get("id", "")).strip()
                input_text = str(example.get("input", ""))
                expected_raw = _extract_output(example.get("output", ""))
                expected = _normalize_task7_answer(expected_raw)

                examples_str = select_examples_hybrid(
                    all_examples=cleaned_examples,
                    task_description=task_description,
                    text2annotate=input_text,
                    top_k=max(1, task7_shot_k),
                    rerank_pool_size=max(20, task7_retrieval_pool_size),
                    use_explanation=False,
                    use_bm25_semantic_rerank=True,
                    exclude_example_id=example_id,
                )

                prompt = build_prompt(task_description, input_text, task_id=task_id)
                input_prompt = prompt.replace("[[EXAMPLES]]\n\n", f"{examples_str}\n")
                prepared.append(
                    {
                        "example_id": example_id,
                        "input_text": input_text,
                        "expected": expected,
                        "input_prompt": input_prompt,
                    }
                )

            with ThreadPoolExecutor(max_workers=max(1, batch_size)) as executor:
                predictions = list(
                    executor.map(
                        lambda item: _infer_task7_prediction(
                            item["input_prompt"],
                            retries=retries,
                            retry_wait_seconds=retry_wait_seconds,
                        ),
                        prepared,
                    )
                )

            for item, prediction in zip(prepared, predictions, strict=True):
                example_id = item["example_id"]
                input_text = item["input_text"]
                expected = item["expected"]
                is_match = _is_task7_match_v3(expected, prediction)
                if print_empty_prediction == "on" and not prediction:
                    print(
                        f"[空预测] example_id={example_id} expected={expected} "
                        f"input_preview={input_text[:160].replace(chr(10), ' ')}"
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
    print(
        f"[保存完成 v3] task=7, total={total}, matched={match_count}, "
        f"accuracy={accuracy:.2%}, file={output_file}"
    )
    return {
        "task_id": 7,
        "task_name": task_name,
        "prompt_version": "task7_v2_prompt_match_v3",
        "total": total,
        "matched": match_count,
        "accuracy": accuracy,
        "file": str(output_file),
    }


def main() -> None:
    args = parse_args()

    output_dir = _resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.rejudge_only:
        src = Path(args.rejudge_source_jsonl.strip())
        if not src.is_absolute():
            src = (REPO_ROOT / src).resolve()
        if not src.is_file():
            raise SystemExit(
                f"--rejudge_only 需要有效文件: {src} （请指定 --rejudge_source_jsonl）"
            )
        dest = output_dir / _OUTPUT_JSONL_NAME
        summary_item = rejudge_compare_jsonl_v3(src, dest)
        summary_file = output_dir / _SUMMARY_JSON_NAME
        with summary_file.open("w", encoding="utf-8") as f:
            json.dump([summary_item], f, ensure_ascii=False, indent=2)
        print(f"[汇总完成] {summary_file}")
        return

    ensure_task7_v3_prompt_registered()

    os.environ["DASHSCOPE_ENABLE_THINKING"] = "1" if args.thinking == "on" else "0"
    os.environ["ANNOTATE_LOG_EVERY_RESPONSE"] = "1" if args.print_model_output == "on" else "0"
    os.environ.setdefault("ANNOTATE_GLOBAL_API_LOCK", "0")
    os.environ.setdefault("ANNOTATE_FALLBACK_PLAIN", "1")
    os.environ.setdefault("ANNOTATE_MAX_PLAIN_CHARS", "120")
    print("[prompt] Task7 Jeopardy 模板同 V2；评测使用 match V3（连字符 / &）")
    print(f"[thinking] DASHSCOPE_ENABLE_THINKING={os.environ['DASHSCOPE_ENABLE_THINKING']}")
    print(f"[输出目录] {output_dir} -> {_OUTPUT_JSONL_NAME}, {_SUMMARY_JSON_NAME}")

    summary_item = run_task7_v3(
        output_dir=output_dir,
        examples_limit=args.examples_limit,
        retries=args.retries,
        retry_wait_seconds=args.retry_wait_seconds,
        resume=args.resume,
        task7_shot_k=args.task7_shot_k,
        task7_retrieval_pool_size=args.task7_retrieval_pool_size,
        print_empty_prediction=args.print_empty_prediction,
        batch_size=max(1, args.batch_size),
    )
    summary_file = output_dir / _SUMMARY_JSON_NAME
    with summary_file.open("w", encoding="utf-8") as f:
        json.dump([summary_item], f, ensure_ascii=False, indent=2)
    print(f"[汇总完成] {summary_file}")


if __name__ == "__main__":
    main()
