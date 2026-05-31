"""

Task7 推理脚本（提示词 V2 + V6 auto-agent 外循环）。



在 ``infer_examples_compare_task7_v2`` 同款 V2 Jeopardy 模板与检索流程之上，增加 **auto-agent**

结构：



- **内循环**：单次调用 DashScope 完成求解（``annotate_nvidia_raw_text`` + ``count_answer``），含 API 失败重试。

- **外循环**：由**审核大模型**阅读 Category+Clue 与候选 ``<label>``，输出 ``<audit_verdict>`` 与

  ``<audit_rationale>``；**不得以 gold 作为停机条件**。若审核为 reject，把**候选答案 + 审核依据**

  一并写入下一轮 **ReAct** 纠错尾缀，再推理。

- 数据集 gold 仍用于离线 ``is_match`` 指标，**不参与**外循环决策。

- 外循环默认 **3 轮**（首轮 + 最多 2 次纠错）；审核 accept 则提前结束。



输出文件名带 ``v6``，便于与 v2 对照。

默认在终端**完整打印**每条样本的外循环过程：轮次说明、追加的 ReAct 尾缀、

模型原始全文（含 Observation/Thought/Act 等）、审核模型输出与 gold 对照（仅日志/指标）；可用 ``--quiet`` 关闭。

写入 JSONL 时默认**省略**各轮 ``raw_model_output`` / ``raw_judge_output``（可用 ``--jsonl_include_raw`` 打开）。

"""



from __future__ import annotations



import argparse

from concurrent.futures import ThreadPoolExecutor

import json

import os

import re

import threading

import time

import unicodedata

from pathlib import Path



from tqdm import tqdm



from method_hyb import annotate_nvidia_raw_text, build_prompt, count_answer, select_examples_hybrid

from method_hyb_prompts import _task_prompt_shell, register_task_prompt



REPO_ROOT = Path(__file__).resolve().parent.parent

TASK7_FILE = "openseek-7_jeopardy_answer_generation_all.json"

TASK7_CANONICAL_DESCRIPTION = (

    "You will be given a trivia clue, and the category it belongs to. "

    "You should answer with the best answer that belongs in the category "

    "and is described by the clue. For simplicity, answers should be in all lower cased letters."

)



_OUTPUT_JSONL_NAME = "openseek-7-examples-compare-task7opt-v6.jsonl"

_SUMMARY_JSON_NAME = "summary_task7opt_v6.json"



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

)





def _prompt_openseek_7_task7_v2(task_description: str, text2annotate: str) -> str:

    return _task_prompt_shell(

        _OPENSEEK_7_JEOPARDY_V2,

        task_description,

        text2annotate,

        annotation_guidelines=_OPENSEEK_7_ANNOTATION_GUIDELINES_V2,

    )





_REACT_CORRECTION_TAIL = (

    "\n\n"

    "### Auto-agent outer-loop correction (ReAct-style revision)\n"

    "An **auditor model** reviewed your previous Jeopardy candidate (without seeing any hidden benchmark gold). "

    "Use the **auditor rationale** below as the primary external signal when revising.\n\n"

    "**Auditor verdict:** {verdict}\n\n"

    "**Auditor rationale (verbatim; fold into your reasoning):**\n{rationale}\n\n"

    "**Previous candidate answer (verbatim):** {prev}\n\n"

    "Use a short **ReAct-style** scratch path in plain text, then output **one** new Jeopardy answer:\n"

    "1. **Observation**: What does the **Category** require as answer type, and what entity/fact does the **Clue** "

    "emphasize?\n"

    "2. **Thought**: How does the **auditor rationale** challenge `{prev}`? What concrete fix satisfies Category+Clue?\n"

    "3. **Act**: Write the revised response **only** inside a single `<label>...</label>` pair (all lower case per "

    "task rules).\n\n"

    "Prefer a **different** specific canonical entity when the auditor + clue warrant it; repeat `{prev}` only if "

    "you conclude it is truly correct after reconciling the auditor feedback with the clue.\n"

)





_TASK7_LLM_JUDGE_PROMPT = (

    "### Role\n"

    "You are an impartial **auditor** for Jeopardy!-style trivia items (Category + Clue → short lowercase answer).\n\n"

    "### Official task definition\n"

    "{task_definition}\n\n"

    "### Instance (Category + Clue only)\n"

    "{instance_input}\n\n"

    "### Candidate answer (from another solver; Jeopardy-style, lowercase)\n"

    "{candidate}\n\n"

    "### Instructions\n"

    "1. Decide whether the candidate satisfies **both** the Category (required answer type) and the Clue "

    "(correct specific entity / phrase / repaired wording).\n"

    "2. **Reject** vague references, wrong contrast branch, category mismatch, or copying irrelevant clue fragments as "

    "the answer when the task expects a canonical entity.\n"

    "3. You **must not** claim you compared against any hidden gold answer; judge only from Category+Clue + general "

    "knowledge.\n\n"

    "### Required machine-readable trailer (tag names exact)\n"

    "After any brief scratch reasoning you like, output exactly one verdict line and one rationale block:\n\n"

    "<audit_verdict>accept</audit_verdict>\n"

    "**OR**\n"

    "<audit_verdict>reject</audit_verdict>\n\n"

    "<audit_rationale>\n"

    "One concise paragraph the solver must read on the next attempt: what is wrong and what to change.\n"

    "</audit_rationale>\n\n"

    "Use **accept** only if no substantive revision is needed.\n"

)





_register_done = False





def ensure_task7_v2_prompt_registered() -> None:

    """与 v2 共用同一套 Task7 模板注册（幂等）。"""

    global _register_done

    if not _register_done:

        register_task_prompt(7, _prompt_openseek_7_task7_v2)

        _register_done = True





def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(

        description=(
            "Task7 V6：V2 提示词 + ICL + auto-agent；"
            "外循环由大模型审核（非 gold），审核依据注入下一轮 ReAct，默认 3 轮。"
        ),

    )

    parser.add_argument("--examples_limit", type=int, default=0, help="最多推理多少条；<=0 表示全部。")

    parser.add_argument("--output_dir", type=str, default="examples", help="结果输出目录。")

    parser.add_argument("--retries", type=int, default=3, help="单次 annotate 失败时的重试次数。")

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

        "--outer_rounds",

        type=int,

        default=3,

        help="外循环最多轮数（含首轮求解）；每轮后由审核模型判定是否继续；默认 3（首轮 + 最多 2 次 ReAct）。",

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

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="关闭详细过程日志（外循环 / ReAct / 原始输出等）；默认会打印完整推理过程。",
    )

    parser.add_argument(
        "--log_raw_max_chars",
        type=int,
        default=24_000,
        help="每条「外循环轮次」打印模型原始全文时的最大字符数；<=0 表示不截断。默认 24000。",
    )

    parser.add_argument(
        "--jsonl_include_raw",
        action="store_true",
        help=(
            "写入 JSONL 时在 outer_trace 各轮保留 raw_model_output 与 raw_judge_output（体积可能极大）；默认省略。"
        ),
    )

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





def _normalize_task7_match_text(text: str) -> str:

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





def _expand_expected_variants(expected: str) -> set[str]:

    base = _normalize_task7_match_text(expected)

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

        v = _normalize_task7_match_text(p)

        if v:

            variants.add(v)

    return variants





def _is_task7_match(expected: str, prediction: str) -> bool:

    pred_norm = _normalize_task7_match_text(prediction)

    if not pred_norm:

        return False

    expected_variants = _expand_expected_variants(expected)

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





def _safe_print(msg: str, log_lock: threading.Lock | None) -> None:

    if log_lock is not None:

        with log_lock:

            print(msg, flush=True)

    else:

        print(msg, flush=True)





def _clip_for_log(text: str | None, max_chars: int) -> str:

    if not text:

        return ""

    s = str(text)

    if max_chars <= 0 or len(s) <= max_chars:

        return s

    return (

        s[:max_chars]

        + f"\n... [日志截断：原始长度 {len(s)} 字符；可通过 --log_raw_max_chars <=0 关闭截断或调大上限]\n"

    )





def _infer_task7_prediction(

    input_prompt: str,

    retries: int,

    retry_wait_seconds: float,

    *,

    verbose_log: bool = False,

    log_lock: threading.Lock | None = None,

    log_prefix: str = "",

    log_raw_max_chars: int = 0,

) -> tuple[str, str | None]:

    """

    内循环：调用模型并解析 ``<label>``；可选打印每次 API 尝试与完整原始回复（含 ReAct 过程）。

    返回 (归一化预测, 最后一次原始助手文本)。

    """

    prediction = ""

    last_raw: str | None = None

    for attempt in range(1, retries + 1):

        try:

            last_raw = annotate_nvidia_raw_text(input_prompt)

            if last_raw is None:

                if verbose_log:

                    _safe_print(

                        f"{log_prefix}[内循环 API] attempt={attempt}/{retries} 返回 None（解析前）",

                        log_lock,

                    )

                raise RuntimeError("annotate_nvidia_raw_text returned None")

            parsed = count_answer(last_raw)

            prediction = "" if parsed is None else str(parsed).strip()

            prediction = _normalize_task7_answer(prediction)

            if verbose_log:

                _safe_print(

                    f"{log_prefix}[内循环] attempt={attempt}/{retries} 解析后 prediction={prediction!r}",

                    log_lock,

                )

                _safe_print(

                    f"{log_prefix}[模型原始输出全文]\n{_clip_for_log(last_raw, log_raw_max_chars)}",

                    log_lock,

                )

            return prediction, last_raw

        except Exception as e:  # noqa: BLE001

            if verbose_log:

                _safe_print(

                    f"{log_prefix}[内循环异常] attempt={attempt}/{retries} error={e!r}",

                    log_lock,

                )

            if attempt >= retries:

                msg = f"[推理失败] attempt={attempt}/{retries} error={e}"

                if verbose_log:

                    _safe_print(f"{log_prefix}{msg}", log_lock)

                else:

                    print(msg)

            else:

                msg = f"[重试] attempt={attempt}/{retries} error={e}"

                if verbose_log:

                    _safe_print(f"{log_prefix}{msg}", log_lock)

                else:

                    print(msg)

                time.sleep(retry_wait_seconds)

    return prediction, last_raw





def _parse_judge_audit_tags(raw: str | None) -> tuple[bool, str, str]:

    """

    从审核模型输出解析 (是否接受, verdict 词, rationale)。

    无合法 verdict 时保守视为 reject。

    """

    text = (raw or "").strip()

    vm = re.search(

        r"<audit_verdict>\s*(accept|reject)\s*</audit_verdict>",

        text,

        flags=re.IGNORECASE | re.DOTALL,

    )

    verdict_word = vm.group(1).lower() if vm else "reject"

    accepted = verdict_word == "accept"

    rm = re.search(

        r"<audit_rationale>\s*(.+?)\s*</audit_rationale>",

        text,

        flags=re.IGNORECASE | re.DOTALL,

    )

    rationale = rm.group(1).strip() if rm else ""

    if not rationale and text:

        rationale = text[:2000]

    return accepted, verdict_word, rationale





def _llm_judge_task7_answer(

    task_definition: str,

    instance_input: str,

    candidate_answer: str,

    retries: int,

    retry_wait_seconds: float,

    *,

    verbose_log: bool,

    log_lock: threading.Lock | None,

    log_prefix: str,

    log_raw_max_chars: int,

) -> tuple[bool, str, str, str | None]:

    """

    调用同一 DashScope 接口作为审核模型。

    返回 (accepted, verdict_word, rationale, raw_output)。

    """

    candidate_display = candidate_answer if candidate_answer.strip() else "(empty)"

    user_prompt = _TASK7_LLM_JUDGE_PROMPT.format(

        task_definition=task_definition.strip(),

        instance_input=instance_input.strip(),

        candidate=candidate_display,

    )

    last_raw: str | None = None

    for attempt in range(1, retries + 1):

        try:

            last_raw = annotate_nvidia_raw_text(user_prompt)

            if last_raw is None:

                raise RuntimeError("annotate_nvidia_raw_text returned None (judge)")

            accepted, verdict_word, rationale = _parse_judge_audit_tags(last_raw)

            if verbose_log:

                _safe_print(

                    f"{log_prefix}[审核模型] attempt={attempt}/{retries} "

                    f"parsed verdict={verdict_word!r} accepted={accepted}",

                    log_lock,

                )

                _safe_print(

                    f"{log_prefix}[审核模型原始输出全文]\n{_clip_for_log(last_raw, log_raw_max_chars)}",

                    log_lock,

                )

                _safe_print(

                    f"{log_prefix}[审核 rationale 摘要]\n{_clip_for_log(rationale, log_raw_max_chars)}",

                    log_lock,

                )

            return accepted, verdict_word, rationale, last_raw

        except Exception as e:  # noqa: BLE001

            if verbose_log:

                _safe_print(

                    f"{log_prefix}[审核异常] attempt={attempt}/{retries} error={e!r}",

                    log_lock,

                )

            if attempt >= retries:

                msg = f"[审核推理失败] attempt={attempt}/{retries} error={e}"

                if verbose_log:

                    _safe_print(f"{log_prefix}{msg}", log_lock)

                else:

                    print(msg)

            else:

                msg = f"[审核重试] attempt={attempt}/{retries} error={e}"

                if verbose_log:

                    _safe_print(f"{log_prefix}{msg}", log_lock)

                else:

                    print(msg)

                time.sleep(retry_wait_seconds)

    return False, "reject", (last_raw or "")[:2000] or "judge API failed after retries", last_raw





def _append_react_correction(

    base_prompt: str,

    previous_prediction: str,

    *,

    judge_verdict: str,

    judge_rationale: str,

) -> str:

    prev_display = previous_prediction if previous_prediction.strip() else "(empty)"

    verdict_display = (judge_verdict or "reject").strip().upper()

    rationale_display = (judge_rationale or "").strip() or "(none)"

    return base_prompt.rstrip() + _REACT_CORRECTION_TAIL.format(

        prev=prev_display,

        verdict=verdict_display,

        rationale=rationale_display,

    )





def _infer_task7_auto_agent_outer_loop(

    base_input_prompt: str,

    expected: str,

    task_definition: str,

    instance_input_text: str,

    retries: int,

    retry_wait_seconds: float,

    max_outer_rounds: int,

    *,

    example_id: str = "",

    verbose_log: bool = True,

    log_lock: threading.Lock | None = None,

    log_raw_max_chars: int = 24_000,

    input_preview_chars: int = 240,

    retain_raw_in_trace: bool = True,

) -> tuple[str, bool, list[dict]]:

    """

    外循环：**审核大模型** accept 则停止；reject 则将审核 verdict + rationale + 候选写入 ReAct 尾缀再推理。

    ``expected`` / ``gold_is_match`` 仅用于离线评测记录，**不参与**停机判断。

    """

    trace: list[dict] = []

    prompt = base_input_prompt

    final_prediction = ""

    rounds = max(1, int(max_outer_rounds))

    prefix = f"[v6 example_id={example_id}] "



    if verbose_log:

        _safe_print(

            f"{prefix}{'=' * 20} 开始 auto-agent 外循环 (最多 {rounds} 轮) {'=' * 20}",

            log_lock,

        )

        _safe_print(

            f"{prefix}离线 gold（仅用于指标对照，外循环不使用） expected={expected!r}",

            log_lock,

        )

        inst_pv = _normalize_text(str(instance_input_text))[:input_preview_chars]

        _safe_print(

            f"{prefix}实例 Category+Clue 预览（审核模型与会话输入，前 {input_preview_chars} 字符）:\n{inst_pv}",

            log_lock,

        )

        pv = _normalize_text(str(base_input_prompt))[:input_preview_chars]

        _safe_print(f"{prefix}完整求解 prompt 预览（前 {input_preview_chars} 字符）:\n{pv}", log_lock)



    for outer in range(1, rounds + 1):

        mode = "首轮（基础 ICL prompt）" if outer == 1 else "ReAct 纠错轮（已追加审核反馈 + outer-loop correction 尾缀）"

        if verbose_log:

            _safe_print(

                f"{prefix}--- 外循环第 {outer}/{rounds} 轮 | {mode} ---",

                log_lock,

            )

            if outer > 1:

                delta = prompt[len(base_input_prompt) :]

                _safe_print(

                    f"{prefix}[相对基础 prompt 追加的 ReAct 尾缀]\n{_clip_for_log(delta, log_raw_max_chars)}",

                    log_lock,

                )



        inner_prefix = f"{prefix}[outer={outer}] "

        prediction, raw_text = _infer_task7_prediction(

            prompt,

            retries,

            retry_wait_seconds,

            verbose_log=verbose_log,

            log_lock=log_lock,

            log_prefix=inner_prefix,

            log_raw_max_chars=log_raw_max_chars,

        )



        judge_prefix = f"{inner_prefix}[judge] "

        accepted, verdict_word, rationale, judge_raw = _llm_judge_task7_answer(

            task_definition,

            instance_input_text,

            prediction,

            retries,

            retry_wait_seconds,

            verbose_log=verbose_log,

            log_lock=log_lock,

            log_prefix=judge_prefix,

            log_raw_max_chars=log_raw_max_chars,

        )



        gold_match = _is_task7_match(expected, prediction)

        step: dict = {

            "outer_round": outer,

            "prediction": prediction,

            "gold_is_match": gold_match,

            "llm_judge_accepted": accepted,

            "llm_judge_verdict": verdict_word,

            "llm_judge_rationale": rationale,

            "used_react_tail": outer > 1,

        }

        if retain_raw_in_trace:

            step["raw_model_output"] = raw_text

            step["raw_judge_output"] = judge_raw

        trace.append(step)



        if verbose_log:

            _safe_print(

                f"{inner_prefix}离线 gold_is_match={gold_match} | 审核 accepted={accepted} | "

                f"verdict={verdict_word!r} | prediction={prediction!r}",

                log_lock,

            )



        final_prediction = prediction

        if accepted:

            if verbose_log:

                _safe_print(

                    f"{prefix}外循环提前结束：第 {outer} 轮 **审核模型 accept**。", log_lock

                )

            break



        if outer >= rounds:

            if verbose_log:

                _safe_print(

                    f"{prefix}已达外循环上限；末轮审核仍为 reject，保留 prediction={final_prediction!r}。", log_lock

                )

            break



        prompt = _append_react_correction(

            base_input_prompt,

            prediction,

            judge_verdict=verdict_word,

            judge_rationale=rationale,

        )

        if verbose_log:

            _safe_print(

                f"{prefix}审核 reject → 已将 verdict+rationale+候选注入下一轮 ReAct 尾缀。", log_lock

            )



    final_ok = _is_task7_match(expected, final_prediction)

    last_accept = bool(trace[-1]["llm_judge_accepted"]) if trace else False

    if verbose_log:

        _safe_print(

            f"{prefix}{'=' * 20} 结束 example_id={example_id} | 离线 gold is_match={final_ok} | "

            f"末轮审核 accepted={last_accept} | prediction={final_prediction!r} {'=' * 20}",

            log_lock,

        )



    return final_prediction, final_ok, trace





def _outer_trace_for_jsonl(outer_trace: list[dict], *, include_raw: bool) -> list[dict]:

    """控制台仍打印全文；JSONL 默认去掉原始模型字段以防体积爆炸。"""

    if include_raw:

        return outer_trace

    drop_keys = frozenset({"raw_model_output", "raw_judge_output"})

    sanitized: list[dict] = []

    for step in outer_trace:

        sanitized.append({k: v for k, v in step.items() if k not in drop_keys})

    return sanitized





def run_task7_v6(

    output_dir: Path,

    examples_limit: int = 0,

    retries: int = 3,

    retry_wait_seconds: float = 2.0,

    resume: bool = False,

    task7_shot_k: int = 8,

    task7_retrieval_pool_size: int = 200,

    print_empty_prediction: str = "on",

    batch_size: int = 8,

    outer_rounds: int = 3,

    verbose_process_log: bool = True,

    log_raw_max_chars: int = 24_000,

    jsonl_include_raw: bool = False,

) -> dict:

    ensure_task7_v2_prompt_registered()



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

        print(f"[断点续跑 v6] task=7, 已完成 {len(done_ids)} 条，继续剩余样本")



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



    outer_rounds_eff = max(1, outer_rounds)

    log_lock = threading.Lock() if verbose_process_log else None



    with output_file.open(mode, encoding="utf-8") as wf:

        for start in tqdm(

            range(0, len(pending_examples), max(1, batch_size)),

            desc=f"Task7 V6 Auto-Agent: {task_name}",

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



            def _run_one(item: dict) -> tuple[str, bool, list[dict]]:

                return _infer_task7_auto_agent_outer_loop(

                    item["input_prompt"],

                    item["expected"],

                    task_description,

                    item["input_text"],

                    retries=retries,

                    retry_wait_seconds=retry_wait_seconds,

                    max_outer_rounds=outer_rounds_eff,

                    example_id=str(item["example_id"]),

                    verbose_log=verbose_process_log,

                    log_lock=log_lock,

                    log_raw_max_chars=log_raw_max_chars,

                    retain_raw_in_trace=verbose_process_log or jsonl_include_raw,

                )



            with ThreadPoolExecutor(max_workers=max(1, batch_size)) as executor:

                results = list(executor.map(_run_one, prepared))



            for item, (prediction, is_match, outer_trace) in zip(prepared, results, strict=True):

                example_id = item["example_id"]

                input_text = item["input_text"]

                expected = item["expected"]

                if print_empty_prediction == "on" and not prediction:

                    print(

                        f"[空预测] example_id={example_id} expected={expected} "

                        f"input_preview={input_text[:160].replace(chr(10), ' ')}"

                    )

                final_llm_ok = bool(outer_trace[-1].get("llm_judge_accepted")) if outer_trace else False

                row = {

                    "example_id": example_id,

                    "input": input_text,

                    "expected_output": expected,

                    "model_output": prediction,

                    "is_match": is_match,

                    "final_llm_judge_accepted": final_llm_ok,

                    "outer_rounds_limit": outer_rounds_eff,

                    "outer_rounds_used": len(outer_trace),

                    "outer_trace": _outer_trace_for_jsonl(outer_trace, include_raw=jsonl_include_raw),

                }

                wf.write(json.dumps(row, ensure_ascii=False) + "\n")

                wf.flush()

                done_ids.add(example_id)



    total, match_count, accuracy = _compute_metrics_from_jsonl(output_file)

    print(

        f"[保存完成 v6] task=7, total={total}, matched={match_count}, "

        f"accuracy={accuracy:.2%}, file={output_file}"

    )

    return {

        "task_id": 7,

        "task_name": task_name,

        "prompt_version": "task7_v6_auto_agent",

        "outer_rounds": outer_rounds_eff,

        "total": total,

        "matched": match_count,

        "accuracy": accuracy,

        "file": str(output_file),

    }





def main() -> None:

    args = parse_args()

    ensure_task7_v2_prompt_registered()



    os.environ["DASHSCOPE_ENABLE_THINKING"] = "1" if args.thinking == "on" else "0"

    os.environ["ANNOTATE_LOG_EVERY_RESPONSE"] = "1" if args.print_model_output == "on" else "0"

    os.environ.setdefault("ANNOTATE_GLOBAL_API_LOCK", "0")

    os.environ.setdefault("ANNOTATE_FALLBACK_PLAIN", "1")

    os.environ.setdefault("ANNOTATE_MAX_PLAIN_CHARS", "120")

    print("[prompt] Task7 Jeopardy 模板已通过 register_task_prompt 覆盖为 v2（v6 共用）")

    print(

        f"[auto-agent v6] 外循环最多 {max(1, args.outer_rounds)} 轮；"

        "每轮后由**审核大模型**判定 accept/reject；reject 时将审核依据写入下一轮 ReAct。"

    )

    if args.quiet:

        print("[日志] 已 --quiet：不打印每轮求解 / 审核 / ReAct 尾缀 / 原始输出过程")

    else:

        print(

            f"[日志] 详细过程已开启（求解模型、审核模型、ReAct 增量尾缀、原始输出）；"

            f"单段原始输出打印上限 log_raw_max_chars={args.log_raw_max_chars}（<=0 不截断）"

        )



    output_dir = _resolve_output_dir(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[输出目录] {output_dir} -> {_OUTPUT_JSONL_NAME}, {_SUMMARY_JSON_NAME}")



    summary_item = run_task7_v6(

        output_dir=output_dir,

        examples_limit=args.examples_limit,

        retries=args.retries,

        retry_wait_seconds=args.retry_wait_seconds,

        resume=args.resume,

        task7_shot_k=args.task7_shot_k,

        task7_retrieval_pool_size=args.task7_retrieval_pool_size,

        print_empty_prediction=args.print_empty_prediction,

        batch_size=max(1, args.batch_size),

        outer_rounds=max(1, args.outer_rounds),

        verbose_process_log=not args.quiet,

        log_raw_max_chars=args.log_raw_max_chars,

        jsonl_include_raw=args.jsonl_include_raw,

    )

    summary_file = output_dir / _SUMMARY_JSON_NAME

    with summary_file.open("w", encoding="utf-8") as f:

        json.dump([summary_item], f, ensure_ascii=False, indent=2)

    print(f"[汇总完成] {summary_file}")





if __name__ == "__main__":

    main()


