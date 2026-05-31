"""
main1 task2 推理：**v5** — 在 ``infer_examples_main1_task2_spacy_v3.py`` 上的两项强化：

1. **工程兜底**：首轮走后照常调用 ``annotate_nvidia``；若解析结果为空，则在同一道题上追加简短 `<label>` 提醒并重试；
   仍为空时对 **助手全文**（``annotate_nvidia_raw_text``）再走一遍 ``count_answer``，最后用正则抽取首个非负整数。
2. **大写预处理（仅 spaCy）**：引用句中含 ASCII 大写字母时，**仅在对 ``en_core_web_sm`` 分词生成 POS 表时**使用 ``sentence.lower()``，
   题干 ``Your input`` 仍保留原始大小写，避免改变用户可见句子。

输出文件（默认 ``v5``）：``openseek-2-examples-main1-compare_spacy_v5_zeroshot.jsonl``。

**Prompt 变种**（``--prompt_variant``，用于与默认 v5 分开推理后再与 v3/v5 等 JSONL 多数票融合）：

- ``v5``：默认 v5（与下方 ``v5a``/``v5b``/``v5c`` 输出文件名不同，避免覆盖）。
- ``v5a_enum``：在 spaCy 表与 v5 任务块之上，**强制先列词表再写标签**（与 v3 思路一致，便于与纯 v5 投票互补）。
- ``v5b_syntax``：**句法优先**的分步计数说明，强调谓语中心/名词投射与 spaCy 行的交叉校验，减少边界词误判。
- ``v5c_rubric``：**边界清单**（零动词句、轻动词/代词、与 spaCy 表一致性）再输出标签。

输出文件名：``compare_spacy_v5_zeroshot`` / ``compare_spacy_v5a_zeroshot`` / ``compare_spacy_v5b_zeroshot`` / ``compare_spacy_v5c_zeroshot``（均带 ``openseek-2-`` 前缀）。

依赖：与 v3 相同（``pip install spacy`` 且 ``python -m spacy download en_core_web_sm``）。
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Callable, Literal, cast

from tqdm import tqdm

from method_hyb import annotate_nvidia as annotate
from method_hyb import annotate_nvidia_raw_text
from method_hyb import count_answer

# ---------------------------------------------------------------------------
# 本地提示词（与 v3 同源结构）
# ---------------------------------------------------------------------------

PromptBuilder = Callable[[str, str], str]

TASK_PROMPT_BUILDERS: dict[int, PromptBuilder] = {}

_DEFAULT_ANNOTATION_GUIDELINES = (
    "### Annotation guidelines\n"
    "1. You may show step-by-step reasoning in plain text **before** the final answer; put all reasoning **outside** "
    "the `<label>...</label>` tags.\n"
    "2. The final answer MUST appear inside a single pair of tags: <label>...</label>.\n"
    "3. Inside the tags, output ONLY one non-negative integer in decimal digits (no extra words or punctuation).\n"
)

_V5A_ENUM_BLOCK = (
    "### Required listing before `<label>` (variant v5a)\n"
    "After any free-form reasoning, you **must** emit a section headed exactly **\"Nouns:\"** or **\"Verbs:\"** "
    "(use the one matching **Your input**), then list **every** surface token you are counting **one per line**, "
    "using the same spelling/casing as in the quoted sentence.\n"
    "Optionally add one line `Count: <integer>` that must equal the number of listed lines.\n"
    "The `<label>...</label>` pair must still contain **only digits** and must equal that count.\n"
    "Do **not** put the token list inside `<label>`.\n"
)

_V5B_SYNTAX_FIRST_BLOCK = (
    "### Counting procedure — syntax-first (variant v5b)\n"
    "1. Mentally bracket the quoted sentence into constituents (subject, predicate VP, objects, complements, "
    "modifiers, participial / reduced clauses).\n"
    "2. **Verb items:** count tokens that **head a verbal predicate** in that structure (finite or non-finite lexical heads). "
    "Exclude auxiliaries, dummy *do*, modals, and light copulas that only link tense/aspect unless they are the sole counted predicate "
    "per the task rules above.\n"
    "3. **Noun items:** count tokens that **head nominal projections** denoting entities/things (common/proper nouns). "
    "Treat scene-predicating `-ing`/`-ed` forms that modify the clause event as **verbal**, not nouns, unless they clearly head a gerund NP.\n"
    "4. **Cross-check with the spaCy table:** after applying (2) or (3), your integer should be consistent with which "
    "**VERB** vs **NOUN/PROPN** rows you actually relied on; if a tag conflicts with syntax + task rules, **follow task rules + syntax**, not the tagger.\n"
)

_V5C_RUBRIC_BLOCK = (
    "### Boundary checklist — variant v5c (before `<label>`)\n"
    "1. **Target kind:** **Your input** asks for **either** nouns **or** verbs—never mix two counts in one answer.\n"
    "2. **Zero verbs is common:** short captions that only describe a static scene may have **0** lexical verbs; do not invent verbs.\n"
    "3. **Pronouns / determiners:** *it/he/she/they/…* and *the/a/…* are **not** nouns for this benchmark’s noun count.\n"
    "4. **Light auxiliaries / copulas:** do not inflate verb counts with bare *be/have/do* unless the task rules treat them as separate counted heads.\n"
    "5. **Consistency with the spaCy table:** your final integer should match the set of **NOUN/PROPN** or **VERB** rows you actually relied on after applying the rules above.\n"
)

Task2PromptVariant = Literal["v5", "v5a_enum", "v5b_syntax", "v5c_rubric"]

_LABEL_RETRY_REMINDER = (
    "\n\n### Critical reminder\n"
    "Your reply MUST contain exactly one pair `<label>...</label>` with **only digits** inside; "
    "do not leave the answer blank.\n"
)


def _task_prompt_shell_zero_shot(
    task_specific_guidelines: str,
    task_description: str,
    text2annotate: str,
    *,
    annotation_guidelines: str | None = None,
) -> str:
    guidelines = (
        annotation_guidelines if annotation_guidelines is not None else _DEFAULT_ANNOTATION_GUIDELINES
    )
    return (
        "### Role\n"
        "You solve OpenSeek benchmark items. Follow the official task definition below.\n\n"
        f"{task_specific_guidelines}\n"
        "### Official task definition\n"
        f"{task_description}\n\n"
        "### Your input (answer for this instance only)\n"
        f"{text2annotate}\n\n"
        f"{guidelines}"
    )


def register_task_prompt(task_id: int, builder: PromptBuilder) -> None:
    TASK_PROMPT_BUILDERS[task_id] = builder


def _build_prompt_default(task_description: str, text2annotate: str) -> str:
    return (
        "### Role Definition\n"
        "You are a data annotation expert.\n\n"
        "### Core Task\n"
        f"{task_description}\n\n"
        "### Text to Annotate\n"
        f"{text2annotate}\n\n"
        "### Final output\n"
        "Wrap the answer in <label>...</label>.\n"
    )


def build_prompt(
    task_description: str,
    text2annotate: str,
    *,
    task_id: int | None = None,
) -> str:
    if task_id is not None:
        custom = TASK_PROMPT_BUILDERS.get(task_id)
        if custom is not None:
            return custom(task_description, text2annotate)
    return _build_prompt_default(task_description, text2annotate)


# ---------------------------------------------------------------------------
# spaCy
# ---------------------------------------------------------------------------

_NLP = None


def _get_nlp():
    global _NLP  # noqa: PLW0603
    if _NLP is None:
        import spacy

        _NLP = spacy.load("en_core_web_sm")
    return _NLP


def _contains_ascii_uppercase(s: str) -> bool:
    return any("A" <= c <= "Z" for c in s)


def _task2_count_target_from_input(text: str) -> str | None:
    m = re.search(
        r"Count\s+the\s+number\s+of\s+(nouns|verbs)\s+in\s+this\s+sentence",
        str(text),
        flags=re.IGNORECASE,
    )
    return None if not m else m.group(1).lower()


def _align_task2_official_definition(task_description: str, text2annotate: str) -> str:
    base = str(task_description).strip()
    target = _task2_count_target_from_input(text2annotate)
    if target == "verbs":
        return (
            f"{base}\n\n"
            "**For this instance:** count **verbs only** in the quoted sentence below. Do not report a noun count."
        )
    if target == "nouns":
        return (
            f"{base}\n\n"
            "**For this instance:** count **nouns only** in the quoted sentence below. Do not report a verb count."
        )
    return (
        f"{base}\n\n"
        "**For this instance:** the instruction below asks explicitly for either noun count or verb count—follow it."
    )


def _extract_sentence_from_task2_input(text: str) -> str | None:
    t = str(text).strip()
    m = re.search(r"Sentence:\s*'(.*)'\.\s*Count\b", t, flags=re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r'Sentence:\s*"(.*)"\.\s*Count\b', t, flags=re.DOTALL)
    if m:
        return m.group(1).strip()
    return None


_SPACY_LEGEND = (
    "### How to read spaCy token rows (`en_core_web_sm`)\n"
    "Each line is: **token** → **`pos_`** (coarse Universal POS) → **`tag_`** (fine-grained English tag from the model).\n"
    "- **`pos_` vs task counting**\n"
    "  - **`NOUN`** (`NN`, `NNS`, …): common nouns. **`PROPN`** (`NNP`, `NNPS`, …): proper nouns / names. "
    "For **noun-count** items, candidates usually appear as **NOUN** or **PROPN** (subject to task rules below—not determiners or pronouns).\n"
    "  - **`VERB`** (`VB`, `VBD`, `VBG`, `VBN`, `VBZ`, …): lexical verb forms. "
    "For **verb-count** items, **VERB** tokens are primary candidates **when they realize the benchmark’s lexical/action predicate**.\n"
    "  - **`AUX`** (`MD`, uses of *be*/*have*/*do* as auxiliary, …): supporting verbs. "
    "This benchmark typically **does not** count bare auxiliaries, light copulas, or modals **by themselves** as separate verbs.\n"
    "  - **`ADJ`**, **`DET`**, **`ADP`**, **`PRON`**, **`PART`**, **`PUNCT`**, **`NUM`**, etc.: generally **not** counted as nouns for noun tasks nor as verbs for verb tasks.\n"
    "- **Gerunds / participles (`VBG`, `VBN`, …)** may correspond to events in the scene; the benchmark often treats them as **verbs** when they denote an action/state, "
    "even inside noun phrases—follow the verb-count rules below when this instance asks for verbs.\n"
    "- **Disclaimer:** spaCy is automatic; if any tag conflicts with the **task-specific rules for this instance**, "
    "follow the task rules, not the tagger.\n"
    "- **v5 note:** if the quoted sentence contains uppercase ASCII letters, the POS token rows below are produced from a "
    "**lower-cased copy** of that sentence **only for tagging stability**; your textual reasoning must still refer to the original casing "
    "as shown in **Your input**.\n"
)


_OPENSEEK_2_TASK_SPECIFIC_VERBS = (
    "### Task-specific output (openseek-2: count verbs — this instance)\n"
    "- This instance asks for **verbs only** in the quoted sentence.\n"
    "- Count **lexical / action predicates**, including many non-finite forms (*driving*, *wearing*, *displayed* in reduced clauses) when they denote the event/state.\n"
    "- **Do not** count **pure auxiliaries**, **light copulas alone** (*is/are/was/were* as mere linking), *have/has* as auxiliary, "
    "*do/does* as dummy auxiliary, or **modals** (*can/may/must/…*) **as standalone verbs** unless the benchmark example pattern treats them differently.\n"
    "- Resolve `-ing`/`-ed` ambiguity by syntax in context: scene predicates → usually **verbs** for this task.\n"
    "- Inside `<label></label>`, output **one non-negative integer** only (decimal digits); no other characters.\n"
)


_OPENSEEK_2_TASK_SPECIFIC_NOUNS = (
    "### Task-specific output (openseek-2: count nouns — this instance)\n"
    "- This instance asks for **nouns only** in the quoted sentence.\n"
    "- Count **common and proper nouns** (people, objects, places, named entities). **`PROPN`** in spaCy usually aligns with proper nouns.\n"
    "- **Do not** count **determiners** (*the/a*), **adjectives**, **pronouns**, **prepositions**, **verbs**, or **pure auxiliaries** as nouns.\n"
    "- Do **not** count an `-ing` verbal noun / gerund heading an action as a noun unless the benchmark convention for this task treats it as nominal—when unsure, "
    "prefer consistency with typical OpenSeek noun lists (objects/entities in the scene).\n"
    "- Inside `<label></label>`, output **one non-negative integer** only (decimal digits); no other characters.\n"
)


_OPENSEEK_2_TASK_SPECIFIC_FALLBACK = (
    "### Task-specific output (openseek-2: count nouns or verbs)\n"
    "- Read **Your input** below: it explicitly asks for either **nouns** or **verbs**—never both in one answer.\n"
    "- **Verb branch:** count lexical/action predicates; exclude bare auxiliaries/copulas/modals per usual OpenSeek verb rules.\n"
    "- **Noun branch:** count common/proper nouns; exclude determiners, adjectives, pronouns.\n"
    "- Inside `<label></label>`, output **one non-negative integer** only.\n"
)


def _spacy_pos_reference_block(sentence: str, *, instance_target: str | None) -> tuple[str, bool]:
    """
    生成 POS 参考块；若对大写句使用了 ``lower()`` 做分词则第二个返回值为 True。
    """
    if not sentence:
        return (
            "### Reference: spaCy POS (en_core_web_sm)\n(Could not isolate the quoted sentence; skipped tagging.)\n",
            False,
        )

    use_lower = _contains_ascii_uppercase(sentence)
    sentence_for_nlp = sentence.lower() if use_lower else sentence

    nlp = _get_nlp()
    doc = nlp(sentence_for_nlp)
    lines = ["{:24} → {:10} ({})".format(tok.text, tok.pos_, tok.tag_) for tok in doc]
    table = "\n".join(lines) if lines else "(empty)"

    focus = ""
    if instance_target == "verbs":
        focus = (
            "**Using this table for this instance:** prioritize tokens whose **`pos_` is `VERB`** after applying the verb-count rules above; "
            "use **`AUX`** rows only when they realize the benchmark’s counted predicate (usually they do not count as extra verbs).\n"
        )
    elif instance_target == "nouns":
        focus = (
            "**Using this table for this instance:** prioritize **`NOUN`** and **`PROPN`** rows after applying the noun-count rules above.\n"
        )
    else:
        focus = "**Using this table:** match spaCy rows to whether this item asks for nouns or verbs (see Your input).\n"

    block = (
        "### Reference: spaCy POS tags (model `en_core_web_sm`)\n"
        f"{focus}\n"
        "```text\n"
        f"{table}\n"
        "```\n"
    )
    return block, use_lower


def _compose_task2_guidelines(
    text2annotate: str,
    *,
    variant: Task2PromptVariant = "v5",
) -> tuple[str, bool]:
    target = _task2_count_target_from_input(text2annotate)
    sent = _extract_sentence_from_task2_input(text2annotate) or ""

    if target == "verbs":
        task_block = _OPENSEEK_2_TASK_SPECIFIC_VERBS
    elif target == "nouns":
        task_block = _OPENSEEK_2_TASK_SPECIFIC_NOUNS
    else:
        task_block = _OPENSEEK_2_TASK_SPECIFIC_FALLBACK

    variant_block = ""
    if variant == "v5a_enum":
        variant_block = _V5A_ENUM_BLOCK + "\n"
    elif variant == "v5b_syntax":
        variant_block = _V5B_SYNTAX_FIRST_BLOCK + "\n"
    elif variant == "v5c_rubric":
        variant_block = _V5C_RUBRIC_BLOCK + "\n"

    pos_block, lowered_for_spacy = _spacy_pos_reference_block(sent, instance_target=target)
    return variant_block + task_block + "\n" + _SPACY_LEGEND + "\n" + pos_block, lowered_for_spacy


def _prompt_openseek_2_spacy_v5(task_description: str, text2annotate: str) -> str:
    return _build_prompt_task2_v5(task_description, text2annotate)[0]


def _build_prompt_task2_v5(
    task_description: str,
    text2annotate: str,
    *,
    variant: Task2PromptVariant = "v5",
) -> tuple[str, bool]:
    """构建完整 prompt，并返回是否对大写句使用了 spaCy lower。"""
    guidelines, lowered_for_spacy = _compose_task2_guidelines(text2annotate, variant=variant)
    definition_for_prompt = _align_task2_official_definition(task_description, text2annotate)
    ann = _DEFAULT_ANNOTATION_GUIDELINES
    if variant == "v5a_enum":
        ann = (
            "### Annotation guidelines\n"
            "1. Put all reasoning and the required **Nouns:** / **Verbs:** list **outside** `<label>...</label>`.\n"
            "2. The final answer MUST appear inside a single pair of tags: <label>...</label>.\n"
            "3. Inside the tags, output ONLY one non-negative integer in decimal digits (no extra words or punctuation).\n"
        )
    prompt = _task_prompt_shell_zero_shot(
        guidelines,
        definition_for_prompt,
        text2annotate,
        annotation_guidelines=ann,
    )
    return prompt, lowered_for_spacy


def _normalize_integer_answer(pred: object) -> str:
    """将模型/解析结果收紧为单个非负整数字符串；无法解析则空串。"""
    if pred is None:
        return ""
    s = str(pred).strip()
    if not s:
        return ""
    if re.fullmatch(r"\d+", s):
        return s
    m = re.search(r"<label>\s*(\d+)\s*</label>", s, flags=re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"\b(\d+)\b", s)
    return m.group(1) if m else ""


def _predict_with_fallbacks(prompt_base: str, retries: int, retry_wait_seconds: float) -> tuple[str, dict[str, Any]]:
    """
    先 ``annotate``（内置 ``count_answer``）；为空则从 attempt≥2 起追加 `<label>` 提醒；
    仍为空则用全文 ``annotate_nvidia_raw_text`` + ``count_answer`` + 数字兜底。
    """
    meta: dict[str, Any] = {"label_reminder_used": False, "raw_fulltext_fallback": False}

    prediction = ""

    for attempt in range(1, retries + 1):
        suffix = _LABEL_RETRY_REMINDER if attempt >= 2 else ""
        meta["label_reminder_used"] = meta["label_reminder_used"] or bool(suffix)
        prompt = prompt_base + suffix
        try:
            raw_pred = annotate(prompt)
            prediction = _normalize_integer_answer(raw_pred)
            if prediction:
                break
        except Exception as e:  # noqa: BLE001
            if attempt >= retries:
                print(f"[推理异常] attempt={attempt}/{retries} err={e}")
            else:
                print(f"[重试] attempt={attempt}/{retries} err={e}")
                time.sleep(retry_wait_seconds)
            continue
        if attempt < retries and not prediction:
            time.sleep(retry_wait_seconds)

    if prediction:
        return prediction, meta

    try:
        whole = annotate_nvidia_raw_text(prompt_base + _LABEL_RETRY_REMINDER)
    except Exception as e:  # noqa: BLE001
        print(f"[raw_text 兜底失败] err={e}")
        whole = None

    if whole:
        meta["raw_fulltext_fallback"] = True
        parsed = count_answer(whole)
        prediction = _normalize_integer_answer(parsed)
        if not prediction:
            prediction = _normalize_integer_answer(whole)

    return prediction, meta


# ---------------------------------------------------------------------------
# 推理
# ---------------------------------------------------------------------------

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


def _variant_to_compare_slug(variant: Task2PromptVariant) -> str:
    return {
        "v5": "spacy_v5",
        "v5a_enum": "spacy_v5a",
        "v5b_syntax": "spacy_v5b",
        "v5c_rubric": "spacy_v5c",
    }[variant]


def _variant_to_summary_stem(variant: Task2PromptVariant) -> str:
    return f"summary_main1_task2_{_variant_to_compare_slug(variant)}_zeroshot"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="main1 task2：v5 = v3 + spaCy 大写句小写分词 + 空标签/解析兜底（infer_examples_main1_task2_spacy_v5）。"
    )
    parser.add_argument("--task_start", type=int, default=2)
    parser.add_argument("--task_end", type=int, default=2)
    parser.add_argument(
        "--examples_limit",
        type=int,
        default=0,
        help="每任务最多推理条数；<=0 为全部。",
    )
    parser.add_argument("--output_dir", type=str, default="examples_main1")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry_wait_seconds", type=float, default=2.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--prompt_variant",
        type=str,
        default="v5",
        choices=["v5", "v5a_enum", "v5b_syntax", "v5c_rubric"],
        help="提示词变种；v5a/v5b/v5c 写入不同 compare JSONL 文件名，便于与 v3/v5 投票融合。",
    )
    parser.add_argument(
        "--emit_meta",
        action="store_true",
        help="在 jsonl 中写入 v5 诊断字段（spacy_lowercase_POS / raw_fulltext_fallback / label_reminder_used）。",
    )
    return parser.parse_args()


def _task_json_path(task_id: int) -> Path:
    if task_id not in TASK_DATA_FILES:
        raise ValueError(f"task_id should be in [1, 8], but got {task_id}.")
    return REPO_ROOT / "data" / TASK_DATA_FILES[task_id]


def _resolve_output_dir(output_dir: str) -> Path:
    p = Path(output_dir)
    return p.resolve() if p.is_absolute() else (REPO_ROOT / p).resolve()


def _normalize_text(text: str) -> str:
    return " ".join(str(text).strip().split())


def _extract_output(output_value: object) -> str:
    if isinstance(output_value, list):
        return "" if not output_value else str(output_value[0]).strip()
    return str(output_value).strip()


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
            eid = str(row.get("example_id", "")).strip()
            if eid:
                done.add(eid)
    return done


def _compute_metrics_from_jsonl(output_file: Path) -> tuple[int, int, float]:
    total = matched = 0
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
    acc = (matched / total) if total else 0.0
    return total, matched, acc


def run_task(
    task_id: int,
    output_dir: Path,
    examples_limit: int = 0,
    retries: int = 3,
    retry_wait_seconds: float = 2.0,
    resume: bool = False,
    *,
    prompt_variant: Task2PromptVariant = "v5",
    emit_meta: bool = False,
) -> dict:
    task_file = _task_json_path(task_id)
    with task_file.open("r", encoding="utf-8") as f:
        task_dict = json.load(f)

    task_name = task_dict["task_name"]
    task_description = task_dict["Definition"][0]
    all_examples = task_dict["examples"]
    if examples_limit > 0:
        all_examples = all_examples[:examples_limit]

    slug = _variant_to_compare_slug(prompt_variant)
    output_file = output_dir / f"openseek-{task_id}-examples-main1-compare_{slug}_zeroshot.jsonl"
    done_ids = _load_done_ids(output_file) if resume else set()
    mode = "a" if resume else "w"

    if done_ids:
        print(f"[断点续跑] task={task_id} variant={prompt_variant}, 已完成 {len(done_ids)} 条")

    with output_file.open(mode, encoding="utf-8") as wf:
        for example in tqdm(
            all_examples,
            desc=f"Task {task_id} task2 {slug} zero-shot: {task_name}",
        ):
            example_id = str(example.get("id", "")).strip()
            if resume and example_id in done_ids:
                continue

            input_text = example.get("input", "")
            expected = _extract_output(example.get("output", ""))

            input_prompt, spacy_lowered = _build_prompt_task2_v5(
                task_description, str(input_text), variant=prompt_variant
            )

            prediction, pmeta = _predict_with_fallbacks(
                input_prompt,
                retries=retries,
                retry_wait_seconds=retry_wait_seconds,
            )

            is_match = _normalize_text(prediction) == _normalize_text(expected)

            record: dict[str, Any] = {
                "example_id": example_id,
                "input": input_text,
                "expected_output": expected,
                "model_output": prediction,
                "is_match": is_match,
            }
            if emit_meta:
                record["v5_spacy_lowercase_POS"] = bool(spacy_lowered)
                record["v5_label_reminder_used"] = bool(pmeta.get("label_reminder_used"))
                record["v5_raw_fulltext_fallback"] = bool(pmeta.get("raw_fulltext_fallback"))

            wf.write(json.dumps(record, ensure_ascii=False) + "\n")
            wf.flush()
            done_ids.add(example_id)

    total, match_count, accuracy = _compute_metrics_from_jsonl(output_file)
    print(
        f"[保存完成] task2 {slug} zeroshot task={task_id} total={total} matched={match_count} "
        f"accuracy={accuracy:.2%} file={output_file}"
    )
    return {
        "task_id": task_id,
        "task_name": task_name,
        "prompt_variant": prompt_variant,
        "total": total,
        "matched": match_count,
        "accuracy": accuracy,
        "file": str(output_file),
    }


def main() -> None:
    register_task_prompt(2, _prompt_openseek_2_spacy_v5)

    args = parse_args()
    prompt_variant = cast(Task2PromptVariant, args.prompt_variant)
    task_start = max(1, args.task_start)
    task_end = min(8, args.task_end)
    if task_start > task_end:
        raise ValueError(f"task_start({task_start}) > task_end({task_end})")

    output_dir = _resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[输出目录] {output_dir}")
    print(
        f"[task2 v5 家族] prompt_variant={prompt_variant}；"
        "spaCy：大写句仅用 lower() 生成分词表；推理：重试 + `<label>` 提醒 + 全文解析兜底。"
    )

    summary: list[dict] = []
    for task_id in range(task_start, task_end + 1):
        summary.append(
            run_task(
                task_id=task_id,
                output_dir=output_dir,
                examples_limit=args.examples_limit,
                retries=args.retries,
                retry_wait_seconds=args.retry_wait_seconds,
                resume=args.resume,
                prompt_variant=prompt_variant,
                emit_meta=args.emit_meta,
            )
        )

    summary_file = output_dir / f"{_variant_to_summary_stem(prompt_variant)}.json"
    with summary_file.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[汇总完成] {summary_file}")


if __name__ == "__main__":
    main()
