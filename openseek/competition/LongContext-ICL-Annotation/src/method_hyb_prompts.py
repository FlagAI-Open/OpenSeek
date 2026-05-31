"""
与 ``method_hyb`` / ``method_hyb_thinking_default`` 配套的标注提示词构造：默认模板 + 按 ``task_id`` 覆盖。

- ``task_id`` 为 1-8 时默认已注册 OpenSeek 八任务专用提示（与 ``main1.TASK_DATA_FILES`` / ``data/openseek-*.json`` 对齐）。
- 仍可在 ``TASK_PROMPT_BUILDERS`` 中改写某 id，或调用 ``register_task_prompt`` 在运行时覆盖。
- 示例占位符统一为 ``[[EXAMPLES]]\\n\\n``，与 ``main1.py`` 里
  ``prompt.replace("[[EXAMPLES]]\\n\\n", examples_str + "\\n\\n")`` 一致。
- 模型推理由 ``annotate_nvidia`` 调用本地 FlagOS（默认 ``http://localhost:9010/v1``，模型 ``Qwen3-4B-ascend-flagos``）。
"""

from __future__ import annotations

from typing import Callable

PromptBuilder = Callable[[str, str], str]

# 任务 id -> 提示词构造函数；未注册的任务走 ``build_prompt`` 的默认实现。
TASK_PROMPT_BUILDERS: dict[int, PromptBuilder] = {}


_DEFAULT_ANNOTATION_GUIDELINES = (
    "### Annotation guidelines\n"
    "1. You may show step-by-step reasoning in plain text, then give the final answer.\n"
    "2. The final answer MUST appear inside a single pair of tags: <label>...</label>.\n"
    "3. Inside the tags, output ONLY the answer string (no quotes around it unless the examples use quotes). "
    "Do not add commentary inside the tags.\n"
    "4. Match the examples: same capitalization, spacing inside list literals, and punctuation.\n"
)


_OPENSEEK_7_ANNOTATION_GUIDELINES = (
    "### Annotation guidelines\n"
    "1. You may show step-by-step reasoning in plain text, then give the final answer.\n"
    "2. The final answer MUST appear inside a single pair of tags: <label>...</label>.\n"
    "3. Inside the tags, output ONLY the answer string. Do not add commentary inside the tags.\n"
    "4. For openseek-7 (Jeopardy-style clues): inside <label> use **all lower case letters**, as required by "
    "the official task definition. Match reference examples for internal spacing and punctuation in the "
    "answer phrase; ignore generic capitalization advice from other OpenSeek tasks.\n"
)


def _task_prompt_shell(
    task_specific_guidelines: str,
    task_description: str,
    text2annotate: str,
    *,
    annotation_guidelines: str | None = None,
) -> str:
    """
    OpenSeek 八任务共用骨架：先任务专属约束，再官方 Definition、示例位、待解输入，
    最后统一要求用 ``<label>`` 包住与 gold 同型的答案（与 ``count_answer`` 解析一致）。
    """
    guidelines = (
        annotation_guidelines
        if annotation_guidelines is not None
        else _DEFAULT_ANNOTATION_GUIDELINES
    )
    return (
        "### Role\n"
        "You solve OpenSeek benchmark items. Follow the official task definition and the reference examples "
        "for answer type, formatting, and level of detail.\n\n"
        f"{task_specific_guidelines}\n"
        "### Official task definition\n"
        f"{task_description}\n\n"
        "### Reference examples (format and conventions)\n"
        "[[EXAMPLES]]\n\n"
        "### Your input (answer for this instance only)\n"
        f"{text2annotate}\n\n"
        f"{guidelines}"
    )


def register_task_prompt(task_id: int, builder: PromptBuilder) -> None:
    """为指定任务注册提示词构造函数（覆盖模块级字典中的同 id 项）。"""
    TASK_PROMPT_BUILDERS[task_id] = builder


# --- OpenSeek 任务 1-8：与 ``data/openseek-*.json`` 及 ``main1.TASK_DATA_FILES`` 对齐 ---

_OPENSEEK_1_CLOSEST_INTEGERS = (
    "### Task-specific output (openseek-1: closest integers)\n"
    "- Goal: return the minimum absolute difference between any two different elements in the integer list.\n"
    "- Reliable method: sort the numbers ascending, then check absolute differences of adjacent elements only; "
    "the smallest adjacent gap is the answer.\n"
    "- Duplicates are valid and immediately imply answer 0. Negative values are handled normally by absolute difference.\n"
    "- Inside <label></label>, output exactly one base-10 integer only (e.g., 0, 1, 33). "
    "No units, commas, extra text, math symbols, or explanation inside the tags.\n"
)

_OPENSEEK_2_COUNT_NOUNS_VERBS = (
    "### Task-specific output (openseek-2: count nouns/verbs)\n"
    "- Read the instruction carefully and count exactly one target POS in the sentence: either nouns or verbs (never both).\n"
    "- For verb counting, prioritize lexical/action predicates (including non-finite forms like driving/wearing/displayed in reduced clauses); "
    "do not count pure auxiliaries or light copulas by themselves (e.g., is/are/was/were, have/has, do/does, modals).\n"
    "- For noun counting, count common/proper nouns (people, objects, places, entities); do not count determiners, adjectives, or pronouns.\n"
    "- Resolve ambiguity by local syntax: -ing/-ed forms that denote an event/state in the scene are usually counted as verbs, "
    "even when they modify a noun phrase.\n"
    "- Inside <label></label>, output a single non-negative integer in decimal digits only; no extra text inside the tags.\n"
)

_OPENSEEK_3_COLLATZ = (
    "### Task-specific output (openseek-3: Collatz step)\n"
    "- Apply exactly one Collatz transform per element (do NOT iterate repeatedly to 1): even n -> n // 2, odd n -> 3*n + 1.\n"
    "- Process each element independently and preserve original order; output length must equal input length.\n"
    "- All outputs must be integers; keep sign handling natural under the arithmetic rules above.\n"
    "- Inside <label></label>, output one Python-style integer list only, matching example spacing "
    "(comma + single space, e.g. [36, 88, 148]); no extra text inside the tags.\n"
)

_OPENSEEK_4_CONCAT_STRINGS = (
    "### Task-specific output (openseek-4: concatenate strings)\n"
    "- Concatenate all list elements strictly in the original order from left to right.\n"
    "- Insert no delimiter at all between elements (no spaces, commas, newlines, or separators unless already inside an element).\n"
    "- Preserve every character from each element exactly, including punctuation, case, and internal whitespace; empty strings contribute nothing.\n"
    "- Inside <label></label>, output only the final concatenated raw string; do not add quotes, brackets, escapes, or explanation.\n"
)

_OPENSEEK_5_TWEET_SADNESS = (
    "### Task-specific output (openseek-5: tweet sadness)\n"
    "- Judge whether the author expresses sadness in the tweet's overall meaning.\n"
    "- Use full textual context first; hashtags, emojis, and punctuation are auxiliary cues only.\n"
    "- Label Sad when there is clear affective evidence of sorrow, grief, loneliness, helplessness, or emotional hurt.\n"
    "- Do not label Sad for neutral statements, jokes, pure sarcasm, or emotions dominated by anger/annoyance/surprise without sadness.\n"
    "- Inside <label></label>, output exactly one label only: Sad or Not sad (match capitalization and spacing exactly).\n"
)

_OPENSEEK_6_MNLI_GENRE = (
    "### Task-specific output (openseek-6: same genre Y/N)\n"
    "- You are given sentence 1, sentence 2, and a candidate genre. Judge whether BOTH sentences fit that same genre in the benchmark sense.\n"
    "- Focus on genre signals (source/register/style: dialogue vs formal writing vs travel guide vs fiction, etc.), not just topic overlap.\n"
    "- Semantic paraphrase does not guarantee same genre; likewise different topics can still be same genre if style/source match.\n"
    "- Use the benchmark genre definitions in the task description as the primary criterion when uncertain.\n"
    "- Inside <label></label>, output exactly one uppercase character: Y or N (no spaces, punctuation, or extra text).\n"
)

_OPENSEEK_7_JEOPARDY = (
    "### Task-specific output (openseek-7: Jeopardy-style answer)\n"
    "- You get a **Category** and a **Clue** (like Jeopardy!): the response must be the short answer that fits "
    "the category and is described by the clue—not a full sentence, not a recap of the clue, and not an essay.\n"
    "- **Use the Category first** to decide what kind of thing is wanted (e.g. person, place, work title, "
    "word/phrase, food, historical event). Then use the Clue to pick the **single best-matching** entity of that kind.\n"
    "- Give the **canonical Jeopardy-style response**: the most specific, widely accepted **proper name or fixed phrase** "
    "(person / place / team / book or film title / band / etc.) when the clue calls for one. Avoid vague glosses such as "
    "\"the director of …\" or \"the king of …\" when a specific name exists.\n"
    "- **Do not treat quoted titles, lyrics, or book/film names that appear inside the clue as the final answer** "
    "unless the clue is explicitly asking for that exact title or line as the response. Do not copy arbitrary "
    "fragments from the clue (e.g. a song lyric, a side character, an album title) when the answer should be a "
    "different entity (e.g. the performer, author, or place).\n"
    "- If several facts in the clue point to different entities, answer what the **Category + clue focus** "
    "most directly asks for (e.g. \"who wrote …\", \"which state …\", \"this word …\").\n"
    "- For categories about correcting a phrase, rhymes, or wordplay, output **only** the corrected word or phrase, "
    "not an explanation.\n"
    "- Inside <label></label>, output only the answer in **all lower case**, with **no** surrounding quotes unless "
    "the reference examples themselves use quotes inside the answer.\n"
    "- For multi-word answers, use a single space between words; omit trailing punctuation unless it is part of a "
    "standard title or name in the examples.\n"
)

_OPENSEEK_8_TRITON = (
    "### Task-specific output (openseek-8: Triton kernel generation)\n"
    "- Implement the requested algorithm faithfully in Triton, including a usable Python wrapper API matching the instruction intent.\n"
    "- Ensure correctness first: align tensor shapes/strides, pointer arithmetic, launch grid, BLOCK parameters, and accumulation dtype.\n"
    "- Enforce memory safety with explicit masks for all potential out-of-bounds loads/stores; handle tail blocks robustly.\n"
    "- Include required runtime checks where appropriate (device/type/contiguity/shape compatibility) as seen in reference style.\n"
    "- Keep output code self-contained and executable in plain Python: imports, @triton.jit kernel(s), and callable wrapper function(s).\n"
    "- Inside <label></label>, output code only (no explanations, no markdown fences, no surrounding commentary).\n"
)


def _prompt_openseek_1(task_description: str, text2annotate: str) -> str:
    return _task_prompt_shell(_OPENSEEK_1_CLOSEST_INTEGERS, task_description, text2annotate)


def _prompt_openseek_2(task_description: str, text2annotate: str) -> str:
    return _task_prompt_shell(_OPENSEEK_2_COUNT_NOUNS_VERBS, task_description, text2annotate)


def _prompt_openseek_3(task_description: str, text2annotate: str) -> str:
    return _task_prompt_shell(_OPENSEEK_3_COLLATZ, task_description, text2annotate)


def _prompt_openseek_4(task_description: str, text2annotate: str) -> str:
    return _task_prompt_shell(_OPENSEEK_4_CONCAT_STRINGS, task_description, text2annotate)


def _prompt_openseek_5(task_description: str, text2annotate: str) -> str:
    return _task_prompt_shell(_OPENSEEK_5_TWEET_SADNESS, task_description, text2annotate)


def _prompt_openseek_6(task_description: str, text2annotate: str) -> str:
    return _task_prompt_shell(_OPENSEEK_6_MNLI_GENRE, task_description, text2annotate)


def _prompt_openseek_7(task_description: str, text2annotate: str) -> str:
    return _task_prompt_shell(
        _OPENSEEK_7_JEOPARDY,
        task_description,
        text2annotate,
        annotation_guidelines=_OPENSEEK_7_ANNOTATION_GUIDELINES,
    )


def _prompt_openseek_8(task_description: str, text2annotate: str) -> str:
    return _task_prompt_shell(_OPENSEEK_8_TRITON, task_description, text2annotate)


TASK_PROMPT_BUILDERS.update(
    {
        1: _prompt_openseek_1,
        2: _prompt_openseek_2,
        3: _prompt_openseek_3,
        4: _prompt_openseek_4,
        5: _prompt_openseek_5,
        6: _prompt_openseek_6,
        7: _prompt_openseek_7,
        8: _prompt_openseek_8,
    }
)


def build_prompt____(task_description: str, text2annotate: str) -> str:
    """
    严格标签输出：最终可见输出仅允许 ``<label>...</label>``（适合要求零旁白的任务）。
    """
    prompt = (
        "### Role Definition\n"
        "You are a professional data annotation expert specializing in long-context text labeling. "
        "Your work must strictly comply with the following rules, with the highest priority given to output format accuracy.\n\n"
        "### Core Annotation Task\n"
        f"{task_description}\n\n"
        "### Non-Negotiable Annotation Rules (Highest Priority)\n"
        "1. **Final Output Mandate**: Your annotation result MUST be wrapped in <label> tags — NO text, symbols, spaces, or explanations are allowed outside the tags.\n"
        "2. **Internal Reasoning Permission**: You may perform logical reasoning, text analysis, or context comprehension internally (in your thought process), but NONE of these thoughts may appear in the final output.\n"
        "3. **Label Format Strictness**: <label> is the opening tag and </label> is the closing tag — they must appear in pairs, with NO extra spaces or characters inside the tags (e.g., <label>  Good Review  </label> is invalid).\n"
        "4. **Prohibited Outputs**: \n"
        "   - ❌ Prohibited: 'After analysis, this is a positive review: <label>Good Review</label>' (extra text outside tags)\n"
        "   - ❌ Prohibited: 'Bad Review' (missing <label> tags entirely)\n"
        "   - ❌ Prohibited: '<label>Bad Review' (unpaired/closing tag missing)\n\n"
        "### Correct vs. Incorrect Examples\n"
        "✅ Correct Example 1: <label>answer</label>\n"
        "✅ Correct Example 2: <label>Bad Review</label>\n"
        "❌ Incorrect Example 1: I think this review is negative → <label>Bad Review</label>\n"
        "❌ Incorrect Example 2: <label>  Neutral Review  </label> (extra spaces inside tags)\n"
        "❌ Incorrect Example 3: Neutral Review (no label tags)\n\n"
        "### Reference Annotation Examples\n"
        "[[EXAMPLES]]\n\n"
        "### Text to Annotate\n"
        f"{text2annotate}\n\n"
        "### Final Output Command (Re-emphasized)\n"
        "You may complete any internal reasoning process, but your FINAL OUTPUT MUST consist solely of the annotation result wrapped in <label> tags (no other content whatsoever).\n"
        "Annotation Result: "
    )
    return prompt


def _build_prompt_default(task_description: str, text2annotate: str) -> str:
    """默认：允许推理过程，最终必须用 ``<label>`` 包裹（与原先 ``method_hyb.build_prompt`` 一致）。"""
    prompt = (
        "### Role Definition\n"
        "You are a professional data annotation expert specialized in long-context text labeling. "
        "Your work must strictly follow the task rules, fully learn from the provided examples, and ensure the final annotation result is 100% enclosed in <label> tags.\n\n"
        "### Core Task\n"
        f"{task_description}\n\n"
        "### Critical Annotation Guidelines\n"
        "1. **Example Learning Requirement**: Thoroughly analyze and fully learn from the annotation logic, format, and criteria in the Examples section. "
        "Your annotation must align with the style, judgment standards, and tag usage shown in the examples.\n"
        "2. **Thinking Process**: You may (and are encouraged to) explain your annotation reasoning step by step (e.g., key information extraction, judgment basis, rule matching).\n"
        "3. **Mandatory Output Rule**: Regardless of any thinking process you provide, your final annotation result MUST be enclosed in <label> tags (this is non-negotiable).\n"
        "   - Correct example: \n"
        "     Reasoning: This review mentions 'excellent quality' and 'very satisfied', which meets the criteria for a Good Review.\n"
        "     <label>Good Review</label>\n"
        "   - Wrong example 1 (missing tags): This review is negative.\n"
        "   - Wrong example 2 (incomplete tags): Bad Review</label>\n"
        "4. **Length Adaptation**: For long texts, maintain complete thinking process and ensure the final <label> tags contain the accurate annotation result (no truncation).\n\n"
        "### Examples (Must Be Fully Followed)\n"
        "[[EXAMPLES]]\n\n"
        "### Text to Annotate\n"
        f"{text2annotate}\n\n"
        "### Final Requirement Summary\n"
        "1. You can (and should) provide clear thinking process for your annotation.\n"
        "2. The final annotation result MUST be wrapped in <label> tags (no exceptions).\n"
        "3. All annotation logic must strictly follow the examples provided above.\n"
    )
    return prompt


def build_prompt_backup(task_description: str, text2annotate: str) -> str:
    """较早的简短模板；示例占位符与主流程一致。"""
    prompt = (
        "You are a data annotation assistant. "
        "Your task is to label the given texts according to the task description "
        "and annotation guidelines provided below.\n\n"
        f"[Task Description]\n {task_description}\n\n"
        "[Examples]\n[[EXAMPLES]]\n\n"
        "Please follow these instructions when labeling:\n"
        "1. **Output Format**: Annotate the text directly by wrapping each labeled "
        "span with <label> tags in the following format: <label> annotation result </label>.\n"
        f"[Task Description (repeat)] \n {task_description}\n\n"
        f"[Input Texts]\n {text2annotate}\n\n"
        "Please output the annotation results: "
    )
    return prompt


def build_prompt(
    task_description: str,
    text2annotate: str,
    *,
    task_id: int | None = None,
) -> str:
    """
    构造用户消息全文（尚未插入 ICL 示例块）。

    ``task_id`` 为 1-8 时使用预置的 OpenSeek 专用模板；其它 id 若在 ``TASK_PROMPT_BUILDERS`` 中有注册则使用该构造函数；
    否则使用默认 CoT 友好模板（``_build_prompt_default``）。
    """
    if task_id is not None:
        custom = TASK_PROMPT_BUILDERS.get(task_id)
        if custom is not None:
            return custom(task_description, text2annotate)
    return _build_prompt_default(task_description, text2annotate)


def build_prompt_for_task(
    task_id: int,
    task_description: str,
    text2annotate: str,
) -> str:
    """等价于 ``build_prompt(..., task_id=task_id)``，便于主程序按任务循环调用。"""
    return build_prompt(task_description, text2annotate, task_id=task_id)
