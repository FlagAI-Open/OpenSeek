from method_hyb import annotate_nvidia
from method_hyb import build_prompt as _base_build_prompt
from method_hyb import select_examples_hybrid


def _task6_same_genre_falsification_block() -> str:
    """Task6 calibration: falsify-first to reduce N->Y bias."""
    return (
        "### Task 6 (Same Genre Y/N) — Falsify-First Rules (High Priority)\n"
        "Goal: decide whether BOTH sentences fit the candidate genre in benchmark sense.\n\n"
        "Decision order (must follow):\n"
        "1) Try to falsify first: check each sentence against the candidate genre independently.\n"
        "2) If either sentence clearly violates genre style/source/register, output N immediately.\n"
        "3) Output Y only when both sentences are positively supported as belonging to that genre.\n\n"
        "Calibration notes:\n"
        "- Prefer genre evidence over semantic overlap; paraphrase/topic similarity alone is not enough for Y.\n"
        "- Typical genre cues: dialogue markers, formality, named entities style, narrative voice, document source.\n"
        "- Borderline cases default to N unless both sides have clear same-genre evidence.\n\n"
        "Output constraint:\n"
        "- Final label must be exactly one uppercase char: Y or N (inside <label> tags).\n\n"
    )


def build_prompt(task_description: str, text2annotate: str, task_id: int | None = None) -> str:
    """Task-aware prompt wrapper; task6 uses falsify-first extra block."""
    if task_id != 6:
        return _base_build_prompt(task_description, text2annotate, task_id=task_id)

    task_extra = _task6_same_genre_falsification_block()
    return (
        "### Role Definition\n"
        "You are a professional data annotation expert specialized in long-context text labeling. "
        "Follow task rules and examples strictly.\n\n"
        "### Core Task\n"
        f"{task_description}\n\n"
        f"{task_extra}"
        "### Critical Annotation Guidelines\n"
        "1. Learn decision boundaries from examples and keep consistency.\n"
        "2. You may reason internally, but final answer must be one label tag.\n"
        "3. Final output MUST be enclosed in <label> tags only.\n\n"
        "### Examples (Must Be Fully Followed)\n"
        "[[EXAMPLES]]\n\n"
        "### Text to Annotate\n"
        f"{text2annotate}\n\n"
        "### Final Requirement Summary\n"
        "Output exactly one label wrapped in <label> tags: <label>Y</label> or <label>N</label>.\n"
    )


def _extract_output(output_value) -> str:
    if isinstance(output_value, list):
        return "" if not output_value else str(output_value[0]).strip()
    return str(output_value).strip()


def select_task6_examples_balanced(
    all_examples: list[dict],
    task_description: str,
    text2annotate: str,
    *,
    top_k: int = 8,
    rerank_pool_size: int = 200,
    exclude_example_id: str | None = None,
    balanced: bool = True,
) -> str:
    """Task6 ICL retrieval; optionally enforce Y/N class balance."""
    cleaned: list[dict] = []
    for ex in all_examples:
        cleaned.append(
            {
                "id": str(ex.get("id", "")).strip(),
                "input": str(ex.get("input", "")),
                "output": [_extract_output(ex.get("output", ""))],
            }
        )

    if not balanced:
        return select_examples_hybrid(
            all_examples=cleaned,
            task_description=task_description,
            text2annotate=text2annotate,
            top_k=max(1, top_k),
            rerank_pool_size=max(20, rerank_pool_size),
            use_explanation=False,
            use_bm25_semantic_rerank=True,
            exclude_example_id=exclude_example_id,
        )

    y_pool = [x for x in cleaned if _extract_output(x.get("output", "")) == "Y"]
    n_pool = [x for x in cleaned if _extract_output(x.get("output", "")) == "N"]
    k = max(2, top_k)
    y_k = k // 2
    n_k = k - y_k

    y_examples = select_examples_hybrid(
        all_examples=y_pool,
        task_description=task_description,
        text2annotate=text2annotate,
        top_k=max(1, y_k),
        rerank_pool_size=max(20, rerank_pool_size),
        use_explanation=False,
        use_bm25_semantic_rerank=True,
        exclude_example_id=exclude_example_id,
    )
    n_examples = select_examples_hybrid(
        all_examples=n_pool,
        task_description=task_description,
        text2annotate=text2annotate,
        top_k=max(1, n_k),
        rerank_pool_size=max(20, rerank_pool_size),
        use_explanation=False,
        use_bm25_semantic_rerank=True,
        exclude_example_id=exclude_example_id,
    )
    merged = (n_examples + y_examples).strip()
    if merged:
        return merged + "\n"
    return merged


__all__ = [
    "annotate_nvidia",
    "build_prompt",
    "select_task6_examples_balanced",
]
