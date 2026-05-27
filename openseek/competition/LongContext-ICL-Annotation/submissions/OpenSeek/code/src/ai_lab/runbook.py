from typing import Any, Dict, List

from src.ai_lab.prompting import format_competition_example


def build_label_description(label_space: List[str], task_type: str, task_name: str = "") -> str:
    if task_name in {"closest_integers", "count_nouns_verbs"}:
        return (
            "Answer type: integer.\n"
            "Return only one base-10 integer. Do not copy an example answer, "
            "do not include units, and do not explain."
        )
    if task_name == "collatz_conjecture":
        return (
            "Answer type: Python-style list of integers.\n"
            "Return only one list literal such as [1, 2, 3]. Do not explain, "
            "do not repeat the list, and do not copy an example answer."
        )
    if task_name == "conala_concat_strings":
        return (
            "Answer type: concatenated result.\n"
            "Return only the final concatenated output required by the task. "
            "Do not describe the operation, do not emit code, and do not copy an example answer."
        )
    if task_name == "semeval_2018_task1_tweet_sadness_detection":
        return (
            "Closed label task.\n"
            "Allowed labels are exactly:\n"
            "- Sad\n"
            "- Not sad\n"
            "Return only one of the allowed labels. Do not return the tweet text."
        )
    if task_name == "jeopardy_answer_generation_all":
        return (
            "Answer type: short Jeopardy answer phrase.\n"
            "Return only the entity, title, place, person, or short phrase that answers the clue. "
            "Do not answer in a full sentence, do not explain, and do not copy an example answer."
        )

    if task_type == "code_generation":
        return (
            "Answer type: source code.\n"
            "Return only the final implementation. Do not include markdown fences, explanations, "
            "or any text outside the code."
        )
    if task_type == "generation":
        return (
            "Open answer task.\n"
            "Return only the final answer text for the current input. Do not explain and do not copy an example answer."
        )
    if not label_space:
        return (
            "Open answer task.\n"
            "Return only the single best final answer for the current input. Do not explain."
        )

    unique_labels = list(dict.fromkeys(str(label).strip() for label in label_space if str(label).strip()))
    unique_count = len(unique_labels)
    total_count = len(label_space)
    unique_ratio = unique_count / max(total_count, 1)

    if any(label.startswith("[") and "," in label for label in unique_labels):
        return (
            "Open list answer task.\n"
            "Return only one final Python-style list answer for the current input. "
            "Do not explain and do not copy an example answer."
        )

    if unique_count >= 8 and unique_ratio >= 0.5:
        return (
            "Open answer task.\n"
            "Return only the final answer string for the current input. Do not explain and do not copy an example answer."
        )

    return "Closed label task.\nAllowed labels are exactly:\n" + "\n".join(f"- {label}" for label in label_space)


def build_examples_block(examples: List[Dict[str, Any]], task_type: str) -> str:
    return "\n\n".join(format_competition_example(example, task_type) for example in examples)
