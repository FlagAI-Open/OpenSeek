from typing import Any, Dict, List


def format_example(example: Dict[str, Any]) -> str:
    output = example.get("label", example.get("output", ""))
    if isinstance(output, list):
        output = output[0] if output else ""
    return (
        f"Input:\n{example.get('input', example.get('text', ''))}\n\n"
        f"Output:\n{output}"
    )


def format_competition_example(example: Dict[str, Any], task_type: str) -> str:
    input_text = example.get("input", example.get("text", ""))
    output = example.get("output", "")
    if isinstance(output, list):
        output = output[0] if output else ""

    if task_type == "code_generation":
        return (
            "### Example\n"
            f"Input:\n{input_text}\n\n"
            f"Reference Output:\n{output}\n"
        )

    return f"# {input_text}\n<label>{output}</label>"


def build_output_instruction(task_type: str) -> str:
    if task_type == "code_generation":
        return (
            "Return only the final code solution. "
            "Do not add markdown fences, explanations, or extra commentary."
        )
    return (
        "Return only the final answer wrapped in <label> and </label>. "
        "Do not add explanations or any text outside the tags."
    )


def build_prompt(
    task_definition: str,
    system_prompt: str,
    output_format: str,
    examples: List[Dict[str, Any]],
    sample: Dict[str, Any],
    task_type: str = "classification",
) -> str:
    example_block = "\n\n".join(
        format_competition_example(example, task_type) for example in examples
    )
    query = sample.get("input", sample.get("text", sample.get("query", "")))
    output_instruction = build_output_instruction(task_type)
    return (
        f"{system_prompt}\n\n"
        f"Task Definition:\n{task_definition}\n\n"
        f"Output Format:\n{output_format}\n\n"
        f"Output Rules:\n{output_instruction}\n\n"
        f"In-Context Examples:\n{example_block}\n\n"
        f"Now annotate the following sample.\n"
        f"Input:\n{query}\n"
    )
