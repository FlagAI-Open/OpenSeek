from method_hyb import annotate_nvidia
from method_hyb import build_prompt as _base_build_prompt


def build_prompt(task_description: str, text2annotate: str, task_id: int | None = None) -> str:
    """Task5 prompt adapter; keeps API compatible with missing legacy file."""
    effective_task_id = 5 if task_id is None else task_id
    return _base_build_prompt(task_description, text2annotate, task_id=effective_task_id)


__all__ = [
    "annotate_nvidia",
    "build_prompt",
]
