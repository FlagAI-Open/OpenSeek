from typing import Any
from strategy_base import BaseStrategy
from strategy_verified import VerifiedProgramStrategy
from strategy_task2 import Task2StructuredStrategy
from strategy_task5 import Task5SadnessStrategy
from strategy_task6 import Task6GenreStrategy
from strategy_task7 import Task7JeopardyStrategy
from strategy_task8 import Task8TritonStrategy

# 所有可用的策略
STRATEGIES = {
    'verified_program': VerifiedProgramStrategy(),
    'task2_structured': Task2StructuredStrategy(),
    'task5_sadness': Task5SadnessStrategy(),
    'task6_genre': Task6GenreStrategy(),
    'task7_jeopardy': Task7JeopardyStrategy(),
    'task8_triton': Task8TritonStrategy(),
}

# 任务 ID 到策略映射
TASK_STRATEGY_MAP = {
    1: 'verified_program',
    2: 'task2_structured',
    3: 'verified_program',
    4: 'verified_program',
    5: 'task5_sadness',
    6: 'task6_genre',
    7: 'task7_jeopardy',
    8: 'task8_triton',
}

def get_strategy(task_id: int) -> BaseStrategy:
    if task_id not in TASK_STRATEGY_MAP:
        raise ValueError(f"No strategy defined for task_id: {task_id}")
    strategy_name = TASK_STRATEGY_MAP[task_id]
    return STRATEGIES[strategy_name]

def normalize_examples(all_examples: list[dict[str, Any]]) -> list[dict[str, str]]:
    """
    格式化原始 JSON 样例。
    """
    normalized = []
    for ex in all_examples:
        input_val = str(ex.get("input", "")).strip()
        output_val = ex.get("output", "")
        if isinstance(output_val, list) and output_val:
            output_val = output_val[0]
        output_val = str(output_val).strip()
        normalized.append({
            "input": input_val,
            "expected": output_val
        })
    return normalized

def select_examples(all_examples: list[dict[str, Any]], example_count: int = 3) -> list[dict[str, str]]:
    """
    选择前 N 个可用样例。
    """
    return normalize_examples(all_examples)[:example_count]
