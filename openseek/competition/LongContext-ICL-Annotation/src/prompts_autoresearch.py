#!/usr/bin/env python3
"""
Task7 可版本化 prompt 注册表（供 autoresearch_prompt.py 读写）。

与 ``method_hyb_prompts`` 的单函数 builder 并存：autoresearch 产出 system/user 模板后
由 ``infer_autoresearch_task7_eval.py`` 注册并评测。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

PromptBuilder = Callable[[str, str], str]

_REGISTRY: dict[int, dict[int, "PromptTemplate"]] = {}


@dataclass
class PromptTemplate:
    task_id: int
    version: int
    description: str
    system: str
    user: str

    def render(self, task_description: str, examples: str, text2annotate: str) -> str:
        user_filled = (
            self.user.replace("{task_description}", task_description)
            .replace("{{task_description}}", task_description)
            .replace("{examples}", examples)
            .replace("{{examples}}", examples)
            .replace("{text2annotate}", text2annotate)
            .replace("{{text2annotate}}", text2annotate)
        )
        sys_part = self.system.strip()
        if sys_part:
            return f"{sys_part}\n\n{user_filled}"
        return user_filled

    def to_builder(self) -> PromptBuilder:
        def _builder(task_description: str, text2annotate: str) -> str:
            return self.render(task_description, "", text2annotate)

        return _builder


def _register(tpl: PromptTemplate) -> None:
    _REGISTRY.setdefault(tpl.task_id, {})[tpl.version] = tpl


def get_prompt(
    task_id: int,
    version: int,
    task_description: str,
    examples: str,
    text2annotate: str,
) -> tuple[str, str]:
    """返回 (system, user_template) 供 meta-prompt 使用。"""
    tpl = _REGISTRY[task_id][version]
    return tpl.system, tpl.user


def get_latest_version(task_id: int) -> int:
    if task_id not in _REGISTRY or not _REGISTRY[task_id]:
        return 0
    return max(_REGISTRY[task_id].keys())


def get_builder(task_id: int, version: int) -> PromptBuilder:
    return _REGISTRY[task_id][version].to_builder()


# --- v1 基线：Jeopardy 短答案 + 小写 ---
_register(
    PromptTemplate(
        task_id=7,
        version=1,
        description="baseline Jeopardy: category+clue, lowercase label",
        system=(
            "You are an expert Jeopardy! contestant. Answer with the shortest correct phrase "
            "that fits the category and clue."
        ),
        user=(
            "### Official task definition\n"
            "{task_description}\n\n"
            "### Reference examples\n"
            "{examples}\n\n"
            "### Your input\n"
            "{text2annotate}\n\n"
            "Put your final answer only inside <label>all_lower_case_answer</label>."
        ),
    )
)

# --- autoresearch generated versions ---

# 兼容 autoresearch 旧 import 名
prompts = __import__(__name__)
