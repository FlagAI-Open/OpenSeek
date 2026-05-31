"""
Task5 对比推理 — 投票融合变体 C。

固定策略：--task5_strip_hashtag off --task5_emoji_mode off --task5_postemoji on
保留话题词前的 #；模型输出后仍可做 emoji 先验校准（见主脚本的 postemoji 逻辑）。

Prompt 侧重：显式提醒保留 # 形式的话题词，与表情同为辅助信号。
"""

from __future__ import annotations

import sys

import infer_examples_compare_task5 as _t5

_t5.TASK5_CANONICAL_DESCRIPTION = (
    'In this task you are given a tweet. You must judge whether the author of the tweet is sad or not. '
    'Label the instances as "Sad" or "Not sad" based on your judgment. '
    "Hashtags may keep a leading #; use them together with emojis as auxiliary signals, "
    "but do not decide from hashtags or emojis alone; the tweet text remains primary."
)

_FORCED = [
    "--task5_strip_hashtag",
    "off",
    "--task5_emoji_mode",
    "off",
    "--task5_postemoji",
    "on",
    "--task5_output_frame",
    "task5vote",
]


def main() -> None:
    sys.argv = [sys.argv[0], *sys.argv[1:], *_FORCED]
    _t5.main()


if __name__ == "__main__":
    main()
