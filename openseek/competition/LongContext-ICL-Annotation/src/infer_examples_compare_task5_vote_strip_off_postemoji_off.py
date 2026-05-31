"""
Task5 对比推理 — 投票融合变体 D。

固定策略：--task5_strip_hashtag off --task5_emoji_mode off --task5_postemoji off
不做 emoji 先验覆盖；话题词保留 #。

Prompt 侧重：要求综合 #、表情与正文，避免单一标签或表情盖过清晰文意。
"""

from __future__ import annotations

import sys

import infer_examples_compare_task5 as _t5

_t5.TASK5_CANONICAL_DESCRIPTION = (
    'In this task you are given a tweet. You must judge whether the author of the tweet is sad or not. '
    'Label the instances as "Sad" or "Not sad" based on your judgment. '
    "Read hashtags (with #), emojis, and the text jointly; avoid letting any single hashtag or emoji "
    "overturn a clear textual reading."
)

_FORCED = [
    "--task5_strip_hashtag",
    "off",
    "--task5_emoji_mode",
    "off",
    "--task5_postemoji",
    "off",
    "--task5_output_frame",
    "task5vote",
]


def main() -> None:
    sys.argv = [sys.argv[0], *sys.argv[1:], *_FORCED]
    _t5.main()


if __name__ == "__main__":
    main()
