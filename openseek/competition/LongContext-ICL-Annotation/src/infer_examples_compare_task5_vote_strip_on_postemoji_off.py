"""
Task5 对比推理 — 投票融合变体 B。

固定策略：--task5_strip_hashtag on --task5_emoji_mode off --task5_postemoji off
输出文件名含 task5vote（无 postemoji 段），便于与其它变体对齐投票。

Prompt 侧重：去掉 # 后的话题词仅作弱线索，以句面语气为主。
"""

from __future__ import annotations

import sys

import infer_examples_compare_task5 as _t5

_t5.TASK5_CANONICAL_DESCRIPTION = (
    'In this task you are given a tweet. You must judge whether the author of the tweet is sad or not. '
    'Label the instances as "Sad" or "Not sad" based on your judgment. '
    "Hashtags may appear without a leading # after normalization; treat them as weak topical cues only. "
    "Prefer the literal meaning and tone of the sentence when cues disagree."
)

_FORCED = [
    "--task5_strip_hashtag",
    "on",
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
