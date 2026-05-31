"""
Task5 对比推理 — 投票融合变体 A。

固定策略：--task5_strip_hashtag on --task5_emoji_mode off --task5_postemoji on
输出文件名含 task5vote 与 postemoji-on-，便于与其它变体 JSONL 对齐做多数票。

用法与 infer_examples_compare_task5.py 相同，可在末尾追加任意原脚本支持的参数（本变体参数会覆盖同名传参）。
"""

from __future__ import annotations

import sys

import infer_examples_compare_task5 as _t5

_t5.TASK5_CANONICAL_DESCRIPTION = (
    'In this task you are given a tweet. You must judge whether the author of the tweet is sad or not. '
    'Label the instances as "Sad" or "Not sad" based on your judgment. '
    "You can get help from hashtags and emojis, but you should not judge only based on them, "
    "and should pay attention to tweet's text as well."
)

_FORCED = [
    "--task5_strip_hashtag",
    "on",
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
