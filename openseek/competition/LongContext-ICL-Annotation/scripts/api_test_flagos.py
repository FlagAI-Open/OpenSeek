#!/usr/bin/env python3
"""测试本地 FlagOS OpenAI 兼容接口（默认 http://localhost:9010/v1）。"""

from __future__ import annotations

import os
import sys

from openai import OpenAI


def main() -> int:
    api_key = os.environ.get("FLAGSCALE_API_KEY", "EMPTY")
    base_url = os.environ.get("FLAGSCALE_BASE_URL", "http://localhost:9010/v1/").rstrip("/")
    model = os.environ.get("FLAGSCALE_MODEL", "Qwen3-4B-ascend-flagos")

    client = OpenAI(api_key=api_key, base_url=base_url)
    prompt = (
        "You are a data annotation assistant. "
        "Reply with one line only, wrapped in <label> tags: <label>ok</label>"
    )
    print(f"[api_test_flagos] base_url={base_url} model={model}")
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=256,
            extra_body={"enable_thinking": False},
            stream=False,
        )
        text = resp.choices[0].message.content or ""
        print("[response]", text)
        return 0
    except Exception as e:
        print("[error]", e, file=sys.stderr)
        print("请先启动 FlagOS 服务，见 readme.md「模型部署」章节。", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
