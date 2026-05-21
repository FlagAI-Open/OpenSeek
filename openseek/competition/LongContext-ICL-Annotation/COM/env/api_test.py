"""Quick API smoke test against the locally hosted Qwen3-4B service.

Endpoint and model id come from ``src/common/paths.py``.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'src'))

from openai import OpenAI
from common.paths import VLLM_BASE_URL, VLLM_MODEL_ID

client = OpenAI(base_url=VLLM_BASE_URL, api_key='EMPTY')

resp = client.chat.completions.create(
    model=VLLM_MODEL_ID,
    messages=[{'role': 'user', 'content': "Say 'hello' and nothing else."}],
    temperature=0.0,
    max_tokens=16,
)
print(resp.choices[0].message.content)
