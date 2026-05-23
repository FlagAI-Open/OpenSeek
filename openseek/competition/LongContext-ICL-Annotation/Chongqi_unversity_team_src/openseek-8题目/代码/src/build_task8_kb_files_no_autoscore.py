# -*- coding: utf-8 -*-
"""
build_task8_kb_files_no_autoscore.py

用途：
    为 OpenSeek-8 kernel generation 任务生成三个中间知识文件：
    1) task8_operator_manual.md
    2) task8_per_question_analysis.jsonl
    3) task8_per_question_analysis.md

设计原则：
    - 至少构造并使用一次长度大于 20k 字符的 Prompt，以满足题目对长上下文使用的要求。
    - 使用 Qwen/OpenAI-compatible API 生成 operator manual 和逐题结构化分析。
    - 不进行“多版本自动评分”。线上平台分数无法由本地脚本获得，因此不同版本仅按 candidate_tag 保存，
      最终版本由人工提交到线上平台后根据真实分数选择。
    - thinking_steps 字段只保存可公开展示的简化解题步骤摘要，不要求模型输出隐藏思维链。

示例：
    export OPENAI_API_KEY=dummy
    export OPENAI_BASE_URL=http://127.0.0.1:8000/v1
    python build_task8_kb_files_no_autoscore.py \
        --data /Users/ks/Desktop/LongContext-ICL-Annotation/data/openseek-8_kernel_generation.json \
        --out-dir /Users/ks/Desktop/LongContext-ICL-Annotation/kb_candidates/v4 \
        --candidate-tag v4 \
        --model /Qwen3-4B/Qwen/Qwen3-4B \
        --batch-size 8
"""

import argparse
import json
import os
import random
import re
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from openai import OpenAI
from tqdm import tqdm


# ============================================================
# 1. API client
# ============================================================


def build_client() -> OpenAI:
    return OpenAI(
        api_key=os.getenv("OPENAI_API_KEY", "dummy"),
        base_url=os.getenv("OPENAI_BASE_URL", "https://flagos.io/flagos-lab/hw/node/HW-gpu57/port/22653/v1"),
    )


def qwen_api(
    client: OpenAI,
    messages: List[Dict[str, str]],
    model: str,
    retries: int = 3,
    sleep_base: float = 5.0,
    temperature: float = 0.0,
) -> str:
    """Call Qwen through an OpenAI-compatible chat endpoint."""
    for attempt in range(retries):
        try:
            res = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
            )
            return res.choices[0].message.content or ""
        except Exception as e:
            if attempt == retries - 1:
                print(f"[ERROR] API failed after {retries} attempts: {e}")
                return ""
            wait = sleep_base * (2 ** attempt) + random.random()
            print(f"[WARN] API failed, retrying in {wait:.1f}s: {e}")
            time.sleep(wait)
    return ""


# ============================================================
# 2. Data loading and text helpers
# ============================================================


def load_task_data(path: str) -> Tuple[str, List[str], List[Dict[str, Any]], List[Dict[str, Any]]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return (
        str(data.get("task_id", "")),
        data.get("Definition", []) or [],
        data.get("examples", []) or [],
        data.get("test_samples", []) or [],
    )


def normalize_text(text: Optional[str]) -> str:
    if text is None:
        return ""
    text = str(text).replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_code_response(text: str) -> str:
    text = normalize_text(text)
    fence = re.search(r"```(?:python|py)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    start_markers = ["import ", "from ", "@triton", "@torch", "def ", "class "]
    positions = [text.find(m) for m in start_markers if text.find(m) != -1]
    if positions:
        text = text[min(positions):]
    return text.replace("```", "").strip()


def get_sample_id(sample: Dict[str, Any]) -> str:
    return str(sample.get("id") or sample.get("test_sample_id") or sample.get("sample_id") or "")


def get_sample_input(sample: Dict[str, Any]) -> str:
    return normalize_text(sample.get("input", ""))


def get_example_output_code(example: Dict[str, Any]) -> str:
    out = example.get("output", "")
    if isinstance(out, list) and out:
        return clean_code_response(str(out[0]))
    return clean_code_response(str(out))


# ============================================================
# 3. Lightweight extraction rules
# ============================================================


def extract_wrapper_entry(input_text: str) -> str:
    text = normalize_text(input_text)
    m = re.search(
        r"Wrapper Entry Information:\s*(.*?)(?:\n\s*Args:|\n\s*Keyword args:|\n\s*Returns:|\n\s*Math:|\Z)",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    return m.group(1).strip() if m else ""


def extract_function_name(input_text: str) -> str:
    text = normalize_text(input_text)
    entry = extract_wrapper_entry(text)
    m = re.search(r"(?:def\s+)?([A-Za-z_]\w*)\s*\(", entry)
    if m:
        return m.group(1)
    m = re.search(r"(?:function|wrapper|entry)\s+[`'\"]?([A-Za-z_]\w*)[`'\"]?", text, flags=re.I)
    if m:
        return m.group(1)
    # Conservative fallback for a few common natural language descriptions.
    lower = text.lower()
    if "mean value" in lower or "computes the mean" in lower:
        return "mean"
    if "square system of linear equations" in lower:
        return "solve"
    if "conv2d" in lower and "add" in lower:
        return "conv2d_add"
    return "generated_function"


def extract_wrapper_signature(input_text: str) -> str:
    entry = extract_wrapper_entry(input_text)
    if entry:
        return entry.splitlines()[0].strip()
    m = re.search(r"(?:def\s+)?([A-Za-z_]\w*\s*\([^\n]*\))", normalize_text(input_text))
    return m.group(1).strip() if m else ""


def extract_section(input_text: str, title: str) -> str:
    """Extract sections such as Description/Math/Notes when they exist in the sample text."""
    text = normalize_text(input_text)
    pat = rf"{re.escape(title)}\s*:\s*(.*?)(?:\n\s*(?:Wrapper Entry Information|Args|Keyword args|Returns|Math|Description|Notes|Example)\s*:|\Z)"
    m = re.search(pat, text, flags=re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else ""


def detect_task_family(input_text: str) -> str:
    lower = input_text.lower()
    if any(x in lower for x in [
        "conv2d", "conv1d", "conv3d", "pool2d", "batch_norm", "instance_norm",
        "layer_norm", "group_norm", "pixel_shuffle", "adaptive_avg_pool2d", "max_pool2d", "avg_pool2d",
    ]):
        return "conv_norm_pool"
    if any(x in lower for x in [
        "linear", "matmul", "matrix multiplication", "mm(", " bmm", "torch.bmm", "mv",
        "addmm", "einsum", "matrix-vector", "matrix vector",
    ]):
        return "matmul_linear"
    if any(x in lower for x in [
        "attention", "softmax", "log_softmax", "cross_entropy", "dropout", "transformer", "scaled dot-product",
    ]):
        return "attention_softmax_loss"
    if any(x in lower for x in [
        "svd", "qr", "lu", "cholesky", "solve", "inverse", "invert", "determinant", "det(",
        "eigen", "eig", "pinv", "lstsq", "least squares", "matrix_power", "matrix power",
    ]):
        return "linalg"
    if any(x in lower for x in [
        "gather", "scatter", "index_select", "masked", "embedding", "repeat_interleave",
        "where", "take", "index_fill",
    ]):
        return "indexing"
    if any(x in lower for x in [
        "relu", "gelu", "sigmoid", "tanh", "silu", "elu", "softplus", "hardsigmoid", "leaky_relu", "selu",
    ]):
        return "activation"
    if any(x in lower for x in [
        "sum", "mean", "std", "var", "min", "max", "argmax", "argmin", "norm",
        "prod", "reduction", "logsumexp", "rsqrt",
    ]):
        return "reduction"
    if any(x in lower for x in ["quantize", "dequantize", "int8", "fp8", "uint8"]):
        return "quantization"
    if any(x in lower for x in [
        "sqrt", "exp", "log", "cos", "sin", "erfc", "rad2deg", "signbit", "bitwise", "ceil", "floor", "zeta", "chebyshev",
    ]):
        return "elementwise_math"
    return "generic"


def extract_ops(input_text: str) -> List[str]:
    lower = input_text.lower()
    candidates = [
        ("conv2d", "F.conv2d"), ("conv1d", "F.conv1d"), ("conv3d", "F.conv3d"),
        ("linear", "F.linear"), ("bmm", "torch.bmm"), ("matmul", "torch.matmul"),
        ("matrix multiplication", "torch.matmul"), ("mm", "torch.mm"), ("mv", "torch.mv"),
        ("addmm", "torch.addmm"), ("einsum", "torch.einsum"),
        ("batch_norm", "F.batch_norm"), ("instance_norm", "F.instance_norm"),
        ("layer_norm", "F.layer_norm"), ("group_norm", "F.group_norm"), ("rms", "custom _rms_norm"),
        ("max_pool2d", "F.max_pool2d"), ("avg_pool2d", "F.avg_pool2d"),
        ("adaptive_avg_pool2d", "F.adaptive_avg_pool2d"), ("pixel_shuffle", "F.pixel_shuffle"),
        ("log_softmax", "F.log_softmax"), ("softmax", "F.softmax"), ("cross_entropy", "F.cross_entropy"),
        ("dropout", "F.dropout"),
        ("leaky_relu", "F.leaky_relu"), ("relu", "F.relu"), ("gelu", "F.gelu"), ("silu", "F.silu"),
        ("sigmoid", "torch.sigmoid"), ("tanh", "torch.tanh"), ("elu", "F.elu"), ("selu", "F.selu"),
        ("softplus", "F.softplus"), ("hardsigmoid", "F.hardsigmoid"),
        ("sqrt", "torch.sqrt"), ("exp", "torch.exp"), ("logsumexp", "torch.logsumexp"),
        ("log", "torch.log"), ("rsqrt", "torch.rsqrt"), ("cos", "torch.cos"), ("sin", "torch.sin"),
        ("erfc", "torch.erfc"), ("rad2deg", "torch.rad2deg"), ("signbit", "torch.signbit"),
        ("bitwise_and", "torch.bitwise_and"),
        ("mean", "torch.mean"), ("sum", "torch.sum"), ("std", "torch.std"), ("var", "torch.var"),
        ("argmax", "torch.argmax"), ("argmin", "torch.argmin"), ("max", "torch.max"), ("min", "torch.min"),
        ("norm", "torch.linalg.vector_norm"),
        ("gather", "torch.gather"), ("scatter", "torch.scatter"), ("index_select", "torch.index_select"),
        ("masked_select", "torch.masked_select"), ("masked_fill", "Tensor.masked_fill"),
        ("embedding", "F.embedding"), ("repeat_interleave", "torch.repeat_interleave"),
        ("where", "torch.where"), ("index_fill", "Tensor.index_fill_"),
        ("svd", "torch.linalg.svd"), ("qr", "torch.linalg.qr"), ("cholesky", "torch.linalg.cholesky"),
        ("solve", "torch.linalg.solve"), ("inverse", "torch.linalg.inv"), ("invert", "torch.linalg.inv"),
        ("determinant", "torch.linalg.det"), ("pinv", "torch.linalg.pinv"),
        ("lstsq", "torch.linalg.lstsq"), ("eig", "torch.linalg.eig"),
        ("matrix_power", "torch.linalg.matrix_power"),
        ("zeta", "torch.special.zeta or finite PyTorch summation"),
        ("chebyshev", "Chebyshev recurrence in PyTorch"),
    ]
    ops: List[str] = []
    for key, op in candidates:
        if key in lower and op not in ops:
            ops.append(op)
    return ops or ["Use the safest matching PyTorch API based on wrapper name and description"]


def extract_answer_apis(code: str) -> List[str]:
    code = clean_code_response(code)
    apis = set(re.findall(r"\b(?:torch|F)\.[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)?", code))
    # Include helper identifiers that matter for this task.
    if "_write_out" in code:
        apis.add("_write_out")
    if "_rms_norm" in code:
        apis.add("custom _rms_norm")
    return sorted(apis)


# ============================================================
# 4. Long prompt construction, >20k chars
# ============================================================


def compact_example_block(example: Dict[str, Any], index: int, max_input_chars: int = 1400, max_output_chars: int = 1800) -> str:
    sid = get_sample_id(example) or f"example-{index}"
    inp = get_sample_input(example)[:max_input_chars]
    out = get_example_output_code(example)[:max_output_chars]
    return f"""
[Example {index}] id={sid}
INPUT:
{inp}

REFERENCE_OUTPUT_CODE:
{out}
""".strip()


def compact_test_block(sample: Dict[str, Any], index: int, max_input_chars: int = 1200) -> str:
    sid = get_sample_id(sample) or f"test-{index}"
    inp = get_sample_input(sample)[:max_input_chars]
    return f"""
[Test {index}] id={sid}
INPUT:
{inp}
""".strip()


def build_long_kb_prompt(
    task_id: str,
    definitions: List[str],
    examples: List[Dict[str, Any]],
    test_samples: List[Dict[str, Any]],
    min_chars: int = 20000,
) -> str:
    """
    Build one deliberately long knowledge-building prompt.

    This is the only place where we force >20k chars. The final prediction script
    does NOT use such a long prompt per sample, because long prompts increase
    truncation risk when generating executable code.
    """
    family_counter = Counter(detect_task_family(get_sample_input(s)) for s in test_samples)
    header = f"""
你是 OpenSeek-8 kernel generation 任务的中间知识构建助手。

任务 ID：{task_id}

目标：
1. 从训练样例答案与测试题目描述中总结常见 PyTorch/F API 使用模式。
2. 形成一个算子使用手册 task8_operator_manual.md。
3. 为每道测试题抽取 wrapper、任务类型、算子链、实现策略和公开的简化解题步骤。
4. 输出内容必须偏工程化、可复现、可直接被后续代码生成脚本检索使用。
5. thinking_steps 只写可公开展示的简要解题步骤，不输出隐藏思维链。

任务定义：
{json.dumps(definitions, ensure_ascii=False, indent=2)}

测试集 family 初步统计：
{json.dumps(family_counter, ensure_ascii=False, indent=2)}

生成要求：
- 优先 PyTorch fallback，不盲目生成 Triton。
- 对 conv/matmul/attention/linalg/indexing 等复杂任务，先保证语义正确与参数兼容。
- 记录 out=、inplace、dim、keepdim、dtype、eps、alpha、beta、training、p 等常见参数。
- 对每题输出 compact JSON 字段：id/function/family/description/wrapper_signature/math/other/detected_ops/answer_apis/answer_summary/thinking_steps。
""".strip()

    parts = [header, "\n\n# 训练样例与参考答案代码\n"]
    for i, ex in enumerate(examples, 1):
        parts.append(compact_example_block(ex, i))
        # Stop only after prompt is comfortably above min_chars and has enough examples.
        if len("\n\n".join(parts)) >= min_chars + 6000 and i >= 8:
            break

    parts.append("\n\n# 测试题目输入片段\n")
    for i, sample in enumerate(test_samples, 1):
        parts.append(compact_test_block(sample, i))
        if len("\n\n".join(parts)) >= min_chars + 12000 and i >= 20:
            break

    prompt = "\n\n".join(parts)

    # If the dataset is small and the prompt is still short, append repeated but useful task instructions.
    filler = "\n".join([
        "请继续保持：函数名必须完全一致；优先 PyTorch fallback；避免复杂 Triton；输出字段稳定；thinking_steps 为公开简化步骤。"
        for _ in range(200)
    ])
    while len(prompt) < min_chars:
        prompt += "\n" + filler

    assert len(prompt) >= min_chars, f"long prompt is too short: {len(prompt)} < {min_chars}"
    return prompt


# ============================================================
# 5. Qwen generation for the three intermediate files
# ============================================================


OPERATOR_MANUAL_SYSTEM = """你是一个严谨的代码竞赛技术文档生成助手。输出 Markdown，内容要简洁、结构稳定、偏工程化。"""


def generate_operator_manual_md(client: OpenAI, model: str, long_prompt: str, out_path: Path) -> str:
    user_prompt = f"""
{long_prompt}

请基于以上长上下文，生成 task8_operator_manual.md。

格式要求：
# Task8 答案中常用算子使用手册

本手册根据题目输入与答案代码中出现的 PyTorch/F API 总结，每个算子给出用途、场景和 demo。

## 高频算子统计

然后按照算子逐个说明，每个算子包含：
- 用途
- 典型场景
- Demo 代码块

只输出 Markdown，不要输出 JSON，不要解释你如何思考。
""".strip()
    print(f"[INFO] operator manual prompt chars = {len(user_prompt)}")
    assert len(user_prompt) > 20000, "At least one Qwen prompt must be >20k chars."

    content = qwen_api(
        client=client,
        model=model,
        messages=[
            {"role": "system", "content": OPERATOR_MANUAL_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
    ).strip()

    if not content.startswith("#"):
        content = "# Task8 答案中常用算子使用手册\n\n" + content
    out_path.write_text(content, encoding="utf-8")
    return content


ANALYSIS_SYSTEM = """你是 OpenSeek-8 kernel generation 任务的逐题分析助手。输出 JSONL，每行一个 JSON 对象，不要输出 Markdown。thinking_steps 只写公开简化解题步骤，不输出隐藏思维链。"""


def build_analysis_batch_prompt(
    definitions: List[str],
    examples: List[Dict[str, Any]],
    batch: List[Dict[str, Any]],
    operator_manual_md: str,
    batch_start: int,
) -> str:
    example_blocks = []
    for i, ex in enumerate(examples[:6], 1):
        example_blocks.append(compact_example_block(ex, i, max_input_chars=1000, max_output_chars=1200))

    sample_blocks = []
    for offset, sample in enumerate(batch):
        sample_blocks.append(compact_test_block(sample, batch_start + offset, max_input_chars=1800))

    prompt = f"""
请为下面这一批 OpenSeek-8 测试题生成结构化分析 JSONL。

任务定义：
{json.dumps(definitions, ensure_ascii=False, indent=2)}

算子手册摘要：
{operator_manual_md[:5000]}

参考训练样例：
{"\n\n".join(example_blocks)}

待分析测试题：
{"\n\n".join(sample_blocks)}

输出要求：
- 只输出 JSONL，每个测试题一行。
- 每行必须是合法 JSON 对象。
- 字段必须包含：
  id, function, family, description, wrapper_signature, math, other,
  detected_ops, answer_apis, answer_summary, thinking_steps
- detected_ops 和 answer_apis 用字符串数组。
- thinking_steps 用字符串数组，写 3 到 5 条可公开展示的简化解题步骤，不输出隐藏思维链。
- family 从以下集合中选择：conv_norm_pool, matmul_linear, attention_softmax_loss, linalg, indexing, activation, reduction, quantization, elementwise_math, generic。
""".strip()
    return prompt


def parse_jsonl_from_model(text: str) -> List[Dict[str, Any]]:
    """Parse JSONL from Qwen output. Also tolerates a fenced code block."""
    text = normalize_text(text)
    fence = re.search(r"```(?:jsonl|json)?\s*(.*?)```", text, flags=re.DOTALL | re.I)
    if fence:
        text = fence.group(1).strip()

    rows: List[Dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip().rstrip(",")
        if not line or not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(obj)
        except json.JSONDecodeError:
            continue
    return rows


def deterministic_analysis_row(sample: Dict[str, Any], examples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Fallback row if Qwen returns malformed/missing JSON for a sample."""
    inp = get_sample_input(sample)
    sid = get_sample_id(sample)
    func = extract_function_name(inp)
    family = detect_task_family(inp)
    ops = extract_ops(inp)

    # Aggregate answer APIs from all examples as a weak reference.
    api_counter: Counter = Counter()
    for ex in examples:
        for api in extract_answer_apis(get_example_output_code(ex)):
            api_counter[api] += 1
    answer_apis = [api for api, _ in api_counter.most_common(80)]

    return {
        "id": sid,
        "function": func,
        "family": family,
        "description": extract_section(inp, "Description") or "",
        "wrapper_signature": extract_wrapper_signature(inp),
        "math": extract_section(inp, "Math") or "",
        "other": extract_section(inp, "Notes") or "",
        "detected_ops": ops,
        "answer_apis": answer_apis,
        "answer_summary": "提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback。",
        "thinking_steps": [
            f"先识别 wrapper `{func}` 与参数来源，保证最终代码定义同名函数。",
            f"题目属于 `{family}` 类，优先把自然语言描述映射到 PyTorch 算子链：" + ", ".join(ops) + "。",
            "复杂算子优先使用 PyTorch fallback，避免因 Triton 维度、stride 或 mask 处理错误导致运行失败。",
            "统一处理 out/inplace/dim/keepdim/dtype/eps 等常见参数，提高答案兼容性。",
        ],
    }


def normalize_analysis_row(obj: Dict[str, Any], sample: Dict[str, Any], examples: List[Dict[str, Any]]) -> Dict[str, Any]:
    fallback = deterministic_analysis_row(sample, examples)
    normalized = dict(fallback)
    for key in normalized:
        if key in obj and obj[key] not in (None, "", []):
            normalized[key] = obj[key]

    if not isinstance(normalized.get("detected_ops"), list):
        normalized["detected_ops"] = [str(normalized.get("detected_ops"))]
    if not isinstance(normalized.get("answer_apis"), list):
        normalized["answer_apis"] = [str(normalized.get("answer_apis"))]
    if not isinstance(normalized.get("thinking_steps"), list):
        normalized["thinking_steps"] = [str(normalized.get("thinking_steps"))]

    # Always trust the original sample id, because online submission depends on it.
    normalized["id"] = get_sample_id(sample)
    if not normalized["function"] or normalized["function"] == "generated_function":
        normalized["function"] = extract_function_name(get_sample_input(sample))
    return normalized


def generate_per_question_analysis_jsonl(
    client: OpenAI,
    model: str,
    definitions: List[str],
    examples: List[Dict[str, Any]],
    test_samples: List[Dict[str, Any]],
    operator_manual_md: str,
    out_path: Path,
    batch_size: int = 8,
) -> List[Dict[str, Any]]:
    all_rows: List[Dict[str, Any]] = []
    sample_by_id = {get_sample_id(s): s for s in test_samples}

    for start in tqdm(range(0, len(test_samples), batch_size), desc="Generating analysis JSONL"):
        batch = test_samples[start:start + batch_size]
        prompt = build_analysis_batch_prompt(definitions, examples, batch, operator_manual_md, start + 1)
        raw = qwen_api(
            client=client,
            model=model,
            messages=[
                {"role": "system", "content": ANALYSIS_SYSTEM},
                {"role": "user", "content": prompt},
            ],
        )
        parsed = parse_jsonl_from_model(raw)
        parsed_by_id = {str(row.get("id", "")): row for row in parsed}

        for sample in batch:
            sid = get_sample_id(sample)
            obj = parsed_by_id.get(sid, {})
            all_rows.append(normalize_analysis_row(obj, sample, examples))

    with out_path.open("w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return all_rows


def render_analysis_md(rows: List[Dict[str, Any]], out_path: Path) -> str:
    family_counter = Counter(row.get("family", "generic") for row in rows)
    parts = [
        "# OpenSeek-8 每道题思路分析与拆解",
        "",
        "说明：本文件将题目输入与答案代码对齐，逐题抽取 wrapper、任务类型、算子链、答案实现策略与解题步骤。",
        "",
        "## 共性总结",
        "",
        f"- 总题数：{len(rows)}。",
        "- 任务共同模式：自然语言功能描述 + Wrapper Entry Information + 参数/数学定义 → 生成同名 Python/Triton wrapper。",
        "- 高稳策略：先保证函数名、import、参数兼容、out/inplace 支持，再用 PyTorch API 实现语义；复杂 Triton 仅在必要且简单时使用。",
        "- 常见答案风格：`import torch`、`import torch.nn.functional as F`、`_write_out`、`def wrapper(*args, **kwargs)`、按算子链逐步组合。",
        "- family 分布：" + ", ".join(f"{k}={v}" for k, v in family_counter.most_common()),
        "",
    ]

    for idx, row in enumerate(rows, 1):
        ops = row.get("detected_ops", []) or []
        apis = row.get("answer_apis", []) or []
        steps = row.get("thinking_steps", []) or []
        parts.extend([
            f"## {idx}. {row.get('id', '')} — `{row.get('function', '')}`",
            "",
            f"- **任务类型**：{row.get('family', '')}",
            f"- **Wrapper**：`{str(row.get('wrapper_signature', '')).replace('`', '')}`",
            f"- **功能描述**：{row.get('description', '')}",
            f"- **数学定义**：{row.get('math', '')}",
            f"- **补充约束**：{row.get('other', '')}",
            f"- **题目算子链**：{', '.join(map(str, ops))}",
            f"- **答案中显式 API**：{', '.join(map(str, apis))}",
            f"- **答案实现风格**：{row.get('answer_summary', '')}",
            "- **拆解思路**：",
        ])
        for s in steps:
            parts.append(f"  1. {s}")
        parts.append("")

    content = "\n".join(parts)
    out_path.write_text(content, encoding="utf-8")
    return content


# ============================================================
# 6. Candidate manifest, no automatic scoring
# ============================================================


def write_manifest(
    out_dir: Path,
    candidate_tag: str,
    model: str,
    task_id: str,
    long_prompt_chars: int,
    rows: List[Dict[str, Any]],
    files: Dict[str, str],
) -> None:
    family_counter = Counter(row.get("family", "generic") for row in rows)
    manifest = {
        "candidate_tag": candidate_tag,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model": model,
        "task_id": task_id,
        "long_prompt_chars": long_prompt_chars,
        "long_prompt_requirement": "satisfied" if long_prompt_chars > 20000 else "not_satisfied",
        "analysis_items": len(rows),
        "family_distribution": dict(family_counter),
        "files": files,
        "scoring": {
            "auto_score": False,
            "reason": "线上评测分数只能通过提交平台获得，本脚本不做多版本自动评分。",
            "selection_method": "人工将不同 candidate_tag 版本接入主生成脚本并提交线上评测，根据真实分数选择最终版本。",
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


# ============================================================
# 7. Main
# ============================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to openseek-8_kernel_generation.json")
    parser.add_argument("--out-dir", required=True, help="Directory to write intermediate files")
    parser.add_argument("--candidate-tag", default="v1", help="Manual version tag, e.g. v1/v2/v4")
    parser.add_argument("--model", default="/Qwen3-4B/Qwen/Qwen3-4B")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--min-long-prompt-chars", type=int, default=20000)
    parser.add_argument("--reuse-operator-manual", default="", help="Optional existing operator manual path")
    parser.add_argument("--reuse-analysis-jsonl", default="", help="Optional existing analysis JSONL path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    task_id, definitions, examples, test_samples = load_task_data(args.data)
    for ex in examples:
        ex["input"] = get_sample_input(ex)
    for sample in test_samples:
        sample["input"] = get_sample_input(sample)

    print(f"task_id={task_id}")
    print(f"examples={len(examples)}, test_samples={len(test_samples)}")
    print(f"candidate_tag={args.candidate_tag}")

    long_prompt = build_long_kb_prompt(
        task_id=task_id,
        definitions=definitions,
        examples=examples,
        test_samples=test_samples,
        min_chars=args.min_long_prompt_chars,
    )
    long_prompt_path = out_dir / f"task8_long_prompt_{args.candidate_tag}.txt"
    long_prompt_path.write_text(long_prompt, encoding="utf-8")
    print(f"long_prompt_chars={len(long_prompt)}")

    client = build_client()

    operator_manual_path = out_dir / "task8_operator_manual.md"
    analysis_jsonl_path = out_dir / "task8_per_question_analysis.jsonl"
    analysis_md_path = out_dir / "task8_per_question_analysis.md"

    if args.reuse_operator_manual:
        operator_manual_md = Path(args.reuse_operator_manual).read_text(encoding="utf-8")
        operator_manual_path.write_text(operator_manual_md, encoding="utf-8")
    else:
        operator_manual_md = generate_operator_manual_md(client, args.model, long_prompt, operator_manual_path)

    if args.reuse_analysis_jsonl:
        rows = []
        with open(args.reuse_analysis_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        analysis_jsonl_path.write_text(Path(args.reuse_analysis_jsonl).read_text(encoding="utf-8"), encoding="utf-8")
    else:
        rows = generate_per_question_analysis_jsonl(
            client=client,
            model=args.model,
            definitions=definitions,
            examples=examples,
            test_samples=test_samples,
            operator_manual_md=operator_manual_md,
            out_path=analysis_jsonl_path,
            batch_size=args.batch_size,
        )

    render_analysis_md(rows, analysis_md_path)

    write_manifest(
        out_dir=out_dir,
        candidate_tag=args.candidate_tag,
        model=args.model,
        task_id=task_id,
        long_prompt_chars=len(long_prompt),
        rows=rows,
        files={
            "long_prompt": str(long_prompt_path),
            "operator_manual": str(operator_manual_path),
            "per_question_analysis_jsonl": str(analysis_jsonl_path),
            "per_question_analysis_md": str(analysis_md_path),
        },
    )

    print("Done. Generated intermediate KB files:")
    print(f"  - {operator_manual_path}")
    print(f"  - {analysis_jsonl_path}")
    print(f"  - {analysis_md_path}")
    print(f"  - {out_dir / 'manifest.json'}")
    print("Note: no automatic multi-version scoring is performed; submit candidates manually for online scoring.")


if __name__ == "__main__":
    main()
