import ast
import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI
from tqdm import tqdm


# ============================================================
# 0. User config
# ============================================================

FILE_PATH = "/Users/ks/Desktop/LongContext-ICL-Annotation/data/openseek-8_kernel_generation.json"
OUT_PATH = "/Users/ks/Desktop/LongContext-ICL-Annotation/outputs/experiment/task8_prompt_kb_fixed_v3_submit.jsonl"

# Prompt knowledge files generated earlier
REASONING_PROMPT_PATH = "/Users/ks/Desktop/LongContext-ICL-Annotation/task8_reasoning_prompt.md"
OPERATOR_MANUAL_PATH = "/Users/ks/Desktop/LongContext-ICL-Annotation/task8_operator_manual.md"
PER_QUESTION_ANALYSIS_PATH = "/Users/ks/Desktop/LongContext-ICL-Annotation/task8_per_question_analysis.jsonl"

MODEL_NAME = "/Qwen3-4B/Qwen/Qwen3-4B"

# Qwen-4B + gateway: start with 1 for stability.
MAX_WORKERS = 8

USE_MODEL = True
ENABLE_REPAIR = True
ENABLE_RULE_FALLBACK = True

# Keep prompt short. Long prompt caused truncated code like:
# def _write_out(...):
#     if out is None:
MAX_REASONING_PROMPT_CHARS = 2500
MAX_OPERATOR_MANUAL_CHARS = 2500
MAX_QUESTION_ANALYSIS_CHARS = 1200


# ============================================================
# 1. API client
# ============================================================

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY", "dummy"),
    base_url=os.getenv(
        "OPENAI_BASE_URL",
        "https://flagos.io/flagos-lab/hw/node/HW-gpu57/port/22653/v1",
    ),
)


# ============================================================
# 2. IO helpers
# ============================================================

def task_data_loader(file_path: str) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return (
        data.get("task_id", ""),
        data.get("examples", []),
        data.get("test_samples", []),
        data.get("Definition", []),
    )


def load_text_file(path: str, default: str = "") -> str:
    if not path or not os.path.exists(path):
        print(f"[WARN] text file not found: {path}")
        return default
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def load_question_analysis(path: str) -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {}

    if not path or not os.path.exists(path):
        print(f"[WARN] question analysis file not found: {path}")
        return results

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                print(f"[WARN] skip invalid analysis JSONL line {line_no}")
                continue

            sid = obj.get("id") or obj.get("test_sample_id") or obj.get("sample_id")
            if sid:
                results[str(sid)] = obj

    return results


def append_jsonl(out_path: str, row: Dict[str, Any]) -> None:
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_existing_predictions(out_path: str) -> Dict[str, Dict[str, Any]]:
    existing: Dict[str, Dict[str, Any]] = {}

    if not os.path.exists(out_path):
        return existing

    with open(out_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                print(f"[WARN] skip invalid JSONL output line {line_no}")
                continue

            sid = obj.get("test_sample_id") or obj.get("id")
            pred = obj.get("prediction", [])

            if sid and isinstance(pred, list) and pred and str(pred[0]).strip():
                existing[str(sid)] = obj

    return existing


def safe_jsonl_row(test_sample_id: str, code: str) -> Dict[str, Any]:
    return {"test_sample_id": test_sample_id, "prediction": code}


# ============================================================
# 3. Text/code cleaning
# ============================================================

def normalize_text(text: Optional[str]) -> str:
    if text is None:
        return ""

    text = str(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_code_response(text: str) -> str:
    if not text:
        return ""

    text = str(text).strip()

    fence = re.search(
        r"```(?:python|py)?\s*(.*?)```",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if fence:
        text = fence.group(1).strip()

    start_markers = ["import ", "from ", "@triton", "@torch", "def ", "class "]
    positions = [text.find(m) for m in start_markers if text.find(m) != -1]
    if positions:
        text = text[min(positions):].strip()

    return text.replace("```", "").strip()


# ============================================================
# 4. Sample parsing / rule extraction
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

    m = re.search(
        r"(?:function|wrapper|entry)\s+[`'\"]?([A-Za-z_]\w*)[`'\"]?",
        text,
        flags=re.IGNORECASE,
    )
    if m:
        return m.group(1)

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
    return ""


def detect_task_family(input_text: str) -> str:
    lower = input_text.lower()

    if any(x in lower for x in [
        "conv2d", "conv1d", "conv3d", "pool2d", "batch_norm",
        "instance_norm", "layer_norm", "group_norm", "pixel_shuffle",
        "adaptive_avg_pool2d", "max_pool2d", "avg_pool2d"
    ]):
        return "conv_norm_pool"

    if any(x in lower for x in [
        "linear", "matmul", "matrix multiplication", "mm(", " bmm",
        "torch.bmm", "mv", "addmm", "einsum", "matrix-vector", "matrix vector"
    ]):
        return "matmul_linear"

    if any(x in lower for x in [
        "attention", "softmax", "log_softmax", "cross_entropy",
        "dropout", "transformer", "scaled dot-product"
    ]):
        return "attention_softmax_loss"

    if any(x in lower for x in [
        "svd", "qr", "lu", "cholesky", "solve", "inverse", "invert",
        "determinant", "det(", "eigen", "eig", "pinv", "lstsq",
        "least squares", "matrix_power", "matrix power"
    ]):
        return "linalg"

    if any(x in lower for x in [
        "gather", "scatter", "index_select", "masked", "embedding",
        "repeat_interleave", "where", "take", "index_fill"
    ]):
        return "indexing"

    if any(x in lower for x in [
        "relu", "gelu", "sigmoid", "tanh", "silu", "elu",
        "softplus", "hardsigmoid", "hard sigmoid", "leaky_relu", "selu"
    ]):
        return "activation"

    if any(x in lower for x in [
        "sum", "mean", "std", "var", "min", "max", "argmax",
        "argmin", "norm", "prod", "reduction", "logsumexp", "rsqrt"
    ]):
        return "reduction"

    if any(x in lower for x in ["quantize", "dequantize", "int8", "fp8", "uint8"]):
        return "quantization"

    if any(x in lower for x in [
        "sqrt", "exp", "log", "cos", "sin", "erfc", "rad2deg",
        "signbit", "bitwise", "ceil", "floor", "zeta", "chebyshev"
    ]):
        return "elementwise_math"

    return "generic"


def extract_ops(input_text: str) -> List[str]:
    lower = input_text.lower()
    ops: List[str] = []

    candidates = [
        ("conv2d", "F.conv2d"),
        ("conv1d", "F.conv1d"),
        ("conv3d", "F.conv3d"),
        ("linear", "F.linear"),
        ("bmm", "torch.bmm"),
        ("matmul", "torch.matmul"),
        ("matrix multiplication", "torch.matmul"),
        ("mm", "torch.mm"),
        ("mv", "torch.mv"),
        ("addmm", "torch.addmm"),
        ("einsum", "torch.einsum"),
        ("batch_norm", "F.batch_norm"),
        ("instance_norm", "F.instance_norm"),
        ("layer_norm", "F.layer_norm"),
        ("group_norm", "F.group_norm"),
        ("rms", "custom _rms_norm"),
        ("max_pool2d", "F.max_pool2d"),
        ("avg_pool2d", "F.avg_pool2d"),
        ("adaptive_avg_pool2d", "F.adaptive_avg_pool2d"),
        ("pixel_shuffle", "F.pixel_shuffle"),
        ("log_softmax", "F.log_softmax"),
        ("softmax", "F.softmax"),
        ("cross_entropy", "F.cross_entropy"),
        ("dropout", "F.dropout"),
        ("leaky_relu", "F.leaky_relu"),
        ("relu", "F.relu"),
        ("gelu", "F.gelu"),
        ("silu", "F.silu"),
        ("sigmoid", "torch.sigmoid"),
        ("tanh", "torch.tanh"),
        ("elu", "F.elu"),
        ("selu", "F.selu"),
        ("softplus", "F.softplus"),
        ("hardsigmoid", "F.hardsigmoid"),
        ("sqrt", "torch.sqrt"),
        ("exp", "torch.exp"),
        ("logsumexp", "torch.logsumexp"),
        ("log", "torch.log"),
        ("rsqrt", "torch.rsqrt"),
        ("cos", "torch.cos"),
        ("sin", "torch.sin"),
        ("erfc", "torch.erfc"),
        ("rad2deg", "torch.rad2deg"),
        ("signbit", "torch.signbit"),
        ("bitwise_and", "torch.bitwise_and"),
        ("mean", "torch.mean"),
        ("sum", "torch.sum"),
        ("std", "torch.std"),
        ("var", "torch.var"),
        ("argmax", "torch.argmax"),
        ("argmin", "torch.argmin"),
        ("max", "torch.max"),
        ("min", "torch.min"),
        ("norm", "torch.linalg.vector_norm"),
        ("gather", "torch.gather"),
        ("scatter", "torch.scatter"),
        ("index_select", "torch.index_select"),
        ("masked_select", "torch.masked_select"),
        ("masked_fill", "Tensor.masked_fill"),
        ("embedding", "F.embedding"),
        ("repeat_interleave", "torch.repeat_interleave"),
        ("where", "torch.where"),
        ("index_fill", "Tensor.index_fill_"),
        ("svd", "torch.linalg.svd"),
        ("qr", "torch.linalg.qr"),
        ("cholesky", "torch.linalg.cholesky"),
        ("solve", "torch.linalg.solve"),
        ("inverse", "torch.linalg.inv"),
        ("invert", "torch.linalg.inv"),
        ("determinant", "torch.linalg.det"),
        ("pinv", "torch.linalg.pinv"),
        ("lstsq", "torch.linalg.lstsq"),
        ("eig", "torch.linalg.eig"),
        ("matrix_power", "torch.linalg.matrix_power"),
        ("zeta", "torch.special.zeta or finite PyTorch summation"),
        ("chebyshev", "Chebyshev recurrence in PyTorch"),
    ]

    for key, op in candidates:
        if key in lower and op not in ops:
            ops.append(op)

    return ops or ["Use the safest matching PyTorch API based on wrapper name and description"]


def build_strategy(family: str, ops: List[str]) -> str:
    if family in {
        "conv_norm_pool", "matmul_linear", "attention_softmax_loss",
        "linalg", "indexing", "activation", "reduction"
    }:
        return (
            "Use PyTorch fallback only. Do not use complex Triton. "
            "Compose the detected operations step by step in the listed order."
        )

    if family == "quantization":
        return (
            "Prefer PyTorch arithmetic implementation. Avoid complex Triton unless the operation is simple row-wise dequantization."
        )

    if family == "elementwise_math":
        return (
            "Use PyTorch elementwise operations. Triton is allowed only for a very simple elementwise kernel, but PyTorch fallback is preferred."
        )

    return "Use the safest PyTorch fallback implementation. Do not generate complex Triton."


def build_ops_text(ops: List[str]) -> str:
    return "\n".join(f"{i + 1}. {op}" for i, op in enumerate(ops))


# ============================================================
# 5. Prompt knowledge selection
# ============================================================

def truncate_text(text: str, max_chars: int) -> str:
    text = normalize_text(text)
    if len(text) <= max_chars:
        return text
    # keep front only; tail often contains irrelevant repeated content
    return text[:max_chars] + "\n\n...[TRUNCATED]..."


def select_operator_manual_section(operator_manual: str, family: str, max_chars: int = MAX_OPERATOR_MANUAL_CHARS) -> str:
    manual = normalize_text(operator_manual)
    if not manual:
        return ""

    keywords_by_family = {
        "activation": ["ReLU", "Sqrt", "Exp", "Log", "Sigmoid", "Tanh", "GELU", "Softplus"],
        "elementwise_math": ["Sqrt", "Exp", "Log", "Sigmoid", "Tanh", "zeta", "chebyshev", "rsqrt"],
        "reduction": ["Reduction", "max", "logsumexp", "rsqrt", "sum_std", "std", "mean", "norm"],
        "indexing": ["Index", "Index Fill", "Index Select", "gather", "masked", "赋值"],
        "linalg": ["solve", "svd", "qr", "cholesky", "det", "inverse", "linalg"],
        "matmul_linear": ["linear", "bmm", "matmul", "mv", "matrix", "softplus_linear", "fused_mv"],
        "conv_norm_pool": ["conv", "conv2d", "batch_norm", "pool", "normalization"],
        "attention_softmax_loss": ["attention", "softmax", "cross_entropy", "dropout"],
        "quantization": ["quantize", "dequantize", "int8", "fp8"],
    }

    keywords = keywords_by_family.get(family, [])
    if not keywords:
        return truncate_text(manual, max_chars)

    selected_lines = []
    for line in manual.splitlines():
        low = line.lower()
        if any(k.lower() in low for k in keywords):
            selected_lines.append(line)

    selected = "\n".join(selected_lines).strip()
    if len(selected) < 300:
        selected = manual

    return truncate_text(selected, max_chars)


def select_question_analysis(sample_id: str, question_analysis: Dict[str, Dict[str, Any]]) -> str:
    obj = question_analysis.get(sample_id, {})
    if not obj:
        return "No precomputed analysis for this sample."
    text = json.dumps(obj, ensure_ascii=False, indent=2)
    return truncate_text(text, MAX_QUESTION_ANALYSIS_CHARS)


# ============================================================
# 6. Fixed helper and prompt templates
# ============================================================

FIXED_HELPER_CODE = """def _write_out(value, out=None):
    if out is None:
        return value
    if isinstance(value, tuple):
        if isinstance(out, tuple):
            for dst, src in zip(out, value):
                dst.copy_(src)
            return out
        return value
    out.copy_(value)
    return out
"""


SYSTEM_PROMPT = """
You are a code-generation model for OpenSeek-8 kernel generation.

Generate one complete executable Python code answer.

Priority:
1. Correctness
2. Importability
3. Exact wrapper function name
4. Robust argument handling
5. PyTorch semantic equivalence
6. Performance

Important:
- Prefer PyTorch fallback over complex Triton.
- Do not blindly write Triton.
- Use Triton only for very simple elementwise kernels.
- Output only Python code.
- No markdown.
- No JSON.
- No explanations.
""".strip()


USER_PROMPT_TEMPLATE = """
Solve this OpenSeek-8 test sample.

A. Short solving guide:
{reasoning_prompt}

B. Relevant operator manual:
{operator_manual_for_sample}

C. Extracted rules for current sample:
Sample ID: {sample_id}
Wrapper function name: {function_name}
Original wrapper signature: {wrapper_signature}
Task family: {family}
Operation chain:
{ops_text}
Strategy: {strategy}

Precomputed analysis:
{analysis_text}

D. Current original test sample:
{current_input}

E. Required output:
Generate ONLY executable Python code.

The code MUST:
- start with import torch
- include import torch.nn.functional as F when useful
- include this exact helper code:

{fixed_helper_code}

- define exactly:
  def {function_name}(*args, **kwargs):
- support positional args and kwargs
- support out= if present
- compose PyTorch operations step by step
- avoid complex Triton
- not be a minimal one-line wrapper
- not output explanation or markdown
- never write assignment expressions inside function call arguments
- use separate assignment lines before function calls
- never write code like: func(arg = value = other)

Final answer:
""".strip()


def build_messages(
    sample: Dict[str, Any],
    definitions: List[str],
    examples: List[Dict[str, Any]],
    reasoning_prompt: str,
    operator_manual: str,
    question_analysis: Dict[str, Dict[str, Any]],
) -> List[Dict[str, str]]:
    current_input = normalize_text(sample.get("input", ""))
    sample_id = str(sample.get("id", ""))

    function_name = extract_function_name(current_input)
    wrapper_signature = extract_wrapper_signature(current_input)
    family = detect_task_family(current_input)
    ops = extract_ops(current_input)
    strategy = build_strategy(family, ops)
    ops_text = build_ops_text(ops)

    reasoning_prompt_for_sample = truncate_text(reasoning_prompt, MAX_REASONING_PROMPT_CHARS)
    operator_manual_for_sample = select_operator_manual_section(operator_manual, family)
    analysis_text = select_question_analysis(sample_id, question_analysis)

    user_prompt = USER_PROMPT_TEMPLATE.format(
        reasoning_prompt=reasoning_prompt_for_sample or "No global reasoning prompt provided.",
        operator_manual_for_sample=operator_manual_for_sample or "No operator manual provided.",
        sample_id=sample_id,
        function_name=function_name,
        wrapper_signature=wrapper_signature or "(not found, use extracted function name)",
        family=family,
        ops_text=ops_text,
        strategy=strategy,
        analysis_text=analysis_text,
        current_input=current_input,
        fixed_helper_code=FIXED_HELPER_CODE,
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


# ============================================================
# 7. Validation
# ============================================================

def has_complex_or_suspicious_triton(code: str, family: str) -> bool:
    if "@triton.jit" not in code and "triton.language" not in code:
        return False

    complex_families = {
        "conv_norm_pool",
        "matmul_linear",
        "attention_softmax_loss",
        "linalg",
        "indexing",
    }
    if family in complex_families:
        return True

    suspicious_patterns = [
        r"isinstance\s*\(",
        r"\.data_ptr\s*\(",
        r"input_shape",
        r"weight_shape",
        r"if\s+.*\s+is\s+not\s+None",
    ]
    return any(re.search(pat, code) for pat in suspicious_patterns)


def validate_code_basic(code: str, expected_func_name: str, family: str) -> Tuple[bool, str]:
    if not code or not code.strip():
        return False, "empty output"

    if "```" in code:
        return False, "markdown fence remains"

    if re.search(r"\bTODO\b|\.\.\.", code):
        return False, "placeholder detected"

    if re.search(r"def\s+\w+\s*\([^)]*\):\s*\n\s*pass\s*(?:\n|$)", code):
        return False, "pass-only function detected"

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"syntax error: {e}"

    func_names = {
        node.name for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    class_names = {
        node.name for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef)
    }

    if expected_func_name not in func_names and expected_func_name not in class_names:
        return False, f"expected wrapper `{expected_func_name}` not defined"

    if has_complex_or_suspicious_triton(code, family):
        return False, f"suspicious Triton generated for family `{family}`"

    # Length check should not apply to linalg:
    # many correct linalg wrappers are naturally short, e.g. torch.linalg.solve/svd/qr.
    length_sensitive_families = {
        "conv_norm_pool",
        "matmul_linear",
        "attention_softmax_loss",
        "indexing",
    }

    # Avoid too short generic stubs for truly complex fused/indexing/attention tasks.
    if family in length_sensitive_families and len(code) < 500:
        return False, f"code too short for complex family `{family}`"

    return True, "ok"


def is_fatal_truncated_syntax(reason: str) -> bool:
    """
    Treat every SyntaxError from the model as fatal and use rule fallback directly.

    For Qwen-4B, repairing syntax errors often wastes time and may produce another
    malformed answer. The rule fallback is usually more stable.
    """
    return "syntax error" in reason.lower()


# ============================================================
# 8. Rule fallback generator
# ============================================================

FALLBACK_TEMPLATE = """
import torch
import torch.nn.functional as F

def _write_out(value, out=None):
    if out is None:
        return value
    if isinstance(value, tuple):
        if isinstance(out, tuple):
            for dst, src in zip(out, value):
                dst.copy_(src)
            return out
        return value
    out.copy_(value)
    return out

def _rms_norm(x, normalized_shape=None, eps=1e-5, weight=None):
    if normalized_shape is None:
        dims = (-1,)
    elif isinstance(normalized_shape, int):
        dims = (-1,)
    else:
        dims = tuple(range(x.dim() - len(tuple(normalized_shape)), x.dim()))
    y = x * torch.rsqrt(torch.mean(x * x, dim=dims, keepdim=True) + eps)
    if weight is not None:
        y = y * weight
    return y

def {func_name}(*args, **kwargs):
    out = kwargs.pop("out", None)
    name = "{func_name}"

    if hasattr(torch, name):
        return _write_out(getattr(torch, name)(*args, **kwargs), out)
    if hasattr(torch.linalg, name):
        return _write_out(getattr(torch.linalg, name)(*args, **kwargs), out)
    if hasattr(torch.special, name):
        return _write_out(getattr(torch.special, name)(*args, **kwargs), out)
    if hasattr(F, name):
        return _write_out(getattr(F, name)(*args, **kwargs), out)

    if "conv2d" in name:
        input = kwargs.get("input", args[0] if len(args) > 0 else None)
        weight = kwargs.get("weight", args[1] if len(args) > 1 else None)
        bias = kwargs.get("bias", args[2] if len(args) > 2 else None)
        stride = kwargs.get("stride", 1)
        padding = kwargs.get("padding", 0)
        dilation = kwargs.get("dilation", 1)
        groups = kwargs.get("groups", 1)
        y = F.conv2d(input, weight, bias, stride, padding, dilation, groups)
        if "add" in name:
            other = kwargs.get("other", args[3] if len(args) > 3 else None)
            if other is not None:
                y = y + kwargs.get("alpha", 1) * other
        if "relu" in name:
            y = F.relu(y, inplace=kwargs.get("inplace", False))
        if "gelu" in name:
            y = F.gelu(y, approximate=kwargs.get("approximate", "none"))
        if "sigmoid" in name:
            y = torch.sigmoid(y)
        return _write_out(y, out)

    if name in ("relu_sqrt", "sqrt_tanh", "sqrt_exp", "exp_sqrt", "log_tanh"):
        x = kwargs.get("input", args[0] if args else None)
        if name == "relu_sqrt":
            y = torch.sqrt(F.relu(x, inplace=kwargs.get("inplace", False)))
        elif name == "sqrt_tanh":
            y = torch.tanh(torch.sqrt(x))
        elif name == "sqrt_exp":
            y = torch.exp(torch.sqrt(x))
        elif name == "exp_sqrt":
            y = torch.sqrt(torch.exp(x))
        else:
            y = torch.tanh(torch.log(x))
        return _write_out(y, out)

    if name in ("add_gelu", "sub_gelu", "mul_relu", "mul_sub"):
        x = args[0]
        other = args[1]
        if name == "add_gelu":
            y = F.gelu(x + kwargs.get("alpha", 1) * other, approximate=kwargs.get("approximate", "none"))
        elif name == "sub_gelu":
            y = F.gelu(x - kwargs.get("alpha", 1) * other, approximate=kwargs.get("approximate", "none"))
        elif name == "mul_relu":
            y = F.relu(x * other, inplace=kwargs.get("inplace", False))
        else:
            y = x * other - kwargs.get("alpha", 1) * args[2]
        return _write_out(y, out)

    if name in ("softmax_log", "softmax_mul", "sigmoid_argmax"):
        x = args[0]
        if name == "softmax_log":
            y = torch.log(F.softmax(x, dim=kwargs.get("dim", -1), dtype=kwargs.get("dtype", None)))
        elif name == "softmax_mul":
            y = F.softmax(x, dim=kwargs.get("dim", -1), dtype=kwargs.get("dtype", None)) * args[1]
        else:
            y = torch.argmax(torch.sigmoid(x), dim=kwargs.get("dim", None), keepdim=kwargs.get("keepdim", False))
        return _write_out(y, out)

    if name in ("log_softmax_linear", "softplus_linear", "tanh_linear", "elu_linear", "dropout_sigmoid_linear"):
        input = args[0]
        weight = args[1]
        bias = args[2] if len(args) > 2 else kwargs.get("bias", None)
        y = F.linear(input, weight, bias)
        if name == "log_softmax_linear":
            y = F.log_softmax(y, dim=kwargs.get("dim", -1), dtype=kwargs.get("dtype", None))
        elif name == "softplus_linear":
            y = F.softplus(y, beta=kwargs.get("beta", 1), threshold=kwargs.get("threshold", 20))
        elif name == "tanh_linear":
            y = torch.tanh(y)
        elif name == "elu_linear":
            y = F.elu(y, alpha=kwargs.get("alpha", 1.0), inplace=kwargs.get("inplace", False))
        else:
            y = F.dropout(torch.sigmoid(y), p=kwargs.get("p", 0.5), training=kwargs.get("training", True), inplace=kwargs.get("inplace", False))
        return _write_out(y, out)

    if name in ("sum_std", "add_mean", "gelu_std", "exp_mean", "min_gelu", "gelu_min"):
        x = args[0]
        dim = kwargs.get("dim", None)
        keepdim = kwargs.get("keepdim", False)
        if name == "sum_std":
            y = torch.sum(x, dim=dim, keepdim=keepdim, dtype=kwargs.get("dtype", None)) + torch.std(x, dim=dim, keepdim=keepdim, correction=kwargs.get("correction", 1))
        elif name == "add_mean":
            y = torch.mean(x + kwargs.get("alpha", 1) * args[1], dim=dim, keepdim=keepdim, dtype=kwargs.get("dtype", None))
        elif name == "gelu_std":
            y = torch.std(F.gelu(x, approximate=kwargs.get("approximate", "none")), dim=dim, keepdim=keepdim, correction=kwargs.get("correction", 1))
        elif name == "exp_mean":
            y = torch.mean(torch.exp(x), dim=dim, keepdim=keepdim, dtype=kwargs.get("dtype", None))
        elif name == "min_gelu":
            y = torch.min(F.gelu(x, approximate=kwargs.get("approximate", "none")), dim=dim, keepdim=keepdim).values if dim is not None else torch.min(F.gelu(x, approximate=kwargs.get("approximate", "none")))
        else:
            base = torch.min(x, dim=dim, keepdim=keepdim).values if dim is not None else torch.min(x)
            y = F.gelu(base, approximate=kwargs.get("approximate", "none"))
        return _write_out(y, out)

    if name in ("solve", "fused_cholesky_solve", "solve_and_add_scaled_vector"):
        if name == "fused_cholesky_solve":
            A = args[0]
            b = args[1]
            L = torch.linalg.cholesky(A)
            b2 = b.unsqueeze(-1) if b.dim() == A.dim() - 1 else b
            y = torch.cholesky_solve(b2, L)
            if b.dim() == A.dim() - 1:
                y = y.squeeze(-1)
        else:
            y = torch.linalg.solve(args[0], args[1])
            if name == "solve_and_add_scaled_vector":
                y = y + args[3] * args[2]
        return _write_out(y, out)

    raise NotImplementedError(f"Generated fallback does not know how to implement {{name}}")
""".strip()


def make_rule_fallback_code(sample: Dict[str, Any]) -> str:
    func_name = extract_function_name(sample.get("input", ""))
    if not re.match(r"^[A-Za-z_]\w*$", func_name):
        func_name = re.sub(r"\W+", "_", func_name)
    return FALLBACK_TEMPLATE.format(func_name=func_name)


# ============================================================
# 9. Model call / repair
# ============================================================

def qwen_api(
    messages: List[Dict[str, str]],
    model: str = MODEL_NAME,
    retries: int = 3,
    sleep_base: float = 6.0,
) -> str:
    for attempt in range(retries):
        try:
            res = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
            )
            return res.choices[0].message.content or ""
        except Exception as e:
            if attempt == retries - 1:
                print(f"\n[ERROR] API call failed after {retries} attempts: {e}")
                return ""
            wait = sleep_base * (2 ** attempt) + random.random()
            time.sleep(wait)
    return ""


def repair_code_with_prompt_kb(
    sample: Dict[str, Any],
    bad_code: str,
    reason: str,
    definitions: List[str],
    examples: List[Dict[str, Any]],
    reasoning_prompt: str,
    operator_manual: str,
    question_analysis: Dict[str, Dict[str, Any]],
    model: str,
) -> str:
    messages = build_messages(
        sample=sample,
        definitions=definitions,
        examples=examples,
        reasoning_prompt=reasoning_prompt,
        operator_manual=operator_manual,
        question_analysis=question_analysis,
    )

    func_name = extract_function_name(sample.get("input", ""))
    family = detect_task_family(sample.get("input", ""))
    ops = extract_ops(sample.get("input", ""))

    repair_prompt = f"""The previous code was invalid.

Validation error:
{reason}

Expected wrapper function:
{func_name}

Detected family:
{family}

Detected operations:
{build_ops_text(ops)}

Previous code:
{bad_code[:3000]}

Regenerate corrected code.

Rules:
- Output ONLY Python code.
- Define def {func_name}(*args, **kwargs):
- Include import torch and import torch.nn.functional as F.
- Include the exact _write_out helper from the prompt.
- Use PyTorch fallback style.
- Do not write complex Triton.
- Never write assignment expressions inside function call arguments.
- Use separate assignment lines before function calls.
- No markdown.
"""
    messages.append({"role": "user", "content": repair_prompt})
    raw = qwen_api(messages, model=model, retries=2, sleep_base=8.0)
    return clean_code_response(raw)


def predict_one(
    sample: Dict[str, Any],
    definitions: List[str],
    examples: List[Dict[str, Any]],
    reasoning_prompt: str,
    operator_manual: str,
    question_analysis: Dict[str, Dict[str, Any]],
    model: str,
    use_model: bool,
    enable_repair: bool,
    enable_rule_fallback: bool,
) -> Dict[str, Any]:
    sid = str(sample.get("id", ""))
    expected_func_name = extract_function_name(sample.get("input", ""))
    family = detect_task_family(sample.get("input", ""))

    code = ""

    if use_model:
        messages = build_messages(
            sample=sample,
            definitions=definitions,
            examples=examples,
            reasoning_prompt=reasoning_prompt,
            operator_manual=operator_manual,
            question_analysis=question_analysis,
        )
        raw = qwen_api(messages, model=model)
        code = clean_code_response(raw)

        ok, reason = validate_code_basic(code, expected_func_name, family)

        if not ok and is_fatal_truncated_syntax(reason):
            print(f"\n[WARN] Fatal syntax output for {sid}, using fallback directly.")
            if enable_rule_fallback:
                return safe_jsonl_row(sid, make_rule_fallback_code(sample))

        if not ok and enable_repair:
            repaired = repair_code_with_prompt_kb(
                sample=sample,
                bad_code=code,
                reason=reason,
                definitions=definitions,
                examples=examples,
                reasoning_prompt=reasoning_prompt,
                operator_manual=operator_manual,
                question_analysis=question_analysis,
                model=model,
            )
            repaired_ok, repaired_reason = validate_code_basic(repaired, expected_func_name, family)

            if repaired_ok:
                code = repaired
                ok, reason = True, "ok after repair"
            else:
                code = repaired
                reason = repaired_reason

        if ok:
            return safe_jsonl_row(sid, code)

        print(f"\n[WARN] Model output invalid for {sid}: {reason}")

    if enable_rule_fallback:
        try:
            fallback = make_rule_fallback_code(sample)
            return safe_jsonl_row(sid, fallback)
        except Exception as e:
            print(f"\n[ERROR] Rule fallback crashed for {sid}: {e}")

    return safe_jsonl_row(sid, code)


# ============================================================
# 10. Main
# ============================================================

def main() -> None:
    task_id, examples, test_samples, definitions = task_data_loader(FILE_PATH)

    reasoning_prompt = load_text_file(REASONING_PROMPT_PATH)
    operator_manual = load_text_file(OPERATOR_MANUAL_PATH)
    question_analysis = load_question_analysis(PER_QUESTION_ANALYSIS_PATH)

    reasoning_prompt = truncate_text(reasoning_prompt, MAX_REASONING_PROMPT_CHARS)
    operator_manual = truncate_text(operator_manual, 15000)

    for ex in examples:
        ex["input"] = normalize_text(ex.get("input", ""))
        out = ex.get("output", "")
        if isinstance(out, list):
            ex["output"] = [clean_code_response(str(x)) for x in out]
        else:
            ex["output"] = [clean_code_response(str(out))]

    for sample in test_samples:
        sample["input"] = normalize_text(sample.get("input", ""))

    print(f"task_id={task_id}")
    print(f"examples={len(examples)}, test_samples={len(test_samples)}")
    print(f"reasoning_prompt_chars={len(reasoning_prompt)}")
    print(f"operator_manual_chars={len(operator_manual)}")
    print(f"question_analysis_items={len(question_analysis)}")
    print(f"model={MODEL_NAME}")
    print(f"use_model={USE_MODEL}, max_workers={MAX_WORKERS}")
    print(f"out_path={OUT_PATH}")

    existing = load_existing_predictions(OUT_PATH)
    pending_samples = [s for s in test_samples if str(s.get("id", "")) not in existing]
    print(f"existing predictions={len(existing)}, pending={len(pending_samples)}")

    if not pending_samples:
        print("All test_samples already generated.")
        return

    worker = partial(
        predict_one,
        definitions=definitions,
        examples=examples,
        reasoning_prompt=reasoning_prompt,
        operator_manual=operator_manual,
        question_analysis=question_analysis,
        model=MODEL_NAME,
        use_model=USE_MODEL,
        enable_repair=ENABLE_REPAIR,
        enable_rule_fallback=ENABLE_RULE_FALLBACK,
    )

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(worker, sample): sample
            for sample in pending_samples
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Generating"):
            sample = futures[future]
            try:
                result = future.result()
            except Exception as e:
                print(f"\n[ERROR] Generation failed: {sample.get('id', '')} | {e}")
                try:
                    result = safe_jsonl_row(str(sample.get("id", "")), make_rule_fallback_code(sample))
                except Exception as fallback_error:
                    print(f"\n[ERROR] Emergency fallback failed: {sample.get('id', '')} | {fallback_error}")
                    result = safe_jsonl_row(str(sample.get("id", "")), "")

            append_jsonl(OUT_PATH, result)

    print(f"Done. JSONL written to: {OUT_PATH}")


if __name__ == "__main__":
    main()
