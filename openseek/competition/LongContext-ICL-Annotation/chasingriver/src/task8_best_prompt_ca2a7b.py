from __future__ import annotations

import ast
import re

from task8_best_common import normalize_text, strip_code_fences, task8_signature_hint


DEFAULT_VARIANT = "pytorch_v1_zero"

VARIANT_CONFIGS = {
    # ─── v2 新方案：修复格式不匹配 ───
    "v2_structured_fs0": {
        "api_mode": "chat_non_thinking",
        "prompt_style": "v2",
        "retrieval_mode": "code_aware",
        "current_request_mode": "full_structured",
        "example_mode": "restructured",
        "max_examples": 0,
        "max_tokens": 4096,
        "stop_at_label_close": True,
        "repair_on_failure": True,
        "prompt_contract": "module_guard",
        "enforce_module_style": True,
        "require_compilable": True,
    },
    # 与 v2_structured_fs0 相同；与 infer_task8 CONFIG `v2_zero_r3_v2` 对齐（官方 hidden 73.3）
    "v2_zero_r3_v2": {
        "api_mode": "chat_non_thinking",
        "prompt_style": "v2",
        "retrieval_mode": "code_aware",
        "current_request_mode": "full_structured",
        "example_mode": "restructured",
        "max_examples": 0,
        "max_tokens": 4096,
        "stop_at_label_close": True,
        "repair_on_failure": True,
        "prompt_contract": "module_guard",
        "enforce_module_style": True,
        "require_compilable": True,
    },
    # Gemini 三阶段主版本：首轮专注“算子数学逻辑草案”，后续轮次在 infer 中做阶段化拼装
    "task8_gemini": {
        "api_mode": "chat_non_thinking",
        "prompt_style": "v2",
        "retrieval_mode": "code_aware",
        "current_request_mode": "full_structured",
        "example_mode": "restructured",
        "max_examples": 0,
        "max_tokens": 4096,
        "stop_at_label_close": True,
        "repair_on_failure": True,
        "prompt_contract": "module_guard",
        "enforce_module_style": True,
        "require_compilable": True,
    },
    # Gemini ablation-1：关闭重构 request（raw）
    "task8_gemini_ab_raw_request": {
        "api_mode": "chat_non_thinking",
        "prompt_style": "v2",
        "retrieval_mode": "code_aware",
        "current_request_mode": "raw",
        "example_mode": "restructured",
        "max_examples": 0,
        "max_tokens": 4096,
        "stop_at_label_close": True,
        "repair_on_failure": True,
        "prompt_contract": "module_guard",
        "enforce_module_style": True,
        "require_compilable": True,
    },
    # Gemini ablation-2：关闭 restructured example
    "task8_gemini_ab_raw_example": {
        "api_mode": "chat_non_thinking",
        "prompt_style": "v2",
        "retrieval_mode": "code_aware",
        "current_request_mode": "full_structured",
        "example_mode": "raw",
        "max_examples": 0,
        "max_tokens": 4096,
        "stop_at_label_close": True,
        "repair_on_failure": True,
        "prompt_contract": "module_guard",
        "enforce_module_style": True,
        "require_compilable": True,
    },
    # Gemini thinking ablation：配置层切换为 chat_thinking
    "task8_gemini_thinking": {
        "api_mode": "chat_thinking",
        "prompt_style": "v2",
        "retrieval_mode": "code_aware",
        "current_request_mode": "full_structured",
        "example_mode": "restructured",
        "max_examples": 0,
        "max_tokens": 4096,
        "stop_at_label_close": True,
        "repair_on_failure": True,
        "prompt_contract": "module_guard",
        "enforce_module_style": True,
        "require_compilable": True,
    },
    "v2_structured_fs1": {
        "api_mode": "chat_non_thinking",
        "prompt_style": "v2",
        "retrieval_mode": "code_aware",
        "current_request_mode": "full_structured",
        "example_mode": "restructured",
        "max_examples": 1,
        "max_tokens": 4096,
        "stop_at_label_close": True,
        "repair_on_failure": True,
        "prompt_contract": "module_guard",
        "enforce_module_style": True,
        "require_compilable": True,
    },
    "v2_structured_fs2": {
        "api_mode": "chat_non_thinking",
        "prompt_style": "v2",
        "retrieval_mode": "code_aware",
        "current_request_mode": "full_structured",
        "example_mode": "restructured",
        "max_examples": 2,
        "max_tokens": 4096,
        "stop_at_label_close": True,
        "repair_on_failure": True,
        "prompt_contract": "module_guard",
        "enforce_module_style": True,
        "require_compilable": True,
    },
    "v2_thinking_fs1": {
        "api_mode": "chat_thinking",
        "prompt_style": "v2",
        "retrieval_mode": "code_aware",
        "current_request_mode": "full_structured",
        "example_mode": "restructured",
        "max_examples": 1,
        "max_tokens": 8192,
        "stop_at_label_close": True,
        "repair_on_failure": True,
        "prompt_contract": "module_guard",
        "enforce_module_style": True,
        "require_compilable": True,
    },
    # ─── v2_conservative：利用 S_Consistency = S_Exec²/S_Call ───
    # 核心策略：宁可输出不能调用的代码，也不输出能调用但算错的代码
    # prompt 强化 PyTorch fallback → 提高 S_Execution 准确率
    "v2_conservative_fs0": {
        "api_mode": "chat_non_thinking",
        "prompt_style": "v2_conservative",
        "retrieval_mode": "code_aware",
        "current_request_mode": "full_structured",
        "example_mode": "restructured",
        "max_examples": 0,
        "max_tokens": 4096,
        "stop_at_label_close": True,
        "repair_on_failure": True,
        "prompt_contract": "module_guard",
        "enforce_module_style": True,
        "require_compilable": True,
    },
    "v2_conservative_fs1": {
        "api_mode": "chat_non_thinking",
        "prompt_style": "v2_conservative",
        "retrieval_mode": "code_aware",
        "current_request_mode": "full_structured",
        "example_mode": "restructured",
        "max_examples": 1,
        "max_tokens": 4096,
        "stop_at_label_close": True,
        "repair_on_failure": True,
        "prompt_contract": "module_guard",
        "enforce_module_style": True,
        "require_compilable": True,
    },
    "v3_siglock_zero": {
        "api_mode": "chat_non_thinking",
        "prompt_style": "v3_siglock",
        "retrieval_mode": "code_aware",
        "current_request_mode": "full_structured",
        "example_mode": "restructured",
        "max_examples": 0,
        "max_tokens": 4096,
        "stop_at_label_close": True,
        "repair_on_failure": True,
        "prompt_contract": "module_guard",
        "enforce_module_style": True,
        "require_compilable": True,
    },
    "v3_safe_gate_zero": {
        "api_mode": "chat_non_thinking",
        "prompt_style": "v3_safe_gate",
        "retrieval_mode": "code_aware",
        "current_request_mode": "full_structured",
        "example_mode": "restructured",
        "max_examples": 0,
        "max_tokens": 4096,
        "stop_at_label_close": True,
        "repair_on_failure": True,
        "prompt_contract": "module_guard",
        "enforce_module_style": True,
        "require_compilable": True,
    },
    "v3_stage_verify_zero": {
        "api_mode": "chat_non_thinking",
        "prompt_style": "v3_stage_verify",
        "retrieval_mode": "code_aware",
        "current_request_mode": "full_structured",
        "example_mode": "restructured",
        "max_examples": 0,
        "max_tokens": 4096,
        "stop_at_label_close": True,
        "repair_on_failure": True,
        "prompt_contract": "module_guard",
        "enforce_module_style": True,
        "require_compilable": True,
    },
    "v4_codeonly_zero": {
        "api_mode": "chat_non_thinking",
        "prompt_style": "v4_codeonly",
        "retrieval_mode": "code_aware",
        "current_request_mode": "full_structured",
        "example_mode": "restructured",
        "max_examples": 0,
        "max_tokens": 4096,
        "stop_at_label_close": True,
        "repair_on_failure": True,
        "prompt_contract": "module_guard",
        "enforce_module_style": True,
        "require_compilable": True,
    },
    "v4_wrapperfirst_zero": {
        "api_mode": "chat_non_thinking",
        "prompt_style": "v4_wrapperfirst",
        "retrieval_mode": "code_aware",
        "current_request_mode": "full_structured",
        "example_mode": "restructured",
        "max_examples": 0,
        "max_tokens": 4096,
        "stop_at_label_close": True,
        "repair_on_failure": True,
        "prompt_contract": "module_guard",
        "enforce_module_style": True,
        "require_compilable": True,
    },
    "v4_compactstrict_zero": {
        "api_mode": "chat_non_thinking",
        "prompt_style": "v4_compactstrict",
        "retrieval_mode": "code_aware",
        "current_request_mode": "compact",
        "example_mode": "restructured",
        "max_examples": 0,
        "max_tokens": 4096,
        "stop_at_label_close": True,
        "repair_on_failure": True,
        "prompt_contract": "module_guard",
        "enforce_module_style": True,
        "require_compilable": True,
    },
    "v4_apicontract_zero": {
        "api_mode": "chat_non_thinking",
        "prompt_style": "v4_apicontract",
        "retrieval_mode": "code_aware",
        "current_request_mode": "full_structured",
        "example_mode": "restructured",
        "max_examples": 0,
        "max_tokens": 4096,
        "stop_at_label_close": True,
        "repair_on_failure": True,
        "prompt_contract": "module_guard",
        "enforce_module_style": True,
        "require_compilable": True,
    },
    "v5_torch_first_zero": {
        "api_mode": "chat_non_thinking",
        "prompt_style": "v5_torch_first",
        "retrieval_mode": "code_aware",
        "current_request_mode": "full_structured",
        "example_mode": "restructured",
        "max_examples": 0,
        "max_tokens": 4096,
        "stop_at_label_close": True,
        "repair_on_failure": True,
        "prompt_contract": "module_guard",
        "enforce_module_style": True,
        "require_compilable": True,
    },
    "v5_api_contract_zero": {
        "api_mode": "chat_non_thinking",
        "prompt_style": "v5_api_contract",
        "retrieval_mode": "code_aware",
        "current_request_mode": "full_structured",
        "example_mode": "restructured",
        "max_examples": 0,
        "max_tokens": 4096,
        "stop_at_label_close": True,
        "repair_on_failure": True,
        "prompt_contract": "module_guard",
        "enforce_module_style": True,
        "require_compilable": True,
    },
    "v5_api_contract_clean_zero": {
        "api_mode": "chat_non_thinking",
        "prompt_style": "v5_clean_api_contract",
        "retrieval_mode": "code_aware",
        "current_request_mode": "compact",
        "example_mode": "restructured",
        "max_examples": 0,
        "max_tokens": 4096,
        "stop_at_label_close": True,
        "repair_on_failure": True,
        "prompt_contract": "module_guard",
        "enforce_module_style": True,
        "require_compilable": True,
    },
    "v6_skeleton_chat_zero": {
        "api_mode": "chat_non_thinking",
        "prompt_style": "v6_skeleton_chat",
        "retrieval_mode": "code_aware",
        "current_request_mode": "compact",
        "example_mode": "restructured",
        "max_examples": 0,
        "max_tokens": 4096,
        "stop_at_label_close": True,
        "repair_on_failure": True,
        "prompt_contract": "module_guard",
        "enforce_module_style": True,
        "require_compilable": True,
    },
    "v7_touch_template_zero": {
        "api_mode": "chat_non_thinking",
        "prompt_style": "v7_touch_template",
        "retrieval_mode": "code_aware",
        "current_request_mode": "compact",
        "example_mode": "restructured",
        "max_examples": 0,
        "max_tokens": 4096,
        "stop_at_label_close": True,
        "repair_on_failure": True,
        "prompt_contract": "module_guard",
        "enforce_module_style": True,
        "require_compilable": True,
    },
    "v7b_pytorch_first_zero": {
        "api_mode": "chat_non_thinking",
        "prompt_style": "v7b_pytorch_first",
        "retrieval_mode": "code_aware",
        "current_request_mode": "compact",
        "example_mode": "restructured",
        "max_examples": 0,
        "max_tokens": 4096,
        "stop_at_label_close": True,
        "repair_on_failure": True,
        "prompt_contract": "module_guard",
        "enforce_module_style": True,
        "require_compilable": True,
    },
    "pytorch_v1_zero": {
        "api_mode": "chat_non_thinking",
        "prompt_style": "pytorch_v1",
        "retrieval_mode": "code_aware",
        "current_request_mode": "compact",
        "example_mode": "restructured",
        "max_examples": 0,
        "max_tokens": 4096,
        "stop_at_label_close": True,
        "repair_on_failure": True,
        "prompt_contract": "module_guard",
        "enforce_module_style": True,
        "require_compilable": True,
    },
    "v8_signature_touch_zero": {
        "api_mode": "chat_non_thinking",
        "prompt_style": "v8_signature_touch_template",
        "retrieval_mode": "code_aware",
        "current_request_mode": "full_structured",
        "example_mode": "restructured",
        "max_examples": 0,
        "max_tokens": 4096,
        "stop_at_label_close": True,
        "repair_on_failure": True,
        "prompt_contract": "module_guard",
        "enforce_module_style": True,
        "require_compilable": True,
    },
    "v9_family_touch_zero": {
        "api_mode": "chat_non_thinking",
        "prompt_style": "v9_family_touch_template",
        "retrieval_mode": "code_aware",
        "current_request_mode": "full_structured",
        "example_mode": "restructured",
        "max_examples": 0,
        "max_tokens": 4096,
        "stop_at_label_close": True,
        "repair_on_failure": True,
        "prompt_contract": "module_guard",
        "enforce_module_style": True,
        "require_compilable": True,
    },
    # ─── 旧方案保留兼容 ───
    "baseline_completion_fs1": {
        "api_mode": "completion",
        "prompt_style": "baseline",
        "retrieval_mode": "lexical",
        "current_request_mode": "raw",
        "example_mode": "raw",
        "max_examples": 1,
        "max_tokens": 4096,
        "stop_at_label_close": False,
        "repair_on_failure": False,
    },
    "strict_chat_fs1_lexical": {
        "api_mode": "chat_non_thinking",
        "prompt_style": "strict",
        "retrieval_mode": "lexical",
        "current_request_mode": "structured",
        "example_mode": "raw",
        "max_examples": 1,
        "max_tokens": 4096,
        "stop_at_label_close": True,
        "repair_on_failure": False,
    },
    "strict_chat_fs1_wrapper": {
        "api_mode": "chat_non_thinking",
        "prompt_style": "strict",
        "retrieval_mode": "wrapper_aware",
        "current_request_mode": "structured",
        "example_mode": "raw",
        "max_examples": 1,
        "max_tokens": 4096,
        "stop_at_label_close": True,
        "repair_on_failure": False,
    },
    "strict_chat_fs0_wrapper": {
        "api_mode": "chat_non_thinking",
        "prompt_style": "strict",
        "retrieval_mode": "wrapper_aware",
        "current_request_mode": "structured",
        "example_mode": "raw",
        "max_examples": 0,
        "max_tokens": 4096,
        "stop_at_label_close": True,
        "repair_on_failure": False,
    },
    "strict_chat_fs1_wrapper_repair": {
        "api_mode": "chat_non_thinking",
        "prompt_style": "strict",
        "retrieval_mode": "wrapper_aware",
        "current_request_mode": "structured",
        "example_mode": "raw",
        "max_examples": 1,
        "max_tokens": 4096,
        "stop_at_label_close": True,
        "repair_on_failure": True,
    },
    "strict_chat_fs1_wrapper_compact": {
        "api_mode": "chat_non_thinking",
        "prompt_style": "strict",
        "retrieval_mode": "wrapper_aware",
        "current_request_mode": "compact",
        "example_mode": "raw",
        "max_examples": 1,
        "max_tokens": 3072,
        "stop_at_label_close": True,
        "repair_on_failure": True,
    },
    "strict_chat_fs1_wrapper_summary": {
        "api_mode": "chat_non_thinking",
        "prompt_style": "strict",
        "retrieval_mode": "wrapper_aware",
        "current_request_mode": "compact",
        "example_mode": "summary",
        "max_examples": 1,
        "max_tokens": 3072,
        "stop_at_label_close": True,
        "repair_on_failure": True,
    },
    "strict_chat_fs0_wrapper_summary": {
        "api_mode": "chat_non_thinking",
        "prompt_style": "strict",
        "retrieval_mode": "wrapper_aware",
        "current_request_mode": "compact",
        "example_mode": "summary",
        "max_examples": 0,
        "max_tokens": 3072,
        "stop_at_label_close": True,
        "repair_on_failure": True,
    },
    "thinking_chat_fs1_wrapper_compact": {
        "api_mode": "chat_thinking",
        "prompt_style": "strict",
        "retrieval_mode": "wrapper_aware",
        "current_request_mode": "compact",
        "example_mode": "raw",
        "max_examples": 1,
        "max_tokens": 6144,
        "stop_at_label_close": True,
        "repair_on_failure": True,
    },
    "strict_chat_fs0_wrapper_compact_module_guard": {
        "api_mode": "chat_non_thinking",
        "prompt_style": "strict",
        "retrieval_mode": "wrapper_aware",
        "current_request_mode": "compact",
        "example_mode": "raw",
        "max_examples": 0,
        "max_tokens": 4096,
        "stop_at_label_close": True,
        "repair_on_failure": True,
        "prompt_contract": "module_guard",
        "enforce_module_style": True,
        "require_compilable": True,
    },
    "strict_chat_fs1_wrapper_compact_module_guard": {
        "api_mode": "chat_non_thinking",
        "prompt_style": "strict",
        "retrieval_mode": "wrapper_aware",
        "current_request_mode": "compact",
        "example_mode": "raw",
        "max_examples": 1,
        "max_tokens": 4096,
        "stop_at_label_close": True,
        "repair_on_failure": True,
        "prompt_contract": "module_guard",
        "enforce_module_style": True,
        "require_compilable": True,
    },
    "strict_chat_fs2_wrapper_compact_module_guard": {
        "api_mode": "chat_non_thinking",
        "prompt_style": "strict",
        "retrieval_mode": "wrapper_aware",
        "current_request_mode": "compact",
        "example_mode": "raw",
        "max_examples": 2,
        "max_tokens": 4096,
        "stop_at_label_close": True,
        "repair_on_failure": True,
        "prompt_contract": "module_guard",
        "enforce_module_style": True,
        "require_compilable": True,
    },
    "thinking_chat_fs1_wrapper_compact_module_guard": {
        "api_mode": "chat_thinking",
        "prompt_style": "strict",
        "retrieval_mode": "wrapper_aware",
        "current_request_mode": "compact",
        "example_mode": "raw",
        "max_examples": 1,
        "max_tokens": 8192,
        "stop_at_label_close": True,
        "repair_on_failure": True,
        "prompt_contract": "module_guard",
        "enforce_module_style": True,
        "require_compilable": True,
    },
    "thinking_chat_fs2_wrapper_compact_module_guard": {
        "api_mode": "chat_thinking",
        "prompt_style": "strict",
        "retrieval_mode": "wrapper_aware",
        "current_request_mode": "compact",
        "example_mode": "raw",
        "max_examples": 2,
        "max_tokens": 8192,
        "stop_at_label_close": True,
        "repair_on_failure": True,
        "prompt_contract": "module_guard",
        "enforce_module_style": True,
        "require_compilable": True,
    },
}

SECTION_MARKERS = (
    "Functional Description",
    "Wrapper Entry Information",
    "Args",
    "Shape",
    "Math",
    "other",
    "After generation",
)

FAMILY_KEYWORDS = {
    "attention": ("attention", "softmax", "query", "key", "value", "causal"),
    "matmul": ("matmul", "matrix multiplication", "bmm", "gemm", "dot product", "mm"),
    "conv": ("conv", "convolution", "conv2d", "filter"),
    "norm": ("layer norm", "layer_norm", "rmsnorm", "rms norm", "normalization"),
    "dequant": ("dequant", "quantiz", "int4", "row-wise"),
    "reduction": ("argmax", "argmin", "reduce", "reduction", "sum", "mean"),
    "fft": ("fft", "fftn", "frequency"),
    "elementwise": (
        "element-wise",
        "elementwise",
        "sigmoid",
        "gelu",
        "relu",
        "dropout",
        "divides each element",
        "masked add",
        "kl divergence",
        "subtraction",
    ),
}

CODE_START_RE = re.compile(r"(?m)^(?:import |from |@triton\.jit|def |class )")
OPEN_LABEL_RE = re.compile(r"^\s*<label>\s*", re.IGNORECASE)
CLOSE_LABEL_RE = re.compile(r"\s*</label>\s*$", re.IGNORECASE)

_EXAMPLE_PREFIX = (
    "You are a expert in writing Triton operators for efficient GPU programming. "
    "Use triton language write a kernel and wrapper according following instruction."
)


def get_variant_config(variant: str | None = None) -> dict:
    key = variant or DEFAULT_VARIANT
    if key not in VARIANT_CONFIGS:
        key = DEFAULT_VARIANT
    return VARIANT_CONFIGS[key]


# ─────────────────────────────────────────────────────────────
#  从 example output 反向提取结构化信息（修复格式不匹配的核心）
# ─────────────────────────────────────────────────────────────

def _extract_wrapper_info_from_code(code: str) -> dict | None:
    """从 example 的输出代码中提取 wrapper 函数签名信息。"""
    try:
        tree = ast.parse(code)
    except Exception:
        return None

    # 找出所有 @triton.jit 装饰的函数名
    jit_funcs: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for deco in node.decorator_list:
                deco_str = ast.unparse(deco) if hasattr(ast, "unparse") else ""
                if "triton" in deco_str and "jit" in deco_str:
                    jit_funcs.add(node.name)
                # 也处理 @triton.autotune 包裹的情况
                if "autotune" in deco_str:
                    jit_funcs.add(node.name)

    # 在 module 顶层找非 jit 函数作为 wrapper
    wrapper_func: ast.FunctionDef | None = None
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.FunctionDef) and node.name not in jit_funcs:
            # 跳过辅助函数（如 get_autotune_config, calculate_settings, is_hip 等）
            if node.name.startswith(("get_", "is_", "calculate_", "heuristics_")):
                continue
            wrapper_func = node

    if wrapper_func is None:
        # 回退：取最后一个非 jit 函数
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.FunctionDef) and node.name not in jit_funcs:
                wrapper_func = node

    if wrapper_func is None:
        return None

    # 提取参数信息
    args_info: list[str] = []
    fn_args = wrapper_func.args
    all_args = fn_args.posonlyargs + fn_args.args
    # 默认值从右对齐
    defaults = fn_args.defaults
    n_no_default = len(all_args) - len(defaults)

    for i, arg in enumerate(all_args):
        if i < n_no_default:
            args_info.append(arg.arg)
        else:
            default = defaults[i - n_no_default]
            try:
                default_str = ast.unparse(default)
            except Exception:
                default_str = "..."
            args_info.append(f"{arg.arg}={default_str}")

    if fn_args.vararg:
        args_info.append(f"*{fn_args.vararg.arg}")
    elif fn_args.kwonlyargs:
        args_info.append("*")

    for i, arg in enumerate(fn_args.kwonlyargs):
        if i < len(fn_args.kw_defaults) and fn_args.kw_defaults[i] is not None:
            try:
                default_str = ast.unparse(fn_args.kw_defaults[i])
            except Exception:
                default_str = "..."
            args_info.append(f"{arg.arg}={default_str}")
        else:
            args_info.append(arg.arg)

    if fn_args.kwarg:
        args_info.append(f"**{fn_args.kwarg.arg}")

    signature = f"{wrapper_func.name}({', '.join(args_info)})"

    return {
        "name": wrapper_func.name,
        "signature": signature,
        "arg_names": [a.arg for a in (fn_args.posonlyargs + fn_args.args + fn_args.kwonlyargs)],
        "kernel_names": sorted(jit_funcs),
    }


def _restructure_example_input(example: dict) -> str:
    """将 example 的叙述式 input 重写为与 test sample 相同的结构化格式。

    这是修复 example/test 格式不匹配的核心函数。
    """
    code = example["output"][0]
    raw_input = example["input"]

    # 提取 wrapper 信息
    wrapper_info = _extract_wrapper_info_from_code(code)

    # 从原始 input 提取功能描述（去掉固定前缀）
    desc = raw_input
    if _EXAMPLE_PREFIX in desc:
        desc = desc.split(_EXAMPLE_PREFIX, 1)[1]
    desc = desc.strip()
    # 取前 3 句作为功能描述
    sentences = re.split(r"(?<=[.!?])\s+", desc.strip())
    func_desc = " ".join(sentences[:3]).strip() if sentences else desc[:300]

    # 构建结构化输入（与 test sample 格式一致）
    parts: list[str] = [
        "You are an expert in Trion programming, capable of writing "
        "corresponding Triton kernels and wrapper functions based on "
        "functional descriptions and function parameters. Ensure that "
        "the wrapper function fully corresponds to the provided function "
        "information.",
    ]

    parts.append(f"Functional Description: {func_desc}")

    if wrapper_info:
        parts.append(f"Wrapper Entry Information: {wrapper_info['signature']}")
    else:
        parts.append(f"Wrapper Entry Information: (see description above)")

    parts.append(
        "After generation: verify if the Triton wrapper aligns with the "
        "provided func_inputs. If not, regenerate."
    )

    return "\n".join(parts)


def _extract_named_section(text: str, name: str) -> str | None:
    marker_pattern = "|".join(re.escape(marker) for marker in SECTION_MARKERS)
    match = re.search(
        rf"{re.escape(name)}:\s*(.*?)(?=\s*(?:{marker_pattern})\s*:|\Z)",
        text,
        re.DOTALL,
    )
    if not match:
        return None
    value = match.group(1).strip()
    return value or None


def extract_wrapper_entry(text: str) -> str | None:
    return _extract_named_section(text, "Wrapper Entry Information")


def _extract_function_names(text: str) -> set[str]:
    patterns = (
        r"Wrapper Entry Information:\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(",
        r"The Python function ['`\"]?([A-Za-z_][A-Za-z0-9_]*)['`\"]?",
        r"The wrapper function ['`\"]?([A-Za-z_][A-Za-z0-9_]*)['`\"]?",
        r"\bfunction\s+`([A-Za-z_][A-Za-z0-9_]*)`",
        r"\bkernel\s+`([A-Za-z_][A-Za-z0-9_]*)`",
    )
    names: set[str] = set()
    for pattern in patterns:
        for match in re.findall(pattern, text):
            names.add(match.lower())
    return names


def _extract_family_tags(text: str) -> set[str]:
    lowered = text.lower()
    tags = set()
    for tag, keywords in FAMILY_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            tags.add(tag)
    return tags


def _extract_wrapper_tokens(text: str) -> set[str]:
    wrapper_entry = extract_wrapper_entry(text)
    if wrapper_entry is None:
        return set()
    return set(normalize_text(wrapper_entry))


def _structured_request(text: str, compact: bool = False, full: bool = False) -> str:
    wrapper_entry = extract_wrapper_entry(text)
    functional = _extract_named_section(text, "Functional Description")
    args_block = _extract_named_section(text, "Args")
    shape_block = _extract_named_section(text, "Shape")
    math_block = _extract_named_section(text, "Math")
    other_block = _extract_named_section(text, "other")

    parts: list[str] = []
    if wrapper_entry:
        parts.append(f"Wrapper Entry Information:\n{wrapper_entry}")
    if functional:
        parts.append(f"Functional Description:\n{functional}")
    if args_block and (full or not compact):
        parts.append(f"Args:\n{args_block}")
    if shape_block:
        parts.append(f"Shape:\n{shape_block}")
    if math_block:
        parts.append(f"Math:\n{math_block}")
    if other_block and (full or not compact):
        parts.append(f"Other Details:\n{other_block}")

    if parts:
        return "\n\n".join(parts)

    cleaned = re.sub(r"\s*After generation:.*$", "", text, flags=re.DOTALL).strip()
    return cleaned


def _first_sentences(text: str, limit: int = 2) -> str:
    normalized = " ".join(text.split())
    parts = re.split(r"(?<=[.!?])\s+", normalized)
    return " ".join(parts[:limit]).strip()


def _summarize_reference_input(text: str) -> str:
    wrapper_hint = task8_signature_hint(text)
    names = sorted(_extract_function_names(text))
    families = sorted(_extract_family_tags(text))
    functional = _extract_named_section(text, "Functional Description")
    summary_parts = [
        f"Wrapper or function hint: {wrapper_hint}",
    ]
    if names:
        summary_parts.append(f"Named symbols: {', '.join(names)}")
    if families:
        summary_parts.append(f"Operator families: {', '.join(families)}")
    if functional:
        summary_parts.append(f"Behavior summary: {_first_sentences(functional, limit=2)}")
    else:
        summary_parts.append(f"Behavior summary: {_first_sentences(text, limit=2)}")
    return "\n".join(summary_parts)


def format_example(example: dict, variant: str | None = None) -> str:
    config = get_variant_config(variant)
    example_mode = config.get("example_mode", "raw")

    if example_mode == "restructured":
        # ★ 核心改动：把 example input 重写为 test-like 结构化格式
        reference_input = _restructure_example_input(example)
    elif example_mode == "summary":
        reference_input = _summarize_reference_input(example["input"])
    else:
        reference_input = example["input"].strip()

    return (
        "### Reference Input\n"
        f"{reference_input}\n\n"
        "### Reference Output\n"
        f"<label>\n{example['output'][0].strip()}\n</label>\n"
    )


def build_prompt(task_description: str, text2annotate: str, variant: str | None = None) -> str:
    config = get_variant_config(variant)
    signature_hint = task8_signature_hint(text2annotate)

    # ─── v2 新 prompt：更简洁、更强约束 ───
    if config["prompt_style"] == "v2":
        request_mode = config["current_request_mode"]
        if request_mode == "full_structured":
            current_request = _structured_request(text2annotate, compact=False, full=True)
        elif request_mode == "compact":
            current_request = _structured_request(text2annotate, compact=True)
        elif request_mode == "structured":
            current_request = _structured_request(text2annotate)
        else:
            current_request = text2annotate

        examples_block = "[[EXAMPLES]]\n\n" if config.get("max_examples", 0) > 0 else ""

        return (
            f"Generate a complete Triton implementation for the following specification.\n\n"
            f"{current_request}\n\n"
            f"{examples_block}"
            f"CRITICAL CONSTRAINTS:\n"
            f"1. The wrapper function MUST be: {signature_hint}\n"
            f"2. Include `import torch`, `import triton`, `import triton.language as tl`.\n"
            f"3. Include at least one @triton.jit kernel with proper tl.load/tl.store.\n"
            f"4. The wrapper MUST launch the kernel via kernel[grid](...).\n"
            f"5. Output ONLY raw Python code between <label> and </label> tags.\n"
            f"6. No markdown, no explanation, no placeholders, no ellipsis.\n"
            f"7. If an operation is hard to do in Triton, use PyTorch in the wrapper.\n"
            f"8. ⚠️ ABSOLUTELY NO `...` (ellipsis), `pass`, `TODO`, or placeholder code.\n"
            f"   Every function body MUST contain actual implementation code.\n"
            f"   If you are unsure, write a simple PyTorch fallback instead of `...`.\n\n"
            f"<label>\n"
            f"FULL_CODE\n"
            f"</label>"
        )

    # ─── v2_conservative：利用 S_Consistency 公式的保守策略 ───
    # S_Consistency = S_Exec² / S_Call → 能调用但算错 比 不能调用 更差
    # 策略：强调正确性优先，大力推 PyTorch fallback
    if config["prompt_style"] == "v2_conservative":
        request_mode = config["current_request_mode"]
        if request_mode == "full_structured":
            current_request = _structured_request(text2annotate, compact=False, full=True)
        elif request_mode == "compact":
            current_request = _structured_request(text2annotate, compact=True)
        elif request_mode == "structured":
            current_request = _structured_request(text2annotate)
        else:
            current_request = text2annotate

        examples_block = "[[EXAMPLES]]\n\n" if config.get("max_examples", 0) > 0 else ""

        return (
            f"Generate a complete Triton implementation for the following specification.\n\n"
            f"{current_request}\n\n"
            f"{examples_block}"
            f"WRAPPER SIGNATURE (MUST match exactly):\n"
            f"  {signature_hint}\n\n"
            f"CORRECTNESS IS THE #1 PRIORITY:\n"
            f"- If you are NOT 100% confident about implementing the algorithm in a Triton kernel,\n"
            f"  implement the core logic using PyTorch operations in the wrapper function.\n"
            f"- A correct wrapper that uses `torch.*` internally is MUCH better than\n"
            f"  an incorrect Triton kernel.\n"
            f"- The @triton.jit kernel can handle just a simple part (e.g., element copy)\n"
            f"  while the wrapper does the complex math with PyTorch.\n"
            f"- NEVER guess or approximate an algorithm. If unsure, use torch.\n\n"
            f"HARD CONSTRAINTS:\n"
            f"1. `import torch`, `import triton`, `import triton.language as tl`.\n"
            f"2. At least one @triton.jit kernel.\n"
            f"3. The wrapper MUST launch the kernel via kernel[grid](...).\n"
            f"4. Output ONLY raw Python code between <label> and </label> tags.\n"
            f"5. No markdown, no explanation, no placeholders.\n"
            f"6. ⚠️ ABSOLUTELY NO `...` (ellipsis), `pass`, `TODO`, or placeholder code.\n"
            f"   Every function body MUST contain actual implementation.\n\n"
            f"<label>\n"
            f"FULL_CODE\n"
            f"</label>"
        )

    if config["prompt_style"] in {"v3_siglock", "v3_safe_gate", "v3_stage_verify"}:
        request_mode = config["current_request_mode"]
        if request_mode == "full_structured":
            current_request = _structured_request(text2annotate, compact=False, full=True)
        elif request_mode == "compact":
            current_request = _structured_request(text2annotate, compact=True)
        elif request_mode == "structured":
            current_request = _structured_request(text2annotate)
        else:
            current_request = text2annotate

        examples_block = "[[EXAMPLES]]\n\n" if config.get("max_examples", 0) > 0 else ""
        base_contract = (
            f"Generate a complete executable Python module for this Triton task.\n\n"
            f"{current_request}\n\n"
            f"{examples_block}"
            f"Target wrapper signature, locked:\n{signature_hint}\n\n"
            "Hard module contract:\n"
            "1. Include exactly these imports if possible: import torch; import triton; import triton.language as tl.\n"
            "2. Include at least one @triton.jit kernel.\n"
            "3. The top-level wrapper function name and parameter names must match the target signature.\n"
            "4. The wrapper must launch a Triton kernel with kernel[grid](...).\n"
            "5. Every tl.load/tl.store that can cross bounds must use a mask.\n"
            "6. If pure Triton is risky, keep the wrapper correct with torch operations and use Triton for a small safe part.\n"
            "7. Output only raw Python code between <label> and </label>; no markdown or prose.\n"
            "8. No ellipsis, no TODO, no omitted helper, no placeholder body.\n\n"
        )
        if config["prompt_style"] == "v3_siglock":
            strategy = (
                "Direction: signature-lock first.\n"
                "- Before algorithm details, preserve the wrapper API exactly.\n"
                "- Do not introduce *args, **kwargs, renamed parameters, nested wrappers, or class-only APIs.\n"
                "- If the requested API has keyword-only args or out=, implement those explicitly.\n"
                "- Prefer a simple correct implementation over a complex incomplete kernel.\n\n"
            )
        elif config["prompt_style"] == "v3_safe_gate":
            strategy = (
                "Direction: consistency-score safe mode.\n"
                "- The score punishes callable wrong code. Do not guess complex math.\n"
                "- Use torch operations in the wrapper for convolutions, linalg, indexing, normalization, dropout, and fused chains when unsure.\n"
                "- Keep a minimal harmless Triton kernel launch for module structure, guarded so CPU inputs do not crash.\n"
                "- Return None only if the requested wrapper cannot be represented as valid Python.\n\n"
            )
        else:
            strategy = (
                "Direction: staged-verification draft.\n"
                "- Write code that is easy for later repair rounds to inspect and patch.\n"
                "- Use clear intermediate variables for shape, output allocation, kernel grid, and final result.\n"
                "- Keep the first version compilable even if some math uses torch fallback.\n"
                "- Make semantic intent visible in code names so static checks can guide repairs.\n\n"
            )
        return base_contract + strategy + "<label>\nFULL_CODE\n</label>"

    if config["prompt_style"] in {"v4_codeonly", "v4_wrapperfirst", "v4_compactstrict", "v4_apicontract"}:
        request_mode = config["current_request_mode"]
        if request_mode == "full_structured":
            current_request = _structured_request(text2annotate, compact=False, full=True)
        elif request_mode == "compact":
            current_request = _structured_request(text2annotate, compact=True)
        elif request_mode == "structured":
            current_request = _structured_request(text2annotate)
        else:
            current_request = text2annotate

        common = (
            "Generate raw Python code only.\n\n"
            f"Specification:\n{current_request}\n\n"
            f"Locked wrapper signature:\n{signature_hint}\n\n"
            "Non-negotiable output rules:\n"
            "- First code line should be an import, not a sentence.\n"
            "- No markdown fences, no explanation, no numbered steps, no English prose.\n"
            "- No comments that explain ideas like 'maybe', 'wait', 'compute', 'how to', 'placeholder'.\n"
            "- No ellipsis, no TODO, no pass-only function bodies.\n"
            "- Include import torch, import triton, import triton.language as tl.\n"
            "- Include at least one @triton.jit kernel and one kernel[grid](...) launch.\n"
            "- Keep wrapper name and parameter names aligned with the locked signature.\n\n"
        )
        if config["prompt_style"] == "v4_codeonly":
            addon = (
                "Direction:\n"
                "- Optimize for code cleanliness and compilability.\n"
                "- Use only short technical comments if absolutely necessary; otherwise no comments.\n"
                "- Prefer simple tensor logic over ambitious Triton logic.\n\n"
            )
        elif config["prompt_style"] == "v4_wrapperfirst":
            addon = (
                "Direction:\n"
                "- Build the wrapper first and keep it correct.\n"
                "- Use torch operations in the wrapper for the main math when the Triton path is uncertain.\n"
                "- Use a small Triton kernel for a safe substep such as copy, pointwise transform, or final writeback.\n\n"
            )
        elif config["prompt_style"] == "v4_compactstrict":
            addon = (
                "Direction:\n"
                "- The request summary is intentionally compact. Focus on wrapper signature, operator family, shape handling, and out= behavior.\n"
                "- Keep code minimal and robust. Avoid extra helper functions unless they are necessary.\n"
                "- When the request is ambiguous, choose the safest implementation that preserves callability.\n\n"
            )
        else:
            addon = (
                "Direction:\n"
                "- API contract first: preserve kw-only args, defaults, out= behavior, tuple returns, and in-place semantics when requested.\n"
                "- Handle broadcasting, dtype propagation, and output allocation explicitly in the wrapper.\n"
                "- If the operator is hard, keep the wrapper behavior correct with torch and keep Triton minimal but valid.\n\n"
            )
        return common + addon + "<label>\nFULL_CODE\n</label>"

    if config["prompt_style"] in {"v5_torch_first", "v5_api_contract"}:
        request_mode = config["current_request_mode"]
        if request_mode == "full_structured":
            current_request = _structured_request(text2annotate, compact=False, full=True)
        elif request_mode == "compact":
            current_request = _structured_request(text2annotate, compact=True)
        elif request_mode == "structured":
            current_request = _structured_request(text2annotate)
        else:
            current_request = text2annotate

        common = (
            "Generate raw Python code only.\n\n"
            f"Specification:\n{current_request}\n\n"
            f"Locked wrapper signature:\n{signature_hint}\n\n"
            "Hard output contract:\n"
            "- The first line must be an import statement.\n"
            "- Include import torch, import triton, and import triton.language as tl.\n"
            "- Define the wrapper exactly with the locked function name, parameter names, order, defaults, and keyword-only marker.\n"
            "- Include at least one real @triton.jit kernel and launch it with kernel[grid](...).\n"
            "- Do not output markdown fences, explanations, analysis text, ellipsis, TODO, or pass-only bodies.\n"
            "- Every function body must contain executable implementation code.\n\n"
        )
        if config["prompt_style"] == "v5_torch_first":
            addon = (
                "Correctness policy:\n"
                "- Prefer exact PyTorch semantics in the wrapper for conv, linalg, normalization, dropout, indexing, pooling, fft, grid_sample, svd, qr, solve, and fused chains.\n"
                "- Use Triton directly only for simple elementwise or simple reduction logic.\n"
                "- If PyTorch computes the main result, still launch a minimal safe Triton kernel on a CUDA tensor result without changing values.\n"
                "- Handle broadcasting, dtype, device, dim, keepdim, training, approximate, and out= behavior explicitly when present.\n"
                "- If out is provided, copy the computed result into out and return out unless the signature requires a tuple.\n\n"
            )
        else:
            addon = (
                "API contract policy:\n"
                "- Preserve kw-only args, defaults, out= behavior, dim/keepdim/dtype/device/training/approximate arguments, tuple returns, and in-place semantics.\n"
                "- For out=None, allocate a result with the correct shape, dtype, and device. For provided out, write with out.copy_(result) and return out when appropriate.\n"
                "- For tuple signatures, return the exact tuple structure requested by the wrapper information.\n"
                "- For in-place style wrappers, mutate the required tensor and return the documented value.\n"
                "- If the operator math is difficult, keep wrapper behavior correct with torch operations and keep Triton minimal but valid.\n\n"
            )
        return common + addon + "<label>\nFULL_CODE\n</label>"

    if config["prompt_style"] == "v5_clean_api_contract":
        request_mode = config["current_request_mode"]
        if request_mode == "full_structured":
            current_request = _structured_request(text2annotate, compact=False, full=True)
        elif request_mode == "compact":
            current_request = _structured_request(text2annotate, compact=True)
        elif request_mode == "structured":
            current_request = _structured_request(text2annotate)
        else:
            current_request = text2annotate
        wrapper_entry = extract_wrapper_entry(text2annotate) or signature_hint
        locked_signature = re.split(r"\.\s*Args:|\s+Args:|->|;", wrapper_entry, maxsplit=1)[0].strip()

        return (
            "/no_think\n"
            "Return raw Python code only. The first character of your answer must be the 'i' in 'import'.\n"
            "Do not write reasoning, notes, markdown, labels, XML tags, bullets, placeholders, or any text outside Python code.\n\n"
            f"Specification:\n{current_request}\n\n"
            f"Locked wrapper signature:\n{locked_signature}\n\n"
            "Non-negotiable module contract:\n"
            "- First line: import torch\n"
            "- Also include: import triton and import triton.language as tl\n"
            "- Define the wrapper exactly with the locked function name, parameter names, order, defaults, and keyword-only marker.\n"
            "- Include at least one real @triton.jit kernel and launch it with kernel[grid](...) from the wrapper.\n"
            "- Every def body must contain executable statements. No ellipsis, TODO, pass-only body, 'code here', or 'Implementation here'.\n"
            "- Preserve out= behavior, dim/keepdim/dtype/device/training/approximate args, tuple returns, and in-place semantics when present.\n\n"
            "Correctness policy:\n"
            "- Use exact torch or torch.nn.functional semantics in the wrapper for complex operations: conv, linalg, norm, dropout, indexing, pooling, fft, grid_sample, svd, qr, solve, and fused chains.\n"
            "- Use Triton directly only for simple elementwise or simple reductions.\n"
            "- If torch computes the main result, still launch a minimal safe Triton kernel on a CUDA tensor result without changing values.\n"
            "- For provided out, write result with out.copy_(result) and return out when the API expects a Tensor.\n"
            "- For tuple APIs, return the exact tuple structure requested by the wrapper information.\n\n"
            "Output the complete Python module now, starting with import torch."
        )

    if config["prompt_style"] == "v6_skeleton_chat":
        request_mode = config["current_request_mode"]
        if request_mode == "full_structured":
            current_request = _structured_request(text2annotate, compact=False, full=True)
        elif request_mode == "compact":
            current_request = _structured_request(text2annotate, compact=True)
        elif request_mode == "structured":
            current_request = _structured_request(text2annotate)
        else:
            current_request = text2annotate
        wrapper_entry = extract_wrapper_entry(text2annotate) or signature_hint
        locked_signature = re.split(r"\.\s*Args:|\s+Args:|->|;", wrapper_entry, maxsplit=1)[0].strip().rstrip(".")
        def_header = f"def {locked_signature}:"

        return (
            "Generate one complete executable Python module for the task below.\n\n"
            f"Specification:\n{current_request}\n\n"
            "Locked wrapper def line. Use this exact line, including names, defaults, and keyword-only marker:\n"
            f"{def_header}\n\n"
            "Required output format:\n"
            "- Output exactly raw Python code between <label> and </label>.\n"
            "- The first code line after <label> must be: import torch\n"
            "- No markdown fences, no reasoning, no prose outside code.\n"
            "- Do not write long explanatory comments. At most 3 short comments in the whole module.\n"
            "- Never leave a function body as comments only. Every def body must contain executable statements.\n\n"
            "Mandatory module skeleton:\n"
            "1. imports: torch, triton, triton.language as tl, and torch.nn.functional as F if useful.\n"
            "2. one small real @triton.jit kernel named _openseek_touch_kernel.\n"
            "3. the kernel must use pid, offsets, mask, tl.load, and tl.store; it may copy output to itself.\n"
            "4. the wrapper must first compute the requested result with exact torch / F semantics when the math is complex.\n"
            "5. if result is a CUDA Tensor with numel > 0, launch _openseek_touch_kernel on result so the module has a real Triton launch.\n"
            "6. preserve out= behavior when present: if out is not None, copy result into out and return out for Tensor APIs.\n"
            "7. preserve tuple returns and in-place behavior when the wrapper information requests them.\n\n"
            "Implementation policy:\n"
            "- For conv, linalg, norm, dropout, indexing, pooling, fft, grid_sample, svd, qr, solve, or fused chains, use torch/F in the wrapper for correctness.\n"
            "- Use Triton direct math only for simple elementwise or simple reductions.\n"
            "- Prefer a short compilable wrapper over ambitious incomplete kernels.\n"
            "- No ellipsis, no pass-only body, no TODO, no pseudo-code, no repeated comments.\n\n"
            "Output exactly:\n<label>\nFULL_CODE\n</label>"
        )

    if config["prompt_style"] == "v7_touch_template":
        request_mode = config["current_request_mode"]
        if request_mode == "full_structured":
            current_request = _structured_request(text2annotate, compact=False, full=True)
        elif request_mode == "compact":
            current_request = _structured_request(text2annotate, compact=True)
        elif request_mode == "structured":
            current_request = _structured_request(text2annotate)
        else:
            current_request = text2annotate
        wrapper_entry = extract_wrapper_entry(text2annotate) or signature_hint
        locked_signature = re.split(r"\.\s*Args:|\s+Args:|->|;", wrapper_entry, maxsplit=1)[0].strip().rstrip(".")
        def_header = f"def {locked_signature}:"

        return (
            "Generate one complete executable Python module for the task below.\n\n"
            f"Specification:\n{current_request}\n\n"
            "Locked wrapper def line. Use this exact line:\n"
            f"{def_header}\n\n"
            "Output format:\n"
            "- Output exactly raw Python code between <label> and </label>.\n"
            "- No markdown fences, no reasoning, no prose outside code.\n"
            "- No comments unless absolutely necessary.\n\n"
            "Copy this helper structure exactly, then write only the requested wrapper below it:\n"
            "import torch\n"
            "import triton\n"
            "import triton.language as tl\n"
            "import torch.nn.functional as F\n\n"
            "@triton.jit\n"
            "def _openseek_touch_kernel(x, n_elements, BLOCK_SIZE: tl.constexpr):\n"
            "    pid = tl.program_id(0)\n"
            "    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)\n"
            "    mask = offsets < n_elements\n"
            "    values = tl.load(x + offsets, mask=mask)\n"
            "    tl.store(x + offsets, values, mask=mask)\n\n"
            "def _openseek_touch(x):\n"
            "    if isinstance(x, torch.Tensor) and x.is_cuda and x.is_contiguous() and x.numel() > 0:\n"
            "        grid = (triton.cdiv(x.numel(), 1024),)\n"
            "        _openseek_touch_kernel[grid](x, x.numel(), BLOCK_SIZE=1024)\n"
            "    return x\n\n"
            "Wrapper policy:\n"
            "- Do not implement complex math inside Triton. Use exact torch/F operations in the wrapper.\n"
            "- For conv, linalg, norm, dropout, indexing, pooling, fft, grid_sample, svd, qr, solve, fused chains: use torch/F.\n"
            "- After computing a Tensor result, call result = _openseek_touch(result) before returning or copying to out.\n"
            "- If out is not None for a Tensor API: compute result, touch result if possible, out.copy_(result), return out.\n"
            "- If the API returns a tuple, touch each Tensor element when practical and return the exact tuple structure.\n"
            "- Keep wrapper name, parameter order, defaults, and keyword-only marker exactly as locked.\n"
            "- No ellipsis, no pass, no TODO, no pseudo-code, no comment-only body.\n\n"
            "Output exactly:\n<label>\nFULL_CODE\n</label>"
        )

    if config["prompt_style"] == "v7b_pytorch_first":
        request_mode = config["current_request_mode"]
        if request_mode == "full_structured":
            current_request = _structured_request(text2annotate, compact=False, full=True)
        elif request_mode == "compact":
            current_request = _structured_request(text2annotate, compact=True)
        elif request_mode == "structured":
            current_request = _structured_request(text2annotate)
        else:
            current_request = text2annotate
        wrapper_entry = extract_wrapper_entry(text2annotate) or signature_hint
        locked_signature = re.split(r"\.\s*Args:|\s+Args:|->|;", wrapper_entry, maxsplit=1)[0].strip().rstrip(".")
        def_header = f"def {locked_signature}:"

        return (
            "/no_think\n"
            "Generate one complete executable Python module for the task below.\n\n"
            f"Specification:\n{current_request}\n\n"
            "Locked wrapper def line. Use this exact line, including names, defaults, and keyword-only marker:\n"
            f"{def_header}\n\n"
            "Output format:\n"
            "- Output exactly raw Python code between <label> and </label>.\n"
            "- The first code line after <label> must be: import torch\n"
            "- Do not output markdown fences, reasoning, prose, labels inside the code, TODO, ellipsis, or pass-only bodies.\n"
            "- Keep comments to zero unless one very short comment is necessary.\n\n"
            "Core strategy: PyTorch reference semantics first.\n"
            "- Implement the wrapper with torch and torch.nn.functional operations that match the provided Reference PyTorch code / torch_code.\n"
            "- Allowed imports only: import torch, and import torch.nn.functional as F when useful.\n"
            "- Do not add custom CUDA/JIT helpers, identity touch helpers, placeholder launches, or unrelated helper kernels.\n"
            "- Prefer a short exact PyTorch implementation over an ambitious custom kernel.\n"
            "- Preserve the locked wrapper name, parameter order, defaults, keyword-only marker, annotations when easy, and return structure.\n\n"
            "Correctness contract:\n"
            "- For simple elementwise/math APIs such as add, sub, mul, div, sqrt, rsqrt, tanh, sigmoid, abs, floor, erf, max, min, argmax, use torch.* APIs, not F.*.\n"
            "- Use F only for neural-network functional APIs such as conv2d, dropout, gelu, relu, leaky_relu, softmax, log_softmax, layer_norm, batch_norm, grid_sample, pooling, normalize, pairwise_distance, and linear.\n"
            "- Preserve out= behavior exactly: compute result first; if out is not None then out.copy_(result) and return out; otherwise return result.\n"
            "- Preserve tuple out behavior: copy each computed tensor into the matching output tensor and return the output tuple.\n"
            "- Preserve inplace behavior when an inplace parameter is present by using the matching torch/F inplace API when available.\n"
            "- Preserve dim, keepdim, dtype, device, layout, memory_format, training, reduction, approximate, eps, alpha, beta, stride, padding, dilation, groups, and broadcasting semantics when present.\n"
            "- For random/dropout functions, call the corresponding torch/F function with the same arguments; do not invent deterministic masks.\n"
            "- For linalg, fft, conv, pooling, normalization, indexing, scatter/gather, sort/topk, special functions, and fused chains, call the matching torch or F API directly.\n"
            "- If an API is under torch.special, torch.linalg, torch.fft, or torch.nn.functional, use that namespace exactly when appropriate.\n"
            "- Return only the requested wrapper result; do not print, benchmark, allocate unrelated tensors, or mutate inputs unless the API is explicitly inplace.\n\n"
            "Minimal module shape:\n"
            "import torch\n"
            "import torch.nn.functional as F\n\n"
            f"{def_header}\n"
            "    # implement exact PyTorch semantics here\n\n"
            "Exact PyTorch patterns to imitate when applicable:\n"
            "def tanh(input, *, out=None): return torch.tanh(input, out=out)\n"
            "def sqrt(input, *, out=None): return torch.sqrt(input, out=out)\n"
            "def div(input, other, *, rounding_mode=None, out=None): return torch.div(input, other, rounding_mode=rounding_mode, out=out)\n"
            "def sub(input, other, *, alpha=1, out=None): return torch.sub(input, other, alpha=alpha, out=out)\n"
            "def relu_sqrt(input, inplace=False, out=None): result = torch.sqrt(torch.relu_(input) if inplace else torch.relu(input)); copy to out if provided.\n\n"
            "Output exactly:\n<label>\nFULL_CODE\n</label>"
        )

    if config["prompt_style"] == "pytorch_v1":
        request_mode = config["current_request_mode"]
        if request_mode == "full_structured":
            current_request = _structured_request(text2annotate, compact=False, full=True)
        elif request_mode == "compact":
            current_request = _structured_request(text2annotate, compact=True)
        elif request_mode == "structured":
            current_request = _structured_request(text2annotate)
        else:
            current_request = text2annotate
        wrapper_entry = extract_wrapper_entry(text2annotate) or signature_hint
        locked_signature = re.split(r"\.\s*Args:|\s+Args:|->|;", wrapper_entry, maxsplit=1)[0].strip().rstrip(".")
        def_header = f"def {locked_signature}:"

        return (
            "/no_think\n"
            "Write one complete executable Python module that matches the reference PyTorch semantics.\n\n"
            f"Task specification:\n{current_request}\n\n"
            "Use this exact wrapper line, including the function name, arguments, defaults, and keyword-only marker:\n"
            f"{def_header}\n\n"
            "Hard output rules:\n"
            "- Return only code inside <label> and </label>.\n"
            "- The first code line must be exactly: import torch\n"
            "- Optional second import: import torch.nn.functional as F\n"
            "- Do not output markdown, prose, FULL_CODE, placeholders, ellipsis, pass-only bodies, or comments instead of code.\n"
            "- The module must define the locked wrapper exactly once at top level.\n\n"
            "Implementation rules:\n"
            "- Use pure PyTorch only. Do not import triton. Do not write @triton.jit, CUDA helpers, touch helpers, or fake kernels.\n"
            "- Prefer the shortest direct PyTorch implementation that computes the requested result.\n"
            "- Preserve output structure exactly: tensor, scalar tensor, tuple/list, or named return as implied by the reference.\n"
            "- Preserve out semantics: compute result; if out is provided, copy each result tensor into the matching out tensor and return out.\n"
            "- Preserve inplace, dim, keepdim, dtype, device, layout, memory_format, training, reduction, approximate, eps, alpha, beta, stride, padding, dilation, groups, and broadcasting semantics.\n"
            "- For dropout/random APIs, call the corresponding torch or F API with the same arguments; do not invent masks.\n\n"
            "Namespace map:\n"
            "- Use torch.* for elementwise/reductions/indexing: add, sub, mul, div, sqrt, rsqrt, tanh, sigmoid, exp, log, abs, floor, ceil, round, erf, i0, signbit, sum, mean, std, var, max, min, argmax, argmin, index_select, gather, scatter, where, sort, topk.\n"
            "- Use F.* for neural-network functionals: conv1d, conv2d, conv3d, linear, dropout, gelu, relu, leaky_relu, elu, silu, softmax, log_softmax, layer_norm, batch_norm, group_norm, grid_sample, pad, pooling, normalize, pairwise_distance.\n"
            "- Use torch.linalg.* for linalg APIs. For LU decomposition use torch.linalg.lu, which returns P, L, U.\n"
            "- Use torch.special.* only for special functions that live there, such as gammaln, digamma, polygamma, airy_ai when available.\n\n"
            "Good patterns:\n"
            "def sqrt(input, *, out=None):\n"
            "    return torch.sqrt(input, out=out)\n\n"
            "def div(input, other, *, rounding_mode=None, out=None):\n"
            "    return torch.div(input, other, rounding_mode=rounding_mode, out=out)\n\n"
            "def fused_index_select_eq(input, dim, index, other, *, out=None):\n"
            "    result = torch.index_select(input, dim, index) == other\n"
            "    if out is not None:\n"
            "        out.copy_(result)\n"
            "        return out\n"
            "    return result\n\n"
            "Now output the actual module:\n<label>\n"
        )

    if config["prompt_style"] in {"v8_signature_touch_template", "v9_family_touch_template"}:
        request_mode = config["current_request_mode"]
        if request_mode == "full_structured":
            current_request = _structured_request(text2annotate, compact=False, full=True)
        elif request_mode == "compact":
            current_request = _structured_request(text2annotate, compact=True)
        elif request_mode == "structured":
            current_request = _structured_request(text2annotate)
        else:
            current_request = text2annotate
        wrapper_entry = extract_wrapper_entry(text2annotate) or signature_hint
        locked_signature = re.split(r"\.\s*Args:|\s+Args:|->|;", wrapper_entry, maxsplit=1)[0].strip().rstrip(".")
        def_header = f"def {locked_signature}:"
        wrapper_info = _extract_wrapper_info_from_code(f"{def_header}\n    pass\n")
        param_names = ", ".join(wrapper_info.get("arg_names", [])) if wrapper_info else "(unavailable)"
        families = sorted(_extract_family_tags(text2annotate))
        family_text = ", ".join(families) if families else "unknown"
        breadcrumb_rule = ""
        if config["prompt_style"] == "v9_family_touch_template":
            breadcrumb_rule = (
                "- Include exactly one short comment immediately above the wrapper: "
                f"# openseek semantic: family={family_text}; shape stride mask math\n"
                "  This comment is allowed; do not add other explanatory comments.\n"
            )

        return (
            "Generate one complete executable Python module for the task below.\n\n"
            f"Specification:\n{current_request}\n\n"
            "Locked wrapper def line. Copy this exact line character-for-character:\n"
            f"{def_header}\n\n"
            "Order-sensitive wrapper parameter names:\n"
            f"{param_names}\n\n"
            "Expected operator family tags:\n"
            f"{family_text}\n\n"
            "Output format:\n"
            "- Output exactly raw Python code between <label> and </label>.\n"
            "- No markdown fences, no reasoning, no prose outside code.\n"
            "- The first code line after <label> must be: import torch\n"
            f"{breadcrumb_rule}"
            "\n"
            "Copy this helper block exactly before the wrapper:\n"
            "import torch\n"
            "import triton\n"
            "import triton.language as tl\n"
            "import torch.nn.functional as F\n\n"
            "@triton.jit\n"
            "def _openseek_touch_kernel(x, n_elements, BLOCK_SIZE: tl.constexpr):\n"
            "    pid = tl.program_id(0)\n"
            "    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)\n"
            "    mask = offsets < n_elements\n"
            "    values = tl.load(x + offsets, mask=mask)\n"
            "    tl.store(x + offsets, values, mask=mask)\n\n"
            "def _openseek_touch(x):\n"
            "    if isinstance(x, torch.Tensor) and x.is_cuda and x.is_contiguous() and x.numel() > 0:\n"
            "        grid = (triton.cdiv(x.numel(), 1024),)\n"
            "        _openseek_touch_kernel[grid](x, x.numel(), BLOCK_SIZE=1024)\n"
            "    return x\n\n"
            "Wrapper implementation policy:\n"
            "- Define the wrapper exactly once using the locked def line. Do not use *args or **kwargs.\n"
            "- Never rename, delete, or reorder wrapper parameters; use the provided parameter names in the body.\n"
            "- Prefer exact torch/F semantics in the wrapper for correctness. Use torch.matmul/bmm/mm for matrix products, "
            "F.conv* for convolution, torch.sum/mean/max/min for reductions, F.softmax/log_softmax for softmax, "
            "F.layer_norm/group_norm/batch_norm or torch.linalg APIs when requested.\n"
            "- Mention shape explicitly in executable code with .shape, size(), numel(), or .dim() when deriving outputs.\n"
            "- Preserve out= behavior: compute result, result = _openseek_touch(result), out.copy_(result), return out.\n"
            "- If the API returns a tuple, touch Tensor elements when practical and return the exact tuple structure.\n"
            "- No ellipsis, no pass, no TODO, no pseudo-code, no comment-only function body.\n\n"
            "Output exactly:\n<label>\nFULL_CODE\n</label>"
        )

    # ─── 旧 prompt 保持兼容 ───
    if config["prompt_style"] == "baseline":
        return (
            "You are solving task 8, Triton kernel generation.\n\n"
            "Use the examples only as structural references. Do not copy APIs or logic that conflict with the current request.\n\n"
            f"Task:\n{task_description}\n\n"
            f"Wrapper focus:\n{signature_hint}\n\n"
            "Examples:\n[[EXAMPLES]]\n\n"
            f"Current request:\n{text2annotate}\n\n"
            "Rules:\n"
            "1. Return complete Python code only.\n"
            "2. Include imports, Triton kernel(s), and a usable wrapper.\n"
            "3. The wrapper must match the described function behavior and arguments.\n"
            "4. Respect masking, broadcasting, and memory-safety details from the request.\n"
            "5. Do not use markdown fences.\n"
            "6. Do not explain.\n\n"
            "Output exactly:\n<label>FULL_CODE</label>"
        )

    request_mode = config["current_request_mode"]
    current_request = text2annotate if request_mode == "raw" else _structured_request(
        text2annotate,
        compact=request_mode == "compact",
    )
    contract_block = ""
    if config.get("prompt_contract") == "module_guard":
        contract_block = (
            "Module-style hard constraints:\n"
            "- Output a complete executable Python module, not a code fragment.\n"
            "- Include `import torch`, `import triton`, and `import triton.language as tl`.\n"
            "- Include at least one `@triton.jit` kernel.\n"
            "- Include a callable wrapper function that follows the target wrapper signature.\n"
            "- The code must be syntactically valid Python (parsable without markdown).\n\n"
        )

    return (
        "You are solving task 8, Triton kernel generation.\n\n"
        "The example is only a structural reference. Never reuse the example wrapper name, tensor names, or algorithm when they conflict with the current request.\n\n"
        f"Task definition:\n{task_description}\n\n"
        f"Target wrapper signature:\n{signature_hint}\n\n"
        "Reference example:\n[[EXAMPLES]]\n\n"
        f"Current request:\n{current_request}\n\n"
        f"{contract_block}"
        "Requirements:\n"
        "1. Return a complete Python module that can be executed directly.\n"
        "2. Include imports, Triton kernel(s), and a usable wrapper aligned to the target wrapper signature.\n"
        "3. Respect tensor shapes, strides, masking, broadcasting, and memory-safety requirements from the request.\n"
        "4. If part of the requested behavior is awkward to fuse into a Triton kernel, keep the wrapper correct and use PyTorch helper ops instead of writing prose.\n"
        "5. No analysis, no bullet points, no markdown fences, and no surrounding explanation.\n"
        "6. The first line must be <label> and the last line must be </label>.\n"
        "7. Everything inside the label must be raw Python code only.\n"
        "8. ⚠️ ABSOLUTELY NO `...` (ellipsis), `pass`, `TODO`, or placeholder code. "
        "Every function body MUST contain actual implementation.\n\n"
        "Return exactly this format:\n"
        "<label>\n"
        "FULL_CODE\n"
        "</label>"
    )


def rank_examples(
    all_examples: list[dict],
    text2annotate: str,
    variant: str | None = None,
) -> list[dict]:
    config = get_variant_config(variant)
    retrieval_mode = config["retrieval_mode"]
    target_tokens = set(normalize_text(text2annotate))
    target_wrapper_tokens = _extract_wrapper_tokens(text2annotate)
    target_names = _extract_function_names(text2annotate)
    target_families = _extract_family_tags(text2annotate)

    def sort_key(example: dict) -> tuple[int, int, int, int, int]:
        text = example["input"]
        lexical_overlap = len(set(normalize_text(text)) & target_tokens)
        length_gap = abs(len(text) - len(text2annotate))
        if retrieval_mode == "lexical":
            return (lexical_overlap, 0, 0, 0, -length_gap)

        if retrieval_mode == "code_aware":
            # ★ 新检索：基于 example OUTPUT 代码的操作族匹配
            code = example.get("output", [""])[0]
            # 用 example 的 input + output 都看操作族
            example_families = _extract_family_tags(text) | _extract_family_tags(code)
            family_overlap = len(example_families & target_families)

            # 从代码中提取 wrapper 信息做对比
            code_tokens = set(normalize_text(code))
            code_overlap = len(code_tokens & target_tokens)

            # 优先短 example（减少 token 消耗）
            code_length = len(code)
            length_penalty = code_length // 500  # 越长扣越多

            return (family_overlap * 10, code_overlap, 0, lexical_overlap, -length_penalty)

        example_wrapper_tokens = _extract_wrapper_tokens(text)
        example_names = _extract_function_names(text)
        example_families = _extract_family_tags(text)
        wrapper_overlap = len(example_wrapper_tokens & target_wrapper_tokens)
        name_overlap = len(example_names & target_names)
        family_overlap = len(example_families & target_families)
        return (name_overlap, family_overlap, wrapper_overlap, lexical_overlap, -length_gap)

    return sorted(all_examples, key=sort_key, reverse=True)


def system_message() -> str:
    return (
        "/no_think "
        "You are a Triton GPU kernel code generator. "
        "You produce complete, executable Python modules with import torch, import triton, "
        "import triton.language as tl, @triton.jit kernels, and wrapper functions. "
        "Output ONLY raw Python code. No markdown, no explanation, no reasoning."
    )


def _looks_like_code(text: str) -> bool:
    stripped = text.lstrip()
    return stripped.startswith(("import ", "from ", "@triton.jit", "def ", "class "))


def _strip_leading_noise(text: str) -> str:
    stripped = strip_code_fences(text).strip()
    stripped = re.sub(r"^\(the code must be in the label\)\s*", "", stripped, flags=re.IGNORECASE)
    stripped = re.sub(r"^/?FULL_CODE\s*", "", stripped, flags=re.IGNORECASE)
    stripped = OPEN_LABEL_RE.sub("", stripped)
    stripped = CLOSE_LABEL_RE.sub("", stripped)
    return stripped.strip()


def _extract_code_from_text(text: str) -> str | None:
    match = CODE_START_RE.search(text)
    if not match:
        return None
    candidate = text[match.start():].strip()
    candidate = CLOSE_LABEL_RE.sub("", candidate).strip()
    return candidate or None


def extract_prediction(prediction: str | None) -> str | None:
    if prediction is None:
        return None

    raw = prediction.strip()
    if not raw:
        return None

    full_label = re.search(r"<label>\s*(.*?)\s*</label>", raw, re.DOTALL | re.IGNORECASE)
    if full_label:
        raw = full_label.group(1)

    raw = _strip_leading_noise(raw)
    if not raw:
        return None
    if not _looks_like_code(raw):
        code = _extract_code_from_text(raw)
        if code is None:
            return None
        raw = code
    raw = strip_code_fences(raw).strip()
    return raw or None


def normalize_prediction(prediction: str | None) -> str | None:
    return extract_prediction(prediction)


def _is_compilable_python(code: str) -> bool:
    try:
        compile(code, "<task8_prediction>", "exec")
        return True
    except Exception:
        return False


def needs_repair(raw_response: str | None, prediction: str | None, config: dict | None = None) -> bool:
    config = config or {}
    if prediction is None:
        return True
    if len(prediction) < 120:
        return True
    if "@triton.jit" not in prediction and "import triton" not in prediction:
        return True
    if config.get("enforce_module_style"):
        lowered = prediction.lower()
        if "import torch" not in lowered:
            return True
        if "import triton" not in lowered:
            return True
        if "def " not in lowered:
            return True
        if "@triton.jit" not in lowered:
            return True
    if config.get("require_compilable") and not _is_compilable_python(prediction):
        return True
    head = (raw_response or "")[:500].lower()
    if any(
        phrase in head
        for phrase in ("okay,", "let me", "i need to", "first,", "the problem says", "we need to")
    ):
        return True
    return False


def build_repair_prompt(original_prompt: str, raw_response: str, strict_module: bool = False) -> str:
    trimmed_response = raw_response.strip()
    if len(trimmed_response) > 3000:
        trimmed_response = trimmed_response[:3000]
    extra = ""
    if strict_module:
        extra = (
            "Also enforce:\n"
            "- Include `import torch`, `import triton`, and `import triton.language as tl`.\n"
            "- Include at least one `@triton.jit` kernel.\n"
            "- Include a wrapper `def ...` function.\n"
            "- Return syntactically valid Python code only.\n\n"
        )
    return (
        "The previous answer did not follow the output format.\n\n"
        "Return the same solution again, but this time follow the format exactly.\n"
        "The first line must be <label>.\n"
        "The last line must be </label>.\n"
        "Everything inside the label must be raw Python code only.\n"
        "No explanation and no markdown fences.\n\n"
        f"{extra}"
        f"Original prompt:\n{original_prompt}\n\n"
        f"Previous answer:\n{trimmed_response}"
    )
