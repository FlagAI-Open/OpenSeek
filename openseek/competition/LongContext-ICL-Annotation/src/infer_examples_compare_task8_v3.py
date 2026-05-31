"""
Task 8 推理与执行器对比 — v3。

在 ``infer_examples_compare_task8.py`` 基础上增加：
1. 更强的输出契约（<label> 内完整可解析 Python、@triton.jit + 可调用 wrapper）；
2. 先剥 thinking/冗余前缀，再 ``_extract_code_candidate``；轻量 AST 门控（syntax / 无顶级 def）在进入重执行器前即可触发 ReAct；
3. ReAct 回合对 ``structure_no_def`` / ``syntax`` 附带行号与代码片段；前向探针失败（``invoke_no_result``）时将完整
   wrapper 实参前向报错写入下一轮修复 prompt，并禁止在 <label> 外输出交互说明。
4. 默认准确率判据为 ``exec_outcome_match``：参考与预测在执行器上的 ``ok`` + ``kind`` 一致（区别于仅二者均通过的 ``exec_dual_pass``）。
5. ICL 检索：默认 ``--task8_icl_retrieval bm25_lexical``；可选 ``semantic``、``jaccard_only``、
   ``bm25_llm``（仅用当前题面做 query：BM25 初筛 → 云端 LLM 精排，few-shot 带完整代码）。
6. 可选 ``tensor_allclose``：在参考与预测均通过简化执行器后，再在 CUDA + PyTorch 下对**同名可调入口**
   用一组固定随机形状探针跑一次前向，要求返回张量在 ``torch.allclose`` 意义下对齐（近似数值一致，
   **非**等价于主办方隐式语义；无法匹配符号名或 arity 不适用探针时将判为不匹配并写入 ``numeric_compare``）。
"""

from __future__ import annotations

import argparse
import ast
import concurrent.futures
import inspect
import json
import os
import re
import tempfile
import time
import traceback
import types
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from tqdm import tqdm

from method_hyb import annotate_nvidia as annotate
from method_hyb import annotate_nvidia_raw_text
from method_hyb import build_prompt, select_examples_bm25_llm_rerank, select_examples_hybrid


REPO_ROOT = Path(__file__).resolve().parent.parent
TASK8_FILE = "openseek-8_kernel_generation.json"
TASK8_CANONICAL_DESCRIPTION = (
    "In this task, you are asked to generate a Triton kernel implementation based on the given instruction. "
    "The output should be valid, executable Python code containing Triton kernel definitions and necessary wrappers."
)

# v3：在官方定义之上追加的硬性输出契约（插入 few-shot 之前，避免与 method_hyb_prompts 中 task8 块重复时仍强调一遍）
TASK8_V3_OUTPUT_CONTRACT = (
    "\n\n### Output contract (task8 v3 — mandatory)\n"
    "- Put **nothing** that must execute outside a **single** pair ``<label>...</label>``.\n"
    "- Inside ``<label>``: **only** Python source that ``ast.parse`` accepts as a full module/snippet.\n"
    "- Include **at least one** module-level ``def`` or ``class`` (not only nested bodies): typically "
    "``@triton.jit`` kernel(s) **plus** a host-side ``def`` wrapper callable after ``exec``.\n"
    "- Follow retrieved examples for imports, BLOCK sizes, and launcher style; "
    "prefer the same layout as examples (imports → kernels → wrapper).\n"
    "- Do **not** put markdown fences, natural-language commentary, or “here is the code” prose **inside** the tags.\n"
)

# 仅用于 ICL 召回：去掉各题共享的固定 role 前言，避免 BM25/向量被同一段模板主导。
# （主 prompt 仍用原始 ``input_text``，不因召回而改写当前题表述。）
_TASK8_RETRIEVAL_BOILERPLATE_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE | re.MULTILINE)
    for p in (
        r"^You are an expert in (?:Trion|Triton) programming, capable of writing corresponding Triton kernels "
        r"and wrapper functions based on functional descriptions and function parameters\. "
        r"Ensure that the wrapper function fully corresponds to the provided function information\.\s*",
        r"^You are a expert in writing Triton operators for efficient GPU programming\. "
        r"Use triton language write a kernel and wrapper according following instruction\.\s*",
    )
)


def _strip_task8_retrieval_boilerplate(text: str) -> str:
    s = str(text or "").replace("\r\n", "\n")
    for pat in _TASK8_RETRIEVAL_BOILERPLATE_PATTERNS:
        s, n = pat.subn("", s, count=1)
        if n:
            break
    return s.lstrip()


def _task8_cleaned_examples_for_retrieval(cleaned_examples: list[dict]) -> list[dict]:
    """浅拷贝：检索与拼进 few-shot 的题面均去掉上述前言，与 query 侧一致。"""
    return [
        {**ex, "input": _strip_task8_retrieval_boilerplate(str(ex.get("input", "")))}
        for ex in cleaned_examples
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Task8 v3：强化格式 + thinking 剥离 + AST 门控 + 增强 ReAct 反馈。"
    )
    parser.add_argument(
        "--infer_split",
        type=str,
        choices=["examples", "test_samples"],
        default="test_samples",
        help="推理数据划分：examples 或 test_samples，默认 test_samples。",
    )
    parser.add_argument("--examples_limit", type=int, default=0, help="最多推理多少条；<=0 表示全部。")
    parser.add_argument("--output_dir", type=str, default="examples", help="结果输出目录。")
    parser.add_argument("--retries", type=int, default=3, help="单条 API 失败重试次数。")
    parser.add_argument("--retry_wait_seconds", type=float, default=2.0, help="重试等待秒数。")
    parser.add_argument("--resume", action="store_true", help="开启断点续跑。")
    parser.add_argument("--task8_shot_k", type=int, default=6, help="每条样本检索示例数，默认 6。")
    parser.add_argument(
        "--task8_retrieval_pool_size",
        type=int,
        default=200,
        help="候选检索池大小（按相似度排序后截断），默认 200。",
    )
    parser.add_argument(
        "--task8_icl_retrieval",
        type=str,
        choices=["bm25_lexical", "semantic", "jaccard_only", "bm25_llm"],
        default="bm25_lexical",
        help="ICL 检索：bm25_lexical=BM25+关键词+Jaccard/ngram（默认）；semantic=向量+Cross-Encoder；"
        "jaccard_only=词面+ngram；bm25_llm=仅用题面 query：BM25 初筛再接云端模型选示例（额外一次 API）。",
    )
    parser.add_argument(
        "--task8_bm25_llm_pool",
        type=int,
        default=20,
        help="bm25_llm：BM25 初筛候选池大小（可被环境变量 ICL_BM25_LLM_POOL 覆盖）。",
    )
    parser.add_argument(
        "--task8_bm25_llm_final_k",
        type=int,
        default=5,
        help="bm25_llm：LLM 精选后送入推理的示例条数上限（会与 --task8_shot_k 取较小值）。",
    )
    parser.add_argument(
        "--thinking",
        type=str,
        choices=["on", "off"],
        default="off",
        help="是否开启模型 thinking（映射到 DASHSCOPE_ENABLE_THINKING）。",
    )
    parser.add_argument(
        "--print_model_output",
        type=str,
        choices=["on", "off"],
        default="off",
        help="是否打印模型原始输出预览（通过 ANNOTATE_LOG_EVERY_RESPONSE）。",
    )
    parser.add_argument(
        "--print_empty_prediction",
        type=str,
        choices=["on", "off"],
        default="on",
        help="当解析结果为空时是否打印样本信息。",
    )
    parser.add_argument(
        "--retrieval_batch_size",
        type=int,
        default=8,
        help="并行推理批大小（按批并发调用模型），默认 8。",
    )
    parser.add_argument(
        "--accuracy_criterion",
        type=str,
        choices=[
            "exec_outcome_match",
            "exec_dual_pass",
            "text",
            "text_and_exec",
            "text_and_exec_outcome",
            "tensor_allclose",
        ],
        default="exec_outcome_match",
        help="准确率判据：exec_outcome_match=参考与预测执行器 outcome 一致(ok+kind，默认)；"
        "exec_dual_pass=二者均单独通过执行器；text=清洗后代码全文等价；"
        "text_and_exec=文本等价且 exec_dual_pass；text_and_exec_outcome=文本等价且 exec_outcome_match。"
        "tensor_allclose=CUDA 上对同名可调入口做随机探针张量一次前向，返回张量 allclose。"
        "(需_torch_与_CUDA_；仅限 examples，有参考答案时。)",
    )
    parser.add_argument(
        "--task8_numeric_rtol",
        type=float,
        default=0.05,
        help="tensor_allclose 模式下 torch.allclose 的 rtol（半精度内核建议略宽），默认 0.05。",
    )
    parser.add_argument(
        "--task8_numeric_atol",
        type=float,
        default=1e-4,
        help="tensor_allclose 模式下 torch.allclose 的 atol，默认 1e-4。",
    )
    parser.add_argument(
        "--max_react_rounds",
        type=int,
        default=5,
        help="执行器/前向探针失败后 ReAct 纠错最大轮数（默认 5）。",
    )
    parser.add_argument(
        "--single_example_id",
        type=str,
        default="",
        help="若非空：仅推理该 ID。",
    )
    parser.add_argument(
        "--single_index",
        type=int,
        default=-1,
        help="若 >=0：仅推理待跑队列中第 N 条（0-based）。",
    )
    return parser.parse_args()


def _narrow_pending_to_single_row(
    pending_items: list[dict],
    *,
    single_example_id: str = "",
    single_index: int = -1,
) -> list[dict]:
    ex_id = str(single_example_id or "").strip()
    if ex_id:
        if not pending_items:
            raise ValueError("单条模式：待推理队列为空。")
        out = [
            row
            for row in pending_items
            if str(row.get("example_id", "") or "").strip() == ex_id
        ]
        if not out:
            raise ValueError(f"单条模式：找不到 example_id={ex_id!r}。")
        return out
    if single_index >= 0:
        if not pending_items:
            raise ValueError("单条模式：待推理队列为空。")
        if single_index >= len(pending_items):
            raise ValueError(f"单条模式：single_index={single_index} 越界。")
        return [pending_items[single_index]]
    return pending_items


def _task_json_path() -> Path:
    return REPO_ROOT / "data" / TASK8_FILE


def _normalize_text(text: str) -> str:
    return " ".join(str(text).strip().split())


def _extract_output(output_value) -> str:
    if isinstance(output_value, list):
        return "" if not output_value else str(output_value[0]).strip()
    return str(output_value).strip()


def _resolve_output_dir(output_dir: str) -> Path:
    p = Path(output_dir)
    if p.is_absolute():
        return p
    return (REPO_ROOT / p).resolve()


def _load_done_ids(output_file: Path) -> set[str]:
    done: set[str] = set()
    if not output_file.exists():
        return done
    with output_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            example_id = str(
                row.get("example_id", "") or row.get("test_sample_id", "")
            ).strip()
            if example_id:
                done.add(example_id)
    return done


def _compute_metrics_from_jsonl(output_file: Path) -> tuple[int, int, float]:
    total = 0
    matched = 0
    if not output_file.exists():
        return total, matched, 0.0
    with output_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            total += 1
            if bool(row.get("is_match", False)):
                matched += 1
    accuracy = (matched / total) if total else 0.0
    return total, matched, accuracy


@dataclass(frozen=True)
class Task8ExecOutcome:
    ok: bool
    kind: str
    message: str


class Task8CodeExecutor:
    def execute(self, code: Optional[str], *, verbose: bool = True) -> Task8ExecOutcome:
        return _task8_executor_run(code, verbose=verbose)


def _task8_executor_run(code: Optional[str], *, verbose: bool = True) -> Task8ExecOutcome:
    prefix = "[执行器]"
    log = lambda msg: print(msg) if verbose else None

    def fail(kind: str, msg: str) -> Task8ExecOutcome:
        log(f"{prefix} {kind}: {msg[:200]}" if verbose and len(msg) > 200 else f"{prefix} {kind}: {msg}")
        return Task8ExecOutcome(ok=False, kind=kind, message=msg)

    def win(kind: str, msg: str) -> Task8ExecOutcome:
        if kind == "ok":
            log(f"{prefix} 通过")
        else:
            log(f"{prefix} {kind}: {msg}")
        return Task8ExecOutcome(ok=True, kind=kind, message=msg)

    if not code or not isinstance(code, str):
        log(f"{prefix} 失败: empty code")
        return fail("empty", "empty code")

    cleaned = _extract_code_candidate(code)
    stripped = cleaned.strip()
    if not stripped:
        log(f"{prefix} 失败: empty code after cleaning")
        return fail("empty", "empty code after cleaning")

    try:
        tree = ast.parse(cleaned)
    except SyntaxError as e:
        log(f"{prefix} 失败: syntax error: {e}")
        return fail("syntax", f"syntax error: {e}")

    has_def = any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        for node in ast.walk(tree)
    )
    if not has_def:
        log(f"{prefix} 失败: no function/class definition found")
        return fail("structure_no_def", "no function/class definition found")

    exec_ns: dict = {"__builtins__": __builtins__}
    tmp_path = ""
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".py", encoding="utf-8", delete=False) as tf:
            tf.write(cleaned)
            tf.flush()
            tmp_path = tf.name
        compiled = compile(cleaned, tmp_path, "exec")
        exec(compiled, exec_ns, exec_ns)
    except ModuleNotFoundError as e:
        missing_name = str(getattr(e, "name", "") or "")
        err = traceback.format_exc(limit=4)
        if missing_name in {"torch", "triton"}:
            log(f"{prefix} 提示: 缺少依赖 {missing_name}，语法与结构已通过")
            return win("optional_dep", f"missing optional dependency: {missing_name}")
        log(f"{prefix} 编译/执行失败:\n{err}")
        return fail("import", err)
    except Exception:
        err = traceback.format_exc(limit=4)
        log(f"{prefix} 编译/执行失败:\n{err}")
        return fail("runtime", err)
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    has_callable = any(
        callable(obj)
        and isinstance(obj, (types.FunctionType, types.BuiltinFunctionType))
        for obj in exec_ns.values()
    )
    if not has_callable:
        log(f"{prefix} 失败: no callable function found in namespace")
        return fail("callable_missing", "no callable function found in generated code namespace")

    return win("ok", "")


def _extract_code_candidate(text: str) -> str:
    if not text:
        return ""
    cleaned = str(text).strip()
    if "<label>" in cleaned and "</label>" in cleaned:
        start = cleaned.find("<label>") + len("<label>")
        end = cleaned.rfind("</label>")
        if end > start:
            cleaned = cleaned[start:end].strip()
    if "```" in cleaned:
        blocks: list[str] = []
        chunks = cleaned.split("```")
        for i in range(1, len(chunks), 2):
            block = chunks[i]
            if block.startswith("python"):
                block = block[len("python") :]
            blocks.append(block.strip())
        if blocks:
            cleaned = max(blocks, key=len)
        else:
            cleaned = cleaned.replace("```python", "").replace("```", "")
    return textwrap.dedent(cleaned).strip()


def _strip_model_thinking_prefix(text: str) -> str:
    """
    剥离常见 thinking / 推理块，避免其进入 ``_extract_code_candidate`` 后污染代码。
    DashScope 的 reasoning 通常在独立字段；此处兼容把推理写进正文的模型。
    """
    if not text:
        return ""
    s = str(text)
    for pat in (
        r"<think>[\s\S]*?</think>",
        r"<thinking>[\s\S]*?</thinking>",
        r"<reasoning>[\s\S]*?</reasoning>",
    ):
        s = re.sub(pat, "", s, flags=re.IGNORECASE)
    return s.strip()


def _ast_quick_check_cleaned(cleaned: str) -> Task8ExecOutcome:
    """对已走 v3 提取管线的代码做 AST 门控（与执行器 parse/def 检查一致，不 exec）。"""
    if not cleaned or not str(cleaned).strip():
        return Task8ExecOutcome(ok=False, kind="empty", message="empty code after v3 extract")
    try:
        tree = ast.parse(cleaned)
    except SyntaxError as e:
        return Task8ExecOutcome(ok=False, kind="syntax", message=f"syntax error: {e}")
    has_def = any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        for node in ast.walk(tree)
    )
    if not has_def:
        return Task8ExecOutcome(ok=False, kind="structure_no_def", message="no function/class definition found")
    return Task8ExecOutcome(ok=True, kind="ok", message="")


def _syntax_error_snippet(exc_text: str, code: str, *, context: int = 2) -> str:
    """从 ``syntax error: ... line N`` 与源码构造片段说明。"""
    m = re.search(r"line\s+(\d+)", str(exc_text), flags=re.IGNORECASE)
    if not m:
        return ""
    try:
        ln = int(m.group(1))
    except ValueError:
        return ""
    lines = str(code).splitlines()
    if not lines or ln < 1:
        return ""
    i = ln - 1
    lo = max(0, i - context)
    hi = min(len(lines), i + context + 1)
    parts: list[str] = []
    for j in range(lo, hi):
        prefix = ">" if j == i else " "
        parts.append(f"{prefix} {j + 1:4d} | {lines[j]}")
    return "\n".join(parts)


def _task8_code_text_equal(lhs: str, rhs: str) -> bool:
    return _normalize_text(_extract_code_candidate(lhs)) == _normalize_text(_extract_code_candidate(rhs))


_TASK8_SKIP_ENTRYPOINT_KEYS: frozenset[str] = frozenset(
    {
        "annotations",
        "torch",
        "triton",
        "tl",
        "math",
        "random",
        "numpy",
        "np",
        "typing",
        "collections",
        "functools",
        "itertools",
        "inspect",
        "contextlib",
    }
)


def _task8_probe_device_and_rng(seed: int) -> tuple[Any, Any]:
    """探针张量与 Generator 使用同一 device（CUDA 可用时用 cuda:0）。"""
    import torch  # noqa: PLC0415

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        rng = torch.Generator(device=device)
    else:
        device = torch.device("cpu")
        rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)
    return device, rng


_TASK8_PROBE_SHAPES_BY_ARITY: dict[int, tuple[tuple[int, ...], ...]] = {
    1: ((48, 48),),
    2: ((8, 128), (8,)),
    3: ((1, 2, 64, 32), (1, 2, 64, 32), (1, 2, 64, 32)),
    4: ((1, 2, 32, 16), (1, 2, 32, 16), (1, 2, 32, 16), (1, 2, 32, 16)),
    5: ((1, 2, 16, 16), (1, 2, 16, 16), (1, 2, 16, 16), (1, 2, 16, 16), (1, 2, 16, 16)),
    6: ((1, 2, 8, 16), (1, 2, 8, 16), (1, 2, 8, 16), (1, 2, 8, 16), (1, 2, 8, 16), (1, 2, 8, 16)),
}


def _task8_probe_tensors_for_params(
    n: int,
    *,
    device: Any,
    rng: Any,
    dtype: Any,
) -> list[Any]:
    import torch

    shapes = _TASK8_PROBE_SHAPES_BY_ARITY.get(n)
    if shapes is None or len(shapes) != n:
        shapes = tuple((8, max(16, min(96, i * 8))) for i in range(1, n + 1))
    tensors: list[Any] = []
    for shape in shapes:
        t = torch.randn(shape, device=device, dtype=dtype, generator=rng)
        tensors.append((t.abs() + 1e-2).clamp(max=512.0))
    return tensors


def _task8_exec_module_namespace(cleaned: str) -> tuple[dict | None, str | None]:
    """与 ``_task8_executor_run`` 一致：在同一规则下写入临时文件并顶层 exec。"""
    if not cleaned.strip():
        return None, "empty code after cleaning"
    try:
        tree = ast.parse(cleaned)
    except SyntaxError as e:
        return None, f"syntax error: {e}"

    has_def = any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        for node in ast.walk(tree)
    )
    if not has_def:
        return None, "no function/class definition found"

    exec_ns: dict = {"__builtins__": __builtins__}
    tmp_path = ""
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".py", encoding="utf-8", delete=False) as tf:
            tf.write(cleaned)
            tf.flush()
            tmp_path = tf.name
        compiled = compile(cleaned, tmp_path, "exec")
        exec(compiled, exec_ns, exec_ns)
    except Exception as exc:  # noqa: BLE001
        err = traceback.format_exc(limit=4)
        msg = getattr(exc, "msg", str(exc))
        combined = msg if isinstance(msg, str) and msg.strip() else err
        return None, combined
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    has_callable = any(
        callable(obj)
        and isinstance(obj, (types.FunctionType, types.BuiltinFunctionType))
        for obj in exec_ns.values()
    )
    if not has_callable:
        return None, "no callable function found in generated code namespace"

    return exec_ns, None


def _task8_entrypoints_from_namespace(ns: dict) -> dict[str, Any]:
    import torch as _torch  # noqa: PLC0415 — 运行时探测

    out: dict[str, Any] = {}
    for k, v in ns.items():
        # 仅跳过 dunder；金标/生成代码里常见 ``_swiglu_fwd`` 等单下划线 wrapper。
        if not isinstance(k, str) or (k.startswith("__") and k.endswith("__")) or k in _TASK8_SKIP_ENTRYPOINT_KEYS:
            continue
        if isinstance(v, type):
            try:
                if issubclass(v, _torch.autograd.Function) and v is not _torch.autograd.Function:
                    out[f"{k}#apply"] = v.apply
            except TypeError:
                pass
        elif isinstance(v, types.FunctionType):
            out[k] = v
    return out


def _task8_usable_positional_parameters(fn: Callable[..., Any]) -> list[inspect.Parameter]:
    sig = inspect.signature(fn)
    params: list[inspect.Parameter] = []
    skip_names = frozenset({"ctx", "self", "cls"})
    for p in sig.parameters.values():
        if p.kind not in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            continue
        if p.name in skip_names:
            continue
        params.append(p)
    return params


def _task8_invoke_and_normalize_tensors(
    fn: Callable[..., Any],
    tensors: list[Any],
) -> tuple[list[Any], str | None]:
    """返回 CPU float 张量列表；若报错则 second 为非空消息。"""
    import torch

    try:
        out = fn(*tensors)
    except Exception:
        err = traceback.format_exc(limit=3)
        return [], err.strip() or "invoke_exception"

    if out is None:
        # 许多 Triton wrapper 原地写输出并 ``return None``，视为前向成功。
        return [], None
    normalized: list[Any] = []
    seq: tuple[Any, ...]
    if torch.is_tensor(out):
        seq = (out,)
    elif isinstance(out, (list, tuple)):
        seq = tuple(out)
    else:
        return [], f"non_tensor_outputs:{type(out).__name__}"

    for item in seq:
        if isinstance(item, torch.Tensor):
            normalized.append(item.detach().float().cpu())
        else:
            return [], "non_tensor_outputs_in_sequence"

    return normalized, None


def _tensor_lists_allclose(
    a: list[Any],
    b: list[Any],
    *,
    rtol: float,
    atol: float,
) -> bool:
    import torch

    if len(a) != len(b) or not a:
        return False
    for x, y in zip(a, b):
        if tuple(x.shape) != tuple(y.shape):
            return False
        x32, y32 = x.float(), y.float()
        if not torch.allclose(x32, y32, rtol=rtol, atol=atol):
            return False
    return True


def _task8_infer_callable_pairs(
    ref_eps: dict[str, Any],
    pred_eps: dict[str, Any],
) -> list[tuple[str, Any, Any]]:
    common = sorted(set(ref_eps.keys()) & set(pred_eps.keys()))
    pairs = [(name, ref_eps[name], pred_eps[name]) for name in common]
    if pairs:
        return pairs
    r_apply = [(k, v) for k, v in ref_eps.items() if k.endswith("#apply")]
    p_apply = [(k, v) for k, v in pred_eps.items() if k.endswith("#apply")]
    if len(r_apply) == 1 and len(p_apply) == 1:
        rl, rf = r_apply[0]
        pl, pf = p_apply[0]
        pairs.append((f"{rl}<->{pl}", rf, pf))
    return pairs


def task8_numeric_allclose_compare(
    ref_code: str,
    pred_code: str,
    *,
    rtol: float,
    atol: float,
    seed: int = 9173,
) -> tuple[bool, dict[str, Any]]:
    """
    在 CUDA + PyTorch 可同时加载两段代码的前提下，对各（推断的）对齐入口跑一次探针，
    **全部**对齐成功则返回 ``(True, detail)``。无法加载、无入口重叠或任一比较失败则为 False。
    """
    detail: dict[str, Any] = {
        "attempted": True,
        "match": False,
        "message": "",
        "pairs_detail": [],
    }
    ref_clean = _extract_code_candidate(str(ref_code or "").strip())
    pred_clean = _extract_code_candidate(str(pred_code or "").strip())
    try:
        import torch  # noqa: PLC0415
    except ModuleNotFoundError:
        detail["message"] = "torch_unavailable"
        return False, detail

    if not torch.cuda.is_available():
        detail["message"] = "cuda_unavailable"
        return False, detail

    ref_ns, ref_err = _task8_exec_module_namespace(ref_clean)
    if ref_ns is None:
        detail["message"] = f"reference_exec_failed: {ref_err}"
        return False, detail

    pred_ns, pred_err = _task8_exec_module_namespace(pred_clean)
    if pred_ns is None:
        detail["message"] = f"prediction_exec_failed: {pred_err}"
        return False, detail

    ref_eps = _task8_entrypoints_from_namespace(ref_ns)
    pred_eps = _task8_entrypoints_from_namespace(pred_ns)
    pairs = _task8_infer_callable_pairs(ref_eps, pred_eps)
    if not pairs:
        detail["message"] = "no_entrypoint_overlap_or_apply_pair"
        return False, detail

    device, rng = _task8_probe_device_and_rng(seed)

    all_ok = True
    for pair_name, rf, pf in pairs:
        pdata: dict[str, Any] = {"name": pair_name, "ok": False, "error": None}
        uref = _task8_usable_positional_parameters(rf)
        upred = _task8_usable_positional_parameters(pf)
        if len(uref) != len(upred):
            pdata["error"] = f"param_count_mismatch ref={len(uref)} pred={len(upred)}"
            all_ok = False
            detail["pairs_detail"].append(pdata)
            continue
        if not uref:
            pdata["error"] = "zero_usable_parameters"
            all_ok = False
            detail["pairs_detail"].append(pdata)
            continue

        n = len(uref)
        try:
            base = _task8_probe_tensors_for_params(n, device=device, rng=rng, dtype=torch.float32)
            tensors_ref = [t.clone().detach().requires_grad_(False) for t in base]
            tensors_pred = [t.clone().detach().requires_grad_(False) for t in base]
            ref_tensors, ref_inv_err = _task8_invoke_and_normalize_tensors(rf, tensors_ref)
            pred_tensors, pred_inv_err = _task8_invoke_and_normalize_tensors(pf, tensors_pred)
            if ref_inv_err or pred_inv_err:
                pdata["error"] = f"invoke ref={ref_inv_err} pred={pred_inv_err}".strip()
                all_ok = False
                detail["pairs_detail"].append(pdata)
                continue
            if not _tensor_lists_allclose(ref_tensors, pred_tensors, rtol=rtol, atol=atol):
                pdata["error"] = "allclose_failed"
                all_ok = False
                pdata["tensor_shapes_ref"] = [list(t.shape) for t in ref_tensors]
                pdata["tensor_shapes_pred"] = [list(t.shape) for t in pred_tensors]
            else:
                pdata["ok"] = True
        except Exception as exc:  # noqa: BLE001
            pdata["error"] = f"probe_exception:{exc}"
            all_ok = False
        detail["pairs_detail"].append(pdata)

    detail["match"] = all_ok
    detail["message"] = "numeric_ok" if all_ok else "numeric_mismatch_or_error"
    return all_ok, detail


def task8_probe_forward_ok(pred_code: str, *, seed: int = 9173) -> tuple[bool, str]:
    """
  对预测代码做一次随机探针前向：至少一个可调入口返回非空张量则视为「有结果」。
  无 torch 时跳过探针（仅依赖执行器）；有 torch 但无 CUDA 时在 CPU 上探针。
    """
    pred_clean = _extract_code_candidate(str(pred_code or "").strip())
    if not pred_clean.strip():
        return False, "empty_code"
    try:
        import torch  # noqa: PLC0415
    except ModuleNotFoundError:
        return True, "torch_unavailable_skip_probe"

    pred_ns, pred_err = _task8_exec_module_namespace(pred_clean)
    if pred_ns is None:
        return False, pred_err or "prediction_exec_failed"

    pred_eps = _task8_entrypoints_from_namespace(pred_ns)
    if not pred_eps:
        return False, "no_callable_entrypoint"

    device, rng = _task8_probe_device_and_rng(seed)
    probe_dtypes = (torch.float32, torch.float16) if device.type == "cuda" else (torch.float32,)

    errors: list[str] = []

    def _try_invoke(name: str, fn: Callable[..., Any], tensors: list[Any]) -> tuple[bool, str]:
        out_tensors, inv_err = _task8_invoke_and_normalize_tensors(fn, tensors)
        if inv_err:
            return False, inv_err[:_TASK8_PROBE_INVOKE_ERROR_MAX_CHARS]
        if out_tensors:
            return True, f"ok_via_{name}"
        return True, f"ok_void_via_{name}"

    for name, fn in pred_eps.items():
        usable = _task8_usable_positional_parameters(fn)
        if not usable:
            try:
                fn()
                return True, f"ok_void_noargs_{name}"
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{name}:noargs:{exc}")
            continue

        n = len(usable)
        for dtype in probe_dtypes:
            try:
                base = _task8_probe_tensors_for_params(n, device=device, rng=rng, dtype=dtype)
                tensors = [t.clone().detach().requires_grad_(False) for t in base]
                ok, detail = _try_invoke(name, fn, tensors)
                if ok:
                    return True, detail
                errors.append(f"{name}@{dtype}:{detail}")
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{name}@{dtype}:{exc}")

    if errors:
        return False, "; ".join(errors[:3])
    return False, "no_invokable_entrypoint_with_tensor_output"


def _task8_validate_prediction(
    code: str,
    executor: Task8CodeExecutor,
    *,
    require_forward_result: bool,
    verbose: bool = True,
    log_prefix: str = "",
) -> tuple[bool, Task8ExecOutcome]:
    """
    AST 门控 + 执行器 + 可选前向探针，供首轮与 ReAct 共用。

    「执行器通过」仅表示：语法正确、能 ``exec``、命名空间有可调用 ``def``。
    若开启 ``require_forward_result``，还会用随机 CUDA 张量**真调一次 wrapper**；
    探针失败时 ``ok=False``、``kind=invoke_no_result``，与执行器失败不是同一层。
    """
    tag = f"{log_prefix} " if log_prefix else ""

    if not str(code or "").strip():
        return False, Task8ExecOutcome(ok=False, kind="empty", message="empty prediction")

    quick = _ast_quick_check_cleaned(code)
    if not quick.ok:
        return False, quick

    outcome = executor.execute(code, verbose=verbose)
    if not outcome.ok:
        return False, outcome

    if require_forward_result:
        if verbose:
            print(f"{tag}[前向探针] 执行器已通过，开始随机张量前向…")
        fwd_ok, fwd_msg = task8_probe_forward_ok(code)
        if not fwd_ok:
            if verbose:
                preview = str(fwd_msg)[:320].replace("\n", " ")
                print(f"{tag}[前向探针] 失败 kind=invoke_no_result preview={preview}")
            return False, Task8ExecOutcome(ok=False, kind="invoke_no_result", message=fwd_msg)
        if verbose:
            print(f"{tag}[前向探针] 通过 ({fwd_msg})")

    return True, outcome


def _infer_compute_task8_accuracy(
    *,
    criterion: str,
    expected_non_empty: bool,
    expected_text: str,
    final_output: str,
    ref_outcome: Optional[Task8ExecOutcome],
    pred_outcome: Task8ExecOutcome,
    numeric_allclose_ok: Optional[bool] = None,
) -> tuple[bool, bool, bool]:
    has_ref = expected_non_empty
    is_text = has_ref and _task8_code_text_equal(final_output, expected_text)
    exec_dual = bool(ref_outcome is not None and ref_outcome.ok and pred_outcome.ok)
    exec_outcome_match = bool(
        ref_outcome is not None
        and ref_outcome.ok == pred_outcome.ok
        and ref_outcome.kind == pred_outcome.kind
    )

    if not has_ref:
        return False, False, False

    if criterion == "text":
        return is_text, is_text, exec_dual
    if criterion == "exec_dual_pass":
        return exec_dual, is_text, exec_dual
    if criterion == "exec_outcome_match":
        return exec_outcome_match, is_text, exec_dual
    if criterion == "text_and_exec":
        merged = is_text and exec_dual
        return merged, is_text, exec_dual
    if criterion == "text_and_exec_outcome":
        return is_text and exec_outcome_match, is_text, exec_dual
    if criterion == "tensor_allclose":
        if ref_outcome is None:
            return False, is_text, exec_dual
        exec_both_ok = bool(ref_outcome.ok and pred_outcome.ok)
        numeric_ok = bool(numeric_allclose_ok) if numeric_allclose_ok is not None else False
        return exec_both_ok and numeric_ok, is_text, exec_dual
    return exec_dual, is_text, exec_dual


def _truncate_for_log(text: str, limit: int = 800) -> str:
    t = str(text or "")
    if len(t) <= limit:
        return t
    return t[:limit] + f"...(+{len(t) - limit} chars)"


def _task8_outcome_as_dict(o: Task8ExecOutcome) -> dict:
    return {
        "ok": o.ok,
        "kind": o.kind,
        "message_truncated": _truncate_for_log(o.message, 1200),
    }


def _task8_build_exec_comparison(
    ref_outcome: Optional[Task8ExecOutcome],
    pred_outcome: Task8ExecOutcome,
    *,
    has_reference: bool,
) -> dict:
    pred_dict = _task8_outcome_as_dict(pred_outcome)
    if not has_reference or ref_outcome is None:
        return {
            "has_reference": False,
            "reference_exec": None,
            "prediction_exec": pred_dict,
            "outcome_categories_match": None,
            "both_pass_executor": False,
            "notes": "无参考答案代码。",
        }
    ref_dict = _task8_outcome_as_dict(ref_outcome)
    categories_match = (
        ref_outcome.ok == pred_outcome.ok and ref_outcome.kind == pred_outcome.kind
    )
    return {
        "has_reference": True,
        "reference_exec": ref_dict,
        "prediction_exec": pred_dict,
        "outcome_categories_match": categories_match,
        "both_pass_executor": bool(ref_outcome.ok and pred_outcome.ok),
    }


def _print_task8_exec_comparison_report(cmp: dict) -> None:
    print("")
    print("========== [执行对比] task8 v3 ==========")
    if not cmp.get("has_reference"):
        pe = cmp.get("prediction_exec") or {}
        print(f"预测: ok={pe.get('ok')} kind={pe.get('kind')}")
        print(_truncate_for_log(str(pe.get("message_truncated", "")), 600))
        print("========================================\n")
        return
    r = cmp.get("reference_exec") or {}
    p = cmp.get("prediction_exec") or {}
    print(f"参考: ok={r.get('ok')} kind={r.get('kind')} | 预测: ok={p.get('ok')} kind={p.get('kind')}")
    print(_truncate_for_log(str(p.get("message_truncated", "")), 2000))
    print("========================================\n")


def _inject_v3_contract_before_examples(prompt: str) -> str:
    marker = "### Reference examples (format and conventions)"
    if marker in prompt:
        return prompt.replace(marker, TASK8_V3_OUTPUT_CONTRACT + "\n" + marker, 1)
    return prompt + TASK8_V3_OUTPUT_CONTRACT


def _build_task8_zero_shot_prompt(task_description: str, input_text: str, task_id: int) -> str:
    p = build_prompt(task_description, input_text, task_id=task_id)
    p = _inject_v3_contract_before_examples(p)
    return p.replace("[[EXAMPLES]]\n\n", "").replace("[[EXAMPLES]]\n", "").replace("[[EXAMPLES]]", "")


_TASK8_PROBE_INVOKE_ERROR_MAX_CHARS = 4000


def _task8_react_traceback_focus(msg: str, *, max_lines: int = 12) -> str:
    """从探针/执行 traceback 中提取最后若干行，便于模型对准根因。"""
    lines = [ln.strip() for ln in str(msg or "").splitlines() if ln.strip()]
    if not lines:
        return ""
    focus = lines[-max_lines:]
    return "\n".join(focus)


def _task8_format_forward_probe_errors(msg: str, *, max_chars: int = 8000) -> str:
    """
    将 ``task8_probe_forward_ok`` 返回的探针失败串格式化为 ReAct prompt 可读文本。
    常见形态：``entry@torch.float32:Traceback ...``，多条以 ``; `` 拼接。
    """
    raw = str(msg or "").strip()
    if not raw:
        return ""
    parts = [p.strip() for p in re.split(r";\s+", raw) if p.strip()]
    if not parts:
        parts = [raw]
    chunks: list[str] = []
    for part in parts:
        tb_idx = part.find("Traceback (most recent call last)")
        if tb_idx >= 0:
            prefix = part[:tb_idx].strip().rstrip(":")
            body = part[tb_idx:].strip()
            if prefix:
                chunks.append(f"entry/dtype: {prefix}\n{body}")
            else:
                chunks.append(body)
        else:
            chunks.append(part)
    text = "\n\n---\n\n".join(chunks) if len(chunks) > 1 else chunks[0]
    return text[:max_chars]


def _task8_react_repair_plan(
    outcome_kind: str,
    outcome_message: str,
) -> tuple[str, str, list[str], list[str]]:
    """
    按校验失败类型 + 报错正文生成 ReAct 修复计划。
    返回 (headline, diagnosis, fix_steps, avoid_steps)。
    """
    kind = str(outcome_kind or "").strip().lower()
    msg = str(outcome_message or "")
    msg_l = msg.lower()

    fixes: list[str] = []
    avoid: list[str] = [
        "Do NOT output markdown fences or natural language outside ``<label>...</label>``.",
        "Do NOT delete the whole program unless necessary; patch the failing part and return the full file.",
    ]

    if kind == "invoke_no_result":
        headline = "Forward probe failed (static executor already passed)"
        diagnosis = (
            "The program **loaded successfully** (syntax + exec + callable wrapper exist). "
            "Failure happened when the validator called your wrapper on **random CUDA tensors** "
            "with a small arity-based shape recipe. Fix the **runtime wrapper / launcher**, not imports-only."
        )
        fixes.extend(
            [
                "Ensure at least one module-level ``def`` wrapper is invocable and runs on CUDA without raising.",
                "If the wrapper writes into an ``out`` tensor in-place, ``return None`` is OK after the kernel launch.",
                "Otherwise return ``torch.Tensor`` or ``tuple``/``list`` of tensors.",
                "Use ``tensor = tensor.contiguous()`` and match ``.dtype`` / ``.device`` before launching Triton.",
                "Keep ``@triton.jit`` kernels separate from host code; launch with ``kernel[grid](...)`` inside the wrapper.",
            ]
        )
        if "cuda" in msg_l and "generator" in msg_l:
            fixes.append(
                "All probe inputs are on ``cuda:0``; do not create CPU tensors when the code checks ``.is_cuda``."
            )
        if "no_callable_entrypoint" in msg_l:
            fixes.extend(
                [
                    "Expose a public or single-underscore wrapper ``def`` (e.g. ``def my_kernel(...)`` or ``def _launch(...)``); "
                    "``@triton.jit`` alone is not enough.",
                ]
            )
        if "rsqrt" in msg_l or "tl." in msg_l or "triton" in msg_l and "language" in msg_l:
            fixes.extend(
                [
                    "**Never** call ``tl.*`` (e.g. ``tl.rsqrt``, ``tl.load``) on ``torch.Tensor`` in the host wrapper.",
                    "In Python use ``torch.rsqrt``, ``torch.nn.functional``, etc.; ``tl.*`` only inside ``@triton.jit`` bodies.",
                ]
            )
        if "assert" in msg_l or "shape" in msg_l or "dimension" in msg_l or "size" in msg_l:
            fixes.extend(
                [
                    "Relax or fix ``assert`` on shapes: probe tensors are small random ranks (1–6 args), not task-specific sizes.",
                    "Prefer deriving sizes from ``tensor.shape`` instead of hard-coded constants when possible.",
                ]
            )
        if "float16" in msg_l or "bfloat16" in msg_l or "dtype" in msg_l:
            fixes.append(
                "Align dtypes: if the wrapper requires ``float16``, cast inputs with ``.to(dtype=torch.float16)`` "
                "or document and handle ``float32`` probes via promotion."
            )
        if "is_cuda" in msg_l or "cuda" in msg_l:
            fixes.append("Move tensors to CUDA inside the wrapper: ``x = x.cuda()`` or ``device=x.device`` consistently.")
        if "autograd" in msg_l or "backward" in msg_l:
            fixes.append(
                "If using ``torch.autograd.Function``, ensure ``forward`` can run under ``torch.no_grad()`` probe "
                "or provide a simple ``def`` wrapper that does not require saved backward context."
            )
        if "block" in msg_l or "grid" in msg_l or "meta[" in msg_l:
            fixes.append(
                "Check Triton ``grid`` and ``BLOCK_*`` constexprs: illegal grid (0 blocks) or missing META keys "
                "will fail at launch."
            )
        if "invoke_exception" in msg_l or "traceback" in msg_l:
            fixes.append("Read the traceback focus below; fix the **first user ``def``** frame, not Triton internals.")

    elif kind == "syntax":
        headline = "SyntaxError in extracted <label> code"
        diagnosis = "Python could not parse the program (unbalanced brackets/quotes, bad indent, etc.)."
        fixes.extend(
            [
                "Fix the exact line indicated; return the **complete** corrected module inside one ``<label>`` pair.",
                "Ensure every ``def`` / ``class`` body is indented with 4 spaces.",
            ]
        )

    elif kind == "structure_no_def":
        headline = "No module-level def/class"
        diagnosis = "Parsed code has no top-level callable definition."
        fixes.extend(
            [
                "Add ``import torch``, ``import triton``, ``import triton.language as tl`` if missing.",
                "Add ``@triton.jit`` kernel(s) **and** at least one top-level ``def`` wrapper that launches them.",
            ]
        )

    elif kind == "callable_missing":
        headline = "Exec succeeded but no user callable in namespace"
        diagnosis = "After exec, no ``types.FunctionType`` was bound at module level (only imports or nested defs)."
        fixes.extend(
            [
                "Define a module-level ``def wrapper(...)`` that is not nested inside another function.",
                "If logic is inside ``class Foo(torch.autograd.Function)``, also expose a thin ``def apply_*(...)`` "
                "that calls ``Foo.apply``.",
            ]
        )

    elif kind == "import":
        headline = "Import / module load error"
        diagnosis = "``exec`` failed while importing or resolving symbols."
        fixes.extend(
            [
                "Only use: ``torch``, ``triton``, ``triton.language as tl``, ``math``, ``typing`` if needed.",
                "Remove references to non-existent local modules or typo package names.",
            ]
        )
        if "triton" in msg_l:
            fixes.append("Keep Triton imports exactly: ``import triton`` and ``import triton.language as tl``.")

    elif kind == "runtime":
        headline = "Runtime error during exec (module load)"
        diagnosis = "Code failed while executing top-level statements (not the forward probe yet)."
        fixes.extend(
            [
                "Avoid running GPU kernels or heavy work at import/top-level; only define functions/classes.",
                "Move launches inside the wrapper ``def``.",
            ]
        )

    elif kind == "empty":
        headline = "Empty extracted code"
        diagnosis = "No Python was found inside ``<label>`` after stripping thinking/markdown."
        fixes.extend(
            [
                "Output exactly one ``<label>...</label>`` containing the full Triton program only.",
            ]
        )

    else:
        headline = f"Validator failure: {kind or 'unknown'}"
        diagnosis = "See validator message and patch accordingly while keeping a complete Triton solution."
        fixes.append("Preserve kernel + wrapper structure from the reference examples in the original prompt.")

    return headline, diagnosis, fixes, avoid


def _build_task8_react_repair_prompt_v3(
    *,
    zero_shot_prompt: str,
    input_text: str,
    last_code: str,
    last_extracted_python: str,
    outcome_kind: str,
    outcome_message: str,
    round_idx: int,
    max_rounds: int,
) -> str:
    last_code = str(last_code or "")
    last_py = str(last_extracted_python or "")
    msg = str(outcome_message or "")
    kind = str(outcome_kind or "")

    snippet_block = ""
    if kind == "syntax":
        snip = _syntax_error_snippet(msg, last_py if last_py.strip() else _extract_code_candidate(last_code))
        if snip:
            snippet_block = (
                "[Highlighted source around SyntaxError]\n"
                "```python\n"
                f"{snip}\n"
                "```\n\n"
            )

    structure_snip = ""
    if kind == "structure_no_def" and last_py.strip():
        ls = last_py.splitlines()
        preview = "\n".join(f"{i + 1:4d} | {line}" for i, line in enumerate(ls[:40]))
        if len(ls) > 40:
            preview += f"\n... ({len(ls) - 40} more lines)"
        structure_snip = (
            "[First 40 lines of extracted <label> code — currently missing module-level def/class]\n"
            "```python\n"
            f"{preview}\n"
            "```\n\n"
        )

    headline, diagnosis, fix_steps, avoid_steps = _task8_react_repair_plan(kind, msg)

    forward_probe_block = ""
    if kind == "invoke_no_result" and msg.strip():
        probe_err_text = _task8_format_forward_probe_errors(msg)
        if probe_err_text:
            forward_probe_block = (
                "[Forward probe failure — executor already passed; wrapper forward on random CUDA tensors failed]\n"
                "The static validator loaded your code and found callable entrypoint(s), but calling the wrapper "
                "with probe tensors raised the error below. **Fix this runtime error** in your wrapper/kernel; "
                "do not only tweak imports or syntax.\n"
                "```text\n"
                f"{probe_err_text}\n"
                "```\n\n"
            )

    traceback_block = ""
    if kind in {"runtime", "import"} or (kind == "invoke_no_result" and not forward_probe_block):
        tb_focus = _task8_react_traceback_focus(msg)
        if tb_focus:
            traceback_block = (
                "[Traceback focus — fix the line in **your** wrapper/kernel, not the validator]\n"
                "```text\n"
                f"{tb_focus}\n"
                "```\n\n"
            )

    fix_block = "\n".join(f"- {s}" for s in fix_steps)
    avoid_block = "\n".join(f"- {s}" for s in avoid_steps)

    return (
        f"{zero_shot_prompt}\n\n"
        "### Repair round (task8 v3) — targeted bugfix\n"
        f"Round {round_idx}/{max_rounds}.\n\n"
        f"**Issue:** {headline}\n\n"
        f"**Diagnosis:** {diagnosis}\n\n"
        "- Reply with **only** one ``<label>...</label>`` pair containing the **full** corrected Python program.\n"
        "- **No** chit-chat, apologies, or markdown outside the tags.\n\n"
        "[Previous raw model output]\n"
        "```text\n"
        f"{last_code[:12000]}\n"
        "```\n\n"
        "[Previous extracted Python (what the validator ran)]\n"
        "```python\n"
        f"{(last_py if last_py.strip() else _extract_code_candidate(last_code))[:14000]}\n"
        "```\n\n"
        f"{snippet_block}"
        f"{structure_snip}"
        f"{forward_probe_block}"
        f"{traceback_block}"
        "[Validator record]\n"
        f"- outcome_kind: ``{kind}``\n"
        f"- message (truncated):\n```\n{msg[:4500]}\n```\n\n"
        "[Required fixes — apply ALL that match the diagnosis]\n"
        f"{fix_block}\n\n"
        "[Do NOT]\n"
        f"{avoid_block}\n\n"
        "[Original task instruction — still must satisfy]\n"
        f"{input_text}\n"
    )


def _raw_to_prediction_code(raw: str) -> str:
    """v3：先剥 thinking，再走标签/代码块提取（与执行器一致）。"""
    stripped = _strip_model_thinking_prefix(raw)
    return _extract_code_candidate(stripped)


def _task8_fetch_model_raw(prompt: str) -> str:
    """
    拉取云端完整文本：`annotate_nvidia_raw_text` 若为空或仅空白，则回退到 ``annotate``
    （经 ``count_answer``）。避免 API 返回空 ``message.content`` 时整条链路 raw_len=0。
    """
    raw_full = annotate_nvidia_raw_text(prompt)
    if raw_full is not None and str(raw_full).strip():
        return str(raw_full).strip()
    raw_prediction = annotate(prompt)
    return "" if raw_prediction is None else str(raw_prediction).strip()


def _infer_task8_item_v3(
    item: dict,
    task_description: str,
    task_id: int,
    cleaned_examples: list[dict],
    task8_shot_k: int,
    task8_retrieval_pool_size: int,
    retries: int,
    retry_wait_seconds: float,
    *,
    enable_react: bool = True,
    max_react_rounds: int = 5,
    require_forward_result: bool = False,
    accuracy_criterion: str = "exec_outcome_match",
    task8_numeric_rtol: float = 0.05,
    task8_numeric_atol: float = 1e-4,
    task8_icl_retrieval: str = "bm25_lexical",
    task8_bm25_llm_pool: int = 20,
    task8_bm25_llm_final_k: int = 5,
) -> dict:
    example_id = item["example_id"]
    input_text = item["input_text"]
    expected = item["expected"]

    retrieval_query = _strip_task8_retrieval_boilerplate(input_text)
    retrieval_corpus = _task8_cleaned_examples_for_retrieval(cleaned_examples)

    mode = str(task8_icl_retrieval or "bm25_lexical").strip().lower()
    if mode == "bm25_llm":
        final_kcap = max(1, int(task8_bm25_llm_final_k))
        final_k = min(max(1, task8_shot_k), final_kcap)
        examples_str = select_examples_bm25_llm_rerank(
            retrieval_corpus,
            task_description,
            retrieval_query,
            top_k=final_k,
            bm25_pool_size=max(1, int(task8_bm25_llm_pool)),
            use_explanation=False,
            exclude_example_id=example_id,
            query_only=True,
            log_llm_selection=True,
        )
    elif mode == "semantic":
        examples_str = select_examples_hybrid(
            all_examples=retrieval_corpus,
            task_description=task_description,
            text2annotate=retrieval_query,
            top_k=max(1, task8_shot_k),
            rerank_pool_size=max(20, task8_retrieval_pool_size),
            use_explanation=False,
            use_bm25_semantic_rerank=True,
            use_bm25_keyword_only=False,
            exclude_example_id=example_id,
        )
    elif mode == "jaccard_only":
        examples_str = select_examples_hybrid(
            all_examples=retrieval_corpus,
            task_description=task_description,
            text2annotate=retrieval_query,
            top_k=max(1, task8_shot_k),
            rerank_pool_size=max(20, task8_retrieval_pool_size),
            use_explanation=False,
            use_bm25_semantic_rerank=False,
            use_bm25_keyword_only=False,
            exclude_example_id=example_id,
        )
    else:
        examples_str = select_examples_hybrid(
            all_examples=retrieval_corpus,
            task_description=task_description,
            text2annotate=retrieval_query,
            top_k=max(1, task8_shot_k),
            rerank_pool_size=max(20, task8_retrieval_pool_size),
            use_explanation=False,
            use_bm25_semantic_rerank=False,
            use_bm25_keyword_only=True,
            exclude_example_id=example_id,
        )

    prompt = build_prompt(task_description, input_text, task_id=task_id)
    prompt = _inject_v3_contract_before_examples(prompt)
    base_prompt = prompt.replace("[[EXAMPLES]]\n\n", f"{examples_str}\n")

    print(
        f"[v3开始] task=8 example_id={example_id} icl_retrieval={mode}"
        + (
            f" bm25_llm_pool={task8_bm25_llm_pool} bm25_llm_final_k={task8_bm25_llm_final_k}"
            if mode == "bm25_llm"
            else ""
        )
    )
    prediction = ""
    last_model_raw = ""
    for attempt in range(1, retries + 1):
        try:
            last_model_raw = _task8_fetch_model_raw(base_prompt)
            prediction = _raw_to_prediction_code(last_model_raw)
            print(
                f"[v3首次] example_id={example_id} attempt={attempt}/{retries} "
                f"raw_len={len(last_model_raw)} code_len={len(prediction)}"
                + (
                    " [提示: raw 非空但提取为空，检查 <label>/``` 或 thinking 包裹]"
                    if last_model_raw.strip() and not prediction.strip()
                    else ""
                )
            )
            if last_model_raw.strip():
                break
            if attempt < retries:
                print(
                    f"[v3首次空响应重试] example_id={example_id} attempt={attempt}/{retries} "
                    f"(raw 与 annotate 均无正文)"
                )
                time.sleep(retry_wait_seconds)
            else:
                break
        except Exception as e:  # noqa: BLE001
            if attempt >= retries:
                print(f"[v3推理失败] example_id={example_id} attempt={attempt}/{retries} error={e}")
            else:
                print(f"[v3重试] example_id={example_id} attempt={attempt}/{retries} error={e}")
                time.sleep(retry_wait_seconds)

    code_executor = Task8CodeExecutor()
    pred_outcome = Task8ExecOutcome(ok=False, kind="empty", message="empty prediction")
    passed = False
    err_msg = ""

    if prediction.strip():
        passed, pred_outcome = _task8_validate_prediction(
            prediction,
            code_executor,
            require_forward_result=require_forward_result,
        )
        err_msg = pred_outcome.message
        if not passed and pred_outcome.kind in {"syntax", "structure_no_def"}:
            print(
                f"[v3 AST门控] example_id={example_id} kind={pred_outcome.kind} "
                f"(跳过 exec 直至 ReAct)"
            )

    if passed:
        print(f"[v3执行成功] example_id={example_id} 首轮通过")
    elif prediction.strip():
        print(
            f"[v3首轮未过] example_id={example_id} kind={pred_outcome.kind} "
            f"preview={str(err_msg)[:280].replace(chr(10), ' ')}"
        )

    react_round = 0
    react_prediction = prediction
    zero_shot_prompt = _build_task8_zero_shot_prompt(task_description, input_text, task_id)

    while enable_react and not passed and react_round < max_react_rounds:
        react_round += 1
        print(f"[v3 ReAct] example_id={example_id} round={react_round}/{max_react_rounds}")
        repair_body = last_model_raw if last_model_raw.strip() else prediction
        react_prompt = _build_task8_react_repair_prompt_v3(
            zero_shot_prompt=zero_shot_prompt,
            input_text=input_text,
            last_code=repair_body,
            last_extracted_python=react_prediction
            if react_prediction.strip()
            else _extract_code_candidate(repair_body),
            outcome_kind=pred_outcome.kind,
            outcome_message=pred_outcome.message,
            round_idx=react_round,
            max_rounds=max_react_rounds,
        )
        try:
            last_model_raw = ""
            for sub_attempt in range(1, max(1, retries) + 1):
                last_model_raw = _task8_fetch_model_raw(react_prompt)
                react_prediction = _raw_to_prediction_code(last_model_raw)
                if last_model_raw.strip():
                    break
                if sub_attempt < max(1, retries):
                    print(
                        f"[v3 ReAct空响应重试] example_id={example_id} round={react_round} "
                        f"sub={sub_attempt}/{retries}"
                    )
                    time.sleep(retry_wait_seconds)
            print(
                f"[v3 ReAct答] example_id={example_id} round={react_round} "
                f"raw_len={len(last_model_raw)} code_len={len(react_prediction)}"
                + (
                    " [提示: raw 非空但提取为空，检查 <label>/``` 或 thinking 包裹]"
                    if last_model_raw.strip() and not react_prediction.strip()
                    else ""
                )
            )
        except Exception as e:  # noqa: BLE001
            print(f"[v3 ReAct失败] example_id={example_id} round={react_round} error={e}")
            break

        if not react_prediction.strip():
            pred_outcome = Task8ExecOutcome(ok=False, kind="empty", message="empty after ReAct extract")
            continue

        passed, pred_outcome = _task8_validate_prediction(
            react_prediction,
            code_executor,
            require_forward_result=require_forward_result,
            verbose=True,
            log_prefix=f"[v3 ReAct r{react_round}]",
        )
        err_msg = pred_outcome.message
        if not passed and pred_outcome.kind in {"syntax", "structure_no_def"}:
            print(f"[v3 ReAct AST] example_id={example_id} kind={pred_outcome.kind}")
        if passed:
            print(f"[v3 ReAct过] example_id={example_id} round={react_round}")
        elif pred_outcome.kind == "invoke_no_result":
            print(
                f"[v3 ReAct前向探针失败] example_id={example_id} round={react_round} "
                f"(执行器已通过，wrapper 实参前向报错) preview={str(err_msg)[:280].replace(chr(10), ' ')}"
            )
        else:
            print(
                f"[v3 ReAct执行器失败] example_id={example_id} round={react_round} "
                f"kind={pred_outcome.kind} preview={str(err_msg)[:280].replace(chr(10), ' ')}"
            )

    final_output = react_prediction if react_round > 0 else prediction

    has_ref_answer = isinstance(expected, str) and expected.strip()
    ref_outcome: Optional[Task8ExecOutcome] = (
        code_executor.execute(expected, verbose=False) if has_ref_answer else None
    )

    if final_output.strip():
        passed, prediction_outcome = _task8_validate_prediction(
            final_output,
            code_executor,
            require_forward_result=require_forward_result,
        )
        err_msg = prediction_outcome.message
    else:
        prediction_outcome = pred_outcome
        passed = False
        err_msg = pred_outcome.message

    numeric_allclose_ok: bool | None = None
    numeric_compare_detail: dict[str, Any] | None = None
    if accuracy_criterion == "tensor_allclose":
        if has_ref_answer and final_output.strip():
            nm, detail = task8_numeric_allclose_compare(
                expected if isinstance(expected, str) else "",
                final_output,
                rtol=task8_numeric_rtol,
                atol=task8_numeric_atol,
            )
            numeric_allclose_ok = nm
            numeric_compare_detail = detail
            print(
                f"[v3数值探针] example_id={example_id} match={detail.get('match')} "
                f"msg={detail.get('message', '')}"
            )

    is_match, is_text_match, exec_dual_pass = _infer_compute_task8_accuracy(
        criterion=accuracy_criterion,
        expected_non_empty=has_ref_answer,
        expected_text=expected if isinstance(expected, str) else "",
        final_output=final_output,
        ref_outcome=ref_outcome,
        pred_outcome=prediction_outcome,
        numeric_allclose_ok=numeric_allclose_ok,
    )

    executor_outcomes_match: bool | None
    if has_ref_answer and ref_outcome is not None:
        executor_outcomes_match = bool(
            ref_outcome.ok == prediction_outcome.ok and ref_outcome.kind == prediction_outcome.kind
        )
    else:
        executor_outcomes_match = None

    exec_cmp = _task8_build_exec_comparison(
        ref_outcome,
        prediction_outcome,
        has_reference=has_ref_answer,
    )

    if has_ref_answer and ref_outcome is not None:
        _rm = _truncate_for_log(ref_outcome.message, 600).replace("\n", "\\n")
        print(
            f"[v3 label执行] example_id={example_id} "
            f"ok={ref_outcome.ok} kind={ref_outcome.kind} msg={_rm}"
        )
    else:
        print(f"[v3 label执行] example_id={example_id} (无参考答案或为空，跳过)")
    _pm = _truncate_for_log(prediction_outcome.message, 600).replace("\n", "\\n")
    print(
        f"[v3 预测执行] example_id={example_id} "
        f"ok={prediction_outcome.ok} kind={prediction_outcome.kind} msg={_pm}"
    )

    print(
        f"[v3完成] example_id={example_id} is_match={is_match} exec_dual_pass={exec_dual_pass} "
        f"executor_outcomes_match={executor_outcomes_match} react_rounds={react_round}"
        + (
            f" numeric_allclose={numeric_compare_detail.get('match')}"
            if numeric_compare_detail
            else ""
        )
    )
    return {
        "example_id": example_id,
        "input": input_text,
        "expected_output": expected,
        "model_output": final_output,
        "pipeline": "task8_compare_v3",
        "accuracy_criterion": accuracy_criterion,
        "numeric_compare": numeric_compare_detail,
        "is_match": is_match,
        "is_text_match": is_text_match,
        "exec_dual_pass": exec_dual_pass,
        "executor_outcomes_match": executor_outcomes_match,
        "exec_comparison": exec_cmp,
        "executor_reference_ok": None if ref_outcome is None else ref_outcome.ok,
        "executor_reference_kind": None if ref_outcome is None else ref_outcome.kind,
        "executor_prediction_ok": prediction_outcome.ok,
        "executor_prediction_kind": prediction_outcome.kind,
        "passed_sanity_check": prediction_outcome.ok,
        "react_rounds": react_round,
        "forward_probe_ok": (
            None
            if not require_forward_result
            else bool(passed and prediction_outcome.kind != "invoke_no_result")
        ),
        "last_error": prediction_outcome.message if not prediction_outcome.ok else "",
    }


def run_task8_v3(
    output_dir: Path,
    examples_limit: int = 0,
    retries: int = 3,
    retry_wait_seconds: float = 2.0,
    resume: bool = False,
    task8_shot_k: int = 6,
    task8_retrieval_pool_size: int = 200,
    print_empty_prediction: str = "on",
    retrieval_batch_size: int = 8,
    accuracy_criterion: str = "exec_outcome_match",
    max_react_rounds: int = 5,
    task8_numeric_rtol: float = 0.05,
    task8_numeric_atol: float = 1e-4,
    single_example_id: str = "",
    single_index: int = -1,
    task8_icl_retrieval: str = "bm25_lexical",
    task8_bm25_llm_pool: int = 20,
    task8_bm25_llm_final_k: int = 5,
) -> dict:
    with _task_json_path().open("r", encoding="utf-8") as f:
        task_dict = json.load(f)

    task_id = 8
    task_name = task_dict["task_name"]
    task_description = TASK8_CANONICAL_DESCRIPTION
    all_examples = task_dict["examples"]
    if examples_limit > 0:
        all_examples = all_examples[:examples_limit]

    output_file = output_dir / f"openseek-{task_id}-examples-compare-task8opt-v3.jsonl"
    done_ids = _load_done_ids(output_file) if resume else set()
    mode = "a" if resume else "w"

    if done_ids:
        print(f"[v3断点续跑] 已完成 {len(done_ids)} 条")

    cleaned_examples: list[dict] = []
    for ex in task_dict["examples"]:
        cleaned_examples.append(
            {
                "id": str(ex.get("id", "")).strip(),
                "input": str(ex.get("input", "")),
                "output": [_extract_output(ex.get("output", ""))],
            }
        )

    pending_items: list[dict] = []
    for example in all_examples:
        example_id = str(example.get("id", "")).strip()
        if resume and example_id in done_ids:
            continue
        pending_items.append(
            {
                "example_id": example_id,
                "input_text": str(example.get("input", "")),
                "expected": _extract_output(example.get("output", "")),
            }
        )

    pending_items = _narrow_pending_to_single_row(
        pending_items,
        single_example_id=single_example_id,
        single_index=single_index,
    )

    bs = max(1, retrieval_batch_size)
    with output_file.open(mode, encoding="utf-8") as wf:
        for i in tqdm(range(0, len(pending_items), bs), desc=f"Task8 v3: {task_name}"):
            chunk = pending_items[i : i + bs]
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(bs, len(chunk))) as tp_executor:
                futures = [
                    tp_executor.submit(
                        _infer_task8_item_v3,
                        item=item,
                        task_description=task_description,
                        task_id=task_id,
                        cleaned_examples=cleaned_examples,
                        task8_shot_k=task8_shot_k,
                        task8_retrieval_pool_size=task8_retrieval_pool_size,
                        retries=retries,
                        retry_wait_seconds=retry_wait_seconds,
                        enable_react=True,
                        max_react_rounds=max_react_rounds,
                        accuracy_criterion=accuracy_criterion,
                        task8_numeric_rtol=task8_numeric_rtol,
                        task8_numeric_atol=task8_numeric_atol,
                        task8_icl_retrieval=task8_icl_retrieval,
                        task8_bm25_llm_pool=task8_bm25_llm_pool,
                        task8_bm25_llm_final_k=task8_bm25_llm_final_k,
                    )
                    for item in chunk
                ]
                infer_results = [f.result() for f in futures]

            for row in infer_results:
                if print_empty_prediction == "on" and not row["model_output"]:
                    print(f"[空预测] example_id={row['example_id']}")
                wf.write(json.dumps(row, ensure_ascii=False) + "\n")
                wf.flush()
                done_ids.add(row["example_id"])
                if str(single_example_id or "").strip() or single_index >= 0:
                    _print_task8_exec_comparison_report(row.get("exec_comparison") or {})

    total, match_count, accuracy = _compute_metrics_from_jsonl(output_file)
    print(
        f"[v3保存] criterion={accuracy_criterion}, total={total}, matched={match_count}, "
        f"accuracy={accuracy:.2%}, file={output_file}"
    )
    return {
        "task_id": 8,
        "task_name": task_name,
        "accuracy_criterion": accuracy_criterion,
        "pipeline": "task8_compare_v3",
        "total": total,
        "matched": match_count,
        "accuracy": accuracy,
        "file": str(output_file),
    }


def run_task8_test_samples_v3(
    output_dir: Path,
    retries: int = 3,
    retry_wait_seconds: float = 2.0,
    resume: bool = False,
    task8_shot_k: int = 6,
    task8_retrieval_pool_size: int = 200,
    retrieval_batch_size: int = 8,
    accuracy_criterion: str = "exec_outcome_match",
    max_react_rounds: int = 5,
    task8_numeric_rtol: float = 0.05,
    task8_numeric_atol: float = 1e-4,
    single_example_id: str = "",
    single_index: int = -1,
    task8_icl_retrieval: str = "bm25_lexical",
    task8_bm25_llm_pool: int = 20,
    task8_bm25_llm_final_k: int = 5,
    require_forward_result: bool = False,
) -> dict:
    with _task_json_path().open("r", encoding="utf-8") as f:
        task_dict = json.load(f)

    task_id = 8
    task_name = task_dict["task_name"]
    task_description = TASK8_CANONICAL_DESCRIPTION
    test_samples = list(task_dict.get("test_samples", []))

    output_file = output_dir / f"openseek-{task_id}-test_samples-predictions-v3.jsonl"
    done_ids = _load_done_ids(output_file) if resume else set()
    mode = "a" if resume else "w"

    cleaned_examples: list[dict] = []
    for ex in task_dict["examples"]:
        cleaned_examples.append(
            {
                "id": str(ex.get("id", "")).strip(),
                "input": str(ex.get("input", "")),
                "output": [_extract_output(ex.get("output", ""))],
            }
        )

    pending_items: list[dict] = []
    for sample in test_samples:
        sample_id = str(sample.get("id", "")).strip()
        if resume and sample_id in done_ids:
            continue
        pending_items.append(
            {
                "example_id": sample_id,
                "input_text": str(sample.get("input", "")),
                "expected": "",
            }
        )

    pending_items = _narrow_pending_to_single_row(
        pending_items,
        single_example_id=single_example_id,
        single_index=single_index,
    )

    bs = max(1, retrieval_batch_size)
    total = 0
    with output_file.open(mode, encoding="utf-8") as wf:
        for i in tqdm(range(0, len(pending_items), bs), desc=f"Task8 v3 test_samples: {task_name}"):
            chunk = pending_items[i : i + bs]
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(bs, len(chunk))) as tp_executor:
                futures = [
                    tp_executor.submit(
                        _infer_task8_item_v3,
                        item=item,
                        task_description=task_description,
                        task_id=task_id,
                        cleaned_examples=cleaned_examples,
                        task8_shot_k=task8_shot_k,
                        task8_retrieval_pool_size=task8_retrieval_pool_size,
                        retries=retries,
                        retry_wait_seconds=retry_wait_seconds,
                        enable_react=True,
                        max_react_rounds=max_react_rounds,
                        accuracy_criterion=accuracy_criterion,
                        task8_numeric_rtol=task8_numeric_rtol,
                        task8_numeric_atol=task8_numeric_atol,
                        task8_icl_retrieval=task8_icl_retrieval,
                        task8_bm25_llm_pool=task8_bm25_llm_pool,
                        task8_bm25_llm_final_k=task8_bm25_llm_final_k,
                        require_forward_result=require_forward_result,
                    )
                    for item in chunk
                ]
                infer_results = [f.result() for f in futures]

            for row in infer_results:
                wf.write(
                    json.dumps(
                        {
                            "test_sample_id": row["example_id"],
                            "prediction": row["model_output"],
                            "exec_comparison": row.get("exec_comparison"),
                            "react_rounds": row.get("react_rounds", 0),
                            "passed_sanity_check": row.get("passed_sanity_check", False),
                            "forward_probe_ok": row.get("forward_probe_ok"),
                            "pipeline": "task8_compare_v3",
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                wf.flush()
                done_ids.add(row["example_id"])
                total += 1
                if str(single_example_id or "").strip() or single_index >= 0:
                    _print_task8_exec_comparison_report(row.get("exec_comparison") or {})

    print(f"[v3 test_samples 保存] total={total}, file={output_file}")
    return {
        "task_id": 8,
        "task_name": task_name,
        "pipeline": "task8_compare_v3",
        "total": total,
        "file": str(output_file),
    }


def main() -> None:
    args = parse_args()
    os.environ["DASHSCOPE_ENABLE_THINKING"] = "1" if args.thinking == "on" else "0"
    os.environ["ANNOTATE_LOG_EVERY_RESPONSE"] = "1" if args.print_model_output == "on" else "0"
    print(f"[v3 thinking] DASHSCOPE_ENABLE_THINKING={os.environ['DASHSCOPE_ENABLE_THINKING']}")

    output_dir = _resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[v3 判据] accuracy_criterion={args.accuracy_criterion} max_react_rounds={args.max_react_rounds} "
          f"task8_numeric_rtol={args.task8_numeric_rtol} task8_numeric_atol={args.task8_numeric_atol} "
          f"task8_icl_retrieval={args.task8_icl_retrieval}")

    single_id = str(args.single_example_id or "").strip()
    if single_id and args.single_index >= 0:
        raise SystemExit("请只指定 --single_example_id 或 --single_index 之一。")

    single_kw = {
        "single_example_id": single_id,
        "single_index": int(args.single_index),
    }

    if args.infer_split == "test_samples":
        summary_item = run_task8_test_samples_v3(
            output_dir=output_dir,
            retries=args.retries,
            retry_wait_seconds=args.retry_wait_seconds,
            resume=args.resume,
            task8_shot_k=args.task8_shot_k,
            task8_retrieval_pool_size=args.task8_retrieval_pool_size,
            retrieval_batch_size=args.retrieval_batch_size,
            accuracy_criterion=args.accuracy_criterion,
            task8_numeric_rtol=args.task8_numeric_rtol,
            task8_numeric_atol=args.task8_numeric_atol,
            max_react_rounds=max(1, args.max_react_rounds),
            task8_icl_retrieval=args.task8_icl_retrieval,
            task8_bm25_llm_pool=args.task8_bm25_llm_pool,
            task8_bm25_llm_final_k=args.task8_bm25_llm_final_k,
            **single_kw,
        )
        summary_file = output_dir / "summary_task8_test_samples_v3.json"
    else:
        summary_item = run_task8_v3(
            output_dir=output_dir,
            examples_limit=args.examples_limit,
            retries=args.retries,
            retry_wait_seconds=args.retry_wait_seconds,
            resume=args.resume,
            task8_shot_k=args.task8_shot_k,
            task8_retrieval_pool_size=args.task8_retrieval_pool_size,
            print_empty_prediction=args.print_empty_prediction,
            retrieval_batch_size=args.retrieval_batch_size,
            accuracy_criterion=args.accuracy_criterion,
            task8_numeric_rtol=args.task8_numeric_rtol,
            task8_numeric_atol=args.task8_numeric_atol,
            max_react_rounds=max(1, args.max_react_rounds),
            task8_icl_retrieval=args.task8_icl_retrieval,
            task8_bm25_llm_pool=args.task8_bm25_llm_pool,
            task8_bm25_llm_final_k=args.task8_bm25_llm_final_k,
            **single_kw,
        )
        summary_file = output_dir / "summary_task8opt_v3.json"

    with summary_file.open("w", encoding="utf-8") as f:
        json.dump([summary_item], f, ensure_ascii=False, indent=2)
    print(f"[v3 汇总] {summary_file}")


if __name__ == "__main__":
    main()
