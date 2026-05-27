#!/usr/bin/env python3
"""Runtime proxy checks for task-8 code-generation predictions.

This is not the official hidden evaluator. It checks the part we can validate:
generated source can be executed, the expected wrapper function exists, and a
small synthetic call returns without an exception in a PyTorch environment.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[0]
sys.path.insert(0, str(SCRIPT_DIR))

from build_task8_fallback_submission import _extract_name_and_args  # noqa: E402


RAW_TASK8 = ROOT / "data/raw/openseek-8_kernel_generation.json"
DEFAULT_PREDICTIONS = ROOT / "outputs/task8_fallback_candidate/predictions/openseek-8-v1.jsonl"


class _FallbackUsed(RuntimeError):
    pass


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_samples(path: Path) -> dict[str, dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {sample["id"]: sample for sample in raw["test_samples"]}


def _matrix(torch: Any, n: int = 4) -> Any:
    eye = torch.eye(n)
    return eye + 0.1 * torch.randn(n, n)


def _positive_tensor(torch: Any, *shape: int) -> Any:
    return torch.rand(*shape) + 0.25


def _param_value(torch: Any, fn_name: str, name: str, default: Any) -> Any:
    lname = name.lower()
    fn = fn_name.lower()

    if default is not inspect._empty and default is not None:
        return default
    if fn == "fused_index_select_eq":
        if lname == "input":
            return torch.randn(4, 4)
        if lname == "index":
            return torch.tensor([0, 1, 2, 3], dtype=torch.long)
        if lname == "other":
            return torch.randn(4, 4)
    if fn == "fused_gather_masked_fill":
        if lname == "input":
            return torch.randn(4, 4)
        if lname == "index":
            return torch.tensor([[0, 1, 2, 3]] * 4, dtype=torch.long)
        if lname == "mask":
            return torch.tensor([[True, False, True, False]] * 4)
    if fn == "fused_mv_sigmoid_sub":
        if lname == "input":
            return torch.randn(4, 4)
        if lname == "vec":
            return torch.randn(4)
        if lname == "other":
            return torch.randn(4)
    if fn == "matrix_vector_dot":
        if lname == "a":
            return torch.randn(4, 4)
        if lname in {"x", "y"}:
            return torch.randn(4)
    if fn == "fused_pairwise_distance_adaptive_avg_pool2d":
        if lname in {"x1", "x2"}:
            return torch.randn(2, 3, 8, 8)
        if lname == "output_size":
            return (4, 4)
    if fn == "fused_avg_pool2d_cosine_similarity":
        if lname in {"x1", "x2"}:
            return torch.randn(2, 3, 8, 8)
        if lname == "kernel_size":
            return 2
    if fn == "permute_copy":
        if lname == "input":
            return torch.randn(2, 3, 4)
        if lname == "dims":
            return (2, 0, 1)
    if fn == "bitwise_and_binomial":
        if lname in {"input", "other"}:
            return torch.randint(0, 4, (4, 4), dtype=torch.int64)
        if lname == "total_count":
            return torch.full((4, 4), 5.0)
        if lname == "probs":
            return torch.full((4, 4), 0.5)
    if lname == "params":
        return [torch.nn.Parameter(torch.randn(2, 2))]
    if lname == "model":
        return torch.nn.Linear(4, 2)
    if lname == "device_type":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if lname in {"input", "x", "y"}:
        if "conv2d" in fn or "pool2d" in fn or "grid_sample" in fn:
            return torch.randn(2, 3, 8, 8)
        if "pool1d" in fn:
            return torch.randn(2, 3, 16)
        if "embedding" in fn:
            return torch.randint(0, 8, (4,), dtype=torch.long)
        if "log" in fn or "sqrt" in fn or "rsqrt" in fn or "zeta" in fn:
            return _positive_tensor(torch, 4, 4)
        if "bitwise" in fn or "signbit" in fn:
            return torch.randint(0, 4, (4, 4), dtype=torch.int64)
        return torch.randn(4, 4)
    if lname in {"input1", "input2"}:
        if "bmm" in fn:
            return torch.randn(2, 4, 4)
        return torch.randn(4, 4)
    if lname in {"other", "other_mul", "other_sub", "divisor"}:
        if "bitwise" in fn:
            return torch.randint(0, 4, (4, 4), dtype=torch.int64)
        return _positive_tensor(torch, 4, 4)
    if lname in {"a", "b", "c"}:
        return _matrix(torch)
    if lname in {"bs"}:
        return torch.randn(4, 2)
    if lname in {"mat1", "mat2"}:
        if "bmm" in fn:
            return torch.randn(2, 4, 4)
        return torch.randn(4, 4)
    if lname in {"vec", "x1", "x2"}:
        return torch.randn(4, 4)
    if lname == "weight":
        if "conv2d" in fn:
            return torch.randn(4, 3, 3, 3)
        if "embedding" in fn:
            return torch.randn(8, 4)
        return torch.randn(4, 4)
    if lname in {"conv_weight"}:
        return torch.randn(4, 3, 3, 3)
    if lname in {"bias", "conv_bias"}:
        if "conv2d" in fn:
            return torch.randn(4)
        return torch.randn(4)
    if lname in {"running_mean", "running_var", "bn_weight", "bn_bias"}:
        if "conv2d" in fn:
            if "var" in lname:
                return torch.ones(4)
            if "weight" in lname:
                return torch.ones(4)
            return torch.zeros(4)
        if "var" in lname:
            return torch.ones(3)
        return torch.zeros(3)
    if lname in {"target", "targets"}:
        return torch.randint(0, 4, (4,), dtype=torch.long)
    if lname in {"index", "input_indices"}:
        return torch.tensor([0, 1, 2, 3], dtype=torch.long)
    if lname in {"mask"}:
        return torch.tensor([[True, False, True, False]] * 4)
    if lname in {"grid"}:
        return torch.rand(2, 4, 4, 2) * 2 - 1
    if lname in {"theta"}:
        return torch.eye(2, 3).unsqueeze(0).repeat(2, 1, 1)
    if lname in {"size"}:
        return (2, 3, 8, 8)
    if lname in {"tensors"}:
        return (torch.randn(2, 2), torch.randn(2, 2))
    if lname in {"normalized_shape"}:
        return (4,)
    if lname in {"output_size"}:
        return (4, 4)
    if lname in {"kernel_size", "pool_kernel_size"}:
        return 2
    if lname in {"dims"}:
        return 1
    if lname in {"dim", "dim_norm"}:
        return 1
    if lname in {"num_groups"}:
        return 1
    if lname in {"n", "k", "steps"}:
        return 2
    if lname in {"start", "end", "alpha", "beta", "value", "exponent"}:
        return 1.0
    if default is not inspect._empty:
        return default
    return torch.randn(4, 4)


def _build_call_args(torch: Any, fn_name: str, fn: Any) -> tuple[list[Any], dict[str, Any]]:
    signature = inspect.signature(fn)
    args: list[Any] = []
    kwargs: dict[str, Any] = {}
    for name, param in signature.parameters.items():
        if param.kind is inspect.Parameter.VAR_POSITIONAL:
            if name == "size":
                args.extend([2, 2])
            elif name == "tensors":
                args.extend([torch.randn(2, 2), torch.randn(2, 2)])
            continue
        if param.kind is inspect.Parameter.VAR_KEYWORD:
            continue
        if param.default is not inspect._empty and name not in {"out"}:
            value = _param_value(torch, fn_name, name, param.default)
        elif name == "out":
            continue
        else:
            value = _param_value(torch, fn_name, name, param.default)
        if param.kind is inspect.Parameter.KEYWORD_ONLY:
            kwargs[name] = value
        else:
            args.append(value)
    return args, kwargs


def check_runtime(raw_path: Path, prediction_path: Path, limit: int | None = None) -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on environment
        return {
            "error": f"PyTorch import failed: {type(exc).__name__}: {exc}",
            "hint": "Run this script inside the OpenBayes PyTorch workspace.",
        }

    samples = load_samples(raw_path)
    rows = load_jsonl(prediction_path)
    if limit is not None:
        rows = rows[:limit]

    details: list[dict[str, Any]] = []
    for row in rows:
        sample_id = row.get("test_sample_id", "")
        sample = samples.get(sample_id)
        expected_name = ""
        if sample:
            expected_name, _ = _extract_name_and_args(sample["input"])
        prediction = str(row.get("prediction", ""))
        detail: dict[str, Any] = {
            "test_sample_id": sample_id,
            "expected_function": expected_name,
            "exec_ok": False,
            "callable_ok": False,
            "call_ok": False,
            "result_tensor_like": False,
            "error": "",
        }
        namespace: dict[str, Any] = {}
        try:
            exec(prediction, namespace)
            detail["exec_ok"] = True
            fn = namespace.get(expected_name)
            detail["callable_ok"] = callable(fn)
            if callable(fn):
                call_args, call_kwargs = _build_call_args(torch, expected_name, fn)
                original_fallback = namespace.get("_fallback_result")
                namespace["_fallback_result"] = lambda *values: (_ for _ in ()).throw(_FallbackUsed("fallback branch used"))
                try:
                    primary_result = fn(*call_args, **call_kwargs)
                    detail["primary_call_ok"] = True
                    detail["primary_result_tensor_like"] = torch.is_tensor(primary_result) or (
                        isinstance(primary_result, (tuple, list))
                        and any(torch.is_tensor(item) for item in primary_result)
                    )
                except _FallbackUsed as exc:
                    detail["primary_error"] = str(exc)
                except Exception as exc:
                    detail["primary_error"] = f"{type(exc).__name__}: {exc}"
                finally:
                    if original_fallback is not None:
                        namespace["_fallback_result"] = original_fallback
                result = fn(*call_args, **call_kwargs)
                detail["call_ok"] = True
                detail["result_tensor_like"] = torch.is_tensor(result) or (
                    isinstance(result, (tuple, list)) and any(torch.is_tensor(item) for item in result)
                )
        except Exception as exc:
            detail["error"] = f"{type(exc).__name__}: {exc}"
        details.append(detail)

    def count(key: str) -> int:
        return sum(1 for item in details if item.get(key))

    return {
        "raw_file": str(raw_path),
        "prediction_file": str(prediction_path),
        "rows": len(details),
        "exec_ok": count("exec_ok"),
        "callable_ok": count("callable_ok"),
        "primary_call_ok": count("primary_call_ok"),
        "primary_result_tensor_like": count("primary_result_tensor_like"),
        "call_ok": count("call_ok"),
        "result_tensor_like": count("result_tensor_like"),
        "details": details,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("prediction_file", nargs="?", default=str(DEFAULT_PREDICTIONS))
    parser.add_argument("--raw", default=str(RAW_TASK8))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    summary = check_runtime(Path(args.raw), Path(args.prediction_file), limit=args.limit)
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(text + "\n", encoding="utf-8")

    print(f"file: {summary.get('prediction_file', args.prediction_file)}")
    if "error" in summary:
        print(summary["error"])
        print(summary.get("hint", ""))
        return
    print(f"rows: {summary['rows']}")
    print(f"exec_ok: {summary['exec_ok']}")
    print(f"callable_ok: {summary['callable_ok']}")
    print(f"primary_call_ok: {summary['primary_call_ok']}")
    print(f"primary_result_tensor_like: {summary['primary_result_tensor_like']}")
    print(f"call_ok: {summary['call_ok']}")
    print(f"result_tensor_like: {summary['result_tensor_like']}")
    failures = [item for item in summary["details"] if not item.get("primary_call_ok")]
    if failures:
        print("first_primary_failures:")
        for item in failures[:10]:
            print(f"- {item['test_sample_id']} {item['expected_function']}: {item.get('primary_error') or item['error']}")


if __name__ == "__main__":
    main()
