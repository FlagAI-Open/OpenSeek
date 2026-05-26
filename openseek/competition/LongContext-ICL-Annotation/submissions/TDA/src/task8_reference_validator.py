import inspect
import json
import re
import zipfile
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"
SYSTEM_DIR = OUTPUT_DIR / "0513_systematic"


def load_rows(zip_name: str) -> dict[int, list[dict[str, Any]]]:
    rows: dict[int, list[dict[str, Any]]] = {}
    with zipfile.ZipFile(OUTPUT_DIR / zip_name) as zf:
        for task_id in range(1, 9):
            rows[task_id] = [
                json.loads(line)
                for line in zf.read(f"openseek-{task_id}-v1.jsonl").decode("utf-8").splitlines()
                if line.strip()
            ]
    return rows


def tensor(shape: tuple[int, ...], positive: bool = False, integer: bool = False) -> torch.Tensor:
    if integer:
        return torch.randint(0, max(shape[-1], 2), shape, dtype=torch.long)
    value = torch.rand(*shape) if positive else torch.randn(*shape)
    return value + 1.0 if positive else value


def normalize_output(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach()
    if isinstance(value, (tuple, list)):
        return type(value)(normalize_output(item) for item in value)
    return value


def outputs_close(left: Any, right: Any) -> bool:
    left = normalize_output(left)
    right = normalize_output(right)
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        if left.shape != right.shape:
            return False
        if left.dtype == torch.bool or right.dtype == torch.bool:
            return torch.equal(left, right)
        return torch.allclose(left.float(), right.float(), atol=1e-4, rtol=1e-4, equal_nan=True)
    if isinstance(left, (tuple, list)) and isinstance(right, (tuple, list)):
        return len(left) == len(right) and all(outputs_close(a, b) for a, b in zip(left, right))
    return left == right


def import_function(code: str, name: str) -> tuple[Any | None, str]:
    namespace: dict[str, Any] = {
        "torch": torch,
        "F": F,
        "Tensor": torch.Tensor,
        "Tuple": tuple,
        "List": list,
        "Union": Any,
    }
    try:
        exec(code, namespace, namespace)
    except Exception as exc:  # noqa: BLE001 - diagnostic path.
        return None, f"exec:{type(exc).__name__}: {exc}"
    fn = namespace.get(name)
    if not callable(fn):
        return None, "missing_function"
    return fn, ""


def call_with_signature(fn: Any, args: dict[str, Any]) -> Any:
    signature = inspect.signature(fn)
    positional = []
    keywords = {}
    for name, parameter in signature.parameters.items():
        if parameter.kind == inspect.Parameter.VAR_POSITIONAL:
            positional.extend(args.get(name, ()))
            continue
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            continue
        if name not in args:
            if parameter.default is inspect.Parameter.empty:
                raise ValueError(f"missing required arg {name}")
            continue
        if parameter.kind in {inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD}:
            positional.append(args[name])
        else:
            keywords[name] = args[name]
    return fn(*positional, **keywords)


def cases_for(name: str) -> list[dict[str, Any]]:
    # Only include cases whose tensor shapes are directly implied by the task interface.
    if name == "fused_bmm_rmsnorm_gelu_dropout_sub":
        return [
            {
                "input1": tensor((2, 3, 4)),
                "input2": tensor((2, 4, 5)),
                "other": tensor((2, 3, 5)),
                "normalized_shape": 5,
                "dropout_p": 0.0,
                "training": False,
            }
        ]
    if name == "fused_cosine_embedding_loss_with_normalization":
        return [
            {
                "input1": tensor((3, 4)),
                "input2": tensor((3, 4)),
                "target": torch.tensor([1, -1, 1]),
                "margin": 0.0,
                "reduction": "mean",
            }
        ]
    if name == "fused_mul_add_logsoftmax_dropout_bmm":
        return [
            {
                "input1": tensor((2, 3, 4)),
                "input2": tensor((2, 3, 4)),
                "other": tensor((2, 3, 4)),
                "mat2": tensor((2, 4, 5)),
                "p": 0.0,
                "training": False,
                "inplace": False,
                "dim": -1,
            }
        ]
    if name == "combined_activation":
        return [
            {"input": tensor((2, 4)), "weight1": tensor((4, 4)), "weight2": tensor((4,)), "bias": tensor((4,))}
        ]
    if name in {"tensordot", "tensordot_rsqrt"}:
        return [
            {"a": tensor((2, 3, 4), positive=name.endswith("rsqrt")), "b": tensor((4, 3, 5), positive=name.endswith("rsqrt")), "dims": ([2, 1], [0, 1])},
            {"a": tensor((2, 3)), "b": tensor((3, 4)), "dims": 1},
        ]
    if name == "fused_pairwise_distance_adaptive_avg_pool2d":
        return [
            {"x1": tensor((2, 3, 6, 6)), "x2": tensor((2, 3, 6, 6)), "output_size": (3, 3), "p": 2.0, "eps": 1e-6, "keepdim": False}
        ]
    if name == "relu_batch_norm_conv2d":
        return [
            {
                "input": tensor((2, 3, 8, 8)),
                "weight": tensor((4, 3, 3, 3)),
                "bias": torch.zeros(4),
                "stride": 1,
                "padding": 1,
                "dilation": 1,
                "groups": 1,
                "running_mean": torch.zeros(4),
                "running_var": torch.ones(4),
                "bn_weight": torch.ones(4),
                "bn_bias": torch.zeros(4),
                "training": False,
            }
        ]
    if name == "matmul":
        return [
            {"input": tensor((2, 3)), "other": tensor((3, 4))},
            {"input": tensor((3,)), "other": tensor((3, 4))},
            {"input": tensor((2, 3)), "other": tensor((3,))},
        ]
    if name == "fused_gather_masked_fill":
        index = torch.tensor([[0, 2], [1, 3]])
        return [
            {
                "input": tensor((2, 4)),
                "dim": 1,
                "index": index,
                "mask": torch.tensor([[True, False], [False, True]]),
                "value": -9.0,
                "sparse_grad": False,
            }
        ]
    if name == "sigmoid_adaptive_avg_pool2d":
        return [{"input": tensor((2, 3, 6, 6)), "output_size": (3, 3)}]
    if name == "fused_bmm_dropout_gelu":
        return [{"input1": tensor((2, 3, 4)), "input2": tensor((2, 4, 5)), "p": 0.0, "training": False, "approximate": "none"}]
    if name == "dropout_sigmoid_linear":
        return [{"input": tensor((3, 4)), "weight": tensor((5, 4)), "bias": tensor((5,)), "p": 0.0, "training": False}]
    return []


def validate_candidates() -> dict[str, Any]:
    candidate_path = SYSTEM_DIR / "task8_qwen_reference_candidates_v3.json"
    candidates = json.loads(candidate_path.read_text(encoding="utf-8"))
    current_rows = load_rows("submission_0513v4.zip")[8]
    current_by_id = {row["test_sample_id"]: row["prediction"] for row in current_rows}
    report = []
    accepted = []
    for candidate in candidates:
        if not candidate["compile_ok"]:
            report.append({**candidate, "validation": "reject_compile"})
            continue
        name = candidate["name"]
        case_list = cases_for(name)
        if not case_list:
            report.append({**candidate, "validation": "reject_no_cases"})
            continue
        new_fn, new_error = import_function(candidate["code"], name)
        current_fn, current_error = import_function(current_by_id[candidate["id"]], name)
        if new_fn is None:
            report.append({**candidate, "validation": "reject_exec", "validation_error": new_error})
            continue
        if current_fn is None:
            report.append({**candidate, "validation": "reject_current_exec", "validation_error": current_error})
            continue
        case_results = []
        changed_behavior = False
        all_new_ok = True
        for args in case_list:
            try:
                new_out = call_with_signature(new_fn, args)
                new_ok = True
                new_msg = ""
            except Exception as exc:  # noqa: BLE001 - diagnostic.
                new_out = None
                new_ok = False
                new_msg = f"{type(exc).__name__}: {exc}"
            try:
                old_out = call_with_signature(current_fn, args)
                old_ok = True
                old_msg = ""
            except Exception as exc:  # noqa: BLE001 - diagnostic.
                old_out = None
                old_ok = False
                old_msg = f"{type(exc).__name__}: {exc}"
            all_new_ok = all_new_ok and new_ok
            if new_ok and old_ok and not outputs_close(new_out, old_out):
                changed_behavior = True
            if new_ok and not old_ok:
                changed_behavior = True
            case_results.append({"new_ok": new_ok, "old_ok": old_ok, "new_msg": new_msg, "old_msg": old_msg})
        validation = "accept" if all_new_ok and changed_behavior else "reject_no_improvement"
        row = {**candidate, "validation": validation, "case_results": case_results}
        report.append(row)
        if validation == "accept":
            accepted.append({"id": candidate["id"], "name": name, "code": candidate["code"], "case_results": case_results})
    result = {"accepted": accepted, "report": report}
    (SYSTEM_DIR / "task8_reference_validation_report_v3.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    result = validate_candidates()
    print(f"accepted {len(result['accepted'])}")
    for item in result["accepted"]:
        print(item["id"], item["name"], item["case_results"])
