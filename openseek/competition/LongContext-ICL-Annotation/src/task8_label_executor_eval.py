"""
Task8 examples 参考答案（label）执行器评估。

对 ``data/openseek-8_kernel_generation.json`` 中每条 example 的 ``output``（金标代码）：
1. 静态执行器（语法 / exec / 可调用入口）
2. 可选前向探针（随机张量跑一次 wrapper，要求返回 torch.Tensor）

统计：
- 可执行率：执行器 ``ok=True`` 的占比
- 有结果率：前向探针成功的占比（默认开启；无 torch 时跳过探针并记为通过）
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# 与 infer_examples_compare_task8_v3 共用同一套执行/探针逻辑
sys.path.insert(0, str(Path(__file__).resolve().parent))
from infer_examples_compare_task8_v3 import (  # noqa: E402
    Task8CodeExecutor,
    _ast_quick_check_cleaned,
    _extract_output,
    _task8_outcome_as_dict,
    task8_probe_forward_ok,
)


def _load_examples(data_path: Path) -> list[dict]:
    with data_path.open("r", encoding="utf-8") as f:
        task_dict = json.load(f)
    return list(task_dict.get("examples", []))


def _eval_one_label(
    example_id: str,
    label_code: str,
    executor: Task8CodeExecutor,
    *,
    require_forward: bool,
) -> dict:
    label_code = str(label_code or "").strip()
    if not label_code:
        return {
            "example_id": example_id,
            "label_len": 0,
            "exec_ok": False,
            "exec_kind": "empty",
            "exec_message": "empty label",
            "forward_ok": False,
            "forward_message": "empty label",
        }

    if not label_code:
        outcome = executor.execute("", verbose=False)
    else:
        quick = _ast_quick_check_cleaned(label_code)
        if not quick.ok:
            outcome = quick
        else:
            outcome = executor.execute(label_code, verbose=False)
    exec_ok = outcome.ok
    exec_d = _task8_outcome_as_dict(outcome)

    if require_forward:
        if not exec_ok:
            fwd_ok, fwd_msg = False, f"skipped_forward: exec_failed ({outcome.kind})"
        else:
            try:
                import torch  # noqa: F401, PLC0415
            except ModuleNotFoundError:
                fwd_ok, fwd_msg = None, "torch_unavailable"
            else:
                fwd_ok, fwd_msg = task8_probe_forward_ok(label_code)
                if fwd_msg == "torch_unavailable_skip_probe":
                    fwd_ok, fwd_msg = None, "torch_unavailable"
    else:
        fwd_ok, fwd_msg = None, "forward_probe_disabled"

    return {
        "example_id": example_id,
        "label_len": len(label_code),
        "exec_ok": exec_ok,
        "exec_kind": exec_d["kind"],
        "exec_message": str(exec_d.get("message_truncated", ""))[:500],
        "forward_ok": fwd_ok,
        "forward_message": str(fwd_msg)[:500],
    }


def _print_summary(rows: list[dict], *, require_forward: bool) -> None:
    total = len(rows)
    if total == 0:
        print("[task8 label eval] 无 examples")
        return

    exec_ok = sum(1 for r in rows if r["exec_ok"])
    empty_cnt = sum(1 for r in rows if r["exec_kind"] == "empty")

    print("\n========== Task8 examples label 执行统计 ==========")
    print(f"样本数: {total}")
    print(f"空 label: {empty_cnt}")
    print(f"可执行 (执行器 ok): {exec_ok} / {total} = {100.0 * exec_ok / total:.2f}%")

    kind_cnt = Counter(r["exec_kind"] for r in rows if not r["exec_ok"])
    if kind_cnt:
        print("\n执行失败 kind 分布:")
        for k, v in kind_cnt.most_common():
            print(f"  {k}: {v}")

    if require_forward:
        torch_na = sum(1 for r in rows if r["forward_ok"] is None)
        fwd_ok = sum(1 for r in rows if r["forward_ok"] is True)
        fwd_fail = sum(1 for r in rows if r["forward_ok"] is False)
        exec_and_fwd = sum(1 for r in rows if r["exec_ok"] and r["forward_ok"] is True)
        probed = total - torch_na

        if torch_na == total:
            print("\n有结果 (前向探针): 未运行（本机无 torch）")
        elif torch_na:
            print(f"\n有结果 (前向探针 ok): {fwd_ok} / {probed} = {100.0 * fwd_ok / probed:.2f}%")
            print(f"  (全量 {fwd_ok}/{total}={100.0*fwd_ok/total:.2f}%；{torch_na} 条因无 torch 未探针)")
        else:
            print(f"\n有结果 (前向探针 ok): {fwd_ok} / {total} = {100.0 * fwd_ok / total:.2f}%")

        exec_probed = [r for r in rows if r["exec_ok"] and r["forward_ok"] is not None]
        if exec_probed:
            exec_fwd = sum(1 for r in exec_probed if r["forward_ok"] is True)
            print(
                f"在可执行且已探针子集上有结果: {exec_fwd} / {len(exec_probed)} "
                f"= {100.0 * exec_fwd / len(exec_probed):.2f}%"
            )
        if fwd_fail:
            print(f"前向探针失败条数: {fwd_fail}")

        fail_msgs = [
            (r["example_id"], r["forward_message"][:120])
            for r in rows
            if r["exec_ok"] and r["forward_ok"] is False
        ][:8]
        if fail_msgs:
            print("\n可执行但前向无结果（最多 8 条）:")
            for eid, msg in fail_msgs:
                print(f"  {eid}: {msg}")

    not_exec = [r for r in rows if not r["exec_ok"]][:8]
    if not_exec:
        print("\n不可执行样例（最多 8 条）:")
        for r in not_exec:
            print(f"  {r['example_id']}: {r['exec_kind']} | {r['exec_message'][:100]}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="评估 task8 examples 金标代码的可执行率与前向有结果率。")
    p.add_argument(
        "--data",
        type=str,
        default=str(REPO_ROOT / "data" / "openseek-8_kernel_generation.json"),
        help="task8 数据 JSON 路径",
    )
    p.add_argument(
        "--output",
        type=str,
        default="",
        help="可选：写出逐条审计 jsonl",
    )
    p.add_argument(
        "--no_forward_probe",
        action="store_true",
        help="不做前向张量探针，仅统计执行器可执行率",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="仅评估前 N 条（0=全部）",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    require_forward = not args.no_forward_probe

    try:
        import torch  # noqa: F401, PLC0415

        cuda = torch.cuda.is_available()
        print(f"[环境] torch={torch.__version__} cuda={cuda}")
    except ModuleNotFoundError:
        print("[环境] torch 未安装：前向探针将按 v3 逻辑跳过（记为通过）")

    examples = _load_examples(data_path)
    if args.limit > 0:
        examples = examples[: args.limit]

    executor = Task8CodeExecutor()
    rows: list[dict] = []
    for ex in examples:
        eid = str(ex.get("id", "")).strip()
        label = _extract_output(ex.get("output", ""))
        rows.append(
            _eval_one_label(
                eid,
                label,
                executor,
                require_forward=require_forward,
            )
        )

    _print_summary(rows, require_forward=require_forward)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as wf:
            for row in rows:
                wf.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"\n[审计] 已写入 {out_path.resolve()}")


if __name__ == "__main__":
    main()
