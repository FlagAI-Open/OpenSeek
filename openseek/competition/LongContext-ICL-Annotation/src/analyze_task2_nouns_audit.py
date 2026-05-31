import argparse
import json
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="分析 task2 nouns 审计复核效果。")
    parser.add_argument(
        "--input_jsonl",
        type=str,
        required=True,
        help="推理输出 jsonl 路径（需包含 first_pass_output / model_output / expected_output）。",
    )
    parser.add_argument(
        "--save_fixed_cases",
        type=str,
        default="off",
        choices=["on", "off"],
        help="是否保存错转对样本到 fixed_cases jsonl。",
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=30,
        help="打印前 N 条错转对样本预览。",
    )
    return parser.parse_args()


def _normalize_text(text: str) -> str:
    return " ".join(str(text).strip().split())


def _to_int(value) -> int | None:
    s = str(value).strip()
    if re.fullmatch(r"\d+", s):
        return int(s)
    return None


def _is_match(pred, expected) -> bool:
    return _normalize_text(pred) == _normalize_text(expected)


def _auto_find_input_jsonl(repo_root: Path) -> Path | None:
    patterns = [
        "**/openseek-2-examples-task2-nouns-prompt-compare.jsonl",
        "**/openseek-2-examples-task2-standalone-compare.jsonl",
        "**/*task2*nouns*compare*.jsonl",
        "**/*task2*compare*.jsonl",
    ]
    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(repo_root.glob(pattern))
    if not candidates:
        return None
    candidates = sorted(set(candidates), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_jsonl).expanduser().resolve()
    if not input_path.exists():
        repo_root = Path(__file__).resolve().parent.parent
        auto_path = _auto_find_input_jsonl(repo_root)
        if auto_path is None:
            raise FileNotFoundError(
                f"输入文件不存在: {input_path}\n"
                "并且在仓库中未发现可用 task2 输出 jsonl。\n"
                "请先运行推理脚本，或检查 --input_jsonl 路径是否正确。"
            )
        print(
            f"[警告] 输入文件不存在，自动切换到最近结果: {auto_path.resolve()}"
        )
        input_path = auto_path.resolve()

    rows: list[dict] = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not rows:
        print("[结果] 无有效样本。")
        return

    total = len(rows)
    first_correct = 0
    final_correct = 0

    nouns_total = 0
    nouns_first_correct = 0
    nouns_final_correct = 0
    nouns_minus1_total = 0
    nouns_minus1_fixed = 0
    verbs_total = 0
    verbs_first_correct = 0
    verbs_final_correct = 0

    fixed_cases: list[dict] = []
    regress_cases: list[dict] = []

    for row in rows:
        expected = row.get("expected_output", "")
        first_pred = row.get("first_pass_output", row.get("model_output", ""))
        final_pred = row.get("model_output", "")
        pos = str(row.get("target_pos", "unknown")).strip().lower()

        first_ok = _is_match(first_pred, expected)
        final_ok = _is_match(final_pred, expected)
        if first_ok:
            first_correct += 1
        if final_ok:
            final_correct += 1

        if pos == "nouns":
            nouns_total += 1
            if first_ok:
                nouns_first_correct += 1
            if final_ok:
                nouns_final_correct += 1

            e = _to_int(expected)
            m1 = _to_int(first_pred)
            if e is not None and m1 is not None and (m1 - e) == -1:
                nouns_minus1_total += 1
                if final_ok:
                    nouns_minus1_fixed += 1
        elif pos == "verbs":
            verbs_total += 1
            if first_ok:
                verbs_first_correct += 1
            if final_ok:
                verbs_final_correct += 1

        if (not first_ok) and final_ok:
            fixed_cases.append(
                {
                    "example_id": row.get("example_id", ""),
                    "target_pos": pos,
                    "input": row.get("input", ""),
                    "expected_output": expected,
                    "first_pass_output": first_pred,
                    "final_output": final_pred,
                    "nouns_audit_applied": row.get("nouns_audit_applied", False),
                }
            )
        elif first_ok and (not final_ok):
            regress_cases.append(
                {
                    "example_id": row.get("example_id", ""),
                    "target_pos": pos,
                    "input": row.get("input", ""),
                    "expected_output": expected,
                    "first_pass_output": first_pred,
                    "final_output": final_pred,
                    "nouns_audit_applied": row.get("nouns_audit_applied", False),
                }
            )

    first_acc = first_correct / total
    final_acc = final_correct / total
    gain = final_acc - first_acc

    nouns_first_acc = (nouns_first_correct / nouns_total) if nouns_total else 0.0
    nouns_final_acc = (nouns_final_correct / nouns_total) if nouns_total else 0.0
    nouns_gain = nouns_final_acc - nouns_first_acc
    verbs_first_acc = (verbs_first_correct / verbs_total) if verbs_total else 0.0
    verbs_final_acc = (verbs_final_correct / verbs_total) if verbs_total else 0.0
    verbs_gain = verbs_final_acc - verbs_first_acc
    minus1_fix_rate = (
        (nouns_minus1_fixed / nouns_minus1_total) if nouns_minus1_total else 0.0
    )

    print(f"[输入] {input_path}")
    print(f"[总体] total={total}")
    print(
        f"[总体] first_acc={first_acc:.4f} final_acc={final_acc:.4f} "
        f"gain={gain:+.4f} ({gain*100:+.2f}pp)"
    )
    print(
        f"[nouns] total={nouns_total} first_acc={nouns_first_acc:.4f} "
        f"final_acc={nouns_final_acc:.4f} gain={nouns_gain:+.4f} ({nouns_gain*100:+.2f}pp)"
    )
    print(
        f"[verbs] total={verbs_total} first_acc={verbs_first_acc:.4f} "
        f"final_acc={verbs_final_acc:.4f} gain={verbs_gain:+.4f} ({verbs_gain*100:+.2f}pp)"
    )
    print(
        f"[nouns -1修复] minus1_total={nouns_minus1_total} "
        f"minus1_fixed={nouns_minus1_fixed} fix_rate={minus1_fix_rate:.4f}"
    )
    print(
        f"[翻转统计] wrong->right={len(fixed_cases)} right->wrong={len(regress_cases)}"
    )

    if fixed_cases:
        print(f"[错转对样本预览] top={max(0, args.top_n)}")
        for item in fixed_cases[: max(0, args.top_n)]:
            print(
                f"- id={item['example_id']} pos={item['target_pos']} "
                f"exp={item['expected_output']} first={item['first_pass_output']} final={item['final_output']}"
            )

    if args.save_fixed_cases == "on":
        out_file = input_path.with_name(input_path.stem + ".fixed_cases.jsonl")
        with out_file.open("w", encoding="utf-8") as wf:
            for row in fixed_cases:
                wf.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[保存] fixed cases -> {out_file}")


if __name__ == "__main__":
    main()
