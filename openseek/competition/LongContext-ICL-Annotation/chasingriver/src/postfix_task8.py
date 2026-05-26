from __future__ import annotations

"""
Task 8 后处理修复脚本：
对已有 submission 中的 task8 predictions 进行编译修复，提高 compile 通过率。

修复策略:
1. 截断到可编译前缀 (去掉不完整的尾部代码)
2. 移除叙述性行 (markdown, <label> 标签等)
3. 补全缺失的 import (torch, triton, triton.language as tl)
4. 截断 + 补 import 组合策略
5. 修复常见缩进错误
"""

import json
import re
import sys
from pathlib import Path


def _is_compilable(code: str) -> bool:
    try:
        compile(code, "<postfix>", "exec")
        return True
    except Exception:
        return False


def _get_compile_error(code: str) -> str | None:
    try:
        compile(code, "<postfix>", "exec")
        return None
    except SyntaxError as e:
        return str(e)


def _strip_narrative_lines(code: str) -> str:
    """移除非代码行（markdown、label 标签、叙述性文本）"""
    cleaned = []
    for line in code.splitlines():
        stripped = line.strip()
        # 跳过 markdown fences
        if stripped.startswith("```"):
            continue
        # 跳过 label 标签
        if stripped in ("<label>", "</label>", "FULL_CODE"):
            continue
        # 跳过 label 开头
        if stripped.startswith("<label>"):
            line = re.sub(r"^\s*<label>\s*", "", line)
        # 跳过 label 结尾
        if stripped.endswith("</label>"):
            line = re.sub(r"\s*</label>\s*$", "", line)
        cleaned.append(line)
    return "\n".join(cleaned).strip()


def _ensure_imports(code: str) -> str:
    """补全缺失的 import 语句"""
    missing = []
    lowered = code.lower()
    if not re.search(r"(?m)^\s*(?:import|from)\s+torch\b", code):
        missing.append("import torch")
    if not re.search(r"(?m)^\s*(?:import|from)\s+triton\b", code):
        missing.append("import triton")
    if not (
        re.search(r"(?m)^\s*import\s+triton\.language\s+as\s+tl\b", code)
        or re.search(r"(?m)^\s*from\s+triton(?:\.language)?\s+import", code)
    ):
        missing.append("import triton.language as tl")
    if not missing:
        return code
    # 插入到 __future__ import 之后
    lines = code.splitlines()
    insert_at = 0
    while insert_at < len(lines) and lines[insert_at].startswith(("from __future__", "#!")):
        insert_at += 1
    new_lines = lines[:insert_at] + missing + [""] + lines[insert_at:]
    return "\n".join(new_lines)


def _truncate_to_compilable(code: str, min_lines: int = 8) -> str | None:
    """从尾部逐行截断，找到最长的可编译前缀"""
    lines = code.splitlines()
    floor = min(min_lines, len(lines))
    for end in range(len(lines), floor - 1, -1):
        candidate = "\n".join(lines[:end]).rstrip()
        if not candidate:
            continue
        if "def " not in candidate:
            continue
        if _is_compilable(candidate):
            return candidate
    return None


def _fix_indentation_block(code: str) -> str | None:
    """尝试修复 'expected an indented block' 错误。
    
    常见原因: 空的 if/else/for/while/def/class 等块没有 pass。
    """
    lines = code.splitlines()
    fixed_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        # 检查是否是需要子块的语句
        if stripped.endswith(":") and re.match(
            r"^\s*(if|elif|else|for|while|def|class|try|except|finally|with)\b",
            stripped,
        ):
            fixed_lines.append(line)
            # 检查下一行是否有适当的缩进
            if i + 1 < len(lines):
                next_line = lines[i + 1]
                current_indent = len(line) - len(line.lstrip())
                next_stripped = next_line.strip()
                if next_stripped:
                    next_indent = len(next_line) - len(next_line.lstrip())
                    if next_indent <= current_indent:
                        # 缺少缩进块，插入 pass
                        fixed_lines.append(" " * (current_indent + 4) + "pass")
                else:
                    # 下一行是空行，检查再下一行
                    if i + 2 < len(lines):
                        next_next = lines[i + 2]
                        nn_stripped = next_next.strip()
                        if nn_stripped:
                            nn_indent = len(next_next) - len(next_next.lstrip())
                            if nn_indent <= current_indent:
                                fixed_lines.append(" " * (current_indent + 4) + "pass")
            else:
                # 最后一行是需要块的语句
                current_indent = len(line) - len(line.lstrip())
                fixed_lines.append(" " * (current_indent + 4) + "pass")
        else:
            fixed_lines.append(line)
        i += 1
    
    result = "\n".join(fixed_lines)
    if _is_compilable(result):
        return result
    return None


def try_fix(prediction: str) -> tuple[str, str]:
    """尝试修复一个 prediction。
    
    Returns: (fixed_code, fix_method) 其中 fix_method 为 "original"/"truncate"/"strip_narrative"/
             "add_imports"/"truncate_imports"/"fix_indent"/"unfixable"
    """
    if prediction is None:
        return None, "null"
    
    if _is_compilable(prediction):
        return prediction, "original"
    
    # 1. 先移除叙述行
    stripped = _strip_narrative_lines(prediction)
    if _is_compilable(stripped):
        return stripped, "strip_narrative"
    
    # 2. 截断到可编译前缀
    truncated = _truncate_to_compilable(stripped)
    if truncated is not None:
        return truncated, "truncate"
    
    # 3. 补 import
    with_imports = _ensure_imports(stripped)
    if _is_compilable(with_imports):
        return with_imports, "add_imports"
    
    # 4. 截断 + 补 import
    truncated2 = _truncate_to_compilable(with_imports)
    if truncated2 is not None:
        return truncated2, "truncate_imports"
    
    # 5. 修复缩进错误
    fixed_indent = _fix_indentation_block(stripped)
    if fixed_indent is not None:
        return fixed_indent, "fix_indent"
    
    # 6. 修复缩进 + 补 import
    fixed_indent2 = _fix_indentation_block(with_imports)
    if fixed_indent2 is not None:
        return fixed_indent2, "fix_indent_imports"
    
    # 7. 截断 + 修复缩进
    for attempt_code in [stripped, with_imports]:
        lines = attempt_code.splitlines()
        for end in range(len(lines), max(8, len(lines) // 2), -1):
            candidate = "\n".join(lines[:end]).rstrip()
            if "def " not in candidate:
                continue
            fixed = _fix_indentation_block(candidate)
            if fixed is not None:
                return fixed, "truncate_fix_indent"
    
    return prediction, "unfixable"


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input jsonl file")
    parser.add_argument("--output", required=True, help="Output jsonl file")
    args = parser.parse_args()
    
    results = []
    with open(args.input) as f:
        for line in f:
            results.append(json.loads(line))
    
    stats = {}
    fixed_results = []
    
    for r in results:
        pred = r.get("prediction")
        fixed_pred, method = try_fix(pred)
        stats[method] = stats.get(method, 0) + 1
        new_r = dict(r)
        new_r["prediction"] = fixed_pred
        fixed_results.append(new_r)
    
    # 统计
    original_ok = sum(1 for r in results if r.get("prediction") and _is_compilable(r["prediction"]))
    fixed_ok = sum(1 for r in fixed_results if r.get("prediction") and _is_compilable(r["prediction"]))
    total = len(results)
    
    print(f"Total: {total}")
    print(f"Original compile OK: {original_ok}/{total} ({original_ok/total*100:.1f}%)")
    print(f"After fix compile OK: {fixed_ok}/{total} ({fixed_ok/total*100:.1f}%)")
    print(f"Fixed: {fixed_ok - original_ok} additional samples")
    print()
    print("Fix method distribution:")
    for k, v in sorted(stats.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")
    
    with open(args.output, "w") as f:
        for r in fixed_results:
            f.write(json.dumps(r) + "\n")
    
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
