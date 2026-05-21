#!/usr/bin/env python3
"""normalize_dataset.py — Task 8 离线数据规整

把 ``data/openseek-8_kernel_generation.json`` 预处理为结构化对齐格式,
便于运行时直接读字段, 不再反复解析。

输入:
    ``data/openseek-8_kernel_generation.json``                     (原始, 不改)

输出（与代码同目录, 提交时随 task8 整目录一起打包）:
    ``src/task8/normalized_data/openseek-8_kernel_generation_normalized.json``  (规整版, 主流程读取)
    ``src/task8/normalized_data/normalize_report.txt``                          (审计报告)

规整后每条 example 字段:
    id, func_name, func_signature, func_desc, math_formula, op_category,
    code, sig_extract_status, raw_input, raw_output

规整后每条 test_sample 字段:
    id, func_name, func_signature, func_desc, math_formula, constraints,
    op_category, raw_input
"""
import os
import re
import sys
import json
import datetime

# 路径中心化: 通过 sys.path 拉取 common.paths
_CUR_DIR = os.path.dirname(os.path.abspath(__file__))           # COM/src/task8
_SRC_DIR = os.path.abspath(os.path.join(_CUR_DIR, '..'))        # COM/src
for _p in (_CUR_DIR, _SRC_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common.paths import (  # noqa: E402
    DATA_DIR,
    TASK8_RAW_DATA_FILENAME,
)

# ------------ 路径 ------------
RAW             = os.path.join(DATA_DIR, TASK8_RAW_DATA_FILENAME)
LOCAL_DATA_DIR  = os.path.join(_CUR_DIR, 'normalized_data')
OUT_DATA        = os.path.join(LOCAL_DATA_DIR, 'openseek-8_kernel_generation_normalized.json')
OUT_REPORT      = os.path.join(LOCAL_DATA_DIR, 'normalize_report.txt')

NORMALIZER_VERSION = "1.0"

# ------------ 系统提示词 ------------
TASK8_SYSTEM_PROMPTS = [
    "You are a expert in writing Triton operators for efficient GPU programming. "
    "Use triton language write a kernel and wrapper according following instruction.",
    "You are an expert in Trion programming, capable of writing corresponding Triton "
    "kernels and wrapper functions based on functional descriptions and function parameters. "
    "Ensure that the wrapper function fully corresponds to the provided function information.",
]


def strip_system_prompt(text: str) -> str:
    for p in TASK8_SYSTEM_PROMPTS:
        if text.startswith(p):
            return text[len(p):].strip()
    # 宽松匹配: 去掉前两句可能的 system prompt
    for p in TASK8_SYSTEM_PROMPTS:
        if p[:40] in text[:200]:
            idx = text.find(p[:40])
            end = text.find('.', idx + len(p[:40]))
            if end != -1:
                remainder = text[end + 1:].lstrip()
                if remainder:
                    return remainder
    return text.strip()


# ------------ 从 test input 抽结构化字段 ------------
def parse_structured_input(text: str) -> dict:
    """解析 test_samples.input 中的结构化字段."""
    info = {
        'func_desc': '',
        'func_signature': '',
        'math_formula': '',
        'constraints': '',
    }

    def _slice(start_marker: str, end_markers: list) -> str:
        """按结构化 marker 切片。

        start_marker 和 end_marker 都仅在行首匹配(前面是 \\n 或字符串开头),
        避免参数列表里的 ``other: torch.Tensor`` 被误识别为段边界。
        """
        # 找到行首的 start_marker
        start_pos = -1
        search_from = 0
        while True:
            p = text.find(start_marker, search_from)
            if p == -1:
                break
            if p == 0 or text[p - 1] == '\n':
                start_pos = p
                break
            search_from = p + 1
        if start_pos == -1:
            return ''
        start = start_pos + len(start_marker)
        end = len(text)
        for m in end_markers:
            search_from = start
            while True:
                pos = text.find(m, search_from)
                if pos == -1:
                    break
                if pos == 0 or text[pos - 1] == '\n':
                    if pos < end:
                        end = pos
                    break
                search_from = pos + 1
        return text[start:end].strip()

    info['func_desc'] = _slice(
        'Functional Description:',
        ['Wrapper Entry Information:', 'Math:', 'other:', 'After generation'],
    )
    info['func_signature'] = _slice(
        'Wrapper Entry Information:',
        ['Math:', 'other:', 'After generation'],
    )
    info['math_formula'] = _slice(
        'Math:', ['other:', 'After generation']
    )
    # 原始 input 中的 `other:` marker 存放参数补充约束 (shape/dtype/training 规则等),
    # 规整时改名为更语义化的 `constraints`.
    info['constraints'] = _slice('other:', ['After generation'])
    return info


def extract_func_name(signature: str) -> str:
    """从签名字符串里抽函数名."""
    if not signature:
        return ''
    s = signature.strip()
    if s.startswith('def '):
        s = s[4:]
    if s.startswith('torch.'):
        s = s[6:]
    m = re.match(r'([A-Za-z_][A-Za-z0-9_\.]*)\s*\(', s)
    if m:
        return m.group(1).replace('.', '_')
    return ''


def clean_wrapper_signature(raw_sig: str) -> tuple[str, str]:
    """把 Wrapper Entry Information 段拆成 (clean_signature, extra_desc).

    输入的 raw_sig 可能形如:
        - "masked_add(grad: torch.Tensor, ...)"                (干净)
        - "relu_conv2d(input, weight, ...) -> Tensor: input (Tensor): ..."
        - "def adaptive_avg_pool2d(output_size) -> Tensor\\nArgs:\\n    ..."
        - "log(input, *, out=None) -> Tensor Args: input (Tensor): ..."

    返回:
        clean_signature: 仅 `func_name(arg1, arg2=default, ...)` 部分, 无 `def`,
                         无返回类型注解, 无尾随描述
        extra_desc:      剩余文本 (返回类型后面的参数说明 / 形状说明等),
                         可合并回 func_desc
    """
    if not raw_sig:
        return '', ''
    text = raw_sig.strip()

    # 1. 去掉 `def ` 前缀 (允许多个空白)
    m = re.match(r'^def\s+', text)
    if m:
        text = text[m.end():]

    # 1.5 去掉函数名开头的 `torch.` 前缀, 并把函数名内的 `.` 替换为 `_`
    #     仅处理 `(` 之前的函数名部分, 不影响参数注解 `input: torch.Tensor`
    #     eg: "torch.permute_copy(input, dims)"  -> "permute_copy(input, dims)"
    #     eg: "linalg.svd(A, *, out=None)"       -> "linalg_svd(A, *, out=None)"
    #     eg: "torch.linalg.cholesky(A, upper=False)" -> "linalg_cholesky(A, upper=False)"
    _paren_idx = text.find('(')
    if _paren_idx > 0:
        _head = text[:_paren_idx]
        _tail = text[_paren_idx:]
        # 先剥 torch. 前缀
        _head = re.sub(r'^torch\.', '', _head)
        # 再把剩余 . 替换为 _ (仅函数名段, 不含参数)
        _head = _head.replace('.', '_')
        text = _head + _tail

    # 2. 定位第一个 `(`
    lp = text.find('(')
    if lp == -1:
        # 无括号, 视为异常, 整串返回为 sig, extra 空
        return text.strip(), ''

    # 3. 括号匹配, 找到与 `(` 配对的 `)` (允许嵌套, 如默认值 `torch.Size([])` )
    depth = 0
    rp = -1
    for i in range(lp, len(text)):
        c = text[i]
        if c == '(':
            depth += 1
        elif c == ')':
            depth -= 1
            if depth == 0:
                rp = i
                break
    if rp == -1:
        # 括号不匹配, 整串返回
        return text.strip(), ''

    sig = text[:rp + 1].strip()
    rest = text[rp + 1:].lstrip()

    # 4. 清理 rest 的开头: 去掉 `-> SomeType` 返回类型声明, 支持多种写法
    #    eg: "-> Tensor: input (Tensor): ..."                    (冒号分隔)
    #    eg: "-> Tensor\nArgs:\n..."                             (换行分隔)
    #    eg: "-> Tensor Args: ..."                               (空格+关键词)
    #    eg: "-> Tensor; input (Tensor): ..."                    (分号分隔)
    #    eg: "-> Tuple[Tensor, Tensor] - logits (Tensor): ..."   (泛型+连字符)
    #    eg: "-> Tensor or (Tensor, LongTensor)"                 (or + 元组)
    #    eg: "-> Tensor A (Tensor): 形状为 ..."                  (空格+参数名)
    patterns_to_strip = [
        # 泛型 + 空格/连字符/关键词/参数名
        r'^->\s*[A-Za-z_][\w\.]*\[[^\]]*\]\s*[-:;]\s*',
        r'^->\s*[A-Za-z_][\w\.]*\[[^\]]*\]\s+(?=Args\b|Keyword\b|[A-Za-z_]\w*\s*\([Tt]ensor|[A-Za-z_]\w*\s+\()',
        r'^->\s*[A-Za-z_][\w\.]*\[[^\]]*\]\s*\n',
        r'^->\s*[A-Za-z_][\w\.]*\[[^\]]*\]\s*$',
        # 类型 or 元组: `Tensor or (Tensor, LongTensor)`
        r'^->\s*[A-Za-z_][\w\.]*\s+or\s+\([^)]*\)\s*[:;\n]?\s*',
        # 简单类型 + 冒号/分号/换行
        r'^->\s*[A-Za-z_][\w\.]*\s*[:;]\s*',
        r'^->\s*[A-Za-z_][\w\.]*\s*\n',
        # 简单类型 + 空格 + docstring 关键词/参数名 (A (Tensor) 等)
        r'^->\s*[A-Za-z_][\w\.]*\s+(?=Args\b|Keyword\b|Example\b|Shape\b|Returns\b|[A-Za-z_]\w*\s*\([Tt]ensor)',
        # 简单类型 (行尾)
        r'^->\s*[A-Za-z_][\w\.]*\s*$',
    ]
    for p in patterns_to_strip:
        m = re.match(p, rest)
        if m:
            rest = rest[m.end():]
            break
    rest = rest.lstrip()
    # 去掉紧跟的冒号或分号
    if rest.startswith(':') or rest.startswith(';') or rest.startswith('-'):
        rest = rest[1:].lstrip()

    # 5. 规整签名: 折叠多余空白
    sig = re.sub(r'\s+', ' ', sig).strip()
    # 6. 规整 extra: 把 `; ` 分隔的参数说明展开成多行, 避免拼接到 func_desc 时生硬
    extra = rest.strip()
    if extra:
        extra = re.sub(r';\s+', '\n', extra)
        extra = re.sub(r'\n{3,}', '\n\n', extra)
    return sig, extra


# ------------ 从 example output 代码反推 wrapper 签名 ------------
def extract_wrapper_signature_from_code(code: str) -> tuple[str, str]:
    """从 Triton code 提取 wrapper 签名, 返回 (signature, status).

    status: ok | failed
    """
    lines = code.strip().split('\n')
    in_kernel = False
    decorator_lines = 0
    candidates = []

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('@triton.jit'):
            in_kernel = True
            decorator_lines = 1
            continue
        if stripped.startswith('@') and (stripped.startswith('@triton.autotune') or 'heuristics' in stripped):
            in_kernel = True
            continue

        if in_kernel and stripped.startswith('def '):
            in_kernel = False  # 进入 kernel def 行, 之后退出 kernel 区块
            # 跳过这个 def 本体到函数结束 (靠缩进判断)
            continue

        if (not in_kernel) and stripped.startswith('def '):
            # 收集多行签名直到闭合的右括号 + 冒号
            sig_parts = []
            paren_depth = 0
            for j in range(i, len(lines)):
                s2 = lines[j].strip()
                sig_parts.append(s2)
                paren_depth += s2.count('(') - s2.count(')')
                if paren_depth <= 0 and ':' in s2:
                    last = sig_parts[-1]
                    # 在最后一个闭括号之后找冒号, 避免像
                    #   def foo(x: int, y: int):
                    # 这种单行签名被 last.index(':') 截断到第一个参数标注的冒号。
                    last_close = last.rfind(')')
                    if last_close != -1:
                        colon_pos = last.find(':', last_close)
                        if colon_pos == -1:
                            colon_pos = last.index(':')
                    else:
                        colon_pos = last.index(':')
                    sig_parts[-1] = last[:colon_pos]
                    break
            sig = ' '.join(sig_parts)
            sig = re.sub(r'\s+', ' ', sig).strip()
            if sig.startswith('def '):
                sig = sig[4:]
            candidates.append(sig)

        # 检测 kernel 内部结束: 缩进回到 0 且非装饰器/非延续
        if in_kernel and stripped and not stripped.startswith('#'):
            current_indent = len(line) - len(line.lstrip())
            if current_indent == 0 and not stripped.startswith('@') and not stripped.startswith('def '):
                in_kernel = False

    if candidates:
        # 取第一个 (wrapper 一般在最外层)
        return candidates[0], 'ok'
    return '', 'failed'


# ------------ 算子类型启发式打标 ------------
OP_CATEGORY_RULES = [
    ('fused', r'^fused[_\-]'),
    ('matmul', r'(bmm|matmul|mm$|_mm_|linear|attn|attention|gemm|dot)'),
    ('normalization', r'(layer_?norm|rms_?norm|batch_?norm|group_?norm|instance_?norm|softmax)'),
    ('activation', r'(relu|sigmoid|tanh|gelu|silu|swish|mish|elu|softplus|leaky_?relu|hardswish|hardsigmoid)'),
    ('elementwise_math', r'(sqrt|exp|log|rsqrt|reciprocal|cos$|sin$|abs$|neg$|ceil|floor|round|pow|square)'),
    ('reduce', r'(sum|mean|prod|max$|min$|argmax|argmin|var|std|norm$|cumsum|cumprod)'),
    ('conv', r'(conv1d|conv2d|conv3d|conv_?transpose)'),
    ('arithmetic', r'^(mul|add|sub|div|mod|floor_divide)(_|$)'),
    ('comparison', r'(eq$|ne$|lt$|gt$|le$|ge$|equal|greater|less)'),
    ('indexing', r'(gather|scatter|index_select|select)'),
    ('shape', r'(cat|stack|split|chunk|reshape|transpose|permute|squeeze|unsqueeze)'),
    ('dropout', r'(dropout)'),
    ('solver', r'(cholesky|lu|qr|svd|solve|inverse|inv$|eig)'),
]


def classify_op(func_name: str) -> str:
    if not func_name:
        return 'unknown'
    name = func_name.lower()
    for cat, pat in OP_CATEGORY_RULES:
        if re.search(pat, name):
            return cat
    return 'other'


# ------------ 主流程 ------------
def main():
    raw = json.load(open(RAW))

    report_lines = []
    report_lines.append(f"normalizer_version: {NORMALIZER_VERSION}")
    report_lines.append(f"normalized_at: {datetime.datetime.now().isoformat(timespec='seconds')}")
    report_lines.append(f"source: {RAW}")
    report_lines.append("")

    # ---- examples ----
    ex_out = []
    ex_sig_fail = []
    ex_cat_counter = {}

    for e in raw.get('examples', []):
        raw_input = e.get('input', '')
        raw_output_list = e.get('output', [])
        raw_output = raw_output_list[0] if isinstance(raw_output_list, list) and raw_output_list else (
            raw_output_list if isinstance(raw_output_list, str) else ''
        )

        desc = strip_system_prompt(raw_input)
        sig, status = extract_wrapper_signature_from_code(raw_output)
        func_name = extract_func_name(sig)
        cat = classify_op(func_name)
        ex_cat_counter[cat] = ex_cat_counter.get(cat, 0) + 1

        if status != 'ok' or not func_name:
            ex_sig_fail.append({'id': e.get('id'), 'status': status, 'sig': sig, 'func_name': func_name})

        ex_out.append({
            'id': e.get('id'),
            'func_name': func_name,
            'func_signature': sig,
            'func_desc': desc,
            'math_formula': 'N/A',
            'op_category': cat,
            'code': raw_output,
            'sig_extract_status': status if func_name else 'failed',
            'raw_input': raw_input,
            'raw_output': raw_output,
        })

    # ---- test_samples ----
    ts_out = []
    ts_no_sig = []
    ts_cat_counter = {}

    for t in raw.get('test_samples', []):
        raw_input = t.get('input', '')
        parsed = parse_structured_input(raw_input)
        # 清洗 func_signature, 剥离 `def` 前缀 / `-> Type` / 尾随参数说明,
        # 剩余描述合并回 func_desc, 使 test_samples 的签名与 examples 保持一致。
        clean_sig, extra_desc = clean_wrapper_signature(parsed['func_signature'])
        merged_desc = parsed['func_desc']
        if extra_desc:
            # 用空行分隔, 避免 "原描述\n-> Tensor; ..." 这种生硬拼接
            merged_desc = (merged_desc + '\n\n' + extra_desc).strip() if merged_desc else extra_desc
        func_name = extract_func_name(clean_sig)
        cat = classify_op(func_name)
        ts_cat_counter[cat] = ts_cat_counter.get(cat, 0) + 1

        if not func_name:
            ts_no_sig.append({'id': t.get('id'), 'sig': clean_sig[:100]})

        ts_out.append({
            'id': t.get('id'),
            'func_name': func_name,
            'func_signature': clean_sig,
            'func_desc': merged_desc,
            'math_formula': parsed['math_formula'],
            'constraints': parsed['constraints'],
            'op_category': cat,
            'raw_input': raw_input,
        })

    # ---- 写输出 ----
    new_data = {
        'task_id': raw.get('task_id'),
        'task_name': raw.get('task_name'),
        'Definition': raw.get('Definition'),
        'License': raw.get('License'),
        'meta': {
            'source_file': os.path.basename(RAW),
            'normalized_at': datetime.datetime.now().isoformat(timespec='seconds'),
            'normalizer_version': NORMALIZER_VERSION,
            'examples_count': len(ex_out),
            'test_samples_count': len(ts_out),
        },
        'examples': ex_out,
        'test_samples': ts_out,
    }
    os.makedirs(os.path.dirname(OUT_DATA), exist_ok=True)
    with open(OUT_DATA, 'w') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)

    # ---- 写审计报告 ----
    report_lines.append(f"examples: total={len(ex_out)}, sig_extract_failed={len(ex_sig_fail)}")
    report_lines.append(f"test_samples: total={len(ts_out)}, no_func_name={len(ts_no_sig)}")
    report_lines.append("")
    report_lines.append("[examples op_category distribution]")
    for k, v in sorted(ex_cat_counter.items(), key=lambda x: -x[1]):
        report_lines.append(f"  {v:4d}  {k}")
    report_lines.append("")
    report_lines.append("[test_samples op_category distribution]")
    for k, v in sorted(ts_cat_counter.items(), key=lambda x: -x[1]):
        report_lines.append(f"  {v:4d}  {k}")
    report_lines.append("")
    if ex_sig_fail:
        report_lines.append("[examples sig_extract_failed (need manual review)]")
        for x in ex_sig_fail[:50]:
            report_lines.append(f"  id={x['id']}  status={x['status']}  sig={x['sig'][:80]!r}")
        if len(ex_sig_fail) > 50:
            report_lines.append(f"  ... {len(ex_sig_fail)-50} more")
        report_lines.append("")
    if ts_no_sig:
        report_lines.append("[test_samples without func_name]")
        for x in ts_no_sig[:50]:
            report_lines.append(f"  id={x['id']}  sig={x['sig']!r}")
        report_lines.append("")

    os.makedirs(os.path.dirname(OUT_REPORT), exist_ok=True)
    with open(OUT_REPORT, 'w') as f:
        f.write('\n'.join(report_lines))

    print(f"wrote: {OUT_DATA}")
    print(f"wrote: {OUT_REPORT}")
    print()
    print('\n'.join(report_lines))


if __name__ == '__main__':
    main()
