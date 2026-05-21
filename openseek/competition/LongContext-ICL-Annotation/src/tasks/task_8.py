"""Task 8: Triton kernel generation — op-type matching + single-turn + retry."""
import sys
import os
import re
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from tasks.base import BaseTask
from retriever import BM25Retriever


PROMPT_TEMPLATE = (
    "You are an expert Triton GPU kernel programmer. "
    "Based on the functional description and the reference examples below, "
    "generate complete, runnable Triton kernel code.\n\n"

    "Requirements:\n"
    "1. Include all necessary imports (torch, triton, triton.language as tl)\n"
    "2. Implement the @triton.jit kernel function with correct block masking\n"
    "3. Implement the complete wrapper function\n"
    "4. Output ONLY the code wrapped in <label></label> tags — NO explanations, NO markdown\n\n"

    "CRITICAL RULES (must follow exactly):\n"
    "- NEVER mix numel() with stride(0): use `input_ptr + offsets` directly (no stride multiplier) for element-wise kernels, OR flatten input with .view(-1) in the wrapper first\n"
    "- NEVER declare runtime parameters (N, alpha, scalar values, string args) as tl.constexpr in @triton.jit — only BLOCK_SIZE should be tl.constexpr\n"
    "- NEVER use tl.pi (does not exist) — define your own constant: 3.141592653589793\n"
    "- NEVER use tl.float32(value) to cast — use direct Python values\n"
    "- NEVER use tl.nearest() — use tl.round() instead\n"
    "- NEVER compare pointers with None inside @triton.jit kernels\n"
    "- NEVER use tl.is_contiguous() or .contiguous() on Triton tensor values inside kernels\n"
    "- Use tl.sqrt, tl.exp, tl.sigmoid, tl.tanh, tl.erf for math operations\n"
    "- Ensure all variables used in @triton.jit are declared as kernel parameters\n"
    "- For 2D operations (matmul, convolution), use proper 2D grid and correct output tensor shape\n\n"

    "CRITICAL: Your ENTIRE output must be exactly:\n"
    "<label>\n"
    "import torch\n"
    "... your code here ...\n"
    "</label>\n\n"

    "### Reference Examples\n"
    "{examples_str}\n"

    "### Task\n"
    "{input_text}\n\n"

    "Generate the complete Triton code, wrapped in <label> tags:\n"
    "<label>\n"
)


# Operation keywords used to classify examples and tests
OP_KEYWORDS = [
    'attention', 'softmax', 'matmul', 'batch_mm', 'bmm',
    'mul', 'add', 'sub', 'reduce', 'sum',
    'norm', 'layernorm', 'rmsnorm', 'conv', 'dequantize', 'quantize',
    'dropout', 'gelu', 'relu', 'swiglu', 'sigmoid', 'concat', 'split',
    'reshape', 'transpose', 'gather', 'scatter', 'argmax', 'argmin',
    'index', 'kldivergence', 'flash', 'rms', 'copy', 'fill',
    'tanh', 'div', 'cross_entropy', 'kld', 'silu',
]

ELEMENTWISE_OPS = {
    'mul', 'add', 'sub', 'relu', 'gelu', 'sigmoid', 'div',
    'fill', 'copy', 'tanh', 'silu', 'sqrt', 'rsqrt', 'log', 'exp',
}
REDUCTION_OPS = {
    'sum', 'argmax', 'argmin', 'reduce',
}
ROW_REDUCTION_OPS = {
    'softmax', 'layernorm', 'rmsnorm', 'norm', 'rms',
}
MATRIX_OPS = {
    'attention', 'matmul', 'flash', 'bmm',
}

KERNEL_TEMPLATES = {
    'element_wise': (
        "### Structural Framework (Element-Wise Kernel)\n"
        "Use this standard framework and fill in the core computation:\n\n"
        "```python\n"
        "import torch\n"
        "import triton\n"
        "import triton.language as tl\n\n"
        "@triton.jit\n"
        "def kernel_name(X_ptr, Out_ptr, N, stride_x, stride_out, BLOCK_SIZE: tl.constexpr):\n"
        "    pid = tl.program_id(axis=0)\n"
        "    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)\n"
        "    mask = offsets < N\n"
        "    x = tl.load(X_ptr + offsets * stride_x, mask=mask)\n"
        "    # === YOUR CORE COMPUTATION HERE ===\n"
        "    out = ...\n"
        "    tl.store(Out_ptr + offsets * stride_out, out, mask=mask)\n\n"
        "def wrapper_name(X):\n"
        "    out = torch.empty_like(X)\n"
        "    N = X.numel()\n"
        "    BLOCK_SIZE = min(1024, triton.next_power_of_2(N))\n"
        "    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)\n"
        "    kernel_name[grid](X, out, N, X.stride(0), out.stride(0), BLOCK_SIZE)\n"
        "    return out\n"
        "```\n\n"
    ),
    'reduction': (
        "### Structural Framework (Reduction Kernel)\n"
        "Use this standard framework with cross-thread reduction:\n\n"
        "```python\n"
        "import torch\n"
        "import triton\n"
        "import triton.language as tl\n\n"
        "@triton.jit\n"
        "def kernel_name(X_ptr, Out_ptr, N, stride_x, BLOCK_SIZE: tl.constexpr):\n"
        "    pid = tl.program_id(axis=0)\n"
        "    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)\n"
        "    mask = offsets < N\n"
        "    x = tl.load(X_ptr + offsets * stride_x, mask=mask, other=0.0)\n"
        "    # === YOUR CORE REDUCTION HERE ===\n"
        "    # e.g., out = tl.sum(x) or tl.max(x)\n"
        "    out = ...\n"
        "    tl.store(Out_ptr + pid, out)\n\n"
        "def wrapper_name(X):\n"
        "    out = torch.empty((X.size(0),), dtype=X.dtype, device=X.device)\n"
        "    N = X.shape[-1]\n"
        "    BLOCK_SIZE = min(1024, triton.next_power_of_2(N))\n"
        "    grid = lambda meta: (X.size(0),)\n"
        "    kernel_name[grid](X, out, N, X.stride(0), BLOCK_SIZE)\n"
        "    return out\n"
        "```\n\n"
    ),
    'row_reduction': (
        "### Structural Framework (Row Reduction Kernel — Softmax/LayerNorm/RMSNorm)\n"
        "Use this standard framework with per-row reduction:\n\n"
        "```python\n"
        "import torch\n"
        "import triton\n"
        "import triton.language as tl\n\n"
        "@triton.jit\n"
        "def kernel_name(X_ptr, Out_ptr, stride_x_row, stride_out_row, N, BLOCK_SIZE: tl.constexpr):\n"
        "    row = tl.program_id(axis=0)\n"
        "    row_start = row * stride_x_row\n"
        "    row_out_start = row * stride_out_row\n"
        "    offsets = tl.arange(0, BLOCK_SIZE)\n"
        "    mask = offsets < N\n"
        "    x = tl.load(X_ptr + row_start + offsets, mask=mask, other=0.0)\n"
        "    # === YOUR ROW-LEVEL REDUCTION HERE ===\n"
        "    # e.g., mean = tl.sum(x) / N, variance, etc.\n"
        "    # === YOUR PER-ELEMENT NORMALIZATION HERE ===\n"
        "    out = ...\n"
        "    tl.store(Out_ptr + row_out_start + offsets, out, mask=mask)\n\n"
        "def wrapper_name(X):\n"
        "    out = torch.empty_like(X)\n"
        "    N = X.shape[-1]\n"
        "    BLOCK_SIZE = triton.next_power_of_2(N)\n"
        "    grid = lambda meta: (X.size(0),)\n"
        "    kernel_name[grid](X, out, X.stride(0), out.stride(0), N, BLOCK_SIZE)\n"
        "    return out\n"
        "```\n\n"
    ),
    'matrix': (
        "### Structural Framework (Matrix Block Kernel — Attention/MatMul)\n"
        "Use this standard framework with block-wise matrix computation:\n\n"
        "```python\n"
        "import torch\n"
        "import triton\n"
        "import triton.language as tl\n\n"
        "@triton.jit\n"
        "def kernel_name(A_ptr, B_ptr, Out_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_om, stride_on, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):\n"
        "    pid_m = tl.program_id(axis=0)\n"
        "    pid_n = tl.program_id(axis=1)\n"
        "    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)\n"
        "    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)\n"
        "    offs_k = tl.arange(0, BLOCK_N)\n"
        "    a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak\n"
        "    b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn\n"
        "    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)\n"
        "    for k in range(0, K, BLOCK_N):\n"
        "        # === YOUR BLOCK-WISE COMPUTATION HERE ===\n"
        "        pass\n"
        "    out_ptrs = Out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on\n"
        "    tl.store(out_ptrs, accumulator)\n\n"
        "def wrapper_name(A, B):\n"
        "    M, K = A.shape\n"
        "    K2, N = B.shape\n"
        "    BLOCK_M, BLOCK_N = 64, 64\n"
        "    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))\n"
        "    out = torch.empty((M, N), dtype=A.dtype, device=A.device)\n"
        "    kernel_name[grid](A, B, out, M, N, K, A.stride(0), A.stride(1), B.stride(0), B.stride(1), out.stride(0), out.stride(1), BLOCK_M, BLOCK_N)\n"
        "    return out\n"
        "```\n\n"
    ),
}


def _extract_purpose(text):
    text_lower = text.lower()
    func_name_match = re.search(r'fused[_-](\w+)', text_lower)
    if func_name_match:
        name = func_name_match.group(1)
        for op in OP_KEYWORDS + ['bmm', 'batch_mm', 'silu', 'cross_entropy', 'sqrt', 'rsqrt', 'log', 'exp']:
            if op in name:
                return op
        parts = name.split('_')
        if parts:
            for op in OP_KEYWORDS:
                if parts[0] == op:
                    return op

    desc_match = re.search(r'Functional Description: (.+?)(?:Wrapper Entry)', text, re.DOTALL)
    if not desc_match:
        desc_match = re.search(r'Functional Description: (.+)', text, re.DOTALL)
    if desc_match:
        desc = desc_match.group(1).lower()
        positions = []
        for op in OP_KEYWORDS:
            m = re.search(r'\b' + re.escape(op) + r'\b', desc)
            if m:
                positions.append((m.start(), op))
        for pattern, label in [
            ('bmm', 'bmm'), ('batch matrix', 'bmm'), ('batch_mm', 'bmm'),
            ('matrix-vector multiplication', 'bmm'), ('matrix-matrix multiplication', 'matmul'),
            ('matrix multiplication', 'matmul'),
            ('silu', 'silu'), ('swish', 'silu'),
            ('square-root', 'sqrt'), ('rsqrt', 'rsqrt'),
            ('logarithm', 'log'), ('log1p', 'log'),
            ('exponential', 'exp'),
            ('cross entropy', 'cross_entropy'), ('cross_entropy', 'cross_entropy'),
            ('convolution', 'conv'), ('2d convolution', 'conv'),
            ('kl divergence', 'kldivergence'),
            ('embedding lookup', 'index'), ('embedding', 'index'),
        ]:
            m = desc.find(pattern)
            if m != -1:
                existing = {op for _, op in positions}
                if label not in existing:
                    positions.append((m, label))
        positions.sort(key=lambda x: x[0])
        if positions:
            return positions[0][1]

    return None


def _extract_ops(text):
    text_lower = text.lower()
    matched = []
    for op in OP_KEYWORDS:
        if re.search(r'\b' + re.escape(op) + r'\b', text_lower):
            matched.append(op)
    if 'bmm' in text_lower or 'batch matrix' in text_lower or 'batch_mm' in text_lower:
        if 'bmm' not in matched:
            matched.append('bmm')
    if ('silu' in text_lower or 'swish' in text_lower) and 'silu' not in matched:
        matched.append('silu')
    if 'cross entropy' in text_lower or 'cross_entropy' in text_lower:
        if 'cross_entropy' not in matched:
            matched.append('cross_entropy')
    if 'kl divergence' in text_lower or 'kl_divergence' in text_lower:
        if 'kld' not in matched:
            matched.append('kld')
    if 'square-root' in text_lower or 'sqrt(' in text_lower or 'rsqrt' in text_lower:
        if 'sqrt' not in matched:
            matched.append('sqrt')
    if 'reciprocal' in text_lower:
        if 'rsqrt' not in matched:
            matched.append('rsqrt')
    if 'logarithm' in text_lower or 'log(' in text_lower or 'log1p' in text_lower:
        if 'log' not in matched:
            matched.append('log')
    if 'exponential' in text_lower or 'exp(' in text_lower:
        if 'exp' not in matched:
            matched.append('exp')
    if 'natural logarithm' in text_lower or 'ln(' in text_lower:
        if 'log' not in matched:
            matched.append('log')
    return matched


def _select_template(purpose, all_ops):
    if purpose:
        if purpose in MATRIX_OPS or purpose in ('batch_mm', 'bmm'):
            return KERNEL_TEMPLATES['matrix']
        if purpose in ROW_REDUCTION_OPS:
            return KERNEL_TEMPLATES['row_reduction']
        if purpose in REDUCTION_OPS:
            return KERNEL_TEMPLATES['reduction']
        if purpose in ELEMENTWISE_OPS or purpose in ('silu', 'sqrt', 'rsqrt', 'log', 'exp'):
            return KERNEL_TEMPLATES['element_wise']

    for op in all_ops:
        if op in MATRIX_OPS or op == 'bmm':
            return KERNEL_TEMPLATES['matrix']
        if op in ROW_REDUCTION_OPS:
            return KERNEL_TEMPLATES['row_reduction']
        if op in REDUCTION_OPS:
            return KERNEL_TEMPLATES['reduction']
        if op in ELEMENTWISE_OPS or op in ('silu', 'sqrt', 'rsqrt', 'log', 'exp'):
            return KERNEL_TEMPLATES['element_wise']
    return None


def _extract_code_from_raw(raw_output: str) -> str:
    if not raw_output:
        return ""

    if '<label>' in raw_output and '</label>' in raw_output:
        last_start = raw_output.rfind('<label>')
        after = raw_output[last_start + 7:]
        end = after.find('</label>')
        if end != -1:
            return after[:end].strip()

    code = raw_output
    for marker in ['</think>', '<|end_of_thought|>']:
        idx = code.find(marker)
        if idx != -1:
            code = code[idx + len(marker):]
            break

    if raw_output.startswith('<|begin_of_thought|>') or raw_output.startswith('<think>'):
        if '</think>' not in raw_output and '<|end_of_thought|>' not in raw_output:
            return ""

    return code.strip()


def _validate_code(code: str) -> tuple:
    if not code:
        return False, "Your output was empty. Please provide Triton kernel code."

    checks = {
        'import triton': 'import triton' in code or 'import torch' in code,
        '@triton.jit': '@triton.jit' in code or 'triton.jit' in code,
        'def ': 'def ' in code,
    }

    missing = [k for k, v in checks.items() if not v]
    if missing:
        return False, f"Your code is missing: {', '.join(missing)}. Include imports, @triton.jit decorator, and function definitions."

    if len(code) < 50:
        return False, "Your code is too short. Provide complete kernel and wrapper functions."

    try:
        compile(code, '<string>', 'exec')
    except SyntaxError as e:
        return False, f"Your code has a syntax error at line {e.lineno}: {e.msg}. Check and fix it."

    return True, ""


class Task8(BaseTask):
    task_id = 8
    DATA_FILE = '../data/openseek-8_kernel_generation.json'
    PROMPT_TEMPLATE = PROMPT_TEMPLATE

    DEFAULT_CFG = {
        "name": "kernel_generation",
        "temperature": 0.3,
        "top_k": 80,
        "target_k": 5,
        "num_votes": 1,
        "max_tokens": 16000,
        "stop_tokens": None,
        "system_prompt": "You are an expert Triton GPU kernel programmer. Generate complete, runnable Triton kernel code based on the functional description and reference examples. Include all imports and implement both the @triton.jit kernel and wrapper function. Output only the code in <label></label> tags.",
        "min_icl_tokens": 16_000,
    }

    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        self.cfg = None
        self.min_icl_tokens = 16_000
        self.max_icl_tokens = 18_000  # generous buffer for padding to reach 16K

    def postprocess(self, prediction: str, raw_output: str = "") -> str:
        if prediction and prediction.strip():
            code = prediction.strip()
        else:
            code = _extract_code_from_raw(raw_output)

        if not code:
            return ""

        # Strip <label> tags
        if '<label>' in code and '</label>' in code:
            start = code.rfind('<label>')
            code = code[start + 7:]
            end = code.find('</label>')
            if end != -1:
                code = code[:end]
        elif '</label>' in code:
            code = code.replace('</label>', '')
        elif '<label>' in code:
            code = code.replace('<label>', '')

        # Strip markdown code blocks
        code = re.sub(r'```python\n?', '', code)
        code = re.sub(r'```\n?', '', code)
        code = code.strip()

        return code

    def split_icl_padding(self, all_examples):
        return all_examples, list(all_examples)

    def prepare(self):
        self.bm25_retriever = BM25Retriever(self.icl_examples)
        self.op_groups = {}
        for idx, ex in enumerate(self.icl_examples):
            ops = _extract_ops(ex['input'])
            for op in ops:
                self.op_groups.setdefault(op, []).append(idx)
        return self

    def _select_examples(self, text, all_ops, purpose, top_k=None):
        """Select examples by op matching. For element-wise ops, exact match only."""
        if top_k is None:
            top_k = self.cfg['top_k']

        is_elementwise = len(all_ops) == 1 and all_ops[0] in ELEMENTWISE_OPS if all_ops else False
        is_fused_simple = len(all_ops) > 1 and all(op in ELEMENTWISE_OPS for op in all_ops)

        seen_ids = set()
        selected = []

        if is_elementwise or is_fused_simple:
            # Simple element-wise: exact op match from op_groups
            for op in all_ops:
                for idx in self.op_groups.get(op, []):
                    ex = self.icl_examples[idx]
                    if id(ex) not in seen_ids:
                        seen_ids.add(id(ex))
                        selected.append(ex)
            # Fill remaining with BM25
            remaining_k = top_k - len(selected)
            if remaining_k > 0:
                bm25_results = self.bm25_retriever.retrieve_top_k(text, top_k=remaining_k + len(selected))
                for ex in bm25_results:
                    if len(selected) >= top_k:
                        break
                    if id(ex) not in seen_ids:
                        selected.append(ex)
                        seen_ids.add(id(ex))
        else:
            # Complex/fused: score by op overlap, skip if no ops detected
            if all_ops:
                fused_examples = []
                for idx, ex in enumerate(self.icl_examples):
                    ex_ops = _extract_ops(ex['input'])
                    if len(ex_ops) > 1:
                        fused_examples.append((idx, ex, ex_ops))

                def fuse_score(ex_ops):
                    test_set = set(all_ops)
                    ex_set = set(ex_ops)
                    overlap = len(test_set & ex_set)
                    has_purpose = purpose in ex_set if purpose else 0
                    return (has_purpose, overlap, len(ex_ops))

                fused_examples.sort(key=lambda x: fuse_score(x[2]), reverse=True)

                for idx, ex, ex_ops in fused_examples:
                    if id(ex) not in seen_ids:
                        seen_ids.add(id(ex))
                        selected.append(ex)

            remaining_k = top_k - len(selected)
            if remaining_k > 0:
                bm25_results = self.bm25_retriever.retrieve_top_k(text, top_k=remaining_k + len(selected))
                for ex in bm25_results:
                    if len(selected) >= top_k:
                        break
                    if id(ex) not in seen_ids:
                        selected.append(ex)
                        seen_ids.add(id(ex))

        return selected

    def _build_examples_str(self, selected, max_tokens=None):
        if max_tokens is None:
            max_tokens = self.min_icl_tokens

        examples_str = ""
        token_count = 0
        for ex in selected:
            output_val = ex['output']
            label_str = output_val[0] if isinstance(output_val, list) and len(output_val) > 0 else output_val
            part = f"Input:\n{ex['input']}\nOutput:\n<label>{label_str}</label>\n\n"
            part_tokens = len(self.tokenizer.encode(part, add_special_tokens=False))
            if token_count + part_tokens > max_tokens:
                break
            examples_str = part + examples_str
            token_count += part_tokens
        return examples_str, token_count

    def _pad_examples(self, examples_str, selected_ids, all_ops=None):
        """Pad examples to min_icl_tokens using remaining padding pool.

        Padding excludes examples that contain complex operations not in the test,
        and is placed at the FAR END (prepended first = farthest from question).
        """
        token_count = len(self.tokenizer.encode(examples_str, add_special_tokens=False))
        if token_count >= self.min_icl_tokens:
            return examples_str, token_count

        # Filter padding pool: exclude examples with complex ops not in the test
        complex_ops = {
            'attention', 'softmax', 'matmul', 'batch_mm', 'bmm',
            'reduce', 'sum', 'argmax', 'argmin',
            'norm', 'layernorm', 'rmsnorm', 'rms',
            'conv', 'dequantize', 'quantize',
            'dropout', 'concat', 'split', 'reshape', 'transpose',
            'gather', 'scatter', 'index',
            'kldivergence', 'kld', 'flash',
            'cross_entropy', 'silu',
        }
        test_ops_set = set(all_ops) if all_ops else set()

        def is_safe_padding(ex):
            """Padding example is safe if it only contains simple ops or matches test ops."""
            ex_ops = _extract_ops(ex['input'])
            if not ex_ops:
                return True  # No detected ops → safe
            # If example only has element-wise ops, it's safe
            if all(op in ELEMENTWISE_OPS for op in ex_ops):
                return True
            # If example has complex ops that aren't in test, skip it
            for op in ex_ops:
                if op in complex_ops and op not in test_ops_set:
                    return False
            return True

        # Exclude already used examples + unsafe examples
        extras = [ex for ex in self.padding_pool
                  if id(ex) not in selected_ids and is_safe_padding(ex)]

        prepend_parts = []
        for ex in extras:
            if token_count >= self.max_icl_tokens:
                break
            output_val = ex['output']
            label_str = output_val[0] if isinstance(output_val, list) and len(output_val) > 0 else output_val
            part = f"Input:\n{ex['input']}\nOutput:\n<label>{label_str}</label>\n\n"
            part_tokens = len(self.tokenizer.encode(part, add_special_tokens=False))
            if token_count + part_tokens > self.max_icl_tokens:
                break
            prepend_parts.append(part)
            token_count += part_tokens

        if prepend_parts:
            # Padding goes at the FRONT (farthest from question)
            examples_str = "".join(prepend_parts) + examples_str
            # Recompute exact token count
            token_count = len(self.tokenizer.encode(examples_str, add_special_tokens=False))

        return examples_str, token_count

    def process_sample(self, test_sample, first_sample=False):
        """Single-turn: select examples by op matching, build prompt with template."""
        text = test_sample['input']
        purpose = _extract_purpose(text)
        all_ops = _extract_ops(text)
        framework_template = _select_template(purpose, all_ops)

        # Select examples: prefer exact op match for element-wise ops
        selected = self._select_examples(text, all_ops, purpose, top_k=self.cfg['top_k'])
        fallback_ex = selected[0] if selected else self.icl_examples[0]
        fallback_output = fallback_ex.get('output', [])
        fallback = fallback_output[0] if isinstance(fallback_output, list) and len(fallback_output) > 0 else fallback_output if fallback_output else ''
        examples_str, token_count = self._build_examples_str(selected)

        # T8 custom padding: fill to exactly 16K using remaining padding pool
        selected_ids = {id(ex) for ex in selected}
        dynamic_examples_str, token_count = self._pad_examples(examples_str, selected_ids, all_ops)

        # Build prompt with template for element-wise ops
        prompt = self.PROMPT_TEMPLATE.format(
            examples_str=dynamic_examples_str,
            input_text=text,
        )

        if framework_template:
            # Place template AFTER examples, right before the task — closest to <label>
            prompt = prompt.replace("### Task", framework_template + "### Task")

        return prompt, dynamic_examples_str, token_count, {'fallback': fallback, 'purpose': purpose, 'template': framework_template is not None}

    def should_retry(self):
        return True

    def run_inference(self, test_sample, call_model):
        """Single-turn generation → validate → retry with feedback → thought extraction fallback."""
        prompt, _, _, info = self.process_sample(test_sample)
        fallback = info.get('fallback', '')

        prediction, raw_output = call_model(prompt, self.cfg)
        raw_outputs = [raw_output]
        prediction = self.postprocess(prediction, raw_output)

        valid, feedback = _validate_code(prediction)
        if valid:
            return prediction, prompt, raw_outputs, [prediction], {}

        # Round 2: retry with feedback and higher temperature
        print(f"  ⚠️ [{feedback}] for sample {test_sample['id'][:30]}, retrying (temp=0.5)...")
        high_temp_cfg = dict(self.cfg)
        high_temp_cfg['temperature'] = 0.5
        base = prompt.rstrip()
        if base.endswith('<label>'):
            base = base[:-len('<label>')]  # strip trailing <label>
        retry_prompt = base.rstrip() + f"\n\nYour previous attempt had issues: {feedback}\nRegenerate the complete code:\n<label>\n"
        prediction2, raw_output2 = call_model(retry_prompt, high_temp_cfg)
        raw_outputs.append(f"[RETRY temp=0.5 with feedback] {raw_output2}")
        prediction2 = self.postprocess(prediction2, raw_output2)

        valid2, feedback2 = _validate_code(prediction2)
        if valid2:
            prediction = prediction2
            raw_outputs[-1] = f"[RETRY SUCCESS] {raw_output2}"
            return prediction, prompt, raw_outputs, [prediction], {}

        # Round 3: second retry with even higher temperature
        print(f"  ⚠️ [{feedback2}] still invalid, second retry (temp=0.6)...")
        high_temp_cfg2 = dict(self.cfg)
        high_temp_cfg2['temperature'] = 0.6
        base2 = retry_prompt.rstrip()
        if base2.endswith('<label>'):
            base2 = base2[:-len('<label>')]
        retry_prompt2 = base2.rstrip() + f"\n\nYour previous attempt still had issues: {feedback2}\nRegenerate the complete code:\n<label>\n"
        prediction3, raw_output3 = call_model(retry_prompt2, high_temp_cfg2)
        raw_outputs.append(f"[RETRY2 temp=0.6] {raw_output3}")
        prediction3 = self.postprocess(prediction3, raw_output3)

        valid3, _ = _validate_code(prediction3)
        if valid3:
            prediction = prediction3
            raw_outputs[-1] = f"[RETRY2 SUCCESS] {raw_output3}"
        else:
            # Fallback: extract answer from thought blocks
            thought_answer = _extract_answer_from_thought(raw_outputs)
            if thought_answer:
                prediction = self.postprocess(thought_answer)
                raw_outputs.append(f"[THOUGHT EXTRACTION] extracted code")
            elif fallback:
                prediction = self.postprocess(fallback)
                raw_outputs.append(f"[FALLBACK to example code]")
            else:
                prediction = prediction3
                raw_outputs.append(f"[ALL FAILED, using last attempt]")

        return prediction, prompt, raw_outputs, [prediction], {}


def _extract_answer_from_thought(raw_outputs):
    """Try to extract code from <think> blocks when all 3 rounds fail."""
    for raw in raw_outputs:
        raw_clean = raw
        for marker in ['</think>', '<|end_of_thought|>']:
            idx = raw.find(marker)
            if idx != -1:
                raw_clean = raw[idx + len(marker):]
                break
        # Look for code after thought
        if raw_clean.strip():
            # Try to extract code between backticks or just use the raw content
            code_blocks = re.findall(r'```python\n(.*?)```', raw_clean, re.DOTALL)
            if code_blocks:
                for block in reversed(code_blocks):
                    if '@triton.jit' in block or 'import triton' in block:
                        return block.strip()
            # If no backticks, check if contains key markers
            if '@triton.jit' in raw_clean and 'def ' in raw_clean:
                return raw_clean.strip()
    return None
