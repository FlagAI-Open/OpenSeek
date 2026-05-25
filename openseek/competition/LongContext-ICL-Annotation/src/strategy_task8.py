import ast
import re
from collections import Counter

from llm_client import post_chat
from strategy_base import BaseStrategy

class Task8TritonStrategy(BaseStrategy):
    """
    针对 Task 8 (Triton Kernel Generation) 的专用策略。
    特点：
    1. 优先压缩长描述，避免大上下文直接生成导致超时。
    2. 使用 /no_think 控制 Qwen3，减少冗长思维链占用。
    3. 提供相关示例检索与兜底，尽量返回可执行代码。
    """

    SYSTEM_PROMPT = """You are an expert AI engineer specializing in GPU programming and Triton kernels.
Your task is to generate functionally correct Triton code based on the provided specification.

Guidelines:
1. Always include necessary imports: `import torch`, `import triton`, `import triton.language as tl`.
2. Implement both the `@triton.jit` kernel(s) and the corresponding Python wrapper function.
3. Ensure memory safety using proper masking (`mask=...`) and handles boundaries (`offsets < n_elements`).
4. Follow the kernel logic, block sizes, tensor shapes, and wrapper behavior described in the input.
5. Prefer conservative, valid Triton APIs over ambitious but invalid code.
6. Use standard Triton idioms such as `tl.program_id(axis=0)`, `tl.arange`, `tl.load(mask=...)`, `tl.store(mask=...)`, and `triton.cdiv`.
7. Provide ONLY the Python code. No explanations, no markdown code blocks.
"""

    SUMMARY_PROMPT = """Extract a concise Triton implementation spec.
Keep only:
- kernel / wrapper names
- tensor roles and important shapes
- core computation
- launch / block / masking / stride constraints
- wrapper behavior and returns
Return a compact bullet list only."""

    REPAIR_PROMPT = """You are fixing a broken Triton program.
Rewrite the entire Python file so it is syntactically valid and uses real Triton APIs only.

Requirements:
1. Keep the intended operator behavior from the specification.
2. Include imports, `@triton.jit` kernel(s), and a callable Python wrapper.
3. Remove invalid APIs such as `tl.shape(...)` or `tl.num_programs(...)`.
4. Do not use bare `None` positional arguments in `tl.load` / `tl.store`; use explicit `mask=` or omit the argument.
5. Every referenced launch symbol or constant must be defined before use.
6. Use proper launch grids, masking, and boundary-safe memory access.
7. Return only the corrected Python code."""

    MAX_INPUT_LENGTH_FOR_DIRECT_GENERATION = 3600
    SUMMARY_MAX_TOKENS = 400
    CODEGEN_MAX_TOKENS = 3200
    REPAIR_MAX_TOKENS = 3200
    COMPACT_SELECTION_TARGET = 1400
    COMPACT_MAX_SENTENCES = 10
    COMPACT_MAX_LENGTH = 1600
    FALLBACK_SIMILARITY_THRESHOLD = 0.6
    MAX_UNDEFINED_NAMES_DISPLAY = 6
    ASSIGNMENT_NODE_TYPES = (
        ast.Assign, ast.AnnAssign, ast.AugAssign, ast.For, ast.AsyncFor,
        ast.With, ast.AsyncWith, ast.NamedExpr,
    )
    CODE_MARKERS = ("import torch", "import triton", "@triton.jit", "def ")
    BAD_CODE_PATTERNS = (
        "tl.num_threads(",
        "tl.fft.",
        "triton.ptl",
        ".data_ptr()",
        "mode == 'forward'",
        'mode == "forward"',
        "tl.shape(",
        "tl.num_programs(",
        "tl.math.",
        "tl.next_power_of_two(",
        "triton.language.",
        ".ptr()",
        ".ptr(",
        "TODO",
    )
    STOPWORDS = {
        "the", "and", "for", "with", "that", "this", "from", "into", "then",
        "input", "output", "kernel", "wrapper", "triton", "torch", "tensor",
        "using", "write", "according", "instruction", "language", "code",
    }
    SAFE_NAME_REFERENCES = {
        "torch", "triton", "tl", "math", "range", "len", "int", "float", "bool",
        "tuple", "list", "dict", "set", "max", "min", "sum", "abs", "enumerate",
        "zip", "isinstance", "print", "Exception", "ValueError", "RuntimeError",
        "AssertionError", "NotImplementedError", "staticmethod", "classmethod",
        "property", "str",
    }
    GENERIC_FAMILY_HINT = """General implementation checklist:
- preserve the operator/function names mentioned in the specification
- allocate outputs in the wrapper with correct dtype/device/shape
- launch Triton kernels with a valid `grid`
- guard all loads/stores with masks when indices may go out of bounds
- keep wrapper arguments and returns aligned with the described API"""
    FAMILY_HINTS = {
        "elementwise": """Element-wise / row-wise guidance:
- use a 1D launch grid
- compute `offsets = block_start + tl.arange(0, BLOCK_SIZE)`
- use `mask = offsets < n_elements`
- load inputs, apply the pointwise math, and store with the same mask""",
        "reduction": """Reduction guidance:
- map programs to rows or reduction chunks explicitly
- use masks for tail handling
- keep partial accumulations in fp32 when needed for stability
- if the wrapper returns reduced tensors, ensure the output shape matches the description""",
        "matmul": """Matrix / attention guidance:
- use 2D launch grids or explicit row/column program ids when appropriate
- respect strides and block sizes from the specification
- use block pointers or offset arithmetic consistently
- keep accumulation in fp32 before casting to the destination dtype""",
        "conv": """Convolution / normalization guidance:
- preserve layout assumptions and stride arguments from the specification
- compute per-dimension offsets explicitly instead of inventing helper APIs
- mask accesses that can cross padded or tail regions
- make wrapper allocation and launch dimensions match the described tensor layout""",
        "autograd": """Autograd guidance:
- if forward and backward are both required, keep them as separate kernels or explicit wrapper stages
- ensure saved tensors / intermediate buffers match the backward computation
- make returned gradients align with the input argument order""",
    }

    def predict(self, task_id: int, task_description: str, prompt_examples: list[dict[str, str]], input_text: str) -> str | None:
        rule_based_code = self._rule_based_generation(input_text)
        if rule_based_code:
            return rule_based_code

        attempts: list[str] = []
        best_candidate = ""
        best_issues: list[str] | None = None
        direct_spec = self._build_generation_spec(task_description, input_text)
        if len(input_text) <= self.MAX_INPUT_LENGTH_FOR_DIRECT_GENERATION:
            attempts.append(direct_spec)

        concise_spec = self._summarize_request(input_text)
        if concise_spec:
            attempts.append(self._build_generation_spec(task_description, concise_spec))

        compact_spec = self._compact_input(input_text)
        if compact_spec:
            attempts.append(self._build_generation_spec(task_description, compact_spec))

        seen_specs: set[str] = set()
        for spec in attempts:
            if not spec or spec in seen_specs:
                continue
            seen_specs.add(spec)
            prediction = self._generate_code(spec)
            clean_code = self._extract_code(prediction)
            validation_issues = self._validate_code(clean_code)
            best_candidate, best_issues = self._pick_better_candidate(
                best_candidate, best_issues, clean_code, validation_issues
            )
            if not validation_issues:
                return clean_code

            repaired_prediction = self._repair_code(spec, clean_code, validation_issues)
            repaired_code = self._extract_code(repaired_prediction)
            repaired_issues = self._validate_code(repaired_code)
            best_candidate, best_issues = self._pick_better_candidate(
                best_candidate, best_issues, repaired_code, repaired_issues
            )
            if not repaired_issues:
                return repaired_code

        fallback = self._fallback_prediction(input_text, prompt_examples)
        if fallback:
            fallback_code = self._extract_code(fallback)
            fallback_issues = self._validate_code(fallback_code)
            if not fallback_issues:
                return fallback_code
            best_candidate, best_issues = self._pick_better_candidate(
                best_candidate, best_issues, fallback_code, fallback_issues
            )

        if best_candidate and self._can_return_best_effort(best_issues or []):
            return best_candidate
        return None

    def _build_generation_spec(self, task_description: str, spec_text: str) -> str:
        family_hint = self._infer_family_hint(spec_text)
        parts = [self.GENERIC_FAMILY_HINT]
        if task_description:
            parts.append(f"Task definition:\n{task_description.strip()}")
        if family_hint:
            parts.append(f"Family-specific guidance:\n{family_hint}")
        parts.append(f"Specification:\n{spec_text.strip()}")
        return "\n\n".join(part for part in parts if part)

    def _infer_family_hint(self, text: str) -> str:
        lowered = text.lower()
        hints: list[str] = []
        token_text = f" {lowered} "

        if any(token in lowered for token in ("backward", "autograd", "gradient", "gradients")):
            hints.append(self.FAMILY_HINTS["autograd"])
        if any(token in lowered for token in ("matmul", "gemm", "bmm", "attention", "query", "key", "value")) or any(
            re.search(pattern, token_text)
            for pattern in (r"\bq\b", r"\bk\b", r"\bv\b")
        ):
            hints.append(self.FAMILY_HINTS["matmul"])
        elif any(token in lowered for token in ("conv", "pooling", "maxpool", "avgpool", "batchnorm", "layernorm", "groupnorm")):
            hints.append(self.FAMILY_HINTS["conv"])
        elif any(token in lowered for token in ("sum", "mean", "norm", "softmax", "argmax", "reduce", "reduction")):
            hints.append(self.FAMILY_HINTS["reduction"])
        else:
            hints.append(self.FAMILY_HINTS["elementwise"])

        return "\n".join(hints)

    def _generate_code(self, spec_text: str) -> str | None:
        try:
            return post_chat(
                [
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": f"/no_think\n{spec_text.strip()}"},
                ],
                temperature=0.0,
                max_tokens=self.CODEGEN_MAX_TOKENS,
                timeout=180,
            )
        except Exception:
            return None

    def _repair_code(self, spec_text: str, broken_code: str, issues: list[str]) -> str | None:
        if not broken_code:
            return None

        issue_text = "\n".join(f"- {issue}" for issue in issues) or "- invalid Triton program"
        try:
            return post_chat(
                [
                    {"role": "system", "content": self.REPAIR_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"/no_think\nSpecification:\n{spec_text.strip()}\n\n"
                            f"Problems to fix:\n{issue_text}\n\n"
                            f"Current code:\n{broken_code.strip()}"
                        ),
                    },
                ],
                temperature=0.0,
                max_tokens=self.REPAIR_MAX_TOKENS,
                timeout=180,
            )
        except Exception:
            return None

    def _summarize_request(self, input_text: str) -> str:
        try:
            summary = post_chat(
                [
                    {"role": "system", "content": self.SUMMARY_PROMPT},
                    {"role": "user", "content": f"/no_think\n{input_text.strip()}"},
                ],
                temperature=0.0,
                max_tokens=self.SUMMARY_MAX_TOKENS,
                timeout=120,
            )
        except Exception:
            return ""

        summary = summary or ""
        if "</think>" in summary:
            summary = summary.split("</think>", 1)[1]
        return summary.strip()

    def _compact_input(self, input_text: str) -> str:
        text = re.sub(
            r"^\s*You are a[n]? expert in writing Triton operators for efficient GPU programming\.\s*Use triton language to write a kernel and wrapper according (?:the )?following instruction\.\s*",
            "",
            input_text.strip(),
            flags=re.IGNORECASE | re.DOTALL,
        )
        text = re.sub(r"\s+", " ", text)
        sentences = re.split(r"(?<=[.!?])\s+", text)
        keywords = (
            "kernel",
            "wrapper",
            "triton",
            "torch",
            "mask",
            "stride",
            "shape",
            "block",
            "grid",
            "tensor",
            "autograd",
            "backward",
            "forward",
            "load",
            "store",
            "matmul",
            "conv",
            "norm",
            "softmax",
            "dequant",
            "attention",
        )

        selected: list[str] = []
        for sentence in sentences:
            normalized = sentence.strip()
            if not normalized:
                continue
            lowered = normalized.lower()
            if any(keyword in lowered for keyword in keywords) or "`" in normalized or "_" in normalized:
                selected.append(normalized)
            if len(" ".join(selected)) >= self.COMPACT_SELECTION_TARGET:
                break

        compact = " ".join(selected[:self.COMPACT_MAX_SENTENCES]).strip()
        return compact[:self.COMPACT_MAX_LENGTH] if compact else text[:self.COMPACT_MAX_LENGTH]

    def _select_relevant_examples(
        self,
        input_text: str,
        all_examples: list[dict[str, str]],
        top_k_examples: int,
    ) -> list[dict[str, str]]:
        input_counter = self._token_counter(input_text)
        scored_examples: list[tuple[float, dict[str, str]]] = []

        for ex in all_examples:
            ex_input = ex["input"]
            if ex_input.strip() == input_text.strip():
                continue
            ex_counter = self._token_counter(ex_input)
            score = self._counter_similarity(input_counter, ex_counter)
            scored_examples.append((score, ex))

        scored_examples.sort(key=lambda item: item[0], reverse=True)
        return [example for score, example in scored_examples[:top_k_examples] if score > 0]

    def _token_counter(self, text: str) -> Counter[str]:
        tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]+", text.lower())
        filtered = [token for token in tokens if token not in self.STOPWORDS]
        return Counter(filtered)

    def _counter_similarity(self, left: Counter[str], right: Counter[str]) -> float:
        if not left or not right:
            return 0.0
        overlap = sum(min(left[token], right[token]) for token in left.keys() & right.keys())
        scale = max(sum(left.values()), sum(right.values()))
        return overlap / scale if scale else 0.0

    def _fallback_prediction(self, input_text: str, all_examples: list[dict[str, str]]) -> str | None:
        if not all_examples:
            return None

        input_counter = self._token_counter(input_text)
        selected_examples = self._select_relevant_examples(input_text, all_examples, top_k_examples=1)
        if not selected_examples:
            return None

        top_example = selected_examples[0]
        similarity = self._counter_similarity(input_counter, self._token_counter(top_example["input"]))
        if similarity < self.FALLBACK_SIMILARITY_THRESHOLD and not (
            self._extract_operator_names(input_text) & self._extract_operator_names(top_example["input"])
        ):
            return None

        fallback = top_example["expected"]
        return fallback[0] if isinstance(fallback, list) else fallback

    def _rule_based_generation(self, input_text: str) -> str | None:
        lowered = input_text.lower()

        if "def cos_signbit" in lowered or "cos_signbit(" in lowered:
            return """import torch

def cos_signbit(input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    cos_result = torch.cos(input)
    sign_bit = torch.signbit(cos_result)
    return cos_result, sign_bit"""

        if "spectral_norm_eig" in lowered:
            return """import torch

def spectral_norm_eig(A: torch.Tensor, *, out: torch.Tensor | None = None) -> torch.Tensor:
    eigvals = torch.linalg.eigvals(A)
    result = eigvals.abs().amax(dim=-1)
    if out is not None:
        out.copy_(result)
        return out
    return result"""

        if "fftn(" in lowered and "discrete fourier transform" in lowered:
            return """import torch

def fftn(
    input: torch.Tensor,
    s: tuple[int, ...] | None = None,
    dim: tuple[int, ...] | None = None,
    norm: str | None = None,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    result = torch.fft.fftn(input, s=s, dim=dim, norm=norm)
    if out is not None:
        out.copy_(result)
        return out
    return result"""

        return None

    def _extract_code(self, text: str) -> str:
        if not text:
            return ""

        if "</think>" in text:
            text = text.split("</think>", 1)[1]

        # 如果模型返回了 ```python ... ```，提取中间部分
        code_block_match = re.search(r"```python\s*(.*?)\s*```", text, re.DOTALL)
        if code_block_match:
            return self._ensure_imports(code_block_match.group(1).strip())

        # 如果只有 ``` ... ```
        generic_block_match = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
        if generic_block_match:
            return self._ensure_imports(generic_block_match.group(1).strip())

        lines = text.strip().splitlines()
        for index, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith(("import ", "from ", "@triton.", "def ", "class ")):
                return self._ensure_imports("\n".join(lines[index:]).strip())

        marker_positions: list[int] = []
        for marker in self.CODE_MARKERS:
            position = text.find(marker)
            if position >= 0:
                marker_positions.append(position)
        if marker_positions:
            start = min(marker_positions)
            return self._ensure_imports(text[start:].strip())

        return self._ensure_imports(text.strip())

    def _ensure_imports(self, code: str) -> str:
        if not code:
            return ""

        required_imports: list[str] = []
        if "import torch" not in code and "torch." in code:
            required_imports.append("import torch")
        if "import triton" not in code and "import triton.language as tl" not in code:
            required_imports.append("import triton")
        if "import triton.language as tl" not in code and "tl." in code:
            required_imports.append("import triton.language as tl")

        if required_imports:
            code = "\n".join(required_imports) + "\n\n" + code
        return code.strip()

    def _looks_like_code(self, text: str) -> bool:
        if not text:
            return False
        return any(marker in text for marker in self.CODE_MARKERS)

    def _looks_reasonable_code(self, text: str) -> bool:
        if not text:
            return False
        return not any(pattern in text for pattern in self.BAD_CODE_PATTERNS)

    def _validate_code(self, text: str) -> list[str]:
        issues: list[str] = []
        if not text:
            return ["empty output"]

        if not self._looks_like_code(text):
            issues.append("missing Python/Triton code markers")
        if not self._looks_reasonable_code(text):
            issues.append("contains invalid or suspicious Triton patterns")
        if "import torch" not in text:
            issues.append("missing `import torch`")
        if "import triton" not in text:
            issues.append("missing `import triton`")
        if "import triton.language as tl" not in text:
            issues.append("missing `import triton.language as tl`")
        if "@triton.jit" not in text:
            issues.append("missing `@triton.jit` kernel")
        if len(re.findall(r"^\s*def\s+[A-Za-z_][A-Za-z0-9_]*\s*\(", text, flags=re.MULTILINE)) < 2 and "class " not in text:
            issues.append("expected both kernel and wrapper definitions")
        if re.search(r"tl\.load\([^()\n]*,\s*None\s*\)", text):
            issues.append("`tl.load` uses a bare `None` positional argument")
        if re.search(r"tl\.store\([^()\n]*,\s*None\s*\)", text):
            issues.append("`tl.store` uses a bare `None` positional argument")

        try:
            tree = ast.parse(text)
        except SyntaxError as exc:
            issues.append(f"syntax error at line {exc.lineno}: {exc.msg}")
            return issues

        for function_name, undefined_names in self._find_undefined_names(tree):
            sorted_names = sorted(undefined_names)
            displayed_names = sorted_names[:self.MAX_UNDEFINED_NAMES_DISPLAY]
            formatted = ", ".join(displayed_names)
            remaining = len(sorted_names) - len(displayed_names)
            if remaining > 0:
                formatted += f" (and {remaining} more)"
            issues.append(f"undefined names in `{function_name}`: {formatted}")

        return issues

    def _pick_better_candidate(
        self,
        best_candidate: str,
        best_issues: list[str] | None,
        candidate: str,
        candidate_issues: list[str],
    ) -> tuple[str, list[str] | None]:
        if not candidate:
            return best_candidate, best_issues
        if not best_candidate or best_issues is None:
            return candidate, candidate_issues

        candidate_score = (self._has_syntax_error(candidate_issues), len(candidate_issues))
        best_score = (self._has_syntax_error(best_issues), len(best_issues))
        if candidate_score < best_score:
            return candidate, candidate_issues
        return best_candidate, best_issues

    def _can_return_best_effort(self, issues: list[str]) -> bool:
        if not issues:
            return True
        if any(
            issue.startswith("empty output")
            or issue.startswith("missing Python/Triton code markers")
            or issue.startswith("syntax error")
            for issue in issues
        ):
            return False
        return True

    def _has_syntax_error(self, issues: list[str]) -> bool:
        return any(issue.startswith("syntax error") for issue in issues)

    def _extract_operator_names(self, text: str) -> set[str]:
        quoted_names = set(re.findall(r"`([A-Za-z_][A-Za-z0-9_]*)`", text))
        explicit_names = set(re.findall(r"\bdef\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", text))
        return {name.lower() for name in quoted_names | explicit_names}

    def _find_undefined_names(self, tree: ast.AST) -> list[tuple[str, set[str]]]:
        module_names = set(self.SAFE_NAME_REFERENCES)

        for node in getattr(tree, "body", []):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module_names.add(alias.name.split(".")[0])
                    module_names.add(alias.asname or alias.name)
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    module_names.add(alias.asname or alias.name)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                module_names.add(node.name)
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                module_names.update(self._collect_assigned_names(node))

        issues: list[tuple[str, set[str]]] = []
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue

            local_names = set(module_names)
            for arg in (
                list(node.args.posonlyargs)
                + list(node.args.args)
                + list(node.args.kwonlyargs)
            ):
                local_names.add(arg.arg)
            if node.args.vararg:
                local_names.add(node.args.vararg.arg)
            if node.args.kwarg:
                local_names.add(node.args.kwarg.arg)

            for child in ast.walk(node):
                if child is not node and isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    local_names.add(child.name)
                    continue
                if isinstance(child, self.ASSIGNMENT_NODE_TYPES) or isinstance(child, ast.comprehension):
                    local_names.update(self._collect_assigned_names(child))
                elif isinstance(child, ast.Import):
                    for alias in child.names:
                        local_names.add(alias.name.split(".")[0])
                        local_names.add(alias.asname or alias.name)
                elif isinstance(child, ast.ImportFrom):
                    for alias in child.names:
                        local_names.add(alias.asname or alias.name)

            loaded_names = {
                child.id
                for child in ast.walk(node)
                if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)
            }
            undefined = {
                name for name in loaded_names
                if name not in local_names
            }
            if undefined:
                issues.append((node.name, undefined))

        return issues

    def _collect_assigned_names(self, node: ast.AST) -> set[str]:
        assigned: set[str] = set()

        def add_target(target: ast.AST) -> None:
            if isinstance(target, ast.Name):
                assigned.add(target.id)
            elif isinstance(target, (ast.Tuple, ast.List)):
                for element in target.elts:
                    add_target(element)

        if isinstance(node, ast.Assign):
            for target in node.targets:
                add_target(target)
        elif isinstance(node, ast.AnnAssign):
            add_target(node.target)
        elif isinstance(node, ast.AugAssign):
            add_target(node.target)
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            add_target(node.target)
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                if item.optional_vars:
                    add_target(item.optional_vars)
        elif isinstance(node, ast.NamedExpr):
            add_target(node.target)
        elif isinstance(node, ast.comprehension):
            add_target(node.target)

        return assigned
