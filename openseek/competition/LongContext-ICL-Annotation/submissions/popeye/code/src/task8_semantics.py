import ast
import re

from task8_retrieval import (
    extract_expected_signature,
    extract_task8_features,
    extract_task8_special_clusters,
    primary_family,
    split_wrapper_tokens,
)


PLACEHOLDER_MARKERS = [
    "placeholder",
    "simplified example",
    "notimplementederror",
    "todo",
    "dummy",
]

TYPING_NAMES = {"Union", "Tuple", "Optional", "List", "Dict", "Set"}
TYPING_IMPORT_RE = re.compile(r"^\s*from\s+typing\s+import\s+(.+)$", re.M)
COMMON_IDENTIFIER_STOPWORDS = {
    "torch",
    "triton",
    "tl",
    "tensor",
    "input",
    "output",
    "ptr",
    "meta",
    "grid",
    "self",
    "device",
    "dtype",
    "shape",
    "block",
    "size",
    "none",
    "true",
    "false",
}

CODE_FAMILY_PATTERNS = {
    "conv2d": [r"\bconv2d\b", r"\bconvolution\b"],
    "bmm": [r"\bbmm\b", r"\bbatched?\s+matmul\b", r"\bbatch matrix multiplication\b"],
    "matmul": [r"\bmatmul\b", r"\bmm\b", r"\bdot\b"],
    "attention": [r"\battention\b", r"\bquery\b", r"\bkey\b", r"\bvalue\b"],
    "sampling": [
        r"\bmultinomial\b",
        r"\bsampling\b",
        r"\bsearchsorted\b",
        r"\brand(?:n|int)?\b",
        r"\buniform\b",
        r"\bprobabilit(?:y|ies)\b",
        r"\bcdf\b",
    ],
    "quantize": [
        r"\bquantiz",
        r"\bint8\b",
        r"\bint4\b",
        r"\buint8\b",
        r"\bfp8\b",
        r"\bscale\b",
        r"\bzero_point\b",
        r"\bpacked?\b",
    ],
    "dequantize": [
        r"\bdequantiz",
        r"\bde-quantiz",
        r"\bzero_point\b",
        r"\bscale\b",
    ],
    "embedding": [
        r"\bembedding\b",
        r"\bindex_select\b",
        r"\bgather\b",
        r"\bembedding_bag\b",
        r"\bindices\b",
    ],
    "cumsum": [r"\bcumsum\b", r"\bcumulative sum\b", r"\bprefix sum\b", r"\bscan\b"],
    "softmax": [r"(?<!log)_softmax\b", r"(?<!log)\bsoftmax\b"],
    "logsoftmax": [r"\blog[_\s-]*softmax\b"],
    "rmsnorm": [r"\brms(?:_|)norm\b", r"\brsqrt\b", r"\broot mean square\b"],
    "layer_norm": [r"\blayer[_\s-]*norm\b", r"\bmean\b", r"\bvariance\b"],
    "dropout": [r"\bdropout\b", r"\bbernoulli\b"],
    "gelu": [r"\bgelu\b", r"\berf\b", r"\btanh\b"],
    "relu": [r"\brelu\b", r"\bmaximum\(.*0"],
    "sigmoid": [r"\bsigmoid\b", r"\b1\s*/\s*\(1\s*\+\s*"],
    "tanh": [r"\btanh\b"],
    "solve": [r"\bsolve\b", r"\blinalg\.solve\b"],
    "lu": [r"\blu\b", r"\blu_factor\b", r"\blu_solve\b"],
    "sum": [r"\bsum\b", r"\breduction\b"],
    "mean": [r"\bmean\b", r"\baverage\b"],
    "max_pool": [r"\bmax[_\s-]*pool\b"],
    "div": [r"\bdiv\b", r"\bdivide\b", r"/"],
    "mul": [r"\bmul\b", r"\bmultiply\b", r"\*"],
    "sub": [r"\bsub\b", r"\bsubtract\b", r"-"],
    "add": [r"\badd\b", r"\bplus\b", r"\+"],
}

CODE_PROPERTY_PATTERNS = {
    "broadcast": [r"\bbroadcast", r"\bexpand\b"],
    "batch": [r"\bbatch\b"],
    "out": [r"\bout\s*=\s*none\b", r"\bout\b"],
    "inplace": [r"\binplace\b", r"_\("],
    "approximate": [r"\bapproximate\b", r"\btanh\b"],
    "groups": [r"\bgroups\b"],
    "padding": [r"\bpadding\b", r"\bpad_h\b", r"\bpad_w\b"],
    "stride": [r"\bstride\b"],
    "dilation": [r"\bdilation\b"],
    "dim": [r"\bdim\b", r"\bdimension\b", r"\baxis\b"],
    "complex": [r"\bcomplex", r"\bcomplex64\b", r"\bcomplex128\b"],
    "bias": [r"\bbias\b"],
    "seed": [r"\bseed\b", r"\bphilox\b"],
    "random": [r"\brandom\b", r"\brand\b"],
    "mask": [r"\bmask\b"],
    "contiguous": [r"\bcontiguous\b"],
    "causal": [r"\bcausal\b"],
    "reduction": [r"\breduction\b", r"\breduce\b"],
    "rounding_mode": [r"\brounding_mode\b", r"\bfloor\b", r"\bceil\b", r"\btrunc\b"],
}

FAMILY_COMPATIBILITY = {
    "bmm": {"matmul"},
    "matmul": {"bmm"},
    "quantize": {"dequantize"},
    "dequantize": {"quantize"},
}

FATAL_BLOCKER_CODES = {
    "empty_code",
    "syntax_error",
    "missing_public_wrapper",
    "contains_pass",
    "contains_noisy_prefix",
    "contains_markdown_fence",
    "placeholder_marker",
    "contains_notimplementederror",
}


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))


def token_f1(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    overlap = len(left & right)
    if overlap == 0:
        return 0.0
    return 2.0 * overlap / (len(left) + len(right))


def decorator_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parts = []
        current = node
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
        return ".".join(reversed(parts))
    return None


def parse_code_tree(code: str) -> ast.AST | None:
    try:
        return ast.parse(code)
    except SyntaxError:
        return None


def get_top_level_functions(tree: ast.AST | None) -> list[ast.FunctionDef]:
    if tree is None:
        return []
    return [node for node in tree.body if isinstance(node, ast.FunctionDef)]


def get_defined_functions(tree: ast.AST | None) -> dict[str, ast.FunctionDef]:
    return {node.name: node for node in get_top_level_functions(tree)}


def get_public_wrapper_name(tree: ast.AST | None) -> str | None:
    funcs = get_top_level_functions(tree)
    if not funcs:
        return None

    wrapper_candidates = []
    for fn in funcs:
        decorators = {decorator_name(node) for node in fn.decorator_list}
        if "triton.jit" in decorators:
            continue
        wrapper_candidates.append(fn.name)

    public_wrappers = [name for name in wrapper_candidates if not name.startswith("_")]
    if public_wrappers:
        return public_wrappers[-1]
    if wrapper_candidates:
        return wrapper_candidates[-1]

    public_funcs = [fn.name for fn in funcs if not fn.name.startswith("_")]
    if public_funcs:
        return public_funcs[-1]
    return funcs[-1].name


def get_imported_typing_names(code: str) -> set[str]:
    imported = set()
    for match in TYPING_IMPORT_RE.finditer(code):
        imported.update(part.strip() for part in match.group(1).split(","))
    return imported


def get_annotation_names(tree: ast.AST | None) -> set[str]:
    if tree is None:
        return set()
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
    return names


def extract_kernel_launches(tree: ast.AST | None) -> list[tuple[str, int, set[str]]]:
    launches = []
    if tree is None:
        return launches

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Subscript):
            continue
        target = func.value
        if isinstance(target, ast.Name):
            kernel_name = target.id
        elif isinstance(target, ast.Attribute):
            kernel_name = target.attr
        else:
            continue
        keyword_names = {kw.arg for kw in node.keywords if kw.arg}
        launches.append((kernel_name, len(node.args), keyword_names))
    return launches


def get_function_name_set(code: str) -> set[str]:
    return {fn.name for fn in get_top_level_functions(parse_code_tree(code))}


def get_api_token_set(code: str) -> set[str]:
    return set(re.findall(r"\b(?:tl|torch|triton)\.[A-Za-z_][A-Za-z0-9_]*", code))


def get_identifier_set(code: str) -> set[str]:
    tree = parse_code_tree(code)
    if tree is None:
        return set()

    identifiers = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            token = node.id.lower()
            if token not in COMMON_IDENTIFIER_STOPWORDS and len(token) > 2:
                identifiers.add(token)
        elif isinstance(node, ast.Attribute):
            token = node.attr.lower()
            if token not in COMMON_IDENTIFIER_STOPWORDS and len(token) > 2:
                identifiers.add(token)
    return identifiers


def count_jit_kernels(code: str) -> int:
    return len(re.findall(r"^@triton\.jit", code, flags=re.M))


def kernel_count_score(pred_code: str, gold_code: str) -> float:
    pred_count = count_jit_kernels(pred_code)
    gold_count = count_jit_kernels(gold_code)
    if pred_count == gold_count:
        return 1.0
    return max(0.0, 1.0 - abs(pred_count - gold_count) / max(gold_count, 1))


def extract_code_tags(text: str, pattern_map: dict[str, list[str]]) -> set[str]:
    lowered = text.lower()
    tags = set()
    for tag, patterns in pattern_map.items():
        if any(re.search(pattern, lowered) for pattern in patterns):
            tags.add(tag)
    return tags


def get_wrapper_arg_names(tree: ast.AST | None, wrapper_name: str | None) -> list[str]:
    if tree is None or not wrapper_name:
        return []
    for fn in get_top_level_functions(tree):
        if fn.name != wrapper_name:
            continue
        arg_names = [arg.arg for arg in fn.args.args]
        arg_names.extend(arg.arg for arg in fn.args.kwonlyargs)
        if fn.args.vararg:
            arg_names.append(fn.args.vararg.arg)
        if fn.args.kwarg:
            arg_names.append(fn.args.kwarg.arg)
        return arg_names
    return []


def extract_task8_code_features(code: str) -> dict:
    tree = parse_code_tree(code)
    wrapper_name = get_public_wrapper_name(tree)
    operator_tags = extract_code_tags(code, CODE_FAMILY_PATTERNS)
    property_tags = extract_code_tags(code, CODE_PROPERTY_PATTERNS)
    special_clusters = extract_task8_special_clusters(code)
    arg_names = get_wrapper_arg_names(tree, wrapper_name)
    wrapper_tokens = split_wrapper_tokens(wrapper_name)
    return {
        "syntax_ok": tree is not None,
        "wrapper_name": wrapper_name,
        "arg_names": arg_names,
        "arg_name_set": {name.lower() for name in arg_names},
        "arg_count": len(arg_names),
        "wrapper_tokens": wrapper_tokens,
        "operator_tags": operator_tags,
        "property_tags": property_tags,
        "special_clusters": special_clusters,
        "primary_family": primary_family(operator_tags),
        "has_out_arg": "out" in {name.lower() for name in arg_names},
        "jit_kernel_count": count_jit_kernels(code),
        "api_tokens": get_api_token_set(code),
        "function_names": get_function_name_set(code),
        "identifiers": get_identifier_set(code),
    }


def families_compatible(expected_family: str | None, observed_family: str | None) -> bool:
    if not expected_family or not observed_family:
        return False
    if expected_family == observed_family:
        return True
    return observed_family in FAMILY_COMPATIBILITY.get(expected_family, set())


def build_family_semantics(sample_input: str, code: str) -> dict:
    input_features = extract_task8_features(sample_input)
    code_features = extract_task8_code_features(code)

    score = 0.0
    matches = []
    mismatches = []

    input_family = input_features["primary_family"]
    code_family = code_features["primary_family"]
    if code_features["operator_tags"]:
        score += 0.05

    if input_family:
        if families_compatible(input_family, code_family):
            score += 0.18
            matches.append("primary_family")
        elif code_family:
            score -= 0.12
            mismatches.append(f"primary_family:{input_family}->{code_family}")
        else:
            score -= 0.05
            mismatches.append("missing_primary_family")

    operator_overlap = input_features["operator_tags"] & code_features["operator_tags"]
    property_overlap = input_features["property_tags"] & code_features["property_tags"]
    cluster_overlap = extract_task8_special_clusters(sample_input) & code_features["special_clusters"]
    arg_overlap = input_features["arg_name_set"] & code_features["arg_name_set"]
    wrapper_overlap = input_features["wrapper_tokens"] & code_features["wrapper_tokens"]

    score += 0.05 * min(len(operator_overlap), 2)
    score += 0.025 * min(len(property_overlap), 2)
    score += 0.05 * min(len(cluster_overlap), 1)
    score += 0.01 * min(len(arg_overlap), 3)
    score += 0.01 * min(len(wrapper_overlap), 2)
    if input_features["has_out_arg"] and code_features["has_out_arg"]:
        score += 0.02

    if input_family == "logsoftmax" and "softmax" in code_features["operator_tags"] and "logsoftmax" not in code_features["operator_tags"]:
        mismatches.append("family_specific:softmax_instead_of_logsoftmax")
        score -= 0.08
    if input_family == "softmax" and "logsoftmax" in code_features["operator_tags"]:
        mismatches.append("family_specific:logsoftmax_instead_of_softmax")
        score -= 0.08
    if input_family == "quantize" and "dequantize" in code_features["operator_tags"] and "quantize" not in code_features["operator_tags"]:
        mismatches.append("family_specific:dequantize_instead_of_quantize")
        score -= 0.08
    if input_family == "dequantize" and "quantize" in code_features["operator_tags"] and "dequantize" not in code_features["operator_tags"]:
        mismatches.append("family_specific:quantize_instead_of_dequantize")
        score -= 0.08

    return {
        "input_family": input_family,
        "code_family": code_family,
        "operator_overlap": sorted(operator_overlap),
        "property_overlap": sorted(property_overlap),
        "cluster_overlap": sorted(cluster_overlap),
        "arg_overlap": sorted(arg_overlap),
        "wrapper_overlap": sorted(wrapper_overlap),
        "match": bool(input_family and families_compatible(input_family, code_family)),
        "score": round(clamp(score, 0.0, 0.35), 4),
        "matches": matches,
        "mismatches": mismatches,
        "input_operator_tags": sorted(input_features["operator_tags"]),
        "code_operator_tags": sorted(code_features["operator_tags"]),
        "input_property_tags": sorted(input_features["property_tags"]),
        "code_property_tags": sorted(code_features["property_tags"]),
    }


def build_input_alignment(sample_input: str, code: str) -> dict:
    input_features = extract_task8_features(sample_input)
    code_features = extract_task8_code_features(code)
    expected_wrapper, _ = extract_expected_signature(sample_input)

    wrapper_match = bool(expected_wrapper and code_features["wrapper_name"] == expected_wrapper)
    wrapper_token_f1 = token_f1(input_features["wrapper_tokens"], code_features["wrapper_tokens"])
    arg_f1 = token_f1(input_features["arg_name_set"], code_features["arg_name_set"])
    operator_f1 = token_f1(input_features["operator_tags"], code_features["operator_tags"])
    property_f1 = token_f1(input_features["property_tags"], code_features["property_tags"])
    score = (
        0.07 * float(wrapper_match)
        + 0.05 * wrapper_token_f1
        + 0.04 * arg_f1
        + 0.02 * operator_f1
        + 0.02 * property_f1
    )
    if input_features["arg_count"] and abs(input_features["arg_count"] - code_features["arg_count"]) <= 1:
        score += 0.02
    return {
        "expected_wrapper": expected_wrapper,
        "wrapper_match": wrapper_match,
        "wrapper_token_f1": round(wrapper_token_f1, 4),
        "arg_f1": round(arg_f1, 4),
        "operator_f1": round(operator_f1, 4),
        "property_f1": round(property_f1, 4),
        "score": round(clamp(score, 0.0, 0.2), 4),
    }


def build_structural_similarity(prediction: str, gold: str) -> dict:
    fn_f1 = token_f1(get_function_name_set(prediction), get_function_name_set(gold))
    api_f1 = token_f1(get_api_token_set(prediction), get_api_token_set(gold))
    ident_f1 = token_f1(get_identifier_set(prediction), get_identifier_set(gold))
    kernel_score = kernel_count_score(prediction, gold)
    score = 0.08 * fn_f1 + 0.06 * api_f1 + 0.04 * ident_f1 + 0.02 * kernel_score
    return {
        "fn_f1": round(fn_f1, 4),
        "api_f1": round(api_f1, 4),
        "ident_f1": round(ident_f1, 4),
        "kernel_score": round(kernel_score, 4),
        "score": round(clamp(score, 0.0, 0.2), 4),
    }


def make_finding(severity: str, code: str, message: str) -> dict:
    return {"severity": severity, "code": code, "message": message}


def audit_task8_code(sample_input: str, code: str) -> dict:
    result = {
        "syntax_ok": True,
        "wrapper_name": None,
        "expected_wrapper": None,
        "issues": [],
        "findings": [],
        "blocker_count": 0,
        "warning_count": 0,
        "blocker_codes": [],
        "warning_codes": [],
    }

    expected_wrapper, _ = extract_expected_signature(sample_input)
    result["expected_wrapper"] = expected_wrapper
    raw_code = (code or "").strip()
    if not raw_code:
        result["syntax_ok"] = False
        result["findings"].append(make_finding("blocker", "empty_code", "Prediction is empty."))
    tree = parse_code_tree(raw_code)
    if raw_code and tree is None:
        result["syntax_ok"] = False
        result["findings"].append(make_finding("blocker", "syntax_error", "Code cannot be parsed by Python AST."))
    if tree is None:
        family = build_family_semantics(sample_input, raw_code)
        result["family"] = family
        _finalize_audit_result(result)
        return result

    defined_functions = get_defined_functions(tree)
    wrapper_name = get_public_wrapper_name(tree)
    result["wrapper_name"] = wrapper_name
    if wrapper_name is None:
        result["findings"].append(make_finding("blocker", "missing_public_wrapper", "No wrapper function found."))
    elif expected_wrapper and wrapper_name != expected_wrapper:
        result["findings"].append(
            make_finding(
                "blocker",
                "wrapper_mismatch",
                f"Expected wrapper {expected_wrapper}, got {wrapper_name}.",
            )
        )

    lowered = raw_code.lower()
    stripped_head = "\n".join(raw_code.splitlines()[:6]).lower()
    if re.search(r"(?m)^\s*pass\s*$", raw_code):
        result["findings"].append(make_finding("blocker", "contains_pass", "Code still contains bare pass."))
    if "notimplementederror" in lowered:
        result["findings"].append(
            make_finding("blocker", "contains_notimplementederror", "Code still contains NotImplementedError.")
        )
    for marker in PLACEHOLDER_MARKERS:
        if marker in lowered:
            result["findings"].append(
                make_finding("blocker", "placeholder_marker", f"Placeholder marker detected: {marker}.")
            )
    if any(token in stripped_head for token in ("answer:", "final answer:", "prediction:", "<label>", "</label>")):
        result["findings"].append(
            make_finding("blocker", "contains_noisy_prefix", "Code starts with answer-like or label tokens.")
        )
    if "```" in raw_code:
        result["findings"].append(
            make_finding("blocker", "contains_markdown_fence", "Markdown fence detected in code output.")
        )
    if "@triton.jit" not in raw_code:
        result["findings"].append(make_finding("warning", "missing_triton_jit", "No @triton.jit kernel found."))

    imported_typing = get_imported_typing_names(raw_code)
    used_names = get_annotation_names(tree)
    missing_typing = sorted(name for name in TYPING_NAMES if name in used_names and name not in imported_typing)
    for name in missing_typing:
        result["findings"].append(
            make_finding("blocker", "missing_typing_import", f"Typing annotation {name} is used but not imported.")
        )

    for kernel_name, positional_count, keyword_names in extract_kernel_launches(tree):
        fn = defined_functions.get(kernel_name)
        if fn is None:
            result["findings"].append(
                make_finding(
                    "blocker",
                    "kernel_launch_without_definition",
                    f"Kernel launch {kernel_name} has no matching definition.",
                )
            )
            continue
        param_names = [arg.arg for arg in fn.args.args]
        matched_keywords = {name for name in keyword_names if name in param_names}
        supplied = positional_count + len(matched_keywords)
        expected = len(param_names)
        if supplied != expected:
            result["findings"].append(
                make_finding(
                    "blocker",
                    "kernel_launch_arg_mismatch",
                    f"Kernel {kernel_name} launch supplied {supplied} args but definition expects {expected}.",
                )
            )

    family = build_family_semantics(sample_input, raw_code)
    result["family"] = family
    if family["input_family"] and family["code_family"] and not family["match"]:
        result["findings"].append(
            make_finding(
                "warning",
                "family_mismatch",
                f"Expected family {family['input_family']}, observed {family['code_family']}.",
            )
        )

    if wrapper_name == "softmax_log":
        if "log_input = tl.math.log" in raw_code and "softmax = log_input - tl.math.log" in raw_code:
            result["findings"].append(
                make_finding("warning", "softmax_log_may_return_log_softmax", "May implement log-softmax variant.")
            )
    if wrapper_name == "sigmoid_adaptive_avg_pool2d":
        if "torch.empty((input.shape[0], output_h, output_w)" in raw_code:
            result["findings"].append(
                make_finding(
                    "warning",
                    "pool2d_output_may_be_missing_channel_dim",
                    "Adaptive pool output may be missing channel dimension.",
                )
            )
    if wrapper_name == "signbit_bitwise_and":
        if "assert input.dtype in [torch.float32" in raw_code and "tl.bitwise_and(input_val, other_val)" in raw_code:
            result["findings"].append(
                make_finding(
                    "warning",
                    "bitwise_and_uses_float_input_without_cast",
                    "bitwise_and appears to use float inputs without cast.",
                )
            )

    _finalize_audit_result(result)
    return result


def _finalize_audit_result(result: dict) -> None:
    blocker_codes = []
    warning_codes = []
    issues = []
    for finding in result["findings"]:
        issues.append(f"{finding['code']}: {finding['message']}")
        if finding["severity"] == "blocker":
            blocker_codes.append(finding["code"])
        else:
            warning_codes.append(finding["code"])
    result["issues"] = issues
    result["blocker_codes"] = blocker_codes
    result["warning_codes"] = warning_codes
    result["blocker_count"] = len(blocker_codes)
    result["warning_count"] = len(warning_codes)


def score_task8_proxy_v1(prediction: str | None, gold: str) -> dict:
    if prediction is None:
        return {"score": 0.0, "reason": "null"}

    pred_code = str(prediction).strip()
    gold_code = str(gold).strip()

    syntax_ok = parse_code_tree(pred_code) is not None
    pred_wrapper = get_public_wrapper_name(parse_code_tree(pred_code))
    gold_wrapper = get_public_wrapper_name(parse_code_tree(gold_code))
    wrapper_match = bool(pred_wrapper and gold_wrapper and pred_wrapper == gold_wrapper)

    fn_f1 = token_f1(get_function_name_set(pred_code), get_function_name_set(gold_code))
    api_f1 = token_f1(get_api_token_set(pred_code), get_api_token_set(gold_code))
    ident_f1 = token_f1(get_identifier_set(pred_code), get_identifier_set(gold_code))
    kernels = kernel_count_score(pred_code, gold_code)

    score = (
        0.25 * float(syntax_ok)
        + 0.20 * float(wrapper_match)
        + 0.20 * fn_f1
        + 0.20 * api_f1
        + 0.10 * ident_f1
        + 0.05 * kernels
    )
    return {
        "score": round(score, 4),
        "reason": (
            f"syntax={syntax_ok},wrapper={wrapper_match},"
            f"fn_f1={fn_f1:.3f},api_f1={api_f1:.3f},ident_f1={ident_f1:.3f},kernels={kernels:.3f}"
        ),
    }


def score_task8_proxy_v2(sample_input: str, prediction: str | None, gold: str | None = None) -> dict:
    if prediction is None or not str(prediction).strip():
        structural = build_structural_similarity("", str(gold).strip()) if gold is not None else build_input_alignment(sample_input, "")
        return {
            "score": 0.0,
            "hard_gate_score": 0.0,
            "family_semantics_score": 0.0,
            "structural_similarity_score": 0.0,
            "reason": "empty",
            "audit": audit_task8_code(sample_input, ""),
            "family": build_family_semantics(sample_input, ""),
            "structural": structural,
        }

    pred_code = str(prediction).strip()
    audit = audit_task8_code(sample_input, pred_code)
    family = audit["family"]

    blocker_codes = set(audit["blocker_codes"])
    hard_gate_score = 0.45
    if blocker_codes & FATAL_BLOCKER_CODES:
        hard_gate_score = 0.0
    elif blocker_codes:
        hard_gate_score = 0.18
    hard_gate_score = max(0.0, hard_gate_score - 0.03 * audit["warning_count"])

    if gold is not None:
        structural = build_structural_similarity(pred_code, str(gold).strip())
    else:
        structural = build_input_alignment(sample_input, pred_code)

    total = clamp(hard_gate_score + family["score"] + structural["score"], 0.0, 1.0)
    return {
        "score": round(total, 4),
        "hard_gate_score": round(hard_gate_score, 4),
        "family_semantics_score": family["score"],
        "structural_similarity_score": structural["score"],
        "reason": (
            f"hard={hard_gate_score:.3f},family={family['score']:.3f},"
            f"struct={structural['score']:.3f},blockers={audit['blocker_count']},warnings={audit['warning_count']}"
        ),
        "audit": audit,
        "family": family,
        "structural": structural,
    }
