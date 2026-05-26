import re
from collections import defaultdict


COMMON_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "into",
    "this",
    "that",
    "these",
    "those",
    "using",
    "given",
    "after",
    "before",
    "input",
    "output",
    "tensor",
    "tensors",
    "function",
    "wrapper",
    "kernel",
    "triton",
    "torch",
    "provided",
    "information",
    "generation",
    "expert",
    "write",
    "writing",
    "corresponding",
    "capable",
    "ensure",
    "fully",
    "aligns",
    "according",
    "instruction",
    "description",
    "entry",
    "args",
    "shape",
    "shapes",
    "math",
    "optional",
    "default",
    "none",
    "true",
    "false",
}

PRIMARY_FAMILY_PRIORITY = [
    "conv2d",
    "bmm",
    "matmul",
    "attention",
    "sampling",
    "quantize",
    "dequantize",
    "embedding",
    "cumsum",
    "logsoftmax",
    "softmax",
    "rmsnorm",
    "layer_norm",
    "dropout",
    "gelu",
    "relu",
    "sigmoid",
    "tanh",
    "solve",
    "lu",
    "sum",
    "mean",
    "max_pool",
    "div",
    "mul",
    "sub",
    "add",
]

OPERATOR_PATTERNS = {
    "conv2d": [r"\bconv2d\b", r"\b2d convolution\b"],
    "bmm": [r"\bbmm\b", r"\bbatch matrix multiplication\b", r"\bbatched matrix multiplication\b"],
    "matmul": [r"\bmatmul\b", r"\bmatrix multiplication\b"],
    "attention": [r"\battention\b", r"\bquery\b", r"\bkey\b", r"\bvalue\b"],
    "sampling": [
        r"\bmultinomial\b",
        r"\bsampling\b",
        r"\bprobability distribution\b",
        r"\brandom samples?\b",
        r"\bsearchsorted\b",
        r"\bcdf\b",
    ],
    "quantize": [r"\bquantiz", r"\bint8\b", r"\bf8\b", r"\bfp8\b", r"\b4-bit\b", r"\b4bit\b", r"\bzero_point\b"],
    "dequantize": [r"\bdequantiz", r"\bde-quantiz"],
    "embedding": [r"\bembedding\b", r"\bgather\b", r"\bindex select\b", r"\bembedding bag\b"],
    "cumsum": [r"\bcumsum\b", r"\bcumulative sum\b", r"\bprefix sum\b", r"\bscan\b"],
    "dropout": [r"\bdropout\b"],
    "gelu": [r"\bgelu\b"],
    "relu": [r"\brelu\b", r"\bleaky relu\b"],
    "sigmoid": [r"\bsigmoid\b"],
    "tanh": [r"\btanh\b"],
    "softmax": [r"(?<!log)\ssoftmax\b", r"\bsoftmax\b"],
    "logsoftmax": [r"\blog[\s_-]*softmax\b"],
    "rmsnorm": [r"\brms[\s_-]*norm\b", r"\brms normalization\b", r"\broot mean square\b"],
    "layer_norm": [r"\blayer[\s_-]*norm\b", r"\blayer normalization\b"],
    "sum": [r"\bsum\b", r"\bsummation\b"],
    "mean": [r"\bmean\b", r"\baverage\b"],
    "max_pool": [r"\bmax[_\s-]*pool"],
    "solve": [r"\bsolve\b", r"\blinear systems?\b"],
    "lu": [r"\blu decomposition\b", r"\blu\b"],
    "div": [r"\bdiv\b", r"\bdivide(?:s|d)?\b", r"\bdivision\b"],
    "mul": [r"\bmul\b", r"\bmultiply\b", r"\bproduct\b"],
    "sub": [r"\bsub\b", r"\bsubtract(?:ion)?\b"],
    "add": [r"\badd\b", r"\bplus\b", r"\bsum of\b"],
}

PROPERTY_PATTERNS = {
    "broadcast": [r"\bbroadcast"],
    "batch": [r"\bbatch"],
    "out": [r"\bout=None\b", r"\bout\b"],
    "inplace": [r"\binplace\b"],
    "approximate": [r"\bapproximate\b", r"\btanh-based approximation\b"],
    "groups": [r"\bgroups\b"],
    "padding": [r"\bpadding\b"],
    "stride": [r"\bstride\b"],
    "dilation": [r"\bdilation\b"],
    "dim": [r"\bdim\b", r"\bdimension\b"],
    "complex": [r"\bcomplex\b"],
    "bias": [r"\bbias\b"],
    "seed": [r"\bseed\b", r"\bseeds\b"],
    "random": [r"\brandom\b"],
    "rounding_mode": [r"\brounding_mode\b", r"\bfloor\b", r"\bceil\b", r"\btrunc\b"],
    "autograd": [r"\bautograd\b", r"\bbackward\b", r"\bforward pass\b"],
    "mask": [r"\bmask\b", r"\bmasking\b"],
    "contiguous": [r"\bcontiguous\b"],
    "causal": [r"\bcausal\b"],
    "reduction": [r"\breduction\b", r"\breduce\b"],
}

SPECIAL_CLUSTER_PATTERNS = {
    "sampling_random": [
        r"\bmultinomial\b",
        r"\bbinomial\b",
        r"\brand\b",
        r"\brandom\b",
        r"\bsearchsorted\b",
        r"\bprobability distribution\b",
        r"\btotal_count\b",
        r"\bgenerator\b",
    ],
    "quantization_pack": [
        r"\bquantiz",
        r"\bdequantiz",
        r"\bscale\b",
        r"\bzero_point\b",
        r"\bint8\b",
        r"\bint4\b",
        r"\bpacked?\b",
    ],
    "embedding_lookup": [
        r"\bembedding\b",
        r"\bgather\b",
        r"\bindex_select\b",
        r"\bindices\b",
        r"\bweight\b",
    ],
    "grid_sample": [
        r"\bgrid_sample\b",
        r"\baffine_grid\b",
        r"\bgrid sampling\b",
        r"\baffine transformation\b",
    ],
    "cumsum_scan": [
        r"\bcumsum\b",
        r"\bcumulative sum\b",
        r"\bprefix sum\b",
        r"\bscan\b",
    ],
}


def simple_tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+", text.lower())


def extract_expected_signature(sample_input: str) -> tuple[str | None, str | None]:
    match = re.search(
        r"Wrapper Entry Information:\s*(?:def\s+)?([A-Za-z_][A-Za-z0-9_.]*)\((.*?)\)(?:\s*->\s*([^\n]+))?",
        sample_input,
        flags=re.DOTALL,
    )
    if not match:
        return None, None
    raw_name = match.group(1).strip()
    wrapper_name = raw_name.split(".")[-1]
    return wrapper_name, match.group(2).strip()


def split_wrapper_tokens(name: str | None) -> set[str]:
    if not name:
        return set()
    return {token for token in name.lower().split("_") if token and token not in COMMON_STOPWORDS}


def parse_arg_names(params: str | None) -> list[str]:
    if not params:
        return []
    arg_names = []
    for chunk in params.split(","):
        token = chunk.strip()
        if not token or token == "*":
            continue
        token = token.split("=", 1)[0].strip()
        token = token.split(":", 1)[0].strip()
        if token:
            arg_names.append(token)
    return arg_names


def extract_pattern_tags(text: str, pattern_map: dict[str, list[str]]) -> set[str]:
    lowered = text.lower()
    tags = set()
    for tag, patterns in pattern_map.items():
        if any(re.search(pattern, lowered) for pattern in patterns):
            tags.add(tag)
    return tags


def primary_family(tags: set[str]) -> str | None:
    for family in PRIMARY_FAMILY_PRIORITY:
        if family in tags:
            return family
    return sorted(tags)[0] if tags else None


def extract_text_tokens(text: str) -> set[str]:
    return {
        token
        for token in simple_tokenize(text)
        if len(token) > 2 and token not in COMMON_STOPWORDS and not token.isdigit()
    }


def extract_task8_features(sample_input: str) -> dict:
    wrapper_name, params = extract_expected_signature(sample_input)
    arg_names = parse_arg_names(params)
    text_tokens = extract_text_tokens(sample_input)
    operator_tags = extract_pattern_tags(sample_input, OPERATOR_PATTERNS)
    property_tags = extract_pattern_tags(sample_input, PROPERTY_PATTERNS)
    wrapper_tokens = split_wrapper_tokens(wrapper_name)
    family = primary_family(operator_tags)
    has_out_arg = "out" in {name.lower() for name in arg_names}
    return {
        "wrapper_name": wrapper_name,
        "arg_names": arg_names,
        "arg_name_set": {name.lower() for name in arg_names},
        "arg_count": len(arg_names),
        "wrapper_tokens": wrapper_tokens,
        "operator_tags": operator_tags,
        "property_tags": property_tags,
        "text_tokens": text_tokens,
        "primary_family": family,
        "has_out_arg": has_out_arg,
    }


def extract_task8_special_clusters(text: str) -> set[str]:
    return extract_pattern_tags(text, SPECIAL_CLUSTER_PATTERNS)


def build_task8_retrieval_context(examples: list[dict]) -> dict:
    token_df = defaultdict(int)
    operator_df = defaultdict(int)
    property_df = defaultdict(int)
    wrapper_df = defaultdict(int)
    parsed_examples = []

    for example in examples:
        features = extract_task8_features(example["input"])
        parsed_examples.append((example, features))
        for token in features["text_tokens"]:
            token_df[token] += 1
        for tag in features["operator_tags"]:
            operator_df[tag] += 1
        for tag in features["property_tags"]:
            property_df[tag] += 1
        for token in features["wrapper_tokens"]:
            wrapper_df[token] += 1

    return {
        "parsed_examples": parsed_examples,
        "token_df": dict(token_df),
        "operator_df": dict(operator_df),
        "property_df": dict(property_df),
        "wrapper_df": dict(wrapper_df),
    }


def build_task8_lexical_retrieval_context(examples: list[dict]) -> dict:
    token_df = defaultdict(int)
    parsed_examples = []
    parsed_examples_with_features = []
    for example in examples:
        tokens = set(simple_tokenize(example["input"]))
        features = extract_task8_features(example["input"])
        clusters = extract_task8_special_clusters(example["input"])
        parsed_examples.append((example, tokens))
        parsed_examples_with_features.append((example, tokens, features, clusters))
        for token in tokens:
            token_df[token] += 1
    return {
        "parsed_examples": parsed_examples,
        "parsed_examples_with_features": parsed_examples_with_features,
        "token_df": dict(token_df),
    }


def reorder_task8_examples_by_lexical_retrieval(
    examples: list[dict],
    text2annotate: str,
    retrieval_context: dict | None = None,
) -> list[dict]:
    if retrieval_context is None:
        retrieval_context = build_task8_lexical_retrieval_context(examples)

    target_tokens = set(simple_tokenize(text2annotate))
    token_df = retrieval_context["token_df"]

    def token_weight(token: str) -> float:
        return 1.0 / (token_df.get(token, 1) ** 0.5)

    scored = []
    for example, ex_tokens in retrieval_context["parsed_examples"]:
        overlap = target_tokens & ex_tokens
        score = sum(token_weight(token) for token in overlap) if overlap else 0.0
        scored.append((score, example))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [example for _, example in scored]


def reorder_task8_examples_by_family_lexical_retrieval(
    examples: list[dict],
    text2annotate: str,
    retrieval_context: dict | None = None,
) -> list[dict]:
    if retrieval_context is None:
        retrieval_context = build_task8_lexical_retrieval_context(examples)

    target_tokens = set(simple_tokenize(text2annotate))
    target_features = extract_task8_features(text2annotate)
    target_clusters = extract_task8_special_clusters(text2annotate)
    token_df = retrieval_context["token_df"]

    def token_weight(token: str) -> float:
        return 1.0 / (token_df.get(token, 1) ** 0.5)

    scored = []
    for example, ex_tokens, ex_features, ex_clusters in retrieval_context["parsed_examples_with_features"]:
        overlap = target_tokens & ex_tokens
        score = sum(token_weight(token) for token in overlap) if overlap else 0.0

        if target_clusters and ex_clusters:
            score += 8.0 * len(target_clusters & ex_clusters)
        elif (
            target_features["primary_family"]
            and target_features["primary_family"] == ex_features["primary_family"]
        ):
            score += 4.0

        if target_features["operator_tags"] and ex_features["operator_tags"]:
            score += 1.5 * len(target_features["operator_tags"] & ex_features["operator_tags"])
        if target_features["property_tags"] and ex_features["property_tags"]:
            score += 0.75 * len(target_features["property_tags"] & ex_features["property_tags"])
        if target_features["wrapper_tokens"] and ex_features["wrapper_tokens"]:
            score += 1.0 * len(target_features["wrapper_tokens"] & ex_features["wrapper_tokens"])
        if target_features["has_out_arg"] and ex_features["has_out_arg"]:
            score += 0.5
        if target_features["arg_name_set"] and ex_features["arg_name_set"]:
            score += 0.5 * len(target_features["arg_name_set"] & ex_features["arg_name_set"])

        scored.append((score, example))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [example for _, example in scored]


def _idf_score(overlap: set[str], df_map: dict[str, int], base_weight: float) -> float:
    score = 0.0
    for token in overlap:
        score += base_weight / (df_map.get(token, 1) ** 0.5)
    return score


def score_task8_example(example_features: dict, target_features: dict, retrieval_context: dict, mode: str = "hybrid") -> float:
    lexical_score = _idf_score(
        target_features["text_tokens"] & example_features["text_tokens"],
        retrieval_context["token_df"],
        base_weight=2.0,
    )
    if mode == "lexical":
        return lexical_score

    score = lexical_score * 0.35 if mode == "structured" else lexical_score
    score += _idf_score(
        target_features["operator_tags"] & example_features["operator_tags"],
        retrieval_context["operator_df"],
        base_weight=4.0,
    )
    score += _idf_score(
        target_features["property_tags"] & example_features["property_tags"],
        retrieval_context["property_df"],
        base_weight=1.5,
    )
    score += _idf_score(
        target_features["wrapper_tokens"] & example_features["wrapper_tokens"],
        retrieval_context["wrapper_df"],
        base_weight=2.5,
    )
    if target_features["primary_family"] and target_features["primary_family"] == example_features["primary_family"]:
        score += 5.0

    if target_features["operator_tags"] and target_features["operator_tags"] <= example_features["operator_tags"]:
        score += 2.0

    arg_overlap = target_features["arg_name_set"] & example_features["arg_name_set"]
    score += 0.8 * len(arg_overlap)

    if target_features["has_out_arg"] and example_features["has_out_arg"]:
        score += 0.5

    if target_features["arg_count"] and abs(target_features["arg_count"] - example_features["arg_count"]) <= 1:
        score += 0.5

    return score


def reorder_task8_examples_by_retrieval(
    examples: list[dict],
    text2annotate: str,
    retrieval_context: dict | None = None,
    mode: str = "hybrid",
) -> list[dict]:
    if retrieval_context is None:
        retrieval_context = build_task8_retrieval_context(examples)

    target_features = extract_task8_features(text2annotate)
    scored = []
    for example, example_features in retrieval_context["parsed_examples"]:
        score = score_task8_example(example_features, target_features, retrieval_context, mode=mode)
        scored.append((score, example))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [example for _, example in scored]
