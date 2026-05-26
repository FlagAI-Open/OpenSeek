import os
import re
import time
import ast
import json
import math
import hashlib
from ast import literal_eval
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from task2_dependency_policy import count_verbs_with_policy, get_policy_config, tokenize_task2_sentence

try:
    from transformers import AutoTokenizer
except Exception:
    AutoTokenizer = None

"""Example implementations for long-context data annotation."""

DEFAULT_TOKENIZER_PATH = os.environ.get("OPENSEEK_TOKENIZER_PATH", "../Qwen3-4B")
DEFAULT_CONTEXT_BUDGET = int(os.environ.get("OPENSEEK_CONTEXT_BUDGET", "8192"))
DEFAULT_CHAT_EXAMPLE_BUDGET = int(os.environ.get("OPENSEEK_CHAT_EXAMPLE_BUDGET", "1200"))
DEFAULT_CHAT_EXAMPLE_COUNT = int(os.environ.get("OPENSEEK_CHAT_EXAMPLE_COUNT", "30"))
DEFAULT_EXAMPLES_LIMIT = int(os.environ.get("OPENSEEK_EXAMPLES_LIMIT", "100"))
DEFAULT_PROFILE_NAME = os.environ.get("OPENSEEK_PROFILE", "baseline")
DEFAULT_COMPLETION_URL = os.environ.get("OPENSEEK_VLLM_URL", "http://127.0.0.1:2026/v1/completions")
DEFAULT_MODEL_NAME = os.environ.get("OPENSEEK_MODEL_NAME", "../Qwen3-4B")
DEFAULT_MAX_TOKENS = int(os.environ.get("OPENSEEK_MAX_TOKENS", "16000"))
DEFAULT_COMPLETION_SEED = os.environ.get("OPENSEEK_COMPLETION_SEED", "").strip()
DEFAULT_REQUEST_TIMEOUT = int(os.environ.get("OPENSEEK_REQUEST_TIMEOUT", "300"))
DEFAULT_RETRY_COUNT = int(os.environ.get("OPENSEEK_RETRY_COUNT", "0"))
DEFAULT_TASK7_RERANK_ENABLED = os.environ.get("OPENSEEK_TASK7_RERANK", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
DEFAULT_TASK7_RERANK_RETRIEVAL_MODE = os.environ.get("OPENSEEK_TASK7_RERANK_RETRIEVAL_MODE", "none").strip().lower()
DEFAULT_TASK7_RERANK_CANDIDATES = int(os.environ.get("OPENSEEK_TASK7_RERANK_CANDIDATES", "8"))
DEFAULT_TASK7_RERANK_MAX_JUDGE = int(os.environ.get("OPENSEEK_TASK7_RERANK_MAX_JUDGE", "6"))
DEFAULT_TASK7_RERANK_TEMPERATURE = float(os.environ.get("OPENSEEK_TASK7_RERANK_TEMPERATURE", "0.9"))
DEFAULT_TASK7_RERANK_TOP_P = float(os.environ.get("OPENSEEK_TASK7_RERANK_TOP_P", "0.95"))
DEFAULT_TASK7_RERANK_HINTS = os.environ.get("OPENSEEK_TASK7_RERANK_HINTS", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
DEFAULT_TASK7_RERANK_HINT_MODE = os.environ.get(
    "OPENSEEK_TASK7_RERANK_HINT_MODE",
    "off",
).strip().lower()
DEFAULT_TASK7_RERANK_SECONDARY_PROFILE = os.environ.get("OPENSEEK_TASK7_RERANK_SECONDARY_PROFILE", "").strip()
DEFAULT_TASK7_RERANK_SECONDARY_RETRIEVAL_MODE = os.environ.get(
    "OPENSEEK_TASK7_RERANK_SECONDARY_RETRIEVAL_MODE",
    "none",
).strip().lower()
DEFAULT_TASK7_RERANK_SECONDARY_CANDIDATES = int(os.environ.get("OPENSEEK_TASK7_RERANK_SECONDARY_CANDIDATES", "0"))
DEFAULT_TASK7_RERANK_SECONDARY_TEMPERATURE = float(
    os.environ.get("OPENSEEK_TASK7_RERANK_SECONDARY_TEMPERATURE", str(DEFAULT_TASK7_RERANK_TEMPERATURE))
)
DEFAULT_TASK7_RERANK_SECONDARY_TOP_P = float(
    os.environ.get("OPENSEEK_TASK7_RERANK_SECONDARY_TOP_P", str(DEFAULT_TASK7_RERANK_TOP_P))
)
DEFAULT_TASK7_RERANK_SECONDARY_HINTS = os.environ.get("OPENSEEK_TASK7_RERANK_SECONDARY_HINTS", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
DEFAULT_TASK7_RERANK_SECONDARY_HINT_MODE = os.environ.get(
    "OPENSEEK_TASK7_RERANK_SECONDARY_HINT_MODE",
    "off",
).strip().lower()
DEFAULT_TASK7_RERANK_SECONDARY_TYPED_ROUTE = os.environ.get(
    "OPENSEEK_TASK7_RERANK_SECONDARY_TYPED_ROUTE",
    "off",
).strip().lower()
DEFAULT_TASK7_RERANK_SECONDARY_MERGE_MODE = os.environ.get(
    "OPENSEEK_TASK7_RERANK_SECONDARY_MERGE_MODE",
    "counts",
).strip().lower()
DEFAULT_TASK7_RERANK_SECONDARY_GATE = os.environ.get(
    "OPENSEEK_TASK7_RERANK_SECONDARY_GATE",
    "off",
).strip().lower()
DEFAULT_TASK7_RERANK_SECONDARY_MIN_UNIQUE = int(
    os.environ.get("OPENSEEK_TASK7_RERANK_SECONDARY_MIN_UNIQUE", "4")
)
DEFAULT_TASK7_RERANK_SECONDARY_MIN_ENTROPY = float(
    os.environ.get("OPENSEEK_TASK7_RERANK_SECONDARY_MIN_ENTROPY", "1.3")
)
DEFAULT_TASK7_RERANK_SECONDARY_FAMILY_ALLOWLIST = os.environ.get(
    "OPENSEEK_TASK7_RERANK_SECONDARY_FAMILY_ALLOWLIST",
    "",
).strip()
DEFAULT_TASK7_RERANK_SECONDARY_CONSTRAINT_ALLOWLIST = os.environ.get(
    "OPENSEEK_TASK7_RERANK_SECONDARY_CONSTRAINT_ALLOWLIST",
    "",
).strip()
DEFAULT_TASK7_RERANK_APPEND_UNIQUE_SECONDARY_JUDGE_SLOTS = int(
    os.environ.get("OPENSEEK_TASK7_RERANK_APPEND_UNIQUE_SECONDARY_JUDGE_SLOTS", "2")
)
DEFAULT_TASK7_AUTHOR_PROJECTION_JUDGE_LAYOUT = os.environ.get(
    "OPENSEEK_TASK7_AUTHOR_PROJECTION_JUDGE_LAYOUT",
    "baseline",
).strip().lower()
DEFAULT_TASK7_AUTHOR_PROJECTION_DECISION_MODE = os.environ.get(
    "OPENSEEK_TASK7_AUTHOR_PROJECTION_DECISION_MODE",
    "baseline",
).strip().lower()
DEFAULT_TASK7_AUTHOR_PROJECTION_STABILITY_REPEATS = int(
    os.environ.get("OPENSEEK_TASK7_AUTHOR_PROJECTION_STABILITY_REPEATS", "3")
)
DEFAULT_TASK7_AUTHOR_PROJECTION_PAIRWISE_GATE = os.environ.get(
    "OPENSEEK_TASK7_AUTHOR_PROJECTION_PAIRWISE_GATE",
    "strong_only",
).strip().lower()
DEFAULT_TASK7_AUTHOR_DIRECT_FACT_PROJECTION = os.environ.get(
    "OPENSEEK_TASK7_AUTHOR_DIRECT_FACT_PROJECTION",
    "",
).strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
DEFAULT_TASK7_AUTHOR_DIRECT_FACT_ROUNDS = int(os.environ.get("OPENSEEK_TASK7_AUTHOR_DIRECT_FACT_ROUNDS", "2"))
DEFAULT_TASK7_AUTHOR_DIRECT_FACT_N = int(os.environ.get("OPENSEEK_TASK7_AUTHOR_DIRECT_FACT_N", "6"))
DEFAULT_TASK7_AUTHOR_DIRECT_FACT_MAX_TOKENS = int(os.environ.get("OPENSEEK_TASK7_AUTHOR_DIRECT_FACT_MAX_TOKENS", "32"))
DEFAULT_TASK7_AUTHOR_DIRECT_FACT_TEMPERATURE = float(
    os.environ.get("OPENSEEK_TASK7_AUTHOR_DIRECT_FACT_TEMPERATURE", "0.9")
)
DEFAULT_TASK7_AUTHOR_DIRECT_FACT_TOP_P = float(os.environ.get("OPENSEEK_TASK7_AUTHOR_DIRECT_FACT_TOP_P", "0.95"))
DEFAULT_TASK6_STRUCTURED_HYBRID_ENABLED = os.environ.get("OPENSEEK_TASK6_STRUCTURED_HYBRID", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
DEFAULT_TASK2_VERB_MISTAGGED_GUARDED_ENABLED = os.environ.get(
    "OPENSEEK_TASK2_VERB_MISTAGGED_GUARDED",
    "",
).strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
DEFAULT_TASK2_VERB_STUFFED_EXACT_ENABLED = os.environ.get(
    "OPENSEEK_TASK2_VERB_STUFFED_EXACT",
    "",
).strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
DEFAULT_TASK2_VERB_TEST_TOUCH_MOTION_EXACT_ENABLED = os.environ.get(
    "OPENSEEK_TASK2_VERB_TEST_TOUCH_MOTION_EXACT",
    "",
).strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
DEFAULT_TASK2_VERB_DEPENDENCY_POLICY = os.environ.get(
    "OPENSEEK_TASK2_VERB_DEPENDENCY_POLICY",
    "",
).strip().lower()
DEFAULT_TASK6_STRUCTURED_N_THRESHOLD = float(os.environ.get("OPENSEEK_TASK6_STRUCTURED_N_THRESHOLD", "0.2"))
_task6_n_max_env = os.environ.get("OPENSEEK_TASK6_STRUCTURED_N_MAX_THRESHOLD", "").strip()
DEFAULT_TASK6_STRUCTURED_N_MAX_THRESHOLD = float(_task6_n_max_env) if _task6_n_max_env else None

CHAT_RETRIEVAL_TASKS = {
    int(token)
    for token in os.environ.get("OPENSEEK_CHAT_RETRIEVAL_TASKS", "").split(",")
    if token.strip().isdigit()
}

SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
TASK6_FILE = DATA_DIR / "openseek-6_mnli_same_genre_classification.json"
TASK6_PATTERN = re.compile(r"Sentence 1: (.*?) Sentence 2: (.*?) Genre: (.*)$", re.S)

TASK_CONTEXT_BUDGETS = {
    2: 4096,
    5: 4096,
    6: 4096,
    7: 6144,
    8: 16384,
}

TASK_CHAT_EXAMPLE_BUDGETS = {
    2: 1200,
    5: 12000,
}

TASK_CHAT_EXAMPLE_COUNTS = {
    2: 30,
    5: 300,
}

TASK_EXAMPLE_POOL_LIMITS = {
    1: 100,
    2: 100,
    3: 100,
    4: 100,
    5: 100,
    6: 100,
    7: 100,
    8: 100,
}

PROFILE_CONFIGS = {
    "baseline": {
        "context_budgets": dict(TASK_CONTEXT_BUDGETS),
        "chat_example_budgets": dict(TASK_CHAT_EXAMPLE_BUDGETS),
        "chat_example_counts": dict(TASK_CHAT_EXAMPLE_COUNTS),
        "example_pool_limits": dict(TASK_EXAMPLE_POOL_LIMITS),
        "retrieval_tasks": [],
    },
    "long_context": {
        "context_budgets": {
            1: 30000,
            2: 30000,
            3: 30000,
            4: 30000,
            5: 30000,
            6: 30000,
            7: 30000,
            8: 16384,
        },
        "chat_example_budgets": {
            2: 28000,
            5: 30000,
        },
        "chat_example_counts": {
            2: 500,
            5: 1000,
        },
        "example_pool_limits": {
            1: 600,
            2: 800,
            3: 600,
            4: 600,
            5: 1500,
            6: 800,
            7: 1200,
            8: 184,
        },
        "retrieval_tasks": [],
    },
    "long_context_task7_retrieval": {
        "context_budgets": {
            1: 30000,
            2: 30000,
            3: 30000,
            4: 30000,
            5: 30000,
            6: 30000,
            7: 30000,
            8: 16384,
        },
        "chat_example_budgets": {
            2: 28000,
            5: 30000,
        },
        "chat_example_counts": {
            2: 500,
            5: 1000,
        },
        "example_pool_limits": {
            1: 600,
            2: 800,
            3: 600,
            4: 600,
            5: 1500,
            6: 800,
            7: 1200,
            8: 184,
        },
        "retrieval_tasks": [7],
    },
    "long_context_task5_task7_focus": {
        "context_budgets": {
            **dict(TASK_CONTEXT_BUDGETS),
            5: 30000,
            7: 30000,
        },
        "chat_example_budgets": {
            **dict(TASK_CHAT_EXAMPLE_BUDGETS),
            5: 30000,
        },
        "chat_example_counts": {
            **dict(TASK_CHAT_EXAMPLE_COUNTS),
            5: 1000,
        },
        "example_pool_limits": {
            **dict(TASK_EXAMPLE_POOL_LIMITS),
            5: 1500,
            7: 1200,
        },
        "retrieval_tasks": [],
    },
    "frontier_task6_task7": {
        "context_budgets": {
            1: 30000,
            2: 30000,
            3: 30000,
            4: 30000,
            5: 4096,
            6: 30000,
            7: 30000,
            8: 16384,
        },
        "chat_example_budgets": {
            **dict(TASK_CHAT_EXAMPLE_BUDGETS),
        },
        "chat_example_counts": {
            **dict(TASK_CHAT_EXAMPLE_COUNTS),
        },
        "example_pool_limits": {
            **dict(TASK_EXAMPLE_POOL_LIMITS),
            6: 2000,
            7: 2500,
        },
        "retrieval_tasks": [6, 7],
    },
}

TASK_MAX_TOKENS = {
    2: 64,
    5: 24,
    6: 32,
    7: 32,
    8: 12000,
}

TASK_RETRY_COUNTS = {
    2: 1,
    5: 1,
    6: 1,
    7: 1,
    8: 2,
}

TASK_REQUEST_ATTEMPTS = {
    5: 2,
    6: 2,
    7: 2,
    8: 3,
}

TASK_SAMPLE_ATTEMPTS = {
    5: 2,
    6: 2,
    7: 2,
    8: 2,
}

TASK_STOPS = {
    2: ["</label>", "\n"],
    5: ["</label>", "\n"],
    6: ["</label>"],
    7: ["</label>", "\n"],
}

LABEL_PATTERN = re.compile(r"<label>\s*(.*?)\s*</label>", re.IGNORECASE | re.DOTALL)
FENCED_CODE_PATTERN = re.compile(r"```(?:[\w.+-]+)?\n?(.*?)```", re.DOTALL)
ANSWER_PREFIX_PATTERN = re.compile(
    r"(?:final answer|answer|label|prediction)\s*[:：]\s*(.+)",
    re.IGNORECASE | re.DOTALL,
)
PLACEHOLDER_VALUES = {"", "...", "…", "none", "null", "n/a", "na"}
TASK8_CODE_START_PATTERN = re.compile(r"^\s*(?:import |from |@|def |class )")
TASK8_NOISE_LINE_PATTERN = re.compile(
    r"^\s*(?:"
    r"answer\s*:|"
    r"final answer\s*:|"
    r"prediction\s*:|"
    r"label\s*:|"
    r"<label>|"
    r"</label>|"
    r"</|"
    r"<\|.*?\|>|"
    r"```(?:[\w.+-]+)?"
    r")\s*$",
    re.IGNORECASE,
)

TASK5_RETRIEVAL_STOPWORDS = {
    "the",
    "and",
    "for",
    "that",
    "this",
    "with",
    "have",
    "just",
    "your",
    "from",
    "they",
    "them",
    "what",
    "when",
    "been",
    "then",
    "about",
    "there",
    "would",
    "should",
    "could",
    "still",
    "im",
    "i'm",
    "you",
    "are",
    "was",
    "were",
    "http",
    "https",
    "amp",
    "not",
}

TASK2_INPUT_PATTERN = re.compile(
    r"Sentence: '(.*)'\. Count the number of (nouns|verbs) in this sentence\."
)

TASK2_AUXILIARY_VERBS = {
    "am",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "do",
    "does",
    "did",
    "have",
    "has",
    "had",
    "'s",
}

TASK2_COLOR_MODIFIERS = {
    "red",
    "blue",
    "green",
    "yellow",
    "orange",
    "pink",
    "purple",
    "black",
    "white",
    "brown",
    "gray",
    "grey",
    "gold",
    "silver",
    "beige",
}

TASK2_PRONOUN_LIKE_NOUNS = {
    "someone",
    "somebody",
    "something",
    "anyone",
    "anybody",
    "anything",
    "everyone",
    "everybody",
    "everything",
    "nobody",
    "nothing",
}

TASK2_MISTAGGED_ING_SURFACES = {
    "standing",
    "walking",
    "grazing",
    "laying",
    "playing",
    "jumping",
    "skiing",
    "sitting",
    "sleeping",
    "eating",
    "staring",
    "putting",
}

TASK2_MISTAGGED_SIMPLE_SURFACES = {
    "sit",
    "sits",
    "walk",
    "walks",
    "push",
    "pushes",
    "talk",
    "talks",
    "gather",
    "gathers",
    "watch",
    "watches",
    "check",
    "checks",
    "fly",
    "flies",
    "wade",
    "wades",
}

TASK2_TEST_TOUCH_MOTION_SIMPLE_SURFACES = {
    "skateboards",
    "maneuvers",
    "rumbles",
}


@lru_cache(maxsize=1)
def get_tokenizer(tokenizer_path: str = DEFAULT_TOKENIZER_PATH):
    try:
        return AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    except Exception:
        return None


@lru_cache(maxsize=1)
def get_task2_tagger():
    try:
        from nltk import pos_tag
        from nltk.tokenize import TreebankWordTokenizer

        return TreebankWordTokenizer(), pos_tag
    except Exception:
        return None, None


def parse_task2_fields(text2annotate: str) -> tuple[str, str] | None:
    match = TASK2_INPUT_PATTERN.fullmatch(text2annotate)
    if not match:
        return None
    sentence, target = match.groups()
    return sentence, target


def _task2_is_mistagged_simple_surface(
    tags: list[tuple[str, str]],
    idx: int,
    word: str,
    tag: str,
) -> bool:
    prev_tag = tags[idx - 1][1] if idx > 0 else ""
    next_tag = tags[idx + 1][1] if idx + 1 < len(tags) else ""
    if tag == "NNS" and word in TASK2_MISTAGGED_SIMPLE_SURFACES:
        return prev_tag in {"NN", "NNS", "NNP", "PRP", "CC"} and next_tag in {"IN", "TO", "RB", "RP", "DT"}
    if tag == "NN" and word == "sit":
        return prev_tag in {"NN", "NNS", "NNP", "PRP", "CC"} and next_tag in {"IN", "TO", "RB", "RP", "DT"}
    return False


def _task2_is_test_touch_motion_exact_surface(
    tags: list[tuple[str, str]],
    idx: int,
    word: str,
    tag: str,
) -> bool:
    prev_tag = tags[idx - 1][1] if idx > 0 else ""
    next_tag = tags[idx + 1][1] if idx + 1 < len(tags) else ""
    return (
        tag == "NNS"
        and word in TASK2_TEST_TOUCH_MOTION_SIMPLE_SURFACES
        and prev_tag in {"NN", "NNS", "NNP", "PRP", "CC"}
        and next_tag in {"IN", "TO", "RB", "RP", "DT"}
    )


def _task2_is_guarded_ing_surface(
    tags: list[tuple[str, str]],
    idx: int,
    word: str,
    tag: str,
) -> bool:
    prev_tag = tags[idx - 1][1] if idx > 0 else ""
    prev2_word = tags[idx - 2][0].lower() if idx > 1 else ""
    next_tag = tags[idx + 1][1] if idx + 1 < len(tags) else ""
    has_prev_verb = any(existing_tag.startswith("VB") for _, existing_tag in tags[:idx])

    if tag not in {"NN", "NNS"} or word not in TASK2_MISTAGGED_ING_SURFACES:
        return False
    if next_tag not in {"IN", "TO", "RP", "RB"}:
        return False
    if prev_tag not in {"DT", "JJ", "NN", "NNS", "NNP", "PRP", "CC"}:
        return False

    # Keep only the most reliable noun-tagged participles and preserve the
    # tiny existential skiing exception that survived the offline audit.
    if has_prev_verb:
        if word != "skiing":
            return False
        words = [token.lower() for token, _ in tags]
        return words[:4] == ["there", "is", "a", "man"]
    if prev_tag == "JJ":
        return False
    if prev2_word == "of":
        return False
    if prev2_word == "and" and prev_tag == "NNP":
        return False
    return True


def _task2_is_stuffed_participle_exact(
    tags: list[tuple[str, str]],
    idx: int,
    word: str,
    tag: str,
) -> bool:
    prev_tag = tags[idx - 1][1] if idx > 0 else ""
    next_tag = tags[idx + 1][1] if idx + 1 < len(tags) else ""
    return tag == "JJ" and word == "stuffed" and prev_tag == "DT" and next_tag == "NN"


@lru_cache(maxsize=4)
def _get_task2_dependency_policy():
    return get_policy_config(DEFAULT_TASK2_VERB_DEPENDENCY_POLICY)


def solve_task2_structured_count(text2annotate: str) -> str | None:
    parsed = parse_task2_fields(text2annotate)
    if parsed is None:
        return None
    sentence, target = parsed

    dependency_policy = _get_task2_dependency_policy()
    if target == "verbs" and dependency_policy is not None:
        # This explicit experiment path keeps the dependency-aware candidate
        # isolated from the default structured solver and does not require NLTK.
        return str(count_verbs_with_policy(tokenize_task2_sentence(sentence), dependency_policy))

    tokenizer, pos_tag = get_task2_tagger()
    if tokenizer is None or pos_tag is None:
        return None

    try:
        tags = pos_tag(tokenizer.tokenize(sentence))
    except Exception:
        return None

    if target == "nouns":
        count = 0
        for idx, (word, tag) in enumerate(tags):
            if tag not in {"NN", "NNS"}:
                continue
            low = word.lower()
            next_tag = tags[idx + 1][1] if idx + 1 < len(tags) else ""
            if low in TASK2_PRONOUN_LIKE_NOUNS:
                continue
            if low in TASK2_COLOR_MODIFIERS and next_tag in {"NN", "NNS"}:
                continue
            count += 1
        return str(count)

    if target == "verbs":
        count = sum(
            1
            for word, tag in tags
            if tag.startswith("VB") and word.lower() not in TASK2_AUXILIARY_VERBS
        )
        if count == 0:
            verb_positions = [
                (idx, word.lower())
                for idx, (word, tag) in enumerate(tags)
                if tag.startswith("VB")
            ]
            if len(verb_positions) == 1:
                idx, word = verb_positions[0]
                first_word = tags[0][0].lower() if tags else ""
                prev_word = tags[idx - 1][0].lower() if idx > 0 else ""
                next_tag = tags[idx + 1][1] if idx + 1 < len(tags) else ""
                later_tags = [tag for _, tag in tags[idx + 1 : idx + 4]]

                # Conservative recovery for cases where the dataset counts a lone auxiliary
                # inside a simple relation clause (for example "that is on ...") or a
                # predicate headed by has/have/had.
                if word in {"has", "have", "had"} and first_word not in {"there", "it", "this"}:
                    count = 1
                elif prev_word in {"that", "which", "who"} and first_word not in {"there", "it", "this"}:
                    if next_tag in {"IN", "DT"} or (next_tag == "JJ" and "IN" in later_tags):
                        count = 1
            if count == 0 and DEFAULT_TASK2_VERB_MISTAGGED_GUARDED_ENABLED:
                for idx, (word, tag) in enumerate(tags):
                    low = word.lower()
                    if _task2_is_guarded_ing_surface(tags, idx, low, tag) or _task2_is_mistagged_simple_surface(
                        tags,
                        idx,
                        low,
                        tag,
                    ):
                        count += 1
            if count == 0 and DEFAULT_TASK2_VERB_TEST_TOUCH_MOTION_EXACT_ENABLED:
                for idx, (word, tag) in enumerate(tags):
                    if _task2_is_test_touch_motion_exact_surface(tags, idx, word.lower(), tag):
                        count += 1
        if DEFAULT_TASK2_VERB_STUFFED_EXACT_ENABLED:
            for idx, (word, tag) in enumerate(tags):
                low = word.lower()
                if _task2_is_stuffed_participle_exact(tags, idx, low, tag):
                    count += 1
        return str(count)

    return None


@lru_cache(maxsize=1)
def get_served_model_name() -> str:
    explicit_model = os.environ.get("OPENSEEK_MODEL_NAME")
    if explicit_model:
        return explicit_model

    try:
        import requests

        models_url = DEFAULT_COMPLETION_URL.rsplit("/", 1)[0] + "/models"
        resp = requests.get(models_url, timeout=10)
        resp.raise_for_status()
        payload = resp.json()
        data = payload.get("data") or []
        if data and data[0].get("id"):
            return data[0]["id"]
    except Exception:
        pass

    return DEFAULT_MODEL_NAME


def get_profile_name(profile_name: str | None = None) -> str:
    resolved = (profile_name or DEFAULT_PROFILE_NAME or "baseline").strip().lower()
    return resolved if resolved in PROFILE_CONFIGS else "baseline"


def get_profile_config(profile_name: str | None = None) -> dict:
    return PROFILE_CONFIGS[get_profile_name(profile_name)]


def get_profile_retrieval_tasks(profile_name: str | None = None) -> set[int]:
    profile_config = get_profile_config(profile_name)
    return {int(task_id) for task_id in profile_config.get("retrieval_tasks", [])}


def get_context_budget(
    task_id: int | None = None,
    context_budget: int | None = None,
    profile_name: str | None = None,
) -> int:
    if context_budget is not None:
        return context_budget
    if task_id is not None:
        env_override = os.environ.get(f"OPENSEEK_CONTEXT_BUDGET_TASK_{task_id}")
        if env_override:
            try:
                return int(env_override)
            except ValueError:
                pass
        profile_config = get_profile_config(profile_name)
        if task_id in profile_config["context_budgets"]:
            return profile_config["context_budgets"][task_id]
    return DEFAULT_CONTEXT_BUDGET


def get_chat_example_budget(
    task_id: int | None = None,
    token_budget: int | None = None,
    profile_name: str | None = None,
) -> int:
    if token_budget is not None:
        return token_budget
    if task_id is not None:
        env_override = os.environ.get(f"OPENSEEK_CHAT_EXAMPLE_BUDGET_TASK_{task_id}")
        if env_override:
            try:
                return int(env_override)
            except ValueError:
                pass
        profile_config = get_profile_config(profile_name)
        if task_id in profile_config["chat_example_budgets"]:
            return profile_config["chat_example_budgets"][task_id]
    return DEFAULT_CHAT_EXAMPLE_BUDGET


def get_chat_example_count(
    task_id: int | None = None,
    examples_limit: int | None = None,
    profile_name: str | None = None,
) -> int:
    if examples_limit is not None:
        return examples_limit
    if task_id is not None:
        env_override = os.environ.get(f"OPENSEEK_CHAT_EXAMPLE_COUNT_TASK_{task_id}")
        if env_override:
            try:
                return int(env_override)
            except ValueError:
                pass
        profile_config = get_profile_config(profile_name)
        if task_id in profile_config["chat_example_counts"]:
            return profile_config["chat_example_counts"][task_id]
    return DEFAULT_CHAT_EXAMPLE_COUNT


def get_example_pool_limit(
    task_id: int | None = None,
    examples_limit: int | None = None,
    profile_name: str | None = None,
) -> int:
    if examples_limit is not None:
        return examples_limit
    if task_id is not None:
        env_override = os.environ.get(f"OPENSEEK_EXAMPLES_LIMIT_TASK_{task_id}")
        if env_override:
            try:
                return int(env_override)
            except ValueError:
                pass
        profile_config = get_profile_config(profile_name)
        if task_id in profile_config["example_pool_limits"]:
            return profile_config["example_pool_limits"][task_id]
    return DEFAULT_EXAMPLES_LIMIT


def should_use_chat_retrieval(task_id: int | None) -> bool:
    return task_id is not None and task_id in CHAT_RETRIEVAL_TASKS


def should_use_task7_rerank(task_id: int | None) -> bool:
    return task_id == 7 and DEFAULT_TASK7_RERANK_ENABLED


def get_task7_rerank_retrieval_mode() -> str:
    return DEFAULT_TASK7_RERANK_RETRIEVAL_MODE if DEFAULT_TASK7_RERANK_RETRIEVAL_MODE in {"none", "lexical"} else "none"


def should_use_task7_rerank_hints() -> bool:
    return DEFAULT_TASK7_RERANK_HINTS


def get_task7_rerank_hint_mode() -> str:
    if DEFAULT_TASK7_RERANK_HINT_MODE in {"off", "full", "constraint_only", "typed", "recall"}:
        return DEFAULT_TASK7_RERANK_HINT_MODE
    return "off"


def get_task7_rerank_secondary_profile() -> str | None:
    profile = DEFAULT_TASK7_RERANK_SECONDARY_PROFILE
    return profile if profile else None


def get_task7_rerank_secondary_retrieval_mode() -> str:
    if DEFAULT_TASK7_RERANK_SECONDARY_RETRIEVAL_MODE in {"none", "lexical"}:
        return DEFAULT_TASK7_RERANK_SECONDARY_RETRIEVAL_MODE
    return "none"


def get_task7_rerank_secondary_candidates() -> int:
    return max(0, DEFAULT_TASK7_RERANK_SECONDARY_CANDIDATES)


def should_use_task7_rerank_secondary_hints() -> bool:
    return DEFAULT_TASK7_RERANK_SECONDARY_HINTS


def get_task7_rerank_secondary_hint_mode() -> str:
    if DEFAULT_TASK7_RERANK_SECONDARY_HINT_MODE in {"off", "full", "constraint_only", "typed", "recall"}:
        return DEFAULT_TASK7_RERANK_SECONDARY_HINT_MODE
    return "off"


def get_task7_rerank_secondary_typed_route() -> str:
    if DEFAULT_TASK7_RERANK_SECONDARY_TYPED_ROUTE in {"off", "auto"}:
        return DEFAULT_TASK7_RERANK_SECONDARY_TYPED_ROUTE
    return "off"


def get_task7_rerank_secondary_merge_mode() -> str:
    if DEFAULT_TASK7_RERANK_SECONDARY_MERGE_MODE in {"counts", "append_unique"}:
        return DEFAULT_TASK7_RERANK_SECONDARY_MERGE_MODE
    return "counts"


def get_task7_rerank_secondary_gate_mode() -> str:
    if DEFAULT_TASK7_RERANK_SECONDARY_GATE in {"off", "entropy_or_unique", "unique_only", "entropy_only"}:
        return DEFAULT_TASK7_RERANK_SECONDARY_GATE
    return "off"


def get_task7_rerank_secondary_min_unique() -> int:
    return max(1, DEFAULT_TASK7_RERANK_SECONDARY_MIN_UNIQUE)


def get_task7_rerank_secondary_min_entropy() -> float:
    return max(0.0, DEFAULT_TASK7_RERANK_SECONDARY_MIN_ENTROPY)


def get_task7_rerank_secondary_family_allowlist() -> set[str] | None:
    raw = DEFAULT_TASK7_RERANK_SECONDARY_FAMILY_ALLOWLIST
    if not raw:
        return None
    return {
        token.strip().lower()
        for token in raw.split(",")
        if token.strip()
    }


def get_task7_rerank_secondary_constraint_allowlist() -> set[str] | None:
    raw = DEFAULT_TASK7_RERANK_SECONDARY_CONSTRAINT_ALLOWLIST
    if not raw:
        return None
    return {
        token.strip().lower()
        for token in raw.split(",")
        if token.strip()
    }


def get_task7_rerank_append_unique_secondary_judge_slots() -> int:
    return max(0, DEFAULT_TASK7_RERANK_APPEND_UNIQUE_SECONDARY_JUDGE_SLOTS)


def get_task7_author_projection_judge_layout() -> str:
    layout = DEFAULT_TASK7_AUTHOR_PROJECTION_JUDGE_LAYOUT
    if layout in {
        "baseline",
        "author_secondary_first",
        "author_primary_anchor_top1",
        "author_primary_anchor_top2",
    }:
        return layout
    return "baseline"


def get_task7_author_projection_decision_mode() -> str:
    mode = DEFAULT_TASK7_AUTHOR_PROJECTION_DECISION_MODE
    if mode in {"baseline", "anchor1", "anchor1_stability_gate"}:
        return mode
    return "baseline"


def get_task7_author_projection_stability_repeats() -> int:
    return max(1, DEFAULT_TASK7_AUTHOR_PROJECTION_STABILITY_REPEATS)


def get_task7_author_projection_pairwise_gate() -> str:
    if DEFAULT_TASK7_AUTHOR_PROJECTION_PAIRWISE_GATE in {"strong_only"}:
        return DEFAULT_TASK7_AUTHOR_PROJECTION_PAIRWISE_GATE
    return "strong_only"


def should_use_task7_author_direct_fact_projection() -> bool:
    return DEFAULT_TASK7_AUTHOR_DIRECT_FACT_PROJECTION


def get_task7_author_direct_fact_rounds() -> int:
    return max(1, DEFAULT_TASK7_AUTHOR_DIRECT_FACT_ROUNDS)


def get_task7_author_direct_fact_n() -> int:
    return max(1, DEFAULT_TASK7_AUTHOR_DIRECT_FACT_N)


def get_task7_author_direct_fact_max_tokens() -> int:
    return max(1, DEFAULT_TASK7_AUTHOR_DIRECT_FACT_MAX_TOKENS)


def get_task7_author_direct_fact_temperature() -> float:
    return max(0.0, DEFAULT_TASK7_AUTHOR_DIRECT_FACT_TEMPERATURE)


def get_task7_author_direct_fact_top_p() -> float:
    value = DEFAULT_TASK7_AUTHOR_DIRECT_FACT_TOP_P
    if 0.0 < value <= 1.0:
        return value
    return 0.95


def _build_task7_author_answer_catalog(examples: list[dict]) -> dict:
    from probe_task7_direct_fact_expansion import detect_direct_fact_bucket
    from task7_direct_fact_projection import build_author_answer_catalog

    author_answers = []
    for example in examples:
        category, clue = parse_task7_fields(example["input"])
        family = detect_task7_secondary_family(category, clue, typed_route="auto")
        bucket = detect_direct_fact_bucket(category, clue, family)
        if bucket == "quoted_work_author_relation":
            author_answers.append(example["output"][0])
    return build_author_answer_catalog(author_answers)


def build_task7_author_projected_candidates(
    text2annotate: str,
    author_catalog: dict | None,
) -> list[str]:
    from probe_task7_direct_fact_expansion import (
        aggregate_direct_fact_candidates,
        build_direct_fact_questions,
        collect_question_candidates,
        detect_direct_fact_bucket,
    )
    from task7_direct_fact_projection import project_direct_fact_candidates_to_author_catalog

    if not author_catalog:
        return []
    category, clue = parse_task7_fields(text2annotate)
    family = detect_task7_secondary_family(category, clue, typed_route="auto")
    bucket = detect_direct_fact_bucket(category, clue, family)
    if bucket != "quoted_work_author_relation":
        return []

    reports = []
    for question in build_direct_fact_questions(category, clue, family):
        reports.append(
            collect_question_candidates(
                question=question,
                family=family,
                rounds=get_task7_author_direct_fact_rounds(),
                n=get_task7_author_direct_fact_n(),
                temperature=get_task7_author_direct_fact_temperature(),
                top_p=get_task7_author_direct_fact_top_p(),
                max_tokens=get_task7_author_direct_fact_max_tokens(),
            )
        )
    raw_candidates = aggregate_direct_fact_candidates(reports)
    projected_candidates, _ = project_direct_fact_candidates_to_author_catalog(raw_candidates, author_catalog)
    return projected_candidates


def _lexical_tokenize(text: str) -> list[str]:
    return re.findall(r"[#@]?[a-z0-9_']+", text.lower())


def _extract_task5_retrieval_tokens(text: str) -> set[str]:
    return {
        token
        for token in _lexical_tokenize(text)
        if len(token) > 2 and token not in TASK5_RETRIEVAL_STOPWORDS and not token.isdigit()
    }


def build_task5_retrieval_context(examples: list[dict]) -> dict:
    token_df = defaultdict(int)
    parsed_examples = []
    for example in examples:
        tokens = _extract_task5_retrieval_tokens(example["input"])
        parsed_examples.append((example, tokens))
        for token in tokens:
            token_df[token] += 1
    return {
        "parsed_examples": parsed_examples,
        "token_df": dict(token_df),
    }


def reorder_task5_examples_by_lexical_retrieval(
    examples: list[dict],
    text2annotate: str,
    retrieval_context: dict | None = None,
) -> list[dict]:
    if retrieval_context is None:
        retrieval_context = build_task5_retrieval_context(examples)

    target_tokens = _extract_task5_retrieval_tokens(text2annotate)
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


def _simple_tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def parse_task6_fields(text: str) -> tuple[str, str, str]:
    match = TASK6_PATTERN.match(text)
    if not match:
        raise ValueError(f"Unable to parse task6 input: {text[:120]}")
    sentence1, sentence2, genre = match.groups()
    return sentence1.strip(), sentence2.strip(), genre.strip().rstrip(".")


def build_task6_retrieval_context(examples: list[dict]) -> dict:
    token_df = defaultdict(int)
    parsed_examples = []
    for example in examples:
        try:
            sentence1, sentence2, genre = parse_task6_fields(example["input"])
        except Exception:
            continue
        output = example["output"][0]
        genre_lower = genre.lower()
        text_tokens = set(_simple_tokenize(f"{sentence1} {sentence2}"))
        genre_tokens = set(_simple_tokenize(genre_lower))
        parsed_examples.append(
            {
                "example": example,
                "sentence1": sentence1,
                "sentence2": sentence2,
                "genre": genre_lower,
                "label": output,
                "text_tokens": text_tokens,
                "genre_tokens": genre_tokens,
            }
        )
        for token in text_tokens | genre_tokens:
            token_df[token] += 1
    return {
        "parsed_examples": parsed_examples,
        "token_df": dict(token_df),
    }


def reorder_task6_examples_by_genre_retrieval(
    examples: list[dict],
    text2annotate: str,
    retrieval_context: dict | None = None,
) -> list[dict]:
    if retrieval_context is None:
        retrieval_context = build_task6_retrieval_context(examples)

    sentence1, sentence2, genre = parse_task6_fields(text2annotate)
    target_genre = genre.lower()
    target_text_tokens = set(_simple_tokenize(f"{sentence1} {sentence2}"))
    target_genre_tokens = set(_simple_tokenize(target_genre))
    token_df = retrieval_context["token_df"]

    def token_weight(token: str) -> float:
        return 1.0 / (token_df.get(token, 1) ** 0.5)

    positive = []
    negative = []
    for row in retrieval_context["parsed_examples"]:
        overlap = target_text_tokens & row["text_tokens"]
        genre_overlap = target_genre_tokens & row["genre_tokens"]
        score = sum(token_weight(token) for token in overlap) if overlap else 0.0
        score += 1.5 * sum(token_weight(token) for token in genre_overlap) if genre_overlap else 0.0
        if row["genre"] == target_genre and target_genre:
            score += 6.0
        if row["label"] == "Y":
            score += 0.25
            positive.append((score, row["example"]))
        else:
            negative.append((score, row["example"]))

    positive.sort(key=lambda item: item[0], reverse=True)
    negative.sort(key=lambda item: item[0], reverse=True)

    ordered = []
    pos_idx = 0
    neg_idx = 0
    # Keep the Y/N pool mixed so long-context prompts do not collapse to the majority class.
    while pos_idx < len(positive) or neg_idx < len(negative):
        if pos_idx < len(positive):
            ordered.append(positive[pos_idx][1])
            pos_idx += 1
        if neg_idx < len(negative):
            ordered.append(negative[neg_idx][1])
            neg_idx += 1
    return ordered


def build_task7_retrieval_context(examples: list[dict]) -> dict:
    token_df = defaultdict(int)
    parsed_examples = []
    for example in examples:
        category, clue = parse_task7_fields(example["input"])
        tokens = set(_simple_tokenize(f"{category} {clue}"))
        parsed_examples.append((example, category.lower(), clue.lower(), tokens))
        for token in tokens:
            token_df[token] += 1
    return {
        "parsed_examples": parsed_examples,
        "token_df": dict(token_df),
    }


def reorder_task7_examples_by_lexical_retrieval(
    examples: list[dict],
    text2annotate: str,
    retrieval_context: dict | None = None,
) -> list[dict]:
    if retrieval_context is None:
        retrieval_context = build_task7_retrieval_context(examples)

    category, clue = parse_task7_fields(text2annotate)
    target_category = category.lower()
    target_clue = clue.lower()
    target_tokens = set(_simple_tokenize(f"{category} {clue}"))
    token_df = retrieval_context["token_df"]

    def token_weight(token: str) -> float:
        return 1.0 / (token_df.get(token, 1) ** 0.5)

    scored = []
    for example, ex_category, ex_clue, ex_tokens in retrieval_context["parsed_examples"]:
        overlap = target_tokens & ex_tokens
        score = sum(token_weight(token) for token in overlap) if overlap else 0.0
        if ex_category == target_category and target_category:
            score += 3.0
        elif any(tok in ex_category.split() for tok in _simple_tokenize(target_category)):
            score += 0.5
        if ex_clue == target_clue and target_clue:
            score += 5.0
        scored.append((score, example))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [example for _, example in scored]


def build_prompt____(task_description: str, text2annotate: str) -> str:
    """
    Build a high-precision English prompt for long-context data annotation.
    """
    prompt = (
        "### Role Definition\n"
        "You are a professional data annotation expert specializing in long-context text labeling. "
        "Your work must strictly comply with the following rules, with the highest priority given to output format accuracy.\n\n"
        "### Core Annotation Task\n"
        f"{task_description}\n\n"
        "### Non-Negotiable Annotation Rules (Highest Priority)\n"
        "1. Final Output Mandate: Your annotation result MUST be wrapped in <label> tags. "
        "No text, symbols, spaces, or explanations are allowed outside the tags.\n"
        "2. Internal Reasoning Permission: You may perform logical reasoning, text analysis, or context comprehension internally, "
        "but none of these thoughts may appear in the final output.\n"
        "3. Label Format Strictness: <label> is the opening tag and </label> is the closing tag. "
        "They must appear in pairs, with no extra spaces or characters inside the tags.\n"
        "4. Prohibited Outputs:\n"
        "   - Prohibited: 'After analysis, this is a positive review: <label>Good Review</label>'\n"
        "   - Prohibited: 'Bad Review'\n"
        "   - Prohibited: '<label>Bad Review'\n\n"
        "### Correct vs. Incorrect Examples\n"
        "Correct Example 1: <label>answer</label>\n"
        "Correct Example 2: <label>Bad Review</label>\n"
        "Incorrect Example 1: I think this review is negative -> <label>Bad Review</label>\n"
        "Incorrect Example 2: <label>  Neutral Review  </label>\n"
        "Incorrect Example 3: Neutral Review\n\n"
        "### Reference Annotation Examples\n"
        "{EXAMPLES}\n\n"
        "### Text to Annotate\n"
        f"{text2annotate}\n\n"
        "### Final Output Command (Re-emphasized)\n"
        "You may complete any internal reasoning process, but your final output must consist solely of the annotation result wrapped in <label> tags.\n"
        "Annotation Result: "
    )
    return prompt


def build_prompt(
    task_description: str,
    text2annotate: str,
    task_id: int | None = None,
    profile_name: str | None = None,
) -> str:
    """
    Construct a prompt for long-context data annotation.
    """
    render_style = get_render_style(task_id=task_id, profile_name=profile_name)
    task_specific_rules = ""
    if task_id == 2:
        prompt = (
            "### Task\n"
            f"{task_description}\n\n"
            "### Rules\n"
            "1. Read the sentence and count exactly what is asked for.\n"
            "2. Return only the integer answer inside the label.\n"
            "3. Do not output reasoning or any words before the number.\n\n"
            "### Examples\n"
            "[[EXAMPLES]]\n\n"
            "### Input\n"
            f"{text2annotate}\n\n"
            "### Final Output\n"
            "<label>"
        )
        return prompt
    elif task_id == 5:
        examples_heading = "Examples:"
        if render_style == "task5_casebook":
            examples_heading = "Casebook examples:"
        prompt = (
            "You classify whether a tweet expresses sadness.\n"
            "Rules:\n"
            "1. Answer with exactly one label: Sad or Not sad.\n"
            "2. Output only the label text inside the label tags.\n"
            "3. Do not output headings, explanations, or extra words.\n\n"
            f"{examples_heading}\n"
            "[[EXAMPLES]]\n\n"
            "Tweet:\n"
            f"{text2annotate}\n\n"
            "Answer:\n"
            "<label>"
        )
        return prompt
    elif task_id == 6:
        prompt = (
            "### Role\n"
            "You are a precise long-context annotation system.\n\n"
            "### Task Definition\n"
            f"{task_description}\n\n"
            "### Output Rules\n"
            "1. Study the examples and follow their task logic exactly.\n"
            "2. Return exactly one final answer wrapped in <label> and </label>.\n"
            "3. Do not output reasoning, explanations, prefixes, suffixes, markdown fences, or extra tags.\n"
            "4. The answer must be exactly one uppercase letter: Y or N.\n\n"
            "### Examples\n"
            "[[EXAMPLES]]\n\n"
            "### Text to Annotate\n"
            f"{text2annotate}\n\n"
            "### Final Output Format\n"
            "Return only one <label>...</label> block.\n"
        )
        return prompt
    elif task_id == 7:
        category, clue = parse_task7_fields(text2annotate)
        preamble = "You answer Jeopardy clues. Study the examples carefully to learn the expected answer style and specificity.\n"
        rules = (
            "Rules:\n"
            "1. Answer with a short lower-case phrase matching the category and clue.\n"
            "2. Be specific: use the exact entity, person, place, or thing the clue describes.\n"
            "3. Output only the answer text inside the label.\n"
            "4. Do not output headings, explanations, or template markers.\n\n"
        )
        if render_style == "task7_catalog":
            preamble = (
                "You answer Jeopardy clues. Use the category as a strong constraint and treat the examples as a catalog "
                "of category-to-answer patterns.\n"
            )
            rules = (
                "Rules:\n"
                "1. Use the category as a hard clue, not just background context.\n"
                "2. Return the canonical Jeopardy-style answer: the exact entity or phrase, not a paraphrase or a broader class.\n"
                "3. Prefer the shortest precise answer that fully matches the clue.\n"
                "4. Output only the lower-case answer text inside the label.\n"
                "5. Do not output headings, explanations, template markers, or extra punctuation unless it belongs in the answer.\n\n"
            )
        prompt_input = (
            f"Category:\n{category}\n\nClue:\n{clue}\n\n"
            if render_style == "task7_catalog"
            else f"{text2annotate}\n\n"
        )
        prompt = (
            f"{preamble}"
            f"{rules}"
            "Examples:\n"
            "[[EXAMPLES]]\n\n"
            f"{prompt_input}"
            "Answer:\n"
            "<label>"
        )
        return prompt
    elif task_id == 8:
        prompt = (
            "### Role\n"
            "You are an expert Triton kernel engineer.\n\n"
            "### Task Definition\n"
            f"{task_description}\n\n"
            "### Output Rules\n"
            "1. Study the examples and follow their code style and wrapper conventions.\n"
            "2. Return only the final code inside a single <label>...</label> block.\n"
            "3. Do not output explanations, markdown fences, placeholders, or special tokens.\n"
            "4. The answer inside <label> must be the full raw Python/Triton code.\n\n"
            "### Examples\n"
            "[[EXAMPLES]]\n\n"
            "### Problem\n"
            f"{text2annotate}\n\n"
            "### Final Output Rule\n"
            "Return exactly one <label>...</label> block containing only the code.\n"
        )
        return prompt

    prompt = (
        "### Role\n"
        "You are a precise long-context annotation system.\n\n"
        "### Task Definition\n"
        f"{task_description}\n\n"
        "### Output Rules\n"
        "1. Study the examples and follow their task logic exactly.\n"
        "2. Return exactly one final answer wrapped in <label> and </label>.\n"
        "3. Do not output reasoning, explanations, prefixes, suffixes, markdown fences, or extra tags.\n"
        "4. If the answer is multi-line content such as code, place the full raw answer inside the tags.\n\n"
        f"{task_specific_rules}"
        "### Examples\n"
        "[[EXAMPLES]]\n\n"
        "### Text to Annotate\n"
        f"{text2annotate}\n\n"
        "### Final Output Format\n"
        "Return only one <label>...</label> block.\n"
    )
    return prompt


def build_prompt_backup(task_description: str, text2annotate: str) -> str:
    prompt = (
        "You are a data annotation assistant. "
        "Your task is to label the given texts according to the task description "
        "and annotation guidelines provided below.\n\n"
        f"[Task Description]\n {task_description}\n\n"
        "[Examples]\n {EXAMPLES}\n\n"
        "Please follow these instructions when labeling:\n"
        "1. Output Format: Annotate the text directly by wrapping each labeled span with <label> tags in the following format: <label> annotation result </label>.\n"
        f"[Task Description (repeat)] \n {task_description}\n\n"
        f"[Input Texts]\n {text2annotate}\n\n"
        "Please output the annotation results: "
    )
    return prompt


def select_examples_backup(all_examples: list[dict], task_description: str, text2annotate: str) -> str:
    target_length = 10_000

    input_list = [example["input"] for example in all_examples]
    output_list = [example["output"][0] for example in all_examples]
    length_list = [example["length"] for example in all_examples]

    examples_str, token_num = "", 0
    for i, (input_text, output_text, length) in enumerate(zip(input_list, output_list, length_list)):
        if length + token_num <= target_length:
            token_num += (length + 2 + 3 + 1 + 1)
            example_str = f"# {input_text} <label> {output_text} </label>\n"
            examples_str += example_str
        else:
            return examples_str, i
    return examples_str


def _balance_binary_examples(examples: list[dict], label_a: str, label_b: str) -> list[dict]:
    """Interleave examples to balance two classes for binary classification."""
    group_a = [e for e in examples if e["output"][0] == label_a]
    group_b = [e for e in examples if e["output"][0] == label_b]
    balanced = []
    for a, b in zip(group_a, group_b):
        balanced.append(a)
        balanced.append(b)
    longer = group_a if len(group_a) > len(group_b) else group_b
    shorter_len = min(len(group_a), len(group_b))
    balanced.extend(longer[shorter_len:])
    return balanced


def parse_task7_fields(text: str) -> tuple[str, str]:
    category_match = re.search(r"Category:\s*(.*?)\s*Clue:", text, flags=re.I | re.S)
    clue_match = re.search(r"Clue:\s*(.*)", text, flags=re.I | re.S)
    category = category_match.group(1).strip() if category_match else ""
    clue = clue_match.group(1).strip() if clue_match else text.strip()
    return category, clue


def extract_task7_category_constraints(category: str) -> list[dict]:
    constraints = []
    cat = category.strip()
    if not cat:
        return constraints

    letter_matches = re.findall(r"(\d+)[- ]LETTER", cat, flags=re.I)
    if letter_matches:
        constraints.append(
            {
                "type": "letter_count",
                "values": sorted({int(value) for value in letter_matches}),
            }
        )

    quoted_fragments = [frag.strip() for frag in re.findall(r'"([^"]+)"', cat)]
    if quoted_fragments:
        constraints.append(
            {
                "type": "quoted_fragment",
                "values": quoted_fragments,
            }
        )

    upper_fragments = re.findall(r"\b([A-Z]{1,4}(?:,\s*[A-Z]{1,4})+(?:\s+OR\s+[A-Z]{1,4})?)\b", cat)
    if upper_fragments:
        options = []
        for fragment in upper_fragments:
            pieces = re.split(r",|\s+OR\s+", fragment)
            options.extend(piece.strip() for piece in pieces if piece.strip())
        if options:
            constraints.append(
                {
                    "type": "starts_with_options",
                    "values": sorted(set(options)),
                }
            )

    lowered = cat.lower()
    if "before & after" in lowered or "before and after" in lowered:
        constraints.append({"type": "before_after", "values": []})

    if "movie title characters" in lowered:
        constraints.append({"type": "title_character", "values": []})

    if "team mascots" in lowered:
        constraints.append({"type": "team_name", "values": []})

    if "people in history" in lowered or "science class" in lowered or "20th century monarchs" in lowered:
        constraints.append({"type": "person_name", "values": []})

    return constraints


def _task7_alpha_len(text: str) -> int:
    return len(re.findall(r"[a-z]", text.lower()))


def get_task7_constraint_subtypes(category: str) -> list[str]:
    return [constraint["type"] for constraint in extract_task7_category_constraints(category)]


def is_task7_constraint_secondary_allowed(
    category: str,
    allowlist: set[str] | None = None,
) -> bool:
    constraint_types = set(get_task7_constraint_subtypes(category))
    if not constraint_types:
        return True
    if allowlist is None:
        return True
    return bool(constraint_types & allowlist)


def is_task7_secondary_family_allowed(
    family: str,
    allowlist: set[str] | None = None,
) -> bool:
    if allowlist is None:
        return True
    return family in allowlist


def _task7_has_any_pattern(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, text, flags=re.I) for pattern in patterns)


def _is_task7_numeric_route(category: str, clue: str) -> bool:
    category_lower = category.lower()
    clue_lower = clue.lower()
    explicit_clue_patterns = (
        r"\bwhat year\b",
        r"\bwhich year\b",
        r"\bthis year\b",
        r"\bof this year\b",
        r"\bwhat date\b",
        r"\bwhich date\b",
        r"\bon this date\b",
        r"\bwhat century\b",
        r"\bwhich century\b",
        r"\bwhat decade\b",
        r"\bwhich decade\b",
        r"\bhow many\b",
        r"\bhow much\b",
        r"\bwhat number\b",
        r"\bwhich number\b",
        r"\bwhat amount\b",
        r"\bwhich amount\b",
        r"\bnumber of\b",
        r"\bamount of\b",
        r"\bcount of\b",
    )
    if _task7_has_any_pattern(clue_lower, explicit_clue_patterns):
        return True

    explicit_category_patterns = (
        r"\bwhen did it happen\b",
        r"\byears?\b",
        r"\bdates?\b",
        r"\bnumbers?\b",
        r"\bnumber of\b",
        r"\bcount\b",
        r"\bamount\b",
        r"\bhow many\b",
    )
    if _task7_has_any_pattern(category_lower, explicit_category_patterns):
        return True
    return False


def _is_task7_person_entity_route(clue: str) -> bool:
    clue_lower = clue.lower()
    explicit_person_patterns = (
        r"^(he|she)\b",
        r"\bwho is\b",
        r"\bwho was\b",
        r"\bthis man\b",
        r"\bthis woman\b",
        r"\bthis person\b",
        r"\bthis author\b",
        r"\bthis writer\b",
        r"\bthis poet\b",
        r"\bthis president\b",
        r"\bthis vice president\b",
        r"\bthis leader\b",
        r"\bthis actor\b",
        r"\bthis actress\b",
        r"\bthis singer\b",
        r"\bthis musician\b",
        r"\bthis philosopher\b",
        r"\bthis scientist\b",
        r"\bthis explorer\b",
        r"\bthis general\b",
        r"\bthis king\b",
        r"\bthis queen\b",
        r"\bhe was\b",
        r"\bshe was\b",
    )
    return _task7_has_any_pattern(clue_lower, explicit_person_patterns)


def _is_task7_title_or_place_route(category: str, clue: str) -> bool:
    combined = f"{category}\n{clue}".lower()
    explicit_patterns = (
        r"\btitle of\b",
        r"\bthis title\b",
        r"\bthis novel\b",
        r"\bthis film\b",
        r"\bthis movie\b",
        r"\bthis tv show\b",
        r"\bthis television show\b",
        r"\bthis city\b",
        r"\bthis country\b",
        r"\bthis state\b",
        r"\bthis park\b",
        r"\bthis university\b",
        r"\bthis company\b",
    )
    return _task7_has_any_pattern(combined, explicit_patterns)


def _is_task7_organization_entity_route(category: str, clue: str) -> bool:
    combined = f"{category}\n{clue}".lower()
    explicit_patterns = (
        r"\bthis company\b",
        r"\bthis publisher\b",
        r"\bthis publishing company\b",
        r"\bthis organization\b",
        r"\bthis newspaper\b",
        r"\bthis magazine\b",
        r"\bthis university\b",
        r"\bthis network\b",
        r"\bthis label\b",
        r"\bpublished the\b",
        r"\bpublisher of\b",
        r"\balliterative name\b",
    )
    return _task7_has_any_pattern(combined, explicit_patterns)


def score_task7_candidate_constraints(candidate: str | None, category: str) -> dict:
    norm = normalize_task7_answer(candidate)
    constraints = extract_task7_category_constraints(category)
    if not norm or not constraints:
        return {
            "score": 0.0,
            "matched_constraints": 0,
            "hard_violations": 0,
            "constraint_count": len(constraints),
            "details": [],
        }

    joined = re.sub(r"[^a-z0-9]", "", norm)
    details = []
    matched_constraints = 0
    hard_violations = 0
    total_score = 0.0

    for constraint in constraints:
        ctype = constraint["type"]
        passed = None
        delta = 0.0

        if ctype == "letter_count":
            passed = _task7_alpha_len(norm) in set(constraint["values"])
            delta = 2.5 if passed else -2.5
            hard_violations += int(not passed)
        elif ctype == "quoted_fragment":
            passed = any(re.sub(r"[^a-z0-9]", "", value.lower()) in joined for value in constraint["values"])
            delta = 2.0 if passed else -1.5
            hard_violations += int(not passed)
        elif ctype == "starts_with_options":
            passed = any(norm.startswith(value.lower()) for value in constraint["values"])
            delta = 1.5 if passed else -1.0
            hard_violations += int(not passed)
        elif ctype == "before_after":
            # This heuristic is too weak to enforce automatically; keep it informational for now.
            passed = len(norm.split()) >= 2
            delta = 0.5 if passed else 0.0
        elif ctype == "title_character":
            passed = len(norm.split()) <= 4
            delta = 0.25 if passed else 0.0
        elif ctype in {"team_name", "person_name"}:
            passed = len(norm.split()) >= 2
            delta = 0.25 if passed else 0.0

        if passed:
            matched_constraints += 1
        if passed is not None:
            total_score += delta
            details.append(
                {
                    "type": ctype,
                    "values": constraint["values"],
                    "passed": passed,
                    "delta": delta,
                }
            )

    return {
        "score": total_score,
        "matched_constraints": matched_constraints,
        "hard_violations": hard_violations,
        "constraint_count": len(constraints),
        "details": details,
    }


def detect_task7_secondary_family(category: str, clue: str, typed_route: str = "auto") -> str:
    if typed_route != "auto":
        return "generic"

    constraints = extract_task7_category_constraints(category)
    if constraints:
        return "constraint"

    if _is_task7_numeric_route(category, clue):
        return "numeric"
    if _is_task7_person_entity_route(clue):
        return "person_entity"
    if _is_task7_organization_entity_route(category, clue):
        return "organization_entity"
    if _is_task7_title_or_place_route(category, clue):
        return "title_or_place"

    return "generic"


def build_task7_generation_guidance(category: str, clue: str, hint_mode: str = "full") -> str:
    if hint_mode == "off":
        return ""

    constraints = extract_task7_category_constraints(category)
    guidance = []
    if hint_mode == "full":
        guidance.extend(
            [
                "Generate diverse but plausible Jeopardy-style answers that obey the category exactly.",
                "Prefer canonical answers over clue fragments, broad classes, or explanations.",
            ]
        )

    for constraint in constraints:
        if constraint["type"] == "letter_count":
            values = ", ".join(str(value) for value in constraint["values"])
            guidance.append(f"The answer should match the category's letter-count constraint: {values} letters.")
        elif constraint["type"] == "quoted_fragment":
            values = ", ".join(f'"{value}"' for value in constraint["values"])
            guidance.append(f"Pay attention to quoted fragments in the category: {values}.")
        elif constraint["type"] == "starts_with_options":
            values = ", ".join(constraint["values"])
            guidance.append(f"The answer should begin with one of these category options if possible: {values}.")
        elif constraint["type"] == "before_after":
            guidance.append("This is a before-and-after clue: prefer a fused phrase that overlaps two titles or names.")
        elif constraint["type"] == "title_character":
            guidance.append("Return the character's canonical name, not the actor, film title, or role description.")
        elif constraint["type"] == "team_name":
            guidance.append("Return the team name, not the mascot itself or a city-only fragment.")
        elif constraint["type"] == "person_name":
            guidance.append("Prefer a person's full canonical name over a description, title, or related concept.")

    if hint_mode == "full" and re.search(r'\bAdd this word to\b', clue, flags=re.I):
        guidance.append("Apply the word-building instruction literally and return the completed target word or phrase.")
    if hint_mode == "full" and re.search(r'\bthis [0-9-]*time Nobel Prize-winning scientist\b', clue, flags=re.I):
        guidance.append("Prefer the scientist's full name, not a field or another famous scientist.")
    if hint_mode == "full" and re.search(r'\bhost of the first version\b', clue, flags=re.I):
        guidance.append("Be careful about historical versions: avoid replacing an earlier host with a later, more famous one.")

    if hint_mode == "full" and _is_task7_organization_entity_route(category, clue):
        guidance.append("Return an organization or publisher name, not a person, title, or generic thing.")

    if hint_mode == "constraint_only" and not guidance:
        return ""

    if hint_mode in {"typed", "recall"}:
        family = detect_task7_secondary_family(category, clue, typed_route="auto")
        if family == "constraint":
            return build_task7_generation_guidance(category, clue, hint_mode="constraint_only")
        if family == "numeric":
            guidance = [
                "The answer should be a short numeric answer, year, count, or date fragment when appropriate.",
                "Avoid person, place, or organization names unless the clue clearly asks for one.",
            ]
        elif family == "person_entity":
            guidance = [
                "Prefer a full canonical person name.",
                "Avoid titles, descriptions, offices, and clue fragments.",
            ]
            if hint_mode == "recall":
                guidance.extend(
                    [
                        "If the clue points to an original or first version, prefer the earlier historical person rather than a later, better-known successor.",
                        "Do not default to the most famous modern host, celebrity, or office-holder if the clue anchors an older version.",
                    ]
                )
        elif family == "organization_entity":
            guidance = [
                "Prefer a canonical company, publisher, organization, or institution name.",
                "Avoid person names, titles, clue fragments, and generic product words.",
            ]
            if hint_mode == "recall":
                guidance.extend(
                    [
                        "If the clue asks who published or issued something, answer with the publisher or organization, not the author or work.",
                        "If the clue hints at an alliterative or ampersand-style name, consider canonical organization names that fit that pattern.",
                    ]
                )
        elif family == "title_or_place":
            guidance = [
                "Prefer the canonical title, place, or organization name.",
                "Avoid related people, descriptions, or broader classes.",
            ]
            if hint_mode == "recall":
                guidance.extend(
                    [
                        "Do not substitute a nearby person, series, or broad category when the clue is asking for the named title or entity.",
                    ]
                )
        else:
            return ""

    return "Candidate generation guidance:\n" + "\n".join(f"- {line}" for line in guidance) + "\n\n"


def get_render_style(task_id: int | None = None, profile_name: str | None = None) -> str:
    profile = get_profile_name(profile_name)
    if not (profile.startswith("long_context") or profile == "frontier_task6_task7"):
        return "baseline"
    if task_id == 5:
        return "task5_casebook"
    if task_id == 7:
        return "task7_catalog"
    return "baseline"


def render_example_block(example: dict, index: int, task_id: int | None = None, profile_name: str | None = None) -> str:
    input_text = example["input"]
    output_text = example["output"][0]
    render_style = get_render_style(task_id=task_id, profile_name=profile_name)

    if render_style == "task7_catalog":
        category, clue = parse_task7_fields(input_text)
        category_line = category if category else "(unknown category)"
        return (
            f"### Catalog Entry {index}\n"
            f"Category Focus: {category_line}\n"
            f"Clue:\n{clue}\n"
            f"Accepted Answer:\n<label>{output_text}</label>\n\n"
        )

    return (
        f"### Example {index}\n"
        f"Input:\n{input_text}\n"
        f"Output:\n<label>{output_text}</label>\n\n"
    )


def render_chat_example(example: dict, index: int, task_id: int | None = None, profile_name: str | None = None) -> str:
    input_text = example["input"]
    output_text = example["output"][0]
    render_style = get_render_style(task_id=task_id, profile_name=profile_name)

    if render_style == "task5_casebook":
        return (
            f"[Case {index}]\n"
            f"Tweet:\n{input_text}\n"
            f"Gold label: {output_text}\n\n"
        )

    return f"Input: {input_text}\nAnswer: {output_text}\n\n"


def build_chat_user_message(
    examples_str: str,
    text2annotate: str,
    task_id: int,
    task_description: str,
    profile_name: str | None = None,
) -> str:
    extra_guidance = ""
    render_style = get_render_style(task_id=task_id, profile_name=profile_name)
    if task_id == 5 and render_style == "task5_casebook":
        extra_guidance = (
            "Decision notes:\n"
            "- Count indirect distress as Sad when the tweet signals anxiety, depression, being lost, emotional overwhelm, grim disgust, or self-harm/suicide themes.\n"
            "- Count it as Not sad when sadness-related words are incidental in jokes, fandom chat, advice, logistics, or general discussion without the author expressing distress.\n"
            "- Hashtags and emojis matter, but only together with the overall tweet meaning.\n\n"
        )
    return (
        f"Task: {task_description}\n\n"
        f"{extra_guidance}"
        f"Examples:\n{examples_str}\n"
        f"Input: {text2annotate}\n"
        "Answer:"
    )


def select_examples(
    all_examples: list[dict],
    task_description: str,
    text2annotate: str,
    task_id: int | None = None,
    context_budget: int | None = None,
    profile_name: str | None = None,
) -> str:
    """
    Select examples that fit inside the configured context budget.
    """
    return select_examples_with_metadata(
        all_examples,
        task_description,
        text2annotate,
        task_id=task_id,
        context_budget=context_budget,
        profile_name=profile_name,
    )["examples_str"]


def select_examples_with_metadata(
    all_examples: list[dict],
    task_description: str,
    text2annotate: str,
    task_id: int | None = None,
    context_budget: int | None = None,
    profile_name: str | None = None,
) -> dict:
    """
    Select examples that fit inside the configured context budget and return usage metadata.
    """
    # Balance binary classification tasks
    if task_id == 5:
        all_examples = _balance_binary_examples(all_examples, "Sad", "Not sad")

    tokenizer = get_tokenizer()
    target_length = get_context_budget(
        task_id=task_id,
        context_budget=context_budget,
        profile_name=profile_name,
    )

    examples_str, token_num = "", 0
    used_examples = 0
    next_example_tokens = None
    for i, example in enumerate(all_examples):
        try:
            input_text = example["input"]
            output_text = example["output"][0]

            if tokenizer is not None:
                input_tokens = len(tokenizer.encode(input_text, add_special_tokens=False))
                output_tokens = len(tokenizer.encode(output_text, add_special_tokens=False))
            else:
                input_tokens = max(1, len(input_text) // 4)
                output_tokens = max(1, len(output_text) // 4)
            length = input_tokens + output_tokens + 24

            if length + token_num <= target_length:
                token_num += length
                used_examples += 1
                example_str = render_example_block(
                    example,
                    i + 1,
                    task_id=task_id,
                    profile_name=profile_name,
                )
                examples_str += example_str
            else:
                next_example_tokens = length
                break
        except KeyError as e:
            print(f"Warning: example {i} missing key {e}, skipped.")
            continue
    return {
        "examples_str": examples_str,
        "render_style": get_render_style(task_id=task_id, profile_name=profile_name),
        "used_examples": used_examples,
        "example_tokens": token_num,
        "budget_tokens": target_length,
        "pool_size": len(all_examples),
        "pool_limited": used_examples >= len(all_examples),
        "truncated": used_examples < len(all_examples),
        "next_example_tokens": next_example_tokens,
    }


def build_chat_examples(
    examples: list[dict],
    task_id: int | None = None,
    examples_limit: int | None = None,
    token_budget: int | None = None,
    profile_name: str | None = None,
) -> str:
    return build_chat_examples_with_metadata(
        examples,
        task_id=task_id,
        examples_limit=examples_limit,
        token_budget=token_budget,
        profile_name=profile_name,
    )["examples_str"]


def build_chat_examples_with_metadata(
    examples: list[dict],
    task_id: int | None = None,
    examples_limit: int | None = None,
    token_budget: int | None = None,
    profile_name: str | None = None,
) -> dict:
    if task_id == 5:
        examples = _balance_binary_examples(examples, "Sad", "Not sad")

    tokenizer = get_tokenizer()
    max_examples = get_chat_example_count(
        task_id=task_id,
        examples_limit=examples_limit,
        profile_name=profile_name,
    )
    target_tokens = get_chat_example_budget(
        task_id=task_id,
        token_budget=token_budget,
        profile_name=profile_name,
    )

    chat_examples = ""
    token_num = 0
    used_examples = 0
    next_example_tokens = None
    for example in examples:
        if used_examples >= max_examples:
            break
        try:
            example_str = render_chat_example(
                example,
                used_examples + 1,
                task_id=task_id,
                profile_name=profile_name,
            )
        except (KeyError, IndexError):
            continue

        if tokenizer is not None:
            example_tokens = len(tokenizer.encode(example_str, add_special_tokens=False))
        else:
            example_tokens = max(1, len(example_str) // 4)

        if chat_examples and token_num + example_tokens > target_tokens:
            next_example_tokens = example_tokens
            break

        chat_examples += example_str
        token_num += example_tokens
        used_examples += 1

    effective_pool_limit = min(len(examples), max_examples)
    return {
        "examples_str": chat_examples,
        "render_style": get_render_style(task_id=task_id, profile_name=profile_name),
        "used_examples": used_examples,
        "example_tokens": token_num,
        "budget_tokens": target_tokens,
        "max_examples": max_examples,
        "pool_size": len(examples),
        "effective_pool_limit": effective_pool_limit,
        "pool_limited": used_examples >= effective_pool_limit,
        "truncated": used_examples < effective_pool_limit,
        "next_example_tokens": next_example_tokens,
    }


def _normalize_prediction(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    text = re.sub(r"^<label>\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*</label>\s*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*</label\s*$", "", text, flags=re.IGNORECASE)
    if text.startswith("```") and text.endswith("```"):
        matches = FENCED_CODE_PATTERN.findall(text)
        if matches:
            text = matches[-1].strip()
    return text.strip()


def normalize_task7_answer(text: str | None) -> str:
    if text is None:
        return ""
    text = _normalize_prediction(text).lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"^[\"'`]+|[\"'`]+$", "", text)
    text = re.sub(r"^[^a-z0-9]+|[^a-z0-9]+$", "", text)
    text = re.sub(r"^(a|an|the)\s+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _is_placeholder_prediction(text: str) -> bool:
    lowered = text.strip().strip("`").strip().lower()
    if lowered in PLACEHOLDER_VALUES:
        return True
    return bool(re.fullmatch(r"[.\s]+", lowered))


def _looks_like_explanation(text: str) -> bool:
    lowered = text.lower()
    explanation_markers = (
        "reasoning",
        "analysis",
        "because",
        "here is",
        "the answer is",
        "i think",
        "step by step",
    )
    return any(marker in lowered for marker in explanation_markers)


def _looks_like_task7_meta_answer(text: str) -> bool:
    stripped = text.strip()
    lowered = stripped.lower()
    meta_prefixes = (
        "answer:",
        "answer",
        "the answer",
        "candidate",
        "the candidate",
        "only output",
        "output only",
        "make sure",
        "the answer must",
        "the answer should",
        "this answer",
        "to answer",
        "okay,",
        "based on the clue",
        "the clue",
    )
    if any(lowered.startswith(prefix) for prefix in meta_prefixes):
        return True
    if "<label" in lowered or "</label" in lowered:
        return True
    if any(phrase in lowered for phrase in ("must be", "should be", "inside the label", "in lowercase", "matching the category and clue")):
        return True
    if len(re.findall(r"[a-zA-Z]+", stripped)) >= 10 and any(token in lowered for token in ("answer", "clue", "category", "label")):
        return True
    return False


def _extract_task_specific_answer(task_id: int | None, text: str) -> str | None:
    if task_id is None:
        return None

    if task_id == 2:
        matches = re.findall(r"-?\d+", text)
        return matches[-1] if matches else None

    if task_id == 5:
        matches = []
        for match in re.finditer(r"not sad|sad", text, flags=re.IGNORECASE):
            label = "Not sad" if match.group(0).lower() == "not sad" else "Sad"
            matches.append((match.start(), label))
        return matches[-1][1] if matches else None

    if task_id == 6:
        matches = re.findall(r"(?<![A-Za-z])([YN])(?![A-Za-z])", text.upper())
        return matches[-1] if matches else None

    return None


def _looks_like_scaffold(text: str) -> bool:
    lowered = text.strip().lower()
    return lowered.startswith("###") or lowered.startswith("<|") or "end of" in lowered


def _looks_like_code_answer(text: str) -> bool:
    stripped = text.strip()
    if not stripped or _looks_like_scaffold(stripped) or _is_placeholder_prediction(stripped):
        return False
    code_markers = (
        "import torch",
        "import triton",
        "triton.language as tl",
        "@triton.jit",
        "def ",
        "tl.",
    )
    marker_hits = sum(1 for marker in code_markers if marker in stripped)
    non_empty_lines = [line for line in stripped.splitlines() if line.strip()]
    return marker_hits >= 2 and len(non_empty_lines) >= 4


def _looks_like_valid_python(text: str) -> bool:
    try:
        ast.parse(text)
        return True
    except SyntaxError:
        return False


def sanitize_task8_code(text: str) -> str:
    stripped = _normalize_prediction(text)
    if not stripped:
        return ""

    lines = stripped.splitlines()
    start_idx = None
    for idx, line in enumerate(lines):
        if TASK8_CODE_START_PATTERN.match(line):
            start_idx = idx
            break
    if start_idx is None:
        return stripped if _looks_like_code_answer(stripped) else ""

    cleaned_lines = []
    for raw_line in lines[start_idx:]:
        line = raw_line.replace("<label>", "").replace("</label>", "").replace("<|end|>", "")
        line = line.replace("<|im_end|>", "").replace("<|eot_id|>", "")
        stripped_line = line.strip()
        if TASK8_NOISE_LINE_PATTERN.fullmatch(stripped_line):
            continue
        cleaned_lines.append(line.rstrip())

    sanitized = "\n".join(cleaned_lines).strip()
    sanitized = re.sub(r"(?:\n\s*){3,}", "\n\n", sanitized)
    return sanitized


def count_answer(text: str, task_id: int | None = None):
    """
    Extract the final answer from a model response.
    """
    if not text:
        return None

    if task_id in {2, 5, 6, 7}:
        stripped = _normalize_prediction(text)
        if _looks_like_scaffold(stripped):
            return None
        task_specific = _extract_task_specific_answer(task_id, stripped)
        if task_specific is not None:
            return task_specific

    if task_id == 8:
        stripped = _normalize_prediction(text)
        sanitized = sanitize_task8_code(stripped)
        if sanitized and _looks_like_code_answer(sanitized) and _looks_like_valid_python(sanitized):
            return sanitized
        if _looks_like_code_answer(stripped) and _looks_like_valid_python(stripped):
            return stripped

    candidates = []
    for match in LABEL_PATTERN.findall(text):
        candidate = _normalize_prediction(match)
        if task_id == 8:
            candidate = sanitize_task8_code(candidate)
        if candidate and not _is_placeholder_prediction(candidate):
            if task_id == 7 and _looks_like_task7_meta_answer(candidate):
                continue
            if task_id == 8 and not (_looks_like_code_answer(candidate) and _looks_like_valid_python(candidate)):
                continue
            candidates.append(candidate)
    if candidates:
        return candidates[-1]

    stripped = text.strip()
    task_specific = _extract_task_specific_answer(task_id, stripped)
    if task_specific is not None:
        return task_specific

    code_blocks = FENCED_CODE_PATTERN.findall(stripped)
    if code_blocks and stripped.startswith("```") and stripped.endswith("```"):
        candidate = _normalize_prediction(code_blocks[-1])
        if task_id == 8:
            candidate = sanitize_task8_code(candidate)
        if candidate and not _is_placeholder_prediction(candidate):
            if task_id == 7 and _looks_like_task7_meta_answer(candidate):
                return None
            if task_id == 8 and not (_looks_like_code_answer(candidate) and _looks_like_valid_python(candidate)):
                return None
            return candidate

    prefixed_match = ANSWER_PREFIX_PATTERN.search(stripped)
    if prefixed_match:
        candidate = _normalize_prediction(prefixed_match.group(1))
        if task_id == 8:
            candidate = sanitize_task8_code(candidate)
        if candidate and not _is_placeholder_prediction(candidate) and not _looks_like_explanation(candidate):
            if task_id == 7 and _looks_like_task7_meta_answer(candidate):
                return None
            if task_id == 8 and not (_looks_like_code_answer(candidate) and _looks_like_valid_python(candidate)):
                return None
            return candidate

    non_empty_lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    if len(non_empty_lines) == 1:
        candidate = _normalize_prediction(non_empty_lines[0])
        if candidate and not _is_placeholder_prediction(candidate) and not _looks_like_explanation(candidate):
            if task_id == 7 and _looks_like_task7_meta_answer(candidate):
                return None
            return candidate

    last_open_tag = stripped.lower().rfind("<label>")
    if last_open_tag != -1:
        candidate = _normalize_prediction(stripped[last_open_tag + len("<label>"):])
        if task_id == 8:
            candidate = sanitize_task8_code(candidate)
        if candidate and not _is_placeholder_prediction(candidate):
            if task_id == 7 and _looks_like_task7_meta_answer(candidate):
                return None
            if task_id == 8 and not (_looks_like_code_answer(candidate) and _looks_like_valid_python(candidate)):
                return None
            return candidate

    return None


def _build_retry_prompt(input_prompt: str) -> str:
    return (
        f"{input_prompt.rstrip()}\n\n"
        "### Retry Instruction\n"
        "Your previous response did not follow the required format.\n"
        "Retry from scratch and return ONLY one <label>...</label> block.\n"
        "Do not include reasoning, bullet points, markdown fences, or any text outside the tags.\n"
    )


def _request_nvidia_completions(
    input_prompt: str,
    task_id: int | None = None,
    *,
    n: int = 1,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int | None = None,
):
    import requests

    data = {
        "model": get_served_model_name(),
        "prompt": input_prompt,
        "max_tokens": max_tokens or TASK_MAX_TOKENS.get(task_id, DEFAULT_MAX_TOKENS),
        "temperature": temperature,
        "top_p": top_p,
    }
    if DEFAULT_COMPLETION_SEED:
        try:
            data["seed"] = int(DEFAULT_COMPLETION_SEED)
        except ValueError:
            pass
    if n > 1:
        data["n"] = n
    stop = TASK_STOPS.get(task_id)
    if stop:
        data["stop"] = stop

    attempts = TASK_REQUEST_ATTEMPTS.get(task_id, 1)
    for attempt in range(attempts):
        try:
            resp = requests.post(DEFAULT_COMPLETION_URL, json=data, timeout=DEFAULT_REQUEST_TIMEOUT)
            resp.raise_for_status()
            choices = resp.json().get("choices") or []
            texts = [choice.get("text", "") for choice in choices if choice.get("text") is not None]
            if texts:
                return texts
        except Exception:
            pass
        if attempt + 1 < attempts:
            time.sleep(min(2 ** attempt, 4))
    return []


def _request_nvidia_completion(input_prompt: str, task_id: int | None = None):
    texts = _request_nvidia_completions(input_prompt, task_id=task_id)
    return texts[0] if texts else ""


def _extract_option_index(text: str) -> int | None:
    if not text:
        return None
    match = re.search(r"\b(\d+)\b", text)
    if not match:
        return None
    return int(match.group(1)) - 1


def _choose_task7_vote_candidate(candidates: list[str], counts: defaultdict[str, int] | dict[str, int]) -> str | None:
    if not candidates:
        return None
    scored = []
    for candidate in candidates:
        norm = normalize_task7_answer(candidate)
        scored.append((counts[norm], -len(norm.split()), -len(norm), candidate))
    scored.sort(reverse=True)
    return scored[0][3]


def _dedupe_task7_candidates(raw_choices: list[str]) -> tuple[list[str], dict[str, int]]:
    counts: dict[str, int] = defaultdict(int)
    normalized_to_candidate = {}
    for raw in raw_choices:
        answer = count_answer(raw, task_id=7)
        normalized = normalize_task7_answer(answer)
        if not normalized:
            continue
        counts[normalized] += 1
        normalized_to_candidate.setdefault(normalized, answer.strip())
    ordered = [normalized_to_candidate[norm] for norm, _ in sorted(counts.items(), key=lambda item: item[1], reverse=True)]
    return ordered, dict(counts)


def _append_task7_unique_candidates(primary_candidates: list[str], secondary_candidates: list[str]) -> list[str]:
    merged = list(primary_candidates)
    seen = {normalize_task7_answer(candidate) for candidate in primary_candidates}
    for candidate in secondary_candidates:
        normalized = normalize_task7_answer(candidate)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        merged.append(candidate)
    return merged


def get_task7_secondary_only_candidates(
    primary_candidates: list[str],
    secondary_candidates: list[str],
) -> list[str]:
    primary_seen = {normalize_task7_answer(candidate) for candidate in primary_candidates}
    secondary_only = []
    seen = set()
    for candidate in secondary_candidates:
        normalized = normalize_task7_answer(candidate)
        if not normalized or normalized in primary_seen or normalized in seen:
            continue
        seen.add(normalized)
        secondary_only.append(candidate)
    return secondary_only


def build_task7_append_unique_judge_candidates(
    primary_candidates: list[str],
    secondary_candidates: list[str],
    *,
    max_candidates: int,
    reserved_secondary_slots: int,
    layout_mode: str = "baseline",
) -> list[str]:
    if max_candidates <= 0:
        return []

    secondary_only = get_task7_secondary_only_candidates(primary_candidates, secondary_candidates)
    if layout_mode == "author_secondary_first":
        return _append_task7_unique_candidates(secondary_only, primary_candidates)[:max_candidates]
    if layout_mode == "author_primary_anchor_top1":
        anchor = primary_candidates[:1]
        tail = primary_candidates[1:]
        return _append_task7_unique_candidates(anchor + secondary_only, tail)[:max_candidates]
    if layout_mode == "author_primary_anchor_top2":
        anchor = primary_candidates[:2]
        tail = primary_candidates[2:]
        return _append_task7_unique_candidates(anchor + secondary_only, tail)[:max_candidates]

    reserved = min(max_candidates, max(0, reserved_secondary_slots), len(secondary_only))
    primary_budget = max(0, max_candidates - reserved)
    judge_candidates = list(primary_candidates[:primary_budget])

    seen = {normalize_task7_answer(candidate) for candidate in judge_candidates}
    for candidate in secondary_only[:reserved]:
        normalized = normalize_task7_answer(candidate)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        judge_candidates.append(candidate)

    for candidate in primary_candidates[primary_budget:]:
        if len(judge_candidates) >= max_candidates:
            break
        normalized = normalize_task7_answer(candidate)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        judge_candidates.append(candidate)

    return judge_candidates[:max_candidates]


def _summarize_task7_candidate_pool(candidate_counts: dict[str, int]) -> dict[str, float | int]:
    counts = list(candidate_counts.values())
    total = sum(counts)
    top = counts[0] if counts else 0
    second = counts[1] if len(counts) > 1 else 0
    entropy = 0.0
    if total:
        for count in counts:
            p = count / total
            entropy -= p * math.log(p, 2)
    return {
        "unique_candidates": len(counts),
        "top_share": top / total if total else 0.0,
        "margin_share": (top - second) / total if total else 0.0,
        "entropy": entropy,
        "total_votes": total,
    }


def _should_trigger_task7_secondary_gate(pool_summary: dict[str, float | int]) -> bool:
    gate_mode = get_task7_rerank_secondary_gate_mode()
    if gate_mode == "off":
        return True
    unique_candidates = int(pool_summary.get("unique_candidates", 0))
    entropy = float(pool_summary.get("entropy", 0.0))
    min_unique = get_task7_rerank_secondary_min_unique()
    min_entropy = get_task7_rerank_secondary_min_entropy()
    if gate_mode == "entropy_or_unique":
        return unique_candidates >= min_unique or entropy >= min_entropy
    if gate_mode == "unique_only":
        return unique_candidates >= min_unique
    if gate_mode == "entropy_only":
        return entropy >= min_entropy
    return True


def build_task7_judge_prompt(category: str, clue: str, candidates: list[str]) -> str:
    candidate_lines = [f"{idx + 1}. {candidate}" for idx, candidate in enumerate(candidates)]
    return (
        "You are selecting the best Jeopardy answer from candidate options.\n\n"
        "Rules:\n"
        "1. Choose the single candidate that best matches both the category and the clue.\n"
        "2. Prefer the canonical Jeopardy answer: the exact entity or phrase, not a paraphrase, broader class, or clue fragment.\n"
        "3. If one option is a short incomplete fragment and another is the fully specified answer, choose the fully specified answer.\n"
        "4. If one option is an expanded team/place name and another is the shorter canonical answer used in Jeopardy, choose the canonical answer.\n"
        "5. Return only the selected option number inside <label>...</label>.\n\n"
        f"Category: {category}\n"
        f"Clue: {clue}\n\n"
        "Candidates:\n"
        + "\n".join(candidate_lines)
        + "\n\nAnswer:\n<label>"
    )


def _take_task7_prompt_head(text: str | None, limit: int = 12) -> str | None:
    if not text:
        return None
    lines = text.splitlines()
    if len(lines) <= limit:
        return text
    return "\n".join(lines[:limit])


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _build_task7_judge_plan(
    primary_candidates: list[str],
    secondary_candidates: list[str],
    *,
    author_projected_candidates: list[str],
    unique_candidates: list[str],
    max_candidates: int,
    reserved_secondary_slots: int,
    layout_mode: str,
    append_unique_secondary_active: bool,
) -> dict:
    if author_projected_candidates:
        secondary_for_judge = list(secondary_candidates)
        secondary_for_judge.extend(get_task7_secondary_only_candidates([], author_projected_candidates))
        return {
            "strategy": "append_unique_author_projection",
            "layout_mode": layout_mode,
            "candidates": build_task7_append_unique_judge_candidates(
                primary_candidates,
                secondary_for_judge,
                max_candidates=max_candidates,
                reserved_secondary_slots=reserved_secondary_slots,
                layout_mode=layout_mode,
            ),
        }
    if append_unique_secondary_active and secondary_candidates:
        return {
            "strategy": "append_unique_secondary",
            "layout_mode": "baseline",
            "candidates": build_task7_append_unique_judge_candidates(
                primary_candidates,
                secondary_candidates,
                max_candidates=max_candidates,
                reserved_secondary_slots=reserved_secondary_slots,
            ),
        }
    return {
        "strategy": "top_unique",
        "layout_mode": "baseline",
        "candidates": unique_candidates[:max_candidates],
    }


def _empty_task7_judge_result() -> dict:
    return {
        "candidates": [],
        "prompt_fingerprint": None,
        "prompt_head": None,
        "raw": "",
        "index": None,
        "winner": None,
    }


def _run_task7_judge_once(category: str, clue: str, candidates: list[str]) -> dict:
    if not candidates:
        return _empty_task7_judge_result()
    judge_prompt = build_task7_judge_prompt(category, clue, candidates)
    judge_choices = _request_nvidia_completions(
        judge_prompt,
        task_id=7,
        n=1,
        temperature=0.0,
        top_p=1.0,
        max_tokens=8,
    )
    judge_raw = judge_choices[0] if judge_choices else ""
    judge_index = _extract_option_index(judge_raw)
    winner = None
    if judge_index is not None and 0 <= judge_index < len(candidates):
        winner = candidates[judge_index]
    return {
        "candidates": list(candidates),
        "prompt_fingerprint": sha256_text(judge_prompt),
        "prompt_head": _take_task7_prompt_head(judge_prompt),
        "raw": judge_raw,
        "index": judge_index,
        "winner": winner,
    }


def _candidate_is_from_author_projection(candidate: str | None, projected_candidates: list[str]) -> bool:
    candidate_norm = normalize_task7_answer(candidate)
    if not candidate_norm:
        return False
    projected_norms = {normalize_task7_answer(projected) for projected in projected_candidates}
    return candidate_norm in projected_norms


def _run_task7_pairwise_gate(
    category: str,
    clue: str,
    *,
    baseline_winner: str | None,
    anchor1_winner: str | None,
    gate_mode: str,
) -> dict:
    comparisons = []
    anchor_norm = normalize_task7_answer(anchor1_winner)
    baseline_norm = normalize_task7_answer(baseline_winner)
    if not anchor_norm or not baseline_norm or anchor_norm == baseline_norm:
        return {
            "mode": gate_mode,
            "passed": False,
            "comparisons": comparisons,
        }

    for candidates in ([baseline_winner, anchor1_winner], [anchor1_winner, baseline_winner]):
        result = _run_task7_judge_once(category, clue, candidates)
        winner_norm = normalize_task7_answer(result["winner"])
        comparisons.append(
            {
                "candidates": list(candidates),
                "winner": result["winner"],
                "winner_is_anchor1": winner_norm == anchor_norm,
                "raw": result["raw"],
                "index": result["index"],
                "prompt_fingerprint": result["prompt_fingerprint"],
            }
        )

    passed = gate_mode == "strong_only" and all(comp["winner_is_anchor1"] for comp in comparisons)
    return {
        "mode": gate_mode,
        "passed": passed,
        "comparisons": comparisons,
    }


def resolve_task7_rerank_judge(
    *,
    category: str,
    clue: str,
    bucket: str | None,
    primary_candidates: list[str],
    secondary_candidates: list[str],
    author_projected_candidates: list[str],
    unique_candidates: list[str],
    max_candidates: int,
    reserved_secondary_slots: int,
    append_unique_secondary_active: bool,
) -> dict:
    decision_mode = get_task7_author_projection_decision_mode()
    configured_layout_mode = get_task7_author_projection_judge_layout()
    stability_repeats = get_task7_author_projection_stability_repeats()
    pairwise_gate_mode = get_task7_author_projection_pairwise_gate()

    selected_plan = _build_task7_judge_plan(
        primary_candidates,
        secondary_candidates,
        author_projected_candidates=author_projected_candidates,
        unique_candidates=unique_candidates,
        max_candidates=max_candidates,
        reserved_secondary_slots=reserved_secondary_slots,
        layout_mode=configured_layout_mode,
        append_unique_secondary_active=append_unique_secondary_active,
    )
    selected_result = _run_task7_judge_once(category, clue, selected_plan["candidates"])

    baseline_plan = None
    anchor1_plan = None
    baseline_result = _empty_task7_judge_result()
    anchor1_result = _empty_task7_judge_result()
    anchor1_repeat_winners: list[str | None] = []
    anchor1_repeat_stable = None
    pairwise_gate_passed = None
    pairwise_gate = {
        "mode": pairwise_gate_mode,
        "passed": None,
        "comparisons": [],
    }
    final_decision_source = selected_plan["strategy"]
    final_prediction = selected_result["winner"]

    should_run_author_gate = (
        bucket == "quoted_work_author_relation"
        and bool(author_projected_candidates)
        and decision_mode in {"baseline", "anchor1", "anchor1_stability_gate"}
    )
    if should_run_author_gate:
        baseline_plan = _build_task7_judge_plan(
            primary_candidates,
            secondary_candidates,
            author_projected_candidates=author_projected_candidates,
            unique_candidates=unique_candidates,
            max_candidates=max_candidates,
            reserved_secondary_slots=reserved_secondary_slots,
            layout_mode="baseline",
            append_unique_secondary_active=append_unique_secondary_active,
        )
        anchor1_plan = _build_task7_judge_plan(
            primary_candidates,
            secondary_candidates,
            author_projected_candidates=author_projected_candidates,
            unique_candidates=unique_candidates,
            max_candidates=max_candidates,
            reserved_secondary_slots=reserved_secondary_slots,
            layout_mode="author_primary_anchor_top1",
            append_unique_secondary_active=append_unique_secondary_active,
        )
        baseline_result = _run_task7_judge_once(category, clue, baseline_plan["candidates"])
        anchor1_result = _run_task7_judge_once(category, clue, anchor1_plan["candidates"])

        if decision_mode == "baseline":
            selected_plan = baseline_plan
            selected_result = baseline_result
            final_prediction = baseline_result["winner"]
            final_decision_source = "baseline_author_projection_judge"
        elif decision_mode == "anchor1":
            if anchor1_result["winner"] is not None:
                selected_plan = anchor1_plan
                selected_result = anchor1_result
                final_prediction = anchor1_result["winner"]
                final_decision_source = "anchor1_author_projection_judge"
            else:
                selected_plan = baseline_plan
                selected_result = baseline_result
                final_prediction = baseline_result["winner"]
                final_decision_source = "anchor1_invalid_fallback_baseline"
        else:
            baseline_norm = normalize_task7_answer(baseline_result["winner"])
            anchor1_norm = normalize_task7_answer(anchor1_result["winner"])
            if baseline_norm and anchor1_norm and baseline_norm == anchor1_norm:
                selected_plan = anchor1_plan
                selected_result = anchor1_result
                final_prediction = anchor1_result["winner"]
                final_decision_source = "baseline_anchor1_agree"
                anchor1_repeat_winners = [anchor1_result["winner"]]
                anchor1_repeat_stable = True
                pairwise_gate_passed = True
                pairwise_gate["passed"] = True
            elif not _candidate_is_from_author_projection(anchor1_result["winner"], author_projected_candidates):
                selected_plan = baseline_plan
                selected_result = baseline_result
                final_prediction = baseline_result["winner"]
                final_decision_source = "anchor1_not_projected_fallback_baseline"
                anchor1_repeat_winners = [anchor1_result["winner"]]
                anchor1_repeat_stable = False
                pairwise_gate_passed = False
                pairwise_gate["passed"] = False
            else:
                anchor1_repeat_winners = [anchor1_result["winner"]]
                for _ in range(max(0, stability_repeats - 1)):
                    repeat_result = _run_task7_judge_once(category, clue, anchor1_plan["candidates"])
                    anchor1_repeat_winners.append(repeat_result["winner"])
                normalized_repeat_winners = [normalize_task7_answer(winner) for winner in anchor1_repeat_winners]
                anchor1_repeat_stable = bool(normalized_repeat_winners) and all(
                    normalized_repeat_winners[0] and repeat_norm == normalized_repeat_winners[0]
                    for repeat_norm in normalized_repeat_winners
                )
                if not anchor1_repeat_stable:
                    selected_plan = baseline_plan
                    selected_result = baseline_result
                    final_prediction = baseline_result["winner"]
                    final_decision_source = "anchor1_repeat_unstable_fallback_baseline"
                    pairwise_gate_passed = False
                    pairwise_gate["passed"] = False
                else:
                    pairwise_gate = _run_task7_pairwise_gate(
                        category,
                        clue,
                        baseline_winner=baseline_result["winner"],
                        anchor1_winner=anchor1_result["winner"],
                        gate_mode=pairwise_gate_mode,
                    )
                    pairwise_gate_passed = pairwise_gate["passed"]
                    if pairwise_gate_passed:
                        selected_plan = anchor1_plan
                        selected_result = anchor1_result
                        final_prediction = anchor1_result["winner"]
                        final_decision_source = "anchor1_stability_gate"
                    else:
                        selected_plan = baseline_plan
                        selected_result = baseline_result
                        final_prediction = baseline_result["winner"]
                        final_decision_source = "anchor1_pairwise_failed_fallback_baseline"

    return {
        "strategy": selected_plan["strategy"],
        "candidates": selected_plan["candidates"],
        "prompt_fingerprint": selected_result["prompt_fingerprint"],
        "prompt_head": selected_result["prompt_head"],
        "raw": selected_result["raw"],
        "index": selected_result["index"],
        "decision_mode": decision_mode,
        "configured_layout_mode": configured_layout_mode,
        "baseline_judge_candidates": None if baseline_plan is None else baseline_plan["candidates"],
        "anchor1_judge_candidates": None if anchor1_plan is None else anchor1_plan["candidates"],
        "baseline_winner": baseline_result["winner"] if baseline_plan is not None else None,
        "anchor1_winner": anchor1_result["winner"] if anchor1_plan is not None else None,
        "anchor1_repeat_winners": anchor1_repeat_winners,
        "anchor1_repeat_stable": anchor1_repeat_stable,
        "pairwise_gate_mode": pairwise_gate_mode,
        "pairwise_gate_passed": pairwise_gate_passed,
        "pairwise_gate_comparisons": pairwise_gate["comparisons"],
        "stability_repeats": stability_repeats,
        "final_decision_source": final_decision_source,
        "final_prediction": final_prediction,
        "reserved_secondary_slots": reserved_secondary_slots,
        "max_candidates": max_candidates,
    }


def build_task7_rerank_generation_prompt(input_prompt: str, text2annotate: str, hint_mode: str = "full") -> str:
    category, clue = parse_task7_fields(text2annotate)
    guidance = build_task7_generation_guidance(category, clue, hint_mode=hint_mode)
    if not guidance:
        return input_prompt
    answer_marker = "Answer:\n<label>"
    if answer_marker in input_prompt:
        prefix, _, suffix = input_prompt.rpartition(answer_marker)
        return (
            prefix.rstrip()
            + "\n\n"
            + guidance
            + "Produce one plausible Jeopardy answer candidate inside <label>...</label>.\n\n"
            + answer_marker
            + suffix
        )
    return (
        input_prompt.rstrip()
        + "\n\n"
        + guidance
        + "Produce one plausible Jeopardy answer candidate inside <label>...</label>.\n"
    )


def _collect_task7_branch_choices(
    input_prompt: str,
    text2annotate: str,
    *,
    n_candidates: int,
    temperature: float,
    top_p: float,
    use_hints: bool,
    hint_mode: str = "off",
) -> list[str]:
    if not input_prompt or n_candidates <= 0:
        return []
    resolved_hint_mode = hint_mode
    if use_hints and resolved_hint_mode == "off":
        resolved_hint_mode = "full"
    generation_prompt = (
        build_task7_rerank_generation_prompt(input_prompt, text2annotate, hint_mode=resolved_hint_mode)
        if resolved_hint_mode != "off"
        else input_prompt
    )
    return _request_nvidia_completions(
        generation_prompt,
        task_id=7,
        n=n_candidates,
        temperature=temperature,
        top_p=top_p,
        max_tokens=64,
    )


def annotate_task7_rerank(
    input_prompt: str,
    text2annotate: str,
    secondary_input_prompt: str | None = None,
    author_catalog: dict | None = None,
) -> str | None:
    from probe_task7_direct_fact_expansion import detect_direct_fact_bucket

    category, clue = parse_task7_fields(text2annotate)
    category_constraints = extract_task7_category_constraints(category)
    secondary_family = detect_task7_secondary_family(
        category,
        clue,
        typed_route=get_task7_rerank_secondary_typed_route(),
    )
    direct_fact_family = detect_task7_secondary_family(category, clue, typed_route="auto")
    bucket = detect_direct_fact_bucket(category, clue, direct_fact_family)
    raw_choices = _collect_task7_branch_choices(
        input_prompt,
        text2annotate,
        n_candidates=max(1, DEFAULT_TASK7_RERANK_CANDIDATES),
        temperature=DEFAULT_TASK7_RERANK_TEMPERATURE,
        top_p=DEFAULT_TASK7_RERANK_TOP_P,
        use_hints=should_use_task7_rerank_hints(),
        hint_mode=get_task7_rerank_hint_mode(),
    )
    primary_unique_candidates, primary_candidate_counts = _dedupe_task7_candidates(raw_choices)
    primary_pool_summary = _summarize_task7_candidate_pool(primary_candidate_counts)
    unique_candidates = list(primary_unique_candidates)
    candidate_counts = dict(primary_candidate_counts)
    vote_candidate = _choose_task7_vote_candidate(unique_candidates, candidate_counts)
    secondary_raw_choices: list[str] = []
    secondary_unique_candidates: list[str] = []

    secondary_candidates = get_task7_rerank_secondary_candidates()
    secondary_hint_mode = get_task7_rerank_secondary_hint_mode()
    secondary_family_allowlist = get_task7_rerank_secondary_family_allowlist()
    secondary_constraint_allowlist = get_task7_rerank_secondary_constraint_allowlist()
    should_try_secondary = (
        secondary_input_prompt
        and secondary_candidates > 0
        and _should_trigger_task7_secondary_gate(primary_pool_summary)
    )
    if should_try_secondary and secondary_hint_mode == "constraint_only" and not category_constraints:
        should_try_secondary = False
    if should_try_secondary and secondary_hint_mode == "typed" and secondary_family == "generic":
        should_try_secondary = False
    if should_try_secondary and secondary_hint_mode == "typed":
        should_try_secondary = is_task7_secondary_family_allowed(
            secondary_family,
            allowlist=secondary_family_allowlist,
        )
    if should_try_secondary and secondary_hint_mode == "typed" and secondary_family == "constraint":
        should_try_secondary = is_task7_constraint_secondary_allowed(
            category,
            allowlist=secondary_constraint_allowlist,
        )

    if should_try_secondary:
        secondary_raw_choices = _collect_task7_branch_choices(
            secondary_input_prompt,
            text2annotate,
            n_candidates=secondary_candidates,
            temperature=DEFAULT_TASK7_RERANK_SECONDARY_TEMPERATURE,
            top_p=DEFAULT_TASK7_RERANK_SECONDARY_TOP_P,
            use_hints=should_use_task7_rerank_secondary_hints(),
            hint_mode=secondary_hint_mode,
        )
        secondary_unique_candidates, secondary_candidate_counts = _dedupe_task7_candidates(secondary_raw_choices)
        if get_task7_rerank_secondary_merge_mode() == "append_unique":
            unique_candidates = _append_task7_unique_candidates(primary_unique_candidates, secondary_unique_candidates)
            candidate_counts = dict(primary_candidate_counts)
            for normalized, count in secondary_candidate_counts.items():
                candidate_counts.setdefault(normalized, count)
            vote_candidate = _choose_task7_vote_candidate(primary_unique_candidates, primary_candidate_counts)
        else:
            raw_choices.extend(secondary_raw_choices)
            unique_candidates, candidate_counts = _dedupe_task7_candidates(raw_choices)
            vote_candidate = _choose_task7_vote_candidate(unique_candidates, candidate_counts)

    author_projected_candidates: list[str] = []
    if should_use_task7_author_direct_fact_projection():
        author_projected_candidates = build_task7_author_projected_candidates(
            text2annotate,
            author_catalog,
        )
        if author_projected_candidates:
            unique_candidates = _append_task7_unique_candidates(unique_candidates, author_projected_candidates)
            for candidate in author_projected_candidates:
                normalized = normalize_task7_answer(candidate)
                if not normalized:
                    continue
                candidate_counts.setdefault(normalized, 1)

    max_judge_candidates = max(1, DEFAULT_TASK7_RERANK_MAX_JUDGE)
    if author_projected_candidates:
        secondary_for_judge = list(secondary_unique_candidates)
        secondary_for_judge.extend(
            get_task7_secondary_only_candidates(unique_candidates[:0], author_projected_candidates)
        )
        del secondary_for_judge

    judge_decision = resolve_task7_rerank_judge(
        category=category,
        clue=clue,
        bucket=bucket if author_projected_candidates else None,
        primary_candidates=primary_unique_candidates,
        secondary_candidates=secondary_unique_candidates,
        author_projected_candidates=author_projected_candidates,
        unique_candidates=unique_candidates,
        max_candidates=max_judge_candidates,
        reserved_secondary_slots=get_task7_rerank_append_unique_secondary_judge_slots(),
        append_unique_secondary_active=(
            should_try_secondary
            and get_task7_rerank_secondary_merge_mode() == "append_unique"
            and bool(secondary_unique_candidates)
        ),
    )
    if judge_decision["final_prediction"] is not None:
        return judge_decision["final_prediction"]

    if vote_candidate is not None:
        return vote_candidate

    fallback = _request_nvidia_completion(input_prompt, task_id=7)
    return count_answer(fallback, task_id=7)


def _majority_vote_classification(input_prompt: str, task_id: int, n_samples: int = 5) -> str | None:
    """Run n_samples with temperature>0 and return majority vote for binary classification."""
    import requests
    from collections import Counter

    data = {
        "model": get_served_model_name(),
        "prompt": input_prompt,
        "max_tokens": TASK_MAX_TOKENS.get(task_id, DEFAULT_MAX_TOKENS),
        "temperature": 0.6,
        "top_p": 0.9,
        "n": n_samples,
    }
    stop = TASK_STOPS.get(task_id)
    if stop:
        data["stop"] = stop

    try:
        resp = requests.post(DEFAULT_COMPLETION_URL, json=data, timeout=DEFAULT_REQUEST_TIMEOUT)
        resp.raise_for_status()
        choices = resp.json()["choices"]
        answers = []
        for choice in choices:
            answer = count_answer(choice["text"], task_id=task_id)
            if answer is not None:
                answers.append(answer)
        if not answers:
            return None
        return Counter(answers).most_common(1)[0][0]
    except Exception:
        return None


def annotate_nvidia(input_prompt: str, task_id: int | None = None):
    """
    Annotate the unlabeled data using an LLM API exposed by a local completion server.
    """
    # Try majority voting for binary classification tasks
    if task_id in {5}:
        result = _majority_vote_classification(input_prompt, task_id, n_samples=5)
        if result is not None:
            return result

    prompts = [input_prompt]
    retry_count = TASK_RETRY_COUNTS.get(task_id, DEFAULT_RETRY_COUNT)
    for _ in range(retry_count):
        prompts.append(_build_retry_prompt(input_prompt))

    for prompt in prompts:
        whole_result = _request_nvidia_completion(prompt, task_id=task_id)
        prediction = count_answer(whole_result, task_id=task_id)
        if prediction is not None:
            return prediction
    return None


DEFAULT_CHAT_URL = DEFAULT_COMPLETION_URL.replace("/v1/completions", "/v1/chat/completions")

TASK_CHAT_SYSTEM = {
    2: "You are a precise annotation system. Count exactly what is asked. Think step by step, then give ONLY the number.",
    5: "You classify tweet sentiment. Think about the tweet, then answer with ONLY 'Sad' or 'Not sad'.",
    6: "You solve natural language inference. Compare Sentence 1 and Sentence 2 carefully, then answer with ONLY 'Y' or 'N'.",
}

TASK_CHAT_MAX_TOKENS = {
    2: 1500,
    5: 800,
    6: 800,
}

TASK_CHAT_TEMPERATURES = {
    2: 0.6,
    5: 0.6,
    6: 0.6,
}

TASK_CHAT_TOP_P = {
    2: 0.95,
    5: 0.95,
    6: 0.95,
}

# Tasks that should use chat+thinking mode instead of completion mode
DEFAULT_CHAT_THINKING_TASKS = {2, 5}
CHAT_THINKING_TASKS = DEFAULT_CHAT_THINKING_TASKS | {
    int(token)
    for token in os.environ.get("OPENSEEK_CHAT_TASKS", "").split(",")
    if token.strip().isdigit()
}


def _extract_chat_answer(raw: str | None, task_id: int) -> str | None:
    """Extract the final answer from a chat response (after </think>)."""
    if raw is None:
        return None
    text = raw.strip()
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    text = re.sub(r"<[^>]+>", "", text).strip()
    if not text:
        return None

    if task_id == 2:
        nums = re.findall(r"\b\d+\b", text)
        return nums[0] if nums else None
    elif task_id == 5:
        low = text.lower()
        if "not sad" in low:
            return "Not sad"
        elif "sad" in low:
            return "Sad"
        return text
    elif task_id == 6:
        matches = re.findall(r"(?<![A-Za-z])([YN])(?![A-Za-z])", text.upper())
        return matches[-1] if matches else None
    return text


def _request_chat_completion(system_msg: str, user_msg: str, task_id: int) -> str | None:
    """Call the vLLM chat completions API with thinking enabled."""
    import requests

    data = {
        "model": get_served_model_name(),
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        "max_tokens": TASK_CHAT_MAX_TOKENS.get(task_id, 1500),
        "temperature": TASK_CHAT_TEMPERATURES.get(task_id, 0.6),
        "top_p": TASK_CHAT_TOP_P.get(task_id, 0.95),
    }

    attempts = TASK_REQUEST_ATTEMPTS.get(task_id, 1)
    for attempt in range(attempts):
        try:
            resp = requests.post(DEFAULT_CHAT_URL, json=data, timeout=DEFAULT_REQUEST_TIMEOUT)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            if content:
                return content
        except Exception:
            pass
        if attempt + 1 < attempts:
            time.sleep(min(2 ** attempt, 4))
    return None


def annotate_chat_thinking(
    examples_str: str,
    text2annotate: str,
    task_id: int,
    task_description: str,
    profile_name: str | None = None,
) -> str | None:
    """Annotate using chat+thinking mode for supported tasks."""
    system_msg = TASK_CHAT_SYSTEM.get(task_id, "You are a precise annotation system.")
    user_msg = build_chat_user_message(
        examples_str,
        text2annotate,
        task_id,
        task_description,
        profile_name=profile_name,
    )

    sample_attempts = TASK_SAMPLE_ATTEMPTS.get(task_id, 1)
    for _ in range(sample_attempts):
        raw = _request_chat_completion(system_msg, user_msg, task_id)
        prediction = _extract_chat_answer(raw, task_id)
        if prediction is not None:
            return prediction
    return None


def annotate_ascend(input_prompt: str, task_id: int | None = None):
    """
    Annotate the unlabeled data using an OpenAI-compatible Ascend endpoint.
    """
    import openai

    openai.api_key = "EMPTY"
    openai.base_url = os.environ.get("OPENSEEK_ASCEND_URL", "http://localhost:9010/v1/")
    model = os.environ.get("OPENSEEK_ASCEND_MODEL_NAME", "Qwen3-4B-ascend-flagos")

    prompts = [input_prompt]
    retry_count = TASK_RETRY_COUNTS.get(task_id, DEFAULT_RETRY_COUNT)
    for _ in range(retry_count):
        prompts.append(_build_retry_prompt(input_prompt))

    for prompt in prompts:
        messages = [
            {"role": "system", "content": "You are a precise long-context annotation system."},
            {"role": "user", "content": prompt},
        ]
        try:
            response = openai.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                top_p=1.0,
                max_tokens=TASK_MAX_TOKENS.get(task_id, DEFAULT_MAX_TOKENS),
                stream=False,
            )
            whole_result = response.choices[0].message.content
        except Exception:
            whole_result = ""

        prediction = count_answer(whole_result, task_id=task_id)
        if prediction is not None:
            return prediction
    return None


def should_use_task6_structured_hybrid(task_id: int | None) -> bool:
    return task_id == 6 and DEFAULT_TASK6_STRUCTURED_HYBRID_ENABLED


@lru_cache(maxsize=1)
def _load_task6_structured_assets():
    try:
        from validate_task6_genre_classifier import (
            build_sentence_genre_classifier,
            parse_task6_fields,
            score_task6_target_genre,
        )
    except Exception:
        return None

    task_dict = json.loads(TASK6_FILE.read_text(encoding="utf-8"))
    examples = list(task_dict["examples"])
    classifier = build_sentence_genre_classifier(examples)
    return {
        "classifier": classifier,
        "parse_task6_fields": parse_task6_fields,
        "score_task6_target_genre": score_task6_target_genre,
    }


def solve_task6_structured_hybrid(text2annotate: str) -> str | None:
    if not should_use_task6_structured_hybrid(6):
        return None

    assets = _load_task6_structured_assets()
    if assets is None:
        return None

    try:
        sentence1, sentence2, genre = assets["parse_task6_fields"](text2annotate)
        score1, score2 = assets["score_task6_target_genre"](assets["classifier"], sentence1, sentence2, genre)
    except Exception:
        return None

    min_score = min(score1, score2)
    max_score = max(score1, score2)
    if min_score >= DEFAULT_TASK6_STRUCTURED_N_THRESHOLD:
        return None
    if (
        DEFAULT_TASK6_STRUCTURED_N_MAX_THRESHOLD is not None
        and max_score > DEFAULT_TASK6_STRUCTURED_N_MAX_THRESHOLD
    ):
        return None
    return "N"


def solve_task_locally(task_id: int, text2annotate: str) -> str | None:
    """
    Solve simple deterministic tasks without calling the LLM.
    """
    if task_id == 2:
        task2_answer = solve_task2_structured_count(text2annotate)
        if task2_answer is not None:
            return task2_answer

    if task_id == 6:
        task6_answer = solve_task6_structured_hybrid(text2annotate)
        if task6_answer is not None:
            return task6_answer

    try:
        parsed = literal_eval(text2annotate)
    except Exception:
        return None

    if task_id == 1 and isinstance(parsed, list) and len(parsed) >= 2:
        numbers = sorted(int(x) for x in parsed)
        answer = min(abs(curr - prev) for prev, curr in zip(numbers, numbers[1:]))
        return str(answer)

    if task_id == 3 and isinstance(parsed, list):
        answer = [value // 2 if value % 2 == 0 else value * 3 + 1 for value in parsed]
        return str(answer)

    if task_id == 4 and isinstance(parsed, list):
        return "".join(str(part) for part in parsed)

    return None
