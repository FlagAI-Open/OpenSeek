import re
from dataclasses import dataclass


TOKEN_PATTERN = re.compile(r"[A-Za-z']+")
BE_AUX = {"is", "are", "was", "were"}
KNOWN_VBN = {"fed", "made", "known", "seen", "born", "gone"}
PROTECTED_STUFFED_DETERMINERS = {"a", "an", "the"}
DISCOURSE_MARKERS = {"while", "as", "and"}
DISCOURSE_FOLLOWUP_VERBS = {
    "drive",
    "drives",
    "drove",
    "flies",
    "fly",
    "go",
    "goes",
    "had",
    "has",
    "have",
    "look",
    "looked",
    "looks",
    "walk",
    "walked",
    "walks",
    "watch",
    "watched",
    "watches",
    "went",
}


@dataclass(frozen=True)
class PolicyConfig:
    name: str
    description: str
    compress_be_chain: bool
    count_trailing_vbg: bool
    count_to_inf: bool
    count_after_discourse_marker: bool
    matches_2026_03_31_spike: bool = False


POLICY_CONFIGS = [
    PolicyConfig(
        name="compressed_only",
        description="Compress the initial be-chain once and stop there.",
        compress_be_chain=True,
        count_trailing_vbg=False,
        count_to_inf=False,
        count_after_discourse_marker=False,
    ),
    PolicyConfig(
        name="compressed_plus_trailing_vbg",
        description="Match the 2026-03-31 spike by compressing be+VBG once and still counting later VBG tails.",
        compress_be_chain=True,
        count_trailing_vbg=True,
        count_to_inf=False,
        count_after_discourse_marker=False,
        matches_2026_03_31_spike=True,
    ),
    PolicyConfig(
        name="compressed_plus_discourse_followup",
        description="Keep the trailing-VBG behavior and add one extra count when while/as/and introduces a finite follow-up predicate.",
        compress_be_chain=True,
        count_trailing_vbg=True,
        count_to_inf=False,
        count_after_discourse_marker=True,
    ),
]

POLICY_BY_NAME = {policy.name: policy for policy in POLICY_CONFIGS}


def tokenize_task2_sentence(sentence: str) -> list[str]:
    return TOKEN_PATTERN.findall(sentence.lower())


def has_protected_stuffed_exact(tokens: list[str]) -> bool:
    for idx in range(1, len(tokens) - 1):
        if tokens[idx] == "stuffed" and tokens[idx - 1] in PROTECTED_STUFFED_DETERMINERS:
            return True
    return False


def surface_chain_count(tokens: list[str]) -> int:
    count = 0
    for idx, token in enumerate(tokens):
        if token in BE_AUX:
            count += 1
            continue
        if token == "being":
            count += 1
            continue
        if token.endswith("ing"):
            count += 1
            continue
        if token.endswith("ed") and idx > 0 and tokens[idx - 1] == "being":
            count += 1
    return count


def is_to_inf_candidate(token: str) -> bool:
    return token.isalpha() and not token.endswith(("ed", "ing")) and token not in BE_AUX


def count_verbs_with_policy(tokens: list[str], policy: PolicyConfig) -> int:
    if has_protected_stuffed_exact(tokens):
        return surface_chain_count(tokens)

    count = 0
    idx = 0
    saw_compressed_be_chain = False

    while idx < len(tokens):
        token = tokens[idx]

        if (
            policy.compress_be_chain
            and token in BE_AUX
            and idx + 2 < len(tokens)
            and tokens[idx + 1] == "being"
            and (tokens[idx + 2].endswith("ed") or tokens[idx + 2] in KNOWN_VBN)
        ):
            count += 1
            idx += 3
            saw_compressed_be_chain = True
            continue

        if (
            policy.compress_be_chain
            and token in BE_AUX
            and idx + 1 < len(tokens)
            and tokens[idx + 1].endswith("ing")
        ):
            count += 1
            idx += 2
            saw_compressed_be_chain = True
            continue

        if token in BE_AUX:
            count += 1
            idx += 1
            continue

        if token == "being":
            count += 1
            idx += 1
            continue

        if token.endswith("ing"):
            if policy.count_trailing_vbg:
                count += 1
            idx += 1
            continue

        if policy.count_to_inf and token == "to" and idx + 1 < len(tokens):
            next_token = tokens[idx + 1]
            if is_to_inf_candidate(next_token) and next_token not in DISCOURSE_MARKERS:
                count += 1
                idx += 2
                continue

        if (
            policy.count_after_discourse_marker
            and saw_compressed_be_chain
            and token in DISCOURSE_MARKERS
        ):
            for tail_token in tokens[idx + 1 :]:
                if tail_token in DISCOURSE_FOLLOWUP_VERBS:
                    count += 1
                    break
                if tail_token in DISCOURSE_MARKERS:
                    break
            idx += 1
            continue

        idx += 1

    return count


def get_policy_config(name: str | None) -> PolicyConfig | None:
    if not name:
        return None
    return POLICY_BY_NAME.get(name.strip().lower())
