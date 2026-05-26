import re
from dataclasses import dataclass

from method import normalize_task7_answer


TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize_author_surface(text: str) -> list[str]:
    return TOKEN_RE.findall(normalize_task7_answer(text))


def build_initials_signature(text: str) -> str:
    tokens = tokenize_author_surface(text)
    if len(tokens) < 2:
        return ""
    surname = tokens[-1]
    initials = "".join(token[0] for token in tokens[:-1] if token)
    if not initials or not surname:
        return ""
    return f"{initials}:{surname}"


def looks_like_initials_surface(text: str) -> bool:
    normalized = normalize_task7_answer(text)
    return bool(re.search(r"\b[a-z](?:\.[a-z])+\.?\b", normalized))


def looks_like_author_name(text: str) -> bool:
    tokens = tokenize_author_surface(text)
    return len(tokens) >= 2 and all(any(ch.isalpha() for ch in token) for token in tokens)


@dataclass(frozen=True)
class ProjectionResult:
    original: str
    projected: str
    reason: str | None


def choose_catalog_surface(candidates: list[str]) -> str:
    if not candidates:
        return ""
    ranked = sorted(
        candidates,
        key=lambda surface: (
            0 if looks_like_initials_surface(surface) else 1,
            len(tokenize_author_surface(surface)),
            len(normalize_task7_answer(surface)),
            normalize_task7_answer(surface),
        ),
    )
    return ranked[0]


def build_author_answer_catalog(example_answers: list[str]) -> dict:
    exact_map = {}
    initials_map = {}
    surname_only_map = {}

    by_initials = {}
    for answer in example_answers:
        normalized = normalize_task7_answer(answer)
        if not normalized:
            continue
        exact_map.setdefault(normalized, answer)

        signature = build_initials_signature(answer)
        if signature:
            by_initials.setdefault(signature, []).append(answer)

        tokens = tokenize_author_surface(answer)
        if len(tokens) == 1 and any(ch.isalpha() for ch in tokens[0]):
            surname_only_map.setdefault(tokens[0], answer)

    for signature, surfaces in by_initials.items():
        initials_map[signature] = choose_catalog_surface(surfaces)

    return {
        "exact_map": exact_map,
        "initials_map": initials_map,
        "surname_only_map": surname_only_map,
    }


def project_author_candidate(candidate: str, catalog: dict) -> ProjectionResult:
    normalized = normalize_task7_answer(candidate)
    if not normalized:
        return ProjectionResult(original=candidate, projected=candidate, reason=None)

    exact_match = catalog["exact_map"].get(normalized)
    if exact_match:
        reason = "exact_catalog_surface" if exact_match != candidate else None
        return ProjectionResult(original=candidate, projected=exact_match, reason=reason)

    signature = build_initials_signature(candidate)
    projected = catalog["initials_map"].get(signature)
    if projected:
        return ProjectionResult(original=candidate, projected=projected, reason="initials_signature")

    tokens = tokenize_author_surface(candidate)
    if len(tokens) >= 2:
        surname_only = catalog["surname_only_map"].get(tokens[-1])
        if surname_only:
            return ProjectionResult(original=candidate, projected=surname_only, reason="surname_only_catalog")

    return ProjectionResult(original=candidate, projected=candidate, reason=None)


def project_direct_fact_candidates_to_author_catalog(
    candidates: list[str],
    catalog: dict,
) -> tuple[list[str], list[dict]]:
    projected_candidates = []
    seen = set()
    projection_trace = []

    for candidate in candidates:
        projection = project_author_candidate(candidate, catalog)
        projected_norm = normalize_task7_answer(projection.projected)
        if projected_norm and projected_norm not in seen:
            seen.add(projected_norm)
            projected_candidates.append(projection.projected)
        projection_trace.append(
            {
                "original": projection.original,
                "projected": projection.projected,
                "reason": projection.reason,
            }
        )

    return projected_candidates, projection_trace
