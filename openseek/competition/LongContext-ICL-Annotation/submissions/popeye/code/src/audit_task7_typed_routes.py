import argparse
import json
import re
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline_files",
        nargs="+",
        required=True,
        help="Baseline task7 holdout JSON files produced by validate_task7_candidate_rerank.py",
    )
    parser.add_argument(
        "--typed_files",
        nargs="+",
        required=True,
        help="Typed-secondary task7 holdout JSON files produced by validate_task7_candidate_rerank.py",
    )
    parser.add_argument("--gate_min_unique", type=int, default=4)
    parser.add_argument("--gate_min_entropy", type=float, default=1.3)
    parser.add_argument("--output_path", type=str, default=None)
    return parser.parse_args()


def load_payloads(paths: list[str]) -> dict[int, dict]:
    payloads = {}
    for raw_path in paths:
        path = Path(raw_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        seed = int(payload["seed"])
        payload["_path"] = str(path)
        payloads[seed] = payload
    return payloads


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


def get_task7_constraint_subtypes(category: str) -> list[str]:
    return [constraint["type"] for constraint in extract_task7_category_constraints(category)]


def _has_any_pattern(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, text, flags=re.I) for pattern in patterns)


def _is_numeric_route(category: str, clue: str) -> bool:
    clue_lower = clue.lower()
    category_lower = category.lower()
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
    if _has_any_pattern(clue_lower, explicit_clue_patterns):
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
    return _has_any_pattern(category_lower, explicit_category_patterns)


def _is_person_entity_route(clue: str) -> bool:
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
    return _has_any_pattern(clue_lower, explicit_person_patterns)


def _is_title_or_place_route(category: str, clue: str) -> bool:
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
    return _has_any_pattern(combined, explicit_patterns)


def detect_task7_secondary_family(category: str, clue: str, typed_route: str = "auto") -> str:
    if typed_route != "auto":
        return "generic"
    if extract_task7_category_constraints(category):
        return "constraint"
    if _is_numeric_route(category, clue):
        return "numeric"
    if _is_person_entity_route(clue):
        return "person_entity"
    if _is_title_or_place_route(category, clue):
        return "title_or_place"
    return "generic"


def row_lookup(payload: dict) -> dict[str, dict]:
    return {row["id"]: row for row in payload.get("rows", [])}


def init_metric_bucket() -> dict:
    return {
        "count": 0,
        "triggered_count": 0,
        "baseline_oracle_hit": 0,
        "typed_oracle_hit": 0,
        "baseline_judge_correct": 0,
        "typed_judge_correct": 0,
        "oracle_gain_count": 0,
        "judge_gain_count": 0,
    }


def finalize_metric_bucket(bucket: dict) -> dict:
    count = bucket["count"]
    triggered = bucket["triggered_count"]
    bucket["baseline_oracle_hit_rate"] = bucket["baseline_oracle_hit"] / count if count else 0.0
    bucket["typed_oracle_hit_rate"] = bucket["typed_oracle_hit"] / count if count else 0.0
    bucket["baseline_judge_accuracy"] = bucket["baseline_judge_correct"] / count if count else 0.0
    bucket["typed_judge_accuracy"] = bucket["typed_judge_correct"] / count if count else 0.0
    bucket["oracle_gain_on_triggered"] = bucket["oracle_gain_count"] / triggered if triggered else 0.0
    bucket["judge_gain_on_triggered"] = bucket["judge_gain_count"] / triggered if triggered else 0.0
    return bucket


def should_trigger_gate(row: dict, min_unique: int, min_entropy: float) -> bool:
    pool_summary = row.get("primary_pool_summary") or {}
    return (
        int(pool_summary.get("unique_candidates", 0)) >= min_unique
        or float(pool_summary.get("entropy", 0.0)) >= min_entropy
    )


def build_audit_row(
    *,
    seed: int,
    typed_row: dict,
    baseline_row: dict | None,
    expected_family: str,
    constraint_subtypes: list[str],
    gate_should_trigger: bool,
) -> dict:
    typed_oracle = bool(typed_row.get("oracle_hit"))
    typed_judge = bool(typed_row.get("judge_correct"))
    baseline_oracle = bool(baseline_row.get("oracle_hit")) if baseline_row else False
    baseline_judge = bool(baseline_row.get("judge_correct")) if baseline_row else False
    return {
        "seed": seed,
        "id": typed_row.get("id"),
        "category": typed_row.get("category"),
        "clue": typed_row.get("clue"),
        "gold": typed_row.get("gold"),
        "routed_family": typed_row.get("secondary_family"),
        "expected_family": expected_family,
        "constraint_subtypes": constraint_subtypes,
        "secondary_gate_triggered": bool(typed_row.get("secondary_gate_triggered")),
        "gate_should_trigger": gate_should_trigger,
        "primary_pool_summary": typed_row.get("primary_pool_summary"),
        "vote_candidate": typed_row.get("vote_candidate"),
        "judge_candidate": typed_row.get("judge_candidate"),
        "oracle_hit": typed_oracle,
        "judge_correct": typed_judge,
        "baseline_vote_candidate": baseline_row.get("vote_candidate") if baseline_row else None,
        "baseline_judge_candidate": baseline_row.get("judge_candidate") if baseline_row else None,
        "baseline_oracle_hit": baseline_oracle,
        "baseline_judge_correct": baseline_judge,
        "oracle_delta_vs_baseline": int(typed_oracle) - int(baseline_oracle),
        "judge_delta_vs_baseline": int(typed_judge) - int(baseline_judge),
    }


def build_report(
    baseline_payloads: dict[int, dict],
    typed_payloads: dict[int, dict],
    *,
    gate_min_unique: int,
    gate_min_entropy: float,
) -> dict:
    paired_seeds = sorted(set(baseline_payloads) & set(typed_payloads))
    family_summary = {
        family: init_metric_bucket()
        for family in ("constraint", "numeric", "person_entity", "title_or_place", "generic")
    }
    constraint_subtype_summary = {
        subtype: init_metric_bucket()
        for subtype in (
            "letter_count",
            "quoted_fragment",
            "starts_with_options",
            "before_after",
            "title_character",
            "team_name",
            "person_name",
        )
    }
    misroute_buckets = {
        "numeric_on_non_numeric": [],
        "title_or_place_on_person_or_generic_entity": [],
        "constraint_without_useful_gain": [],
        "generic_skipped_but_divergent": [],
    }
    paired_rows = []

    for seed in paired_seeds:
        baseline_payload = baseline_payloads[seed]
        typed_payload = typed_payloads[seed]
        baseline_rows = row_lookup(baseline_payload)

        for typed_row in typed_payload.get("rows", []):
            baseline_row = baseline_rows.get(typed_row["id"])
            constraint_subtypes = typed_row.get("constraint_subtypes") or get_task7_constraint_subtypes(
                typed_row.get("category", "")
            )
            expected_family = detect_task7_secondary_family(
                typed_row.get("category", ""),
                typed_row.get("clue", ""),
                typed_route="auto",
            )
            gate_should_trigger = should_trigger_gate(
                typed_row,
                min_unique=gate_min_unique,
                min_entropy=gate_min_entropy,
            )
            audit_row = build_audit_row(
                seed=seed,
                typed_row=typed_row,
                baseline_row=baseline_row,
                expected_family=expected_family,
                constraint_subtypes=constraint_subtypes,
                gate_should_trigger=gate_should_trigger,
            )
            paired_rows.append(audit_row)

            family = typed_row.get("secondary_family", "generic")
            family_summary.setdefault(family, init_metric_bucket())
            family_summary[family]["count"] += 1
            family_summary[family]["triggered_count"] += int(bool(typed_row.get("secondary_gate_triggered")))
            family_summary[family]["baseline_oracle_hit"] += int(bool(baseline_row and baseline_row.get("oracle_hit")))
            family_summary[family]["typed_oracle_hit"] += int(bool(typed_row.get("oracle_hit")))
            family_summary[family]["baseline_judge_correct"] += int(
                bool(baseline_row and baseline_row.get("judge_correct"))
            )
            family_summary[family]["typed_judge_correct"] += int(bool(typed_row.get("judge_correct")))
            if typed_row.get("secondary_gate_triggered"):
                family_summary[family]["oracle_gain_count"] += audit_row["oracle_delta_vs_baseline"]
                family_summary[family]["judge_gain_count"] += audit_row["judge_delta_vs_baseline"]

            for subtype in constraint_subtypes:
                constraint_subtype_summary.setdefault(subtype, init_metric_bucket())
                bucket = constraint_subtype_summary[subtype]
                bucket["count"] += 1
                bucket["triggered_count"] += int(bool(typed_row.get("secondary_gate_triggered")))
                bucket["baseline_oracle_hit"] += int(bool(baseline_row and baseline_row.get("oracle_hit")))
                bucket["typed_oracle_hit"] += int(bool(typed_row.get("oracle_hit")))
                bucket["baseline_judge_correct"] += int(bool(baseline_row and baseline_row.get("judge_correct")))
                bucket["typed_judge_correct"] += int(bool(typed_row.get("judge_correct")))
                if typed_row.get("secondary_gate_triggered"):
                    bucket["oracle_gain_count"] += audit_row["oracle_delta_vs_baseline"]
                    bucket["judge_gain_count"] += audit_row["judge_delta_vs_baseline"]

            if family == "numeric" and expected_family != "numeric":
                misroute_buckets["numeric_on_non_numeric"].append(audit_row)
            if family == "title_or_place" and expected_family in {"person_entity", "generic"}:
                misroute_buckets["title_or_place_on_person_or_generic_entity"].append(audit_row)
            if (
                family == "constraint"
                and typed_row.get("secondary_gate_triggered")
                and audit_row["oracle_delta_vs_baseline"] <= 0
                and audit_row["judge_delta_vs_baseline"] <= 0
            ):
                misroute_buckets["constraint_without_useful_gain"].append(audit_row)
            if (
                family == "generic"
                and not typed_row.get("secondary_gate_triggered")
                and gate_should_trigger
            ):
                misroute_buckets["generic_skipped_but_divergent"].append(audit_row)

    for bucket in family_summary.values():
        finalize_metric_bucket(bucket)
    for bucket in constraint_subtype_summary.values():
        finalize_metric_bucket(bucket)

    constraint_allowlist = sorted(
        subtype
        for subtype, bucket in constraint_subtype_summary.items()
        if bucket["oracle_gain_count"] > 0 and bucket["judge_gain_count"] >= 0
    )

    return {
        "paired_seeds": paired_seeds,
        "baseline_files": {seed: payload["_path"] for seed, payload in baseline_payloads.items()},
        "typed_files": {seed: payload["_path"] for seed, payload in typed_payloads.items()},
        "gate": {
            "min_unique": gate_min_unique,
            "min_entropy": gate_min_entropy,
        },
        "constraint_allowlist": constraint_allowlist,
        "family_summary": family_summary,
        "constraint_subtype_summary": constraint_subtype_summary,
        "misroute_buckets": misroute_buckets,
        "rows": paired_rows,
    }


def main():
    args = parse_args()
    baseline_payloads = load_payloads(args.baseline_files)
    typed_payloads = load_payloads(args.typed_files)
    report = build_report(
        baseline_payloads,
        typed_payloads,
        gate_min_unique=args.gate_min_unique,
        gate_min_entropy=args.gate_min_entropy,
    )
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output_path:
        Path(args.output_path).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
