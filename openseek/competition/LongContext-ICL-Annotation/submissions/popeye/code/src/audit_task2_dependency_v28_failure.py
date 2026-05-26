import json
from collections import Counter, defaultdict
from pathlib import Path

from method import parse_task2_fields
from spike_task2_dependency_aware import (
    classify_target_cluster,
    classify_target_subcluster,
    has_protected_stuffed_exact,
    tokenize,
)


SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
OUTPUTS_DIR = PROJECT_DIR / "outputs"
WORK_LOGS_DIR = OUTPUTS_DIR / "work_logs"
DATA_PATH = PROJECT_DIR / "data" / "openseek-2_count_nouns_verbs.json"

BASE_VERSION = "v27"
BASE_SCORE = 73.05
BASE_TASK2_PATH = (
    OUTPUTS_DIR
    / "final_submission_v27_task2stuffed_on_v25_candidate"
    / "openseek-2-v1.jsonl"
)

CANDIDATE_VERSION = "v28"
REPORTED_OFFICIAL_SCORE = 71.68
CANDIDATE_TASK2_PATH = (
    OUTPUTS_DIR
    / "final_submission_v28_task2dependency_discourse_candidate"
    / "openseek-2-v1.jsonl"
)

MAX_TINY_TEST_DIFF = 12
BE_AUX = {"is", "are", "was", "were"}


def load_jsonl_rows(path: Path) -> dict[str, dict]:
    return {
        row["test_sample_id"]: row
        for row in (
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }


def load_test_samples() -> dict[str, dict]:
    task_dict = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    return {row["id"]: row for row in task_dict["test_samples"]}


def classify_failure_bucket(tokens: list[str], old_prediction: int, new_prediction: int) -> str:
    protected = has_protected_stuffed_exact(tokens)
    cluster = None if protected else classify_target_cluster(tokens)
    if protected:
        return "protected_stuffed_exact"
    if cluster is not None:
        subcluster = classify_target_subcluster(tokens, cluster)
        return f"target_family:{cluster}:{subcluster}"

    has_be = any(token in BE_AUX for token in tokens)
    has_ing = any(token.endswith("ing") for token in tokens)
    starts_there_is = len(tokens) >= 2 and tokens[0] == "there" and tokens[1] in BE_AUX
    starts_this_is = len(tokens) >= 2 and tokens[0] in {"this", "it"} and tokens[1] in BE_AUX

    if new_prediction < old_prediction:
        if not has_ing and not has_be:
            return "outside_target:bare_finite_dropped"
        if not has_ing and has_be:
            return "outside_target:copula_or_helper_dropped"
        if has_ing and not has_be:
            return "outside_target:non_be_ing_undercount"
        return "outside_target:mixed_be_ing_undercount"

    if starts_there_is or starts_this_is:
        return "outside_target:existential_or_this_is_promoted"
    if has_ing and not has_be:
        return "outside_target:non_be_ing_promoted"
    if has_be and not has_ing:
        return "outside_target:copula_or_helper_promoted"
    if has_ing and has_be:
        return "outside_target:mixed_be_ing_promoted"
    return "outside_target:sideways_or_other"


def build_report() -> dict:
    base_rows = load_jsonl_rows(BASE_TASK2_PATH)
    candidate_rows = load_jsonl_rows(CANDIDATE_TASK2_PATH)
    test_samples = load_test_samples()

    changed_rows = []
    direction_counter = Counter()
    bucket_counter = Counter()
    discourse_marker_counter = Counter()
    sample_rows_by_bucket: dict[str, list[dict]] = defaultdict(list)

    for sample_id, candidate_row in candidate_rows.items():
        base_row = base_rows[sample_id]
        if candidate_row["prediction"] == base_row["prediction"]:
            continue

        sample = test_samples[sample_id]
        sentence, target = parse_task2_fields(sample["input"])
        tokens = tokenize(sentence)
        old_prediction = int(base_row["prediction"])
        new_prediction = int(candidate_row["prediction"])
        bucket = classify_failure_bucket(tokens, old_prediction, new_prediction)

        changed = {
            "id": sample_id,
            "sentence": sentence,
            "target": target,
            "old_prediction": old_prediction,
            "new_prediction": new_prediction,
            "delta": new_prediction - old_prediction,
            "bucket": bucket,
        }
        changed_rows.append(changed)
        direction_counter[(base_row["prediction"], candidate_row["prediction"])] += 1
        bucket_counter[bucket] += 1

        lowered = f" {sentence.lower()} "
        for marker in (" while ", " as ", " and ", " to ", " that ", " who ", " which "):
            if marker in lowered:
                discourse_marker_counter[marker.strip()] += 1

        if len(sample_rows_by_bucket[bucket]) < 8:
            sample_rows_by_bucket[bucket].append(changed)

    changed_rows.sort(key=lambda row: (abs(row["delta"]), row["old_prediction"], row["new_prediction"]), reverse=True)

    outside_target_count = sum(
        count for bucket, count in bucket_counter.items() if bucket.startswith("outside_target:")
    )
    target_family_count = sum(
        count for bucket, count in bucket_counter.items() if bucket.startswith("target_family:")
    )
    down_rows = sum(row["delta"] < 0 for row in changed_rows)
    up_rows = sum(row["delta"] > 0 for row in changed_rows)
    net_delta_sum = sum(row["delta"] for row in changed_rows)
    official_delta_vs_base = round(REPORTED_OFFICIAL_SCORE - BASE_SCORE, 2)

    gate_results = {
        "official_result_positive": REPORTED_OFFICIAL_SCORE > BASE_SCORE,
        "tiny_test_diff": len(changed_rows) <= MAX_TINY_TEST_DIFF,
        "outside_target_scope_zero": outside_target_count == 0,
        "target_scope_majority": target_family_count >= outside_target_count,
    }

    return {
        "generated_on": "2026-04-01",
        "baseline": {
            "version": BASE_VERSION,
            "official_score": BASE_SCORE,
            "task2_file": str(BASE_TASK2_PATH),
        },
        "candidate": {
            "version": CANDIDATE_VERSION,
            "reported_official_score": REPORTED_OFFICIAL_SCORE,
            "official_delta_vs_base": official_delta_vs_base,
            "task2_file": str(CANDIDATE_TASK2_PATH),
        },
        "gate_policy": {
            "max_tiny_test_diff": MAX_TINY_TEST_DIFF,
            "require_no_scope_leak_outside_target_family": True,
            "require_target_family_changes_to_dominate": True,
        },
        "summary": {
            "changed_test_row_count": len(changed_rows),
            "up_rows": up_rows,
            "down_rows": down_rows,
            "net_prediction_delta_sum": net_delta_sum,
            "outside_target_scope_change_count": outside_target_count,
            "target_family_change_count": target_family_count,
            "outside_target_scope_change_rate": round(outside_target_count / len(changed_rows), 6)
            if changed_rows
            else 0.0,
            "marker_counter_on_changed_rows": dict(sorted(discourse_marker_counter.items())),
        },
        "direction_counter": {
            f"{old}->{new}": count
            for (old, new), count in sorted(direction_counter.items())
        },
        "bucket_counter": dict(bucket_counter.most_common()),
        "sample_rows_by_bucket": dict(sample_rows_by_bucket),
        "gate_results": gate_results,
        "decision": {
            "result": "dependency_candidate_rejected",
            "primary_failure_mode": "global_scope_replacement_outside_target_family",
            "why": [
                "91 of 93 changed rows landed outside the intended dependency-aware target family.",
                "The dominant error mode was bare finite verbs dropping to zero rather than target-cluster follow-up counting.",
                "The candidate also promoted non-be participles and existential/this-is rows that were never part of the intended focus bundle.",
            ],
            "next_constraints_for_v28b": [
                "Never replace the global task2 verb solver with the dependency policy.",
                "Force the dependency-aware path to be a no-op outside an explicit target-family allowlist.",
                "Keep to_inf_tail deferred by default.",
                "Require submit-time task2 test diff to stay within tiny-patch territory before promotion.",
            ],
        },
    }


def render_markdown(report: dict) -> str:
    lines = [
        "# Task2 Dependency v28 Failure Audit",
        "",
        f"- Baseline: `{report['baseline']['version']} = {report['baseline']['official_score']}`",
        f"- Candidate: `{report['candidate']['version']} = {report['candidate']['reported_official_score']}`",
        f"- Official delta vs baseline: `{report['candidate']['official_delta_vs_base']}`",
        "",
        "## Summary",
        "",
        f"- Changed task2 test rows: `{report['summary']['changed_test_row_count']}`",
        f"- Up rows: `{report['summary']['up_rows']}`",
        f"- Down rows: `{report['summary']['down_rows']}`",
        f"- Net prediction delta sum: `{report['summary']['net_prediction_delta_sum']}`",
        f"- Outside target-family changes: `{report['summary']['outside_target_scope_change_count']}`",
        f"- Target-family changes: `{report['summary']['target_family_change_count']}`",
        f"- Outside target-family rate: `{report['summary']['outside_target_scope_change_rate']}`",
        "",
        "## Gate Failure",
        "",
    ]

    for key, value in report["gate_results"].items():
        lines.append(f"- `{key}` = `{value}`")

    lines.extend(
        [
            "",
            "## Failure Buckets",
            "",
            "| Bucket | Rows |",
            "|---|---:|",
        ]
    )
    for bucket, count in report["bucket_counter"].items():
        lines.append(f"| `{bucket}` | `{count}` |")

    lines.extend(["", "## Direction Counts", ""])
    for direction, count in report["direction_counter"].items():
        lines.append(f"- `{direction}`: `{count}`")

    lines.extend(["", "## Sample Rows", ""])
    for bucket, rows in report["sample_rows_by_bucket"].items():
        lines.append(f"- `{bucket}`")
        for row in rows[:4]:
            lines.append(
                f"  - `{row['old_prediction']} -> {row['new_prediction']}` :: {row['sentence']}"
            )

    lines.extend(["", "## Next Constraints", ""])
    for item in report["decision"]["next_constraints_for_v28b"]:
        lines.append(f"- {item}")

    return "\n".join(lines) + "\n"


def main() -> None:
    report = build_report()
    json_path = WORK_LOGS_DIR / "task2_dependency_v28_failure_audit_2026-04-01.json"
    md_path = WORK_LOGS_DIR / "task2_dependency_v28_failure_audit_2026-04-01.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
