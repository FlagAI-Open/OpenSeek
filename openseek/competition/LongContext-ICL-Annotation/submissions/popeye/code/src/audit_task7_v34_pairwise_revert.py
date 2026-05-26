import argparse
import json
from pathlib import Path

from method import _run_task7_judge_once, parse_task7_fields
from audit_task7_v34_changed_rows import classify_change


SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUTS_DIR = PROJECT_DIR / "outputs"
WORK_LOGS_DIR = OUTPUTS_DIR / "work_logs"

TASK7_DATA_PATH = DATA_DIR / "openseek-7_jeopardy_answer_generation_all.json"
BASE_PATH = OUTPUTS_DIR / "final_submission_v30_task8clean2_on_v27_candidate" / "openseek-7-v1.jsonl"
CANDIDATE_PATH = OUTPUTS_DIR / "task7_author_projection_stability_live_candidate" / "openseek-7-v1.jsonl"
DEFAULT_OUTPUT_JSON = WORK_LOGS_DIR / "task7_v34_pairwise_revert_audit_2026-04-09.json"
DEFAULT_OUTPUT_MD = WORK_LOGS_DIR / "task7_v34_pairwise_revert_audit_2026-04-09.md"

TARGET_LOCK_IDS = {
    "openseek-7-3deb283e13ec4b5393b1f1d61b426a4e",
    "openseek-7-88c5e426f6b84e959de64d77e0862297",
    "openseek-7-02e748b0da7f4492864df5ab8b804556",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--bucket", type=str, default="semantic_jump")
    parser.add_argument("--output_path", type=str, default=str(DEFAULT_OUTPUT_JSON))
    parser.add_argument("--markdown_path", type=str, default=str(DEFAULT_OUTPUT_MD))
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> dict[str, dict]:
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows[row["test_sample_id"]] = row
    return rows


def load_test_samples() -> dict[str, dict]:
    payload = load_json(TASK7_DATA_PATH)
    return {row["id"]: row for row in payload["test_samples"]}


def run_pairwise(category: str, clue: str, old_prediction: str, new_prediction: str) -> dict:
    ab = _run_task7_judge_once(category, clue, [old_prediction, new_prediction])
    ba = _run_task7_judge_once(category, clue, [new_prediction, old_prediction])
    old_win_count = int(ab["winner"] == old_prediction) + int(ba["winner"] == old_prediction)
    new_win_count = int(ab["winner"] == new_prediction) + int(ba["winner"] == new_prediction)

    if old_win_count == 2:
        recommendation = "revert_to_v30"
    elif new_win_count == 2:
        recommendation = "keep_v34"
    else:
        recommendation = "manual_review"

    return {
        "ab": {
            "order": [old_prediction, new_prediction],
            "winner": ab["winner"],
            "raw": ab["raw"],
            "index": ab["index"],
            "winner_is_old": ab["winner"] == old_prediction,
            "winner_is_new": ab["winner"] == new_prediction,
        },
        "ba": {
            "order": [new_prediction, old_prediction],
            "winner": ba["winner"],
            "raw": ba["raw"],
            "index": ba["index"],
            "winner_is_old": ba["winner"] == old_prediction,
            "winner_is_new": ba["winner"] == new_prediction,
        },
        "old_win_count": old_win_count,
        "new_win_count": new_win_count,
        "recommendation": recommendation,
    }


def build_changed_rows(base_rows: dict[str, dict], candidate_rows: dict[str, dict]) -> list[dict]:
    changed_rows = []
    for sample_id, candidate_row in candidate_rows.items():
        base_row = base_rows.get(sample_id)
        if base_row is None or base_row["prediction"] == candidate_row["prediction"]:
            continue
        classification = classify_change(base_row["prediction"], candidate_row["prediction"])
        changed_rows.append(
            {
                "id": sample_id,
                "old_prediction": base_row["prediction"],
                "new_prediction": candidate_row["prediction"],
                **classification,
            }
        )
    changed_rows.sort(key=lambda row: (row["bucket"], row["id"]))
    return changed_rows


def pick_rows(changed_rows: list[dict], *, bucket: str, limit: int) -> list[dict]:
    filtered = [
        row
        for row in changed_rows
        if row["bucket"] == bucket and row["id"] not in TARGET_LOCK_IDS
    ]
    return filtered[:limit]


def build_report(limit: int, bucket: str) -> dict:
    base_rows = load_jsonl(BASE_PATH)
    candidate_rows = load_jsonl(CANDIDATE_PATH)
    test_samples = load_test_samples()
    changed_rows = build_changed_rows(base_rows, candidate_rows)

    selected_rows = pick_rows(changed_rows, bucket=bucket, limit=limit)
    audited_rows = []
    recommendation_counts: dict[str, int] = {}

    for triage_row in selected_rows:
        sample_id = triage_row["id"]
        test_sample = test_samples[sample_id]
        category, clue = parse_task7_fields(test_sample["input"])
        old_prediction = base_rows[sample_id]["prediction"]
        new_prediction = candidate_rows[sample_id]["prediction"]
        pairwise = run_pairwise(category, clue, old_prediction, new_prediction)
        audited_row = {
            "id": sample_id,
            "category": category,
            "clue": clue,
            "old_prediction": old_prediction,
            "new_prediction": new_prediction,
            "triage_bucket": triage_row["bucket"],
            "char_similarity": triage_row.get("char_similarity"),
            "pairwise": pairwise,
        }
        audited_rows.append(audited_row)
        recommendation = pairwise["recommendation"]
        recommendation_counts[recommendation] = recommendation_counts.get(recommendation, 0) + 1

    return {
        "generated_on": "2026-04-09",
        "base_version": "v30",
        "candidate_version": "v34",
        "audited_bucket": bucket,
        "audited_row_count": len(audited_rows),
        "recommendation_counts": recommendation_counts,
        "rows": audited_rows,
        "next_step_hint": (
            "Rows with recommendation `revert_to_v30` are the first shortlist for a narrow Task7-on-v34 revert pack. "
            "Rows with `manual_review` should be checked before any revert package is built."
        ),
    }


def render_markdown(report: dict) -> str:
    lines = [
        "# Task7 v34 Pairwise Revert Audit",
        "",
        f"- Audited bucket: `{report['audited_bucket']}`",
        f"- Audited row count: `{report['audited_row_count']}`",
        "",
        "## Recommendation counts",
        "",
    ]
    for key, value in sorted(report["recommendation_counts"].items()):
        lines.append(f"- `{key}`: `{value}`")

    lines.extend(["", "## Rows", ""])
    for row in report["rows"]:
        lines.append(
            f"- `{row['id']}`: `{row['old_prediction']}` vs `{row['new_prediction']}` -> `{row['pairwise']['recommendation']}` "
            f"(old `{row['pairwise']['old_win_count']}` / new `{row['pairwise']['new_win_count']}`)"
        )

    lines.extend(["", "## Next-step hint", "", f"- {report['next_step_hint']}", ""])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    report = build_report(limit=args.limit, bucket=args.bucket)
    output_path = Path(args.output_path)
    markdown_path = Path(args.markdown_path)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
