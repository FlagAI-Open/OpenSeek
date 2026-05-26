import json
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

from method import normalize_task7_answer


SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
OUTPUTS_DIR = PROJECT_DIR / "outputs"
WORK_LOGS_DIR = OUTPUTS_DIR / "work_logs"

BASE_PATH = OUTPUTS_DIR / "final_submission_v30_task8clean2_on_v27_candidate" / "openseek-7-v1.jsonl"
CANDIDATE_PATH = OUTPUTS_DIR / "task7_author_projection_stability_live_candidate" / "openseek-7-v1.jsonl"
OUTPUT_JSON = WORK_LOGS_DIR / "task7_v34_changed_row_triage_2026-04-09.json"
OUTPUT_MD = WORK_LOGS_DIR / "task7_v34_changed_row_triage_2026-04-09.md"

TARGET_ROWS = {
    "openseek-7-3deb283e13ec4b5393b1f1d61b426a4e",
    "openseek-7-88c5e426f6b84e959de64d77e0862297",
    "openseek-7-02e748b0da7f4492864df5ab8b804556",
}


def load_jsonl(path: Path) -> dict[str, dict]:
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows[row["test_sample_id"]] = row
    return rows


def normalize_surface(text: str) -> str:
    lowered = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii").lower()
    lowered = re.sub(r"\b(a|an|the)\b", " ", lowered)
    lowered = re.sub(r"[^a-z0-9]+", " ", lowered)
    return " ".join(lowered.split())


def normalize_compact(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", normalize_surface(text))


def tokenize(text: str) -> list[str]:
    return [token for token in normalize_surface(text).split() if token]


def classify_change(old_prediction: str, new_prediction: str) -> dict:
    old_norm = normalize_task7_answer(old_prediction) or ""
    new_norm = normalize_task7_answer(new_prediction) or ""
    old_surface = normalize_surface(old_prediction)
    new_surface = normalize_surface(new_prediction)
    old_compact = normalize_compact(old_prediction)
    new_compact = normalize_compact(new_prediction)
    old_tokens = set(tokenize(old_prediction))
    new_tokens = set(tokenize(new_prediction))
    shared = sorted(old_tokens & new_tokens)
    union = old_tokens | new_tokens
    jaccard = round(len(shared) / len(union), 4) if union else 0.0
    char_similarity = round(SequenceMatcher(None, old_compact, new_compact).ratio(), 4) if old_compact or new_compact else 0.0

    if old_norm == new_norm:
        bucket = "normalized_equal"
    elif old_surface == new_surface or old_compact == new_compact:
        bucket = "surface_equivalent"
    elif char_similarity >= 0.9:
        bucket = "surface_close_edit"
    elif old_tokens and new_tokens and jaccard >= 0.5:
        bucket = "high_overlap_rewrite"
    elif old_tokens and new_tokens and jaccard > 0.0:
        bucket = "partial_overlap_rewrite"
    else:
        bucket = "semantic_jump"

    return {
        "bucket": bucket,
        "old_norm": old_norm,
        "new_norm": new_norm,
        "old_surface": old_surface,
        "new_surface": new_surface,
        "old_compact": old_compact,
        "new_compact": new_compact,
        "shared_tokens": shared,
        "token_jaccard": jaccard,
        "char_similarity": char_similarity,
    }


def main() -> None:
    base_rows = load_jsonl(BASE_PATH)
    candidate_rows = load_jsonl(CANDIDATE_PATH)

    changed_rows = []
    bucket_counts: dict[str, int] = {}
    target_rows = []

    for sample_id, candidate_row in candidate_rows.items():
        base_row = base_rows.get(sample_id)
        if base_row is None:
            continue
        if base_row["prediction"] == candidate_row["prediction"]:
            continue

        classification = classify_change(base_row["prediction"], candidate_row["prediction"])
        row = {
            "id": sample_id,
            "old_prediction": base_row["prediction"],
            "new_prediction": candidate_row["prediction"],
            **classification,
        }
        changed_rows.append(row)
        bucket_counts[classification["bucket"]] = bucket_counts.get(classification["bucket"], 0) + 1
        if sample_id in TARGET_ROWS:
            target_rows.append(row)

    changed_rows.sort(key=lambda row: (row["bucket"], row["id"]))
    semantic_jump_rows = [row for row in changed_rows if row["bucket"] == "semantic_jump"]
    normalized_equal_rows = [row for row in changed_rows if row["bucket"] == "normalized_equal"]
    surface_equivalent_rows = [row for row in changed_rows if row["bucket"] == "surface_equivalent"]
    surface_close_edit_rows = [row for row in changed_rows if row["bucket"] == "surface_close_edit"]

    report = {
        "generated_on": "2026-04-09",
        "base_version": "v30",
        "candidate_version": "v34",
        "changed_row_count": len(changed_rows),
        "bucket_counts": bucket_counts,
        "target_rows": target_rows,
        "surface_like_examples": {
            "normalized_equal": normalized_equal_rows[:20],
            "surface_equivalent": surface_equivalent_rows[:20],
            "surface_close_edit": surface_close_edit_rows[:20],
        },
        "high_risk_examples": semantic_jump_rows[:40],
        "next_step_hint": (
            "Use the semantic_jump bucket as the first review queue for a possible narrow revert pack on top of v34. "
            "Keep the known target rows locked."
        ),
    }
    OUTPUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Task7 v34 Changed Row Triage",
        "",
        "- Base: `v30`",
        "- Candidate: `v34`",
        f"- Changed rows: `{report['changed_row_count']}`",
        "",
        "## Bucket counts",
        "",
    ]
    for bucket, count in sorted(bucket_counts.items()):
        lines.append(f"- `{bucket}`: `{count}`")
    lines.extend(
        [
            "",
            "## Locked target rows",
            "",
        ]
    )
    for row in target_rows:
        lines.append(f"- `{row['id']}`: `{row['old_prediction']}` -> `{row['new_prediction']}` (`{row['bucket']}`)")
    lines.extend(
        [
            "",
            "## High-risk examples",
            "",
        ]
    )
    for row in semantic_jump_rows[:20]:
        lines.append(f"- `{row['id']}`: `{row['old_prediction']}` -> `{row['new_prediction']}`")
    lines.extend(
        [
            "",
            "## Next-step hint",
            "",
            f"- {report['next_step_hint']}",
            "",
        ]
    )
    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
