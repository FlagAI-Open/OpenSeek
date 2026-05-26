import argparse
import json
import re
from collections import Counter
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audit_path",
        type=str,
        required=True,
        help="Path to a task7 error audit JSON produced by audit_task7_errors.py.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Optional path to save the constraint probe report as JSON.",
    )
    return parser.parse_args()


def normalize_answer(text: str | None) -> str:
    if text is None:
        return ""
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"^[\"'`]+|[\"'`]+$", "", text)
    text = re.sub(r"^[^a-z0-9]+|[^a-z0-9]+$", "", text)
    text = re.sub(r"^(a|an|the)\s+", "", text)
    return re.sub(r"\s+", " ", text).strip()


def alpha_len(text: str) -> int:
    return len(re.findall(r"[a-z]", text.lower()))


def extract_constraints(category: str) -> list[dict]:
    constraints = []
    cat = category.strip()

    letter_matches = re.findall(r"(\d+)[- ]LETTER", cat, flags=re.I)
    if letter_matches:
        constraints.append(
            {
                "type": "letter_count",
                "values": sorted({int(value) for value in letter_matches}),
            }
        )

    quoted_fragments = [frag.lower() for frag in re.findall(r'"([^"]+)"', cat)]
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
            options.extend(piece.strip().lower() for piece in pieces if piece.strip())
        if options:
            constraints.append(
                {
                    "type": "starts_with_options",
                    "values": sorted(set(options)),
                }
            )

    return constraints


def check_constraint(answer: str, constraint: dict) -> bool:
    norm = normalize_answer(answer)
    if not norm:
        return False

    if constraint["type"] == "letter_count":
        return alpha_len(norm) in set(constraint["values"])

    if constraint["type"] == "quoted_fragment":
        joined = re.sub(r"[^a-z0-9]", "", norm)
        return any(re.sub(r"[^a-z0-9]", "", value) in joined for value in constraint["values"])

    if constraint["type"] == "starts_with_options":
        return any(norm.startswith(value) for value in constraint["values"])

    return False


def summarize_report(report: dict) -> dict:
    overall_constraint_types = Counter()
    overall_gold_hits = Counter()
    overall_pred_hits = Counter()
    actionable_counter = Counter()
    per_file = []

    for file_report in report["reports"]:
        file_counter = Counter()
        gold_counter = Counter()
        pred_counter = Counter()
        actionable_examples = []

        for row in file_report["mismatch_examples"]:
            constraints = extract_constraints(row["category"])
            if not constraints:
                continue
            for constraint in constraints:
                ctype = constraint["type"]
                file_counter[ctype] += 1
                overall_constraint_types[ctype] += 1

                gold_ok = check_constraint(row["gold"], constraint)
                pred_ok = check_constraint(row["prediction"], constraint)
                gold_counter[ctype] += int(gold_ok)
                pred_counter[ctype] += int(pred_ok)
                overall_gold_hits[ctype] += int(gold_ok)
                overall_pred_hits[ctype] += int(pred_ok)

                if gold_ok and not pred_ok:
                    actionable_counter[ctype] += 1
                    actionable_examples.append(
                        {
                            "id": row["id"],
                            "category": row["category"],
                            "constraint_type": ctype,
                            "constraint_values": constraint["values"],
                            "gold": row["gold"],
                            "prediction": row["prediction"],
                        }
                    )

        per_file.append(
            {
                "file": file_report["file"],
                "avg_score": file_report["avg_score"],
                "constraint_type_counter": dict(file_counter),
                "gold_satisfy_counter": dict(gold_counter),
                "prediction_satisfy_counter": dict(pred_counter),
                "actionable_examples": actionable_examples[:15],
            }
        )

    return {
        "aggregate": {
            "constraint_type_counter": dict(overall_constraint_types),
            "gold_satisfy_counter": dict(overall_gold_hits),
            "prediction_satisfy_counter": dict(overall_pred_hits),
            "actionable_counter": dict(actionable_counter),
        },
        "files": per_file,
    }


def main():
    args = parse_args()
    audit_report = json.loads(Path(args.audit_path).read_text(encoding="utf-8"))
    summary = summarize_report(audit_report)

    if args.output_path:
        Path(args.output_path).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
