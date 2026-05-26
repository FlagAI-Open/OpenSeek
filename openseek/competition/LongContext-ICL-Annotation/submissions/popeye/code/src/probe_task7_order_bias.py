import argparse
import json
from collections import Counter, defaultdict
from itertools import islice
from pathlib import Path

from validate_task7_candidate_rerank import extract_option_index, run_judge


PERMUTATION_MODES = ("original", "reverse", "secondary_first")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--triggered_bundle", type=str, required=True)
    parser.add_argument("--permutations", nargs="+", default=list(PERMUTATION_MODES))
    parser.add_argument("--output_path", type=str, default=None)
    return parser.parse_args()


def load_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def permute_candidates(row: dict, mode: str) -> list[str]:
    candidates = list(row.get("judge_candidate_order") or row.get("judge_candidates_visible") or [])
    secondary_norms = set(row.get("secondary_only_candidate_norms") or [])
    if mode == "original":
        return candidates
    if mode == "reverse":
        return list(reversed(candidates))
    if mode == "secondary_first":
        secondary = [c for c in candidates if normalize(c) in secondary_norms]
        primary = [c for c in candidates if normalize(c) not in secondary_norms]
        return secondary + primary
    raise ValueError(f"Unsupported permutation mode: {mode}")


def normalize(text: str | None) -> str:
    return (text or "").strip().lower()


def evaluate_row(row: dict, permutations: list[str]) -> dict:
    per_mode = {}
    selected_norms = []
    flips_involve_secondary = False
    baseline_norm = normalize(row.get("judge_candidate"))
    secondary_norms = set(row.get("secondary_only_candidate_norms") or [])

    for mode in permutations:
        candidates = permute_candidates(row, mode)
        judge_candidate, judge_raw = run_judge(
            category=row.get("category") or "",
            clue=row.get("clue") or "",
            candidates=candidates,
            judge_mode=row.get("judge_mode") or "completion",
        )
        if judge_candidate is None:
            judge_candidate = candidates[0] if candidates else None
        selected_norm = normalize(judge_candidate)
        selected_norms.append(selected_norm)
        per_mode[mode] = {
            "candidate_order": candidates,
            "judge_raw": judge_raw,
            "judge_candidate": judge_candidate,
            "judge_candidate_normalized": selected_norm,
            "judge_correct": selected_norm == normalize(row.get("gold")),
            "selected_secondary_only": selected_norm in secondary_norms,
        }
        if baseline_norm and selected_norm != baseline_norm and (baseline_norm in secondary_norms or selected_norm in secondary_norms):
            flips_involve_secondary = True

    distinct = {norm for norm in selected_norms if norm}
    return {
        "seed": row.get("seed"),
        "id": row.get("id"),
        "secondary_family": row.get("secondary_family"),
        "constraint_subtypes": row.get("constraint_subtypes") or [],
        "gold": row.get("gold"),
        "baseline_judge_candidate": row.get("judge_candidate"),
        "baseline_judge_correct": bool(row.get("judge_correct")),
        "permutations": per_mode,
        "top_choice_flip": len(distinct) > 1,
        "disagreement_count": max(0, len(distinct) - 1),
        "disagreement_rate": 1.0 if len(distinct) > 1 else 0.0,
        "flips_involve_secondary_only": flips_involve_secondary,
    }


def build_report(triggered_bundle: dict, permutations: list[str]) -> dict:
    rows = [evaluate_row(row, permutations) for row in triggered_bundle.get("rows", [])]
    family_totals = Counter()
    family_flips = Counter()
    subtype_totals = Counter()
    subtype_flips = Counter()
    total_flips = 0
    total_secondary_flip_rows = 0
    for row in rows:
        family = row.get("secondary_family") or "unknown"
        family_totals[family] += 1
        family_flips[family] += int(row["top_choice_flip"])
        total_flips += int(row["top_choice_flip"])
        total_secondary_flip_rows += int(row["flips_involve_secondary_only"])
        for subtype in row.get("constraint_subtypes") or []:
            subtype_totals[subtype] += 1
            subtype_flips[subtype] += int(row["top_choice_flip"])

    family_summary = {
        family: {
            "row_count": family_totals[family],
            "flip_count": family_flips[family],
            "disagreement_rate": family_flips[family] / family_totals[family] if family_totals[family] else 0.0,
        }
        for family in sorted(family_totals)
    }
    subtype_summary = {
        subtype: {
            "row_count": subtype_totals[subtype],
            "flip_count": subtype_flips[subtype],
            "disagreement_rate": subtype_flips[subtype] / subtype_totals[subtype] if subtype_totals[subtype] else 0.0,
        }
        for subtype in sorted(subtype_totals)
    }

    return {
        "triggered_bundle": triggered_bundle.get("report_files"),
        "permutations": permutations,
        "row_count": len(rows),
        "top_choice_flip_count": total_flips,
        "disagreement_rate": total_flips / len(rows) if rows else 0.0,
        "flips_involving_secondary_only_count": total_secondary_flip_rows,
        "family_summary": family_summary,
        "constraint_subtype_summary": subtype_summary,
        "rows": rows,
    }


def main():
    args = parse_args()
    triggered_bundle = load_json(args.triggered_bundle)
    report = build_report(triggered_bundle, args.permutations)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output_path:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
