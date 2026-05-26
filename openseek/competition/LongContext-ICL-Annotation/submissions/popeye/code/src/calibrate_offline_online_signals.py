import json
from collections import defaultdict
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
OUTPUTS_DIR = PROJECT_DIR / "outputs"
WORK_LOGS_DIR = OUTPUTS_DIR / "work_logs"


LINE_DEFS = {
    "task2_structured": {
        "label_target": "trusted",
        "promotion_gate": "promotable",
        "keywords": ["task2structured"],
        "versions": ["v18", "v22", "v23"],
    },
    "task7_rerank_main": {
        "label_target": "conditional",
        "promotion_gate": "promotable",
        "keywords": ["task7rerank"],
        "versions": ["v19"],
    },
    "task6_structured_hybrid": {
        "label_target": "misleading",
        "promotion_gate": "reject",
        "keywords": ["task6structuredhybrid"],
        "versions": ["v20"],
    },
    "task7_gated_secondary": {
        "label_target": "misleading",
        "promotion_gate": "reject",
        "keywords": ["task7u7gate"],
        "versions": ["v21"],
    },
    "task5_guided": {
        "label_target": "misleading",
        "promotion_gate": "exploratory",
        "keywords": ["task5guided"],
        "versions": ["v17"],
    },
    "task5_longchat": {
        "label_target": "conditional",
        "promotion_gate": "exploratory",
        "keywords": ["task5longchat"],
        "versions": ["v10"],
    },
    "task8_patch_family": {
        "label_target": "misleading",
        "promotion_gate": "exploratory",
        "keywords": ["task8patch", "task8lexical", "task8family", "task8proxy"],
        "versions": [],
    },
}


def load_ledger() -> dict:
    path = WORK_LOGS_DIR / "official_score_ledger_2026-03-29.json"
    return json.loads(path.read_text(encoding="utf-8"))


def match_line(entry: dict) -> str | None:
    candidate_dir = (entry.get("candidate_dir") or "").lower()
    version = entry.get("version")
    for line_name, spec in LINE_DEFS.items():
        if version in spec["versions"]:
            return line_name
        if any(keyword in candidate_dir for keyword in spec["keywords"]):
            return line_name
    return None


def main():
    ledger = load_ledger()
    grouped = defaultdict(list)
    for entry in ledger["entries"]:
        line_name = match_line(entry)
        if line_name:
            grouped[line_name].append(entry)

    reports = []
    for line_name, entries in grouped.items():
        offline_positive = sum(1 for entry in entries if entry["offline_claimed_direction"] == "positive")
        online_positive = sum(1 for entry in entries if entry["actual_online_direction"] == "positive")
        online_negative = sum(1 for entry in entries if entry["actual_online_direction"] == "negative")
        avg_delta = round(sum(entry["official_delta_vs_base"] or 0.0 for entry in entries) / len(entries), 4)
        spec = LINE_DEFS[line_name]
        reports.append(
            {
                "line_name": line_name,
                "entries": [entry["version"] for entry in entries],
                "target_tasks": sorted(
                    {
                        task
                        for entry in entries
                        for task in (entry.get("replaced_tasks") or {}).keys()
                    }
                ),
                "offline_positive_count": offline_positive,
                "online_positive_count": online_positive,
                "online_negative_count": online_negative,
                "avg_official_delta_vs_base": avg_delta,
                "calibration_label": spec["label_target"],
                "promotion_gate": spec["promotion_gate"],
                "rationale": build_rationale(line_name, entries, spec["label_target"]),
            }
        )

    report = {
        "generated_on": "2026-03-29",
        "stable_official_baseline": "v23",
        "line_reports": sorted(reports, key=lambda item: item["line_name"]),
        "default_rules": {
            "non_promotable_before_next_phase": ["task6_structured_hybrid", "task7_gated_secondary"],
            "next_promotable_frontier_candidates": ["task2_structured", "task7_rerank_main"],
        },
    }

    json_path = WORK_LOGS_DIR / "offline_online_calibration_2026-03-29.json"
    md_path = WORK_LOGS_DIR / "offline_online_calibration_2026-03-29.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def build_rationale(line_name: str, entries: list[dict], label: str) -> str:
    versions = ", ".join(entry["version"] for entry in entries)
    if line_name == "task2_structured":
        return f"{versions} all improved online after positive offline evidence, so task2 is now the strongest calibrated structure-first line."
    if line_name == "task7_rerank_main":
        return f"{versions} improved online, but follow-up task7 variants failed, so the signal is real but fragile."
    if line_name == "task6_structured_hybrid":
        return f"{versions} had non-negative two-seed offline evidence but still regressed online."
    if line_name == "task7_gated_secondary":
        return f"{versions} had careful offline gating but still failed to beat the v19 task7 base online."
    if line_name == "task5_guided":
        return f"{versions} looked strong offline yet underperformed online, so it should stay exploratory."
    if line_name == "task5_longchat":
        return f"{versions} established the old stable base, but later task5-only follow-ups did not continue the gain."
    return f"{versions} does not have reliable offline-to-online transfer yet."


def render_markdown(report: dict) -> str:
    lines = [
        "# Offline/Online Calibration",
        "",
        f"- Generated on: `{report['generated_on']}`",
        f"- Stable official baseline: `{report['stable_official_baseline']}`",
        "",
        "## Line Classification",
        "",
        "| Line | Versions | Label | Gate | Avg Delta | Notes |",
        "|---|---|---|---|---:|---|",
    ]
    for row in report["line_reports"]:
        lines.append(
            f"| `{row['line_name']}` | `{', '.join(row['entries'])}` | `{row['calibration_label']}` | "
            f"`{row['promotion_gate']}` | `{row['avg_official_delta_vs_base']}` | {row['rationale']} |"
        )
    lines.extend(
        [
            "",
            "## Default Rules",
            "",
            f"- Non-promotable before next phase: `{', '.join(report['default_rules']['non_promotable_before_next_phase'])}`",
            f"- Next promotable frontier candidates: `{', '.join(report['default_rules']['next_promotable_frontier_candidates'])}`",
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
