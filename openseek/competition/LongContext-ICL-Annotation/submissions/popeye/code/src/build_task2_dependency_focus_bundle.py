import json
from pathlib import Path

from spike_task2_dependency_aware import (
    WORK_LOGS_DIR,
    classify_target_cluster,
    classify_target_subcluster,
    dependency_aware_fallback_count,
    has_protected_stuffed_exact,
    load_dataset,
    parse_task2_row,
    surface_chain_count,
    tokenize,
)


FOCUS_SUBCLUSTERS = {
    ("be_being_vbn", "passive_chain"),
    ("rel_be_vbg", "plain"),
    ("rel_be_vbg", "directional_tail"),
    ("be_vbg", "plain"),
    ("be_vbg", "directional_tail"),
    ("be_vbg", "serial_vbg"),
    ("be_vbg", "secondary_vbg"),
    ("rel_be_vbg", "secondary_vbg"),
}

DEFER_SUBCLUSTERS = {
    ("be_vbg", "to_inf_tail"),
    ("rel_be_vbg", "serial_vbg"),
    ("rel_be_vbg", "to_inf_tail"),
}


def collect_rows() -> dict[tuple[str, str], list[dict]]:
    dataset = load_dataset()
    grouped: dict[tuple[str, str], list[dict]] = {}

    for row in dataset["examples"]:
        parsed = parse_task2_row(row["input"])
        if parsed is None:
            continue
        sentence, target = parsed
        if target != "verbs":
            continue

        tokens = tokenize(sentence)
        if has_protected_stuffed_exact(tokens):
            continue

        cluster = classify_target_cluster(tokens)
        if cluster is None:
            continue
        subcluster = classify_target_subcluster(tokens, cluster)
        key = (cluster, subcluster)
        grouped.setdefault(key, []).append(
            {
                "id": row["id"],
                "gold": int(row["output"][0]),
                "sentence": sentence,
                "surface_count": surface_chain_count(tokens),
                "spike_count": dependency_aware_fallback_count(tokens),
            }
        )

    return grouped


def summarize_rows(rows: list[dict]) -> dict:
    improved = []
    worsened = []
    unchanged = []
    for row in rows:
        surface_error = abs(row["surface_count"] - row["gold"])
        spike_error = abs(row["spike_count"] - row["gold"])
        enriched = row | {
            "surface_error": surface_error,
            "spike_error": spike_error,
        }
        if spike_error < surface_error:
            improved.append(enriched)
        elif spike_error > surface_error:
            worsened.append(enriched)
        else:
            unchanged.append(enriched)

    return {
        "row_count": len(rows),
        "improve_count": len(improved),
        "worse_count": len(worsened),
        "same_count": len(unchanged),
        "sample_improvements": improved[:10],
        "sample_worsenings": worsened[:10],
        "sample_same": unchanged[:5],
        "all_row_ids": [row["id"] for row in rows],
    }


def build_report() -> dict:
    grouped = collect_rows()

    focus_sections = {}
    defer_sections = {}

    for key in sorted(FOCUS_SUBCLUSTERS):
        rows = grouped.get(key, [])
        focus_sections[f"{key[0]}:{key[1]}"] = {
            "cluster": key[0],
            "subcluster": key[1],
            **summarize_rows(rows),
        }

    for key in sorted(DEFER_SUBCLUSTERS):
        rows = grouped.get(key, [])
        defer_sections[f"{key[0]}:{key[1]}"] = {
            "cluster": key[0],
            "subcluster": key[1],
            **summarize_rows(rows),
        }

    focus_row_count = sum(section["row_count"] for section in focus_sections.values())
    defer_row_count = sum(section["row_count"] for section in defer_sections.values())

    return {
        "generated_on": "2026-03-31",
        "purpose": "Materialize the next dependency-aware review set so the next offline iteration can stay on the cleanest subclusters and avoid mixed to-infinitive buckets.",
        "focus_subclusters": sorted(f"{cluster}:{subcluster}" for cluster, subcluster in FOCUS_SUBCLUSTERS),
        "defer_subclusters": sorted(f"{cluster}:{subcluster}" for cluster, subcluster in DEFER_SUBCLUSTERS),
        "focus_bundle": {
            "row_count": focus_row_count,
            "sections": focus_sections,
        },
        "defer_bundle": {
            "row_count": defer_row_count,
            "sections": defer_sections,
        },
        "next_step_boundary": {
            "keep_for_next_offline_iteration": [
                "plain be+VBG",
                "directional-tail be+VBG",
                "relative-clause plain be+VBG",
                "relative-clause directional-tail be+VBG",
                "passive-chain be being + VBN",
                "selected serial/secondary VBG buckets",
            ],
            "defer_until_separate_policy": [
                "be+VBG + to-infinitive tails",
                "very small relative serial buckets",
                "very small relative to-infinitive buckets",
            ],
        },
    }


def render_markdown(report: dict) -> str:
    lines = [
        "# Task2 Dependency Focus Bundle",
        "",
        f"- Focus bundle rows: `{report['focus_bundle']['row_count']}`",
        f"- Defer bundle rows: `{report['defer_bundle']['row_count']}`",
        "",
        "## Focus Subclusters",
        "",
    ]

    for name, section in report["focus_bundle"]["sections"].items():
        lines.append(
            f"- `{name}` rows=`{section['row_count']}` improved=`{section['improve_count']}` worsened=`{section['worse_count']}`"
        )

    lines.extend(["", "## Deferred Subclusters", ""])
    for name, section in report["defer_bundle"]["sections"].items():
        lines.append(
            f"- `{name}` rows=`{section['row_count']}` improved=`{section['improve_count']}` worsened=`{section['worse_count']}`"
        )

    lines.extend(["", "## Boundary", ""])
    for item in report["next_step_boundary"]["keep_for_next_offline_iteration"]:
        lines.append(f"- Keep: {item}")
    for item in report["next_step_boundary"]["defer_until_separate_policy"]:
        lines.append(f"- Defer: {item}")
    return "\n".join(lines) + "\n"


def main() -> None:
    report = build_report()
    json_path = WORK_LOGS_DIR / "task2_dependency_focus_bundle_2026-03-31.json"
    md_path = WORK_LOGS_DIR / "task2_dependency_focus_bundle_2026-03-31.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
