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


def build_report() -> dict:
    dataset = load_dataset()
    rows_by_subcluster: dict[tuple[str, str], list[dict]] = {}

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
        rows_by_subcluster.setdefault(key, []).append(
            {
                "id": row["id"],
                "gold": int(row["output"][0]),
                "sentence": sentence,
                "surface_count": surface_chain_count(tokens),
                "spike_count": dependency_aware_fallback_count(tokens),
            }
        )

    cluster_reports = {}
    recommended_focus = []
    hold_for_later = []
    for (cluster, subcluster), rows in sorted(rows_by_subcluster.items()):
        improved = 0
        worsened = 0
        same = 0
        surface_exact = 0
        spike_exact = 0
        examples_improved = []
        examples_worsened = []

        for row in rows:
            surface_err = abs(row["surface_count"] - row["gold"])
            spike_err = abs(row["spike_count"] - row["gold"])
            if row["surface_count"] == row["gold"]:
                surface_exact += 1
            if row["spike_count"] == row["gold"]:
                spike_exact += 1
            if spike_err < surface_err:
                improved += 1
                if len(examples_improved) < 4:
                    examples_improved.append(row)
            elif spike_err > surface_err:
                worsened += 1
                if len(examples_worsened) < 4:
                    examples_worsened.append(row)
            else:
                same += 1

        row_count = len(rows)
        spike_exact_rate = round(spike_exact / row_count, 6)
        worsen_rate = round(worsened / row_count, 6)
        promote = improved > worsened and worsen_rate <= 0.2 and row_count >= 3

        summary = {
            "cluster": cluster,
            "subcluster": subcluster,
            "row_count": row_count,
            "surface_exact_rate": round(surface_exact / row_count, 6),
            "spike_exact_rate": spike_exact_rate,
            "improve_count": improved,
            "worse_count": worsened,
            "same_count": same,
            "recommended_status": "focus_next" if promote else "hold_or_split_further",
            "sample_improvements": examples_improved,
            "sample_worsenings": examples_worsened,
        }
        cluster_reports[f"{cluster}:{subcluster}"] = summary

        compact = {
            "cluster": cluster,
            "subcluster": subcluster,
            "row_count": row_count,
            "spike_exact_rate": spike_exact_rate,
            "improve_count": improved,
            "worse_count": worsened,
        }
        if promote:
            recommended_focus.append(compact)
        else:
            hold_for_later.append(compact)

    recommended_focus.sort(key=lambda row: (row["spike_exact_rate"], row["row_count"]), reverse=True)
    hold_for_later.sort(key=lambda row: (row["worse_count"], -row["spike_exact_rate"], row["row_count"]), reverse=True)

    return {
        "generated_on": "2026-03-31",
        "purpose": "Refine the dependency-aware task2 spike into subclusters so the next offline iteration keeps only the cleanest structure-aware buckets.",
        "subcluster_reports": cluster_reports,
        "recommended_focus_next": recommended_focus,
        "hold_or_split_further": hold_for_later,
    }


def render_markdown(report: dict) -> str:
    lines = [
        "# Task2 Dependency Subcluster Audit",
        "",
        "## Recommended Focus",
        "",
        "| Cluster | Subcluster | Rows | Spike exact | Improved | Worsened |",
        "|---|---|---:|---:|---:|---:|",
    ]

    for row in report["recommended_focus_next"]:
        lines.append(
            f"| `{row['cluster']}` | `{row['subcluster']}` | `{row['row_count']}` | "
            f"`{row['spike_exact_rate']}` | `{row['improve_count']}` | `{row['worse_count']}` |"
        )

    lines.extend(["", "## Hold Or Split Further", ""])
    for row in report["hold_or_split_further"]:
        lines.append(
            f"- `{row['cluster']}:{row['subcluster']}` rows=`{row['row_count']}` spike_exact=`{row['spike_exact_rate']}` "
            f"improved=`{row['improve_count']}` worsened=`{row['worse_count']}`"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    report = build_report()
    json_path = WORK_LOGS_DIR / "task2_dependency_subcluster_audit_2026-03-31.json"
    md_path = WORK_LOGS_DIR / "task2_dependency_subcluster_audit_2026-03-31.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
