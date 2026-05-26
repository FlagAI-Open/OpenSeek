import json
from dataclasses import asdict

from build_task2_dependency_focus_bundle import DEFER_SUBCLUSTERS, FOCUS_SUBCLUSTERS
from spike_task2_dependency_aware import (
    WORK_LOGS_DIR,
    classify_target_cluster,
    classify_target_subcluster,
    load_dataset,
    parse_task2_row,
)
from task2_dependency_policy import (
    POLICY_CONFIGS,
    PolicyConfig,
    count_verbs_with_policy,
    has_protected_stuffed_exact,
    surface_chain_count,
    tokenize_task2_sentence,
)


PRIMARY_FOCUS_SUBCLUSTERS = {
    ("be_being_vbn", "passive_chain"),
    ("be_vbg", "plain"),
    ("be_vbg", "directional_tail"),
    ("rel_be_vbg", "plain"),
    ("rel_be_vbg", "directional_tail"),
}

ANCHOR_POLICY = "compressed_plus_trailing_vbg"
CANDIDATE_POLICY = "compressed_plus_discourse_followup"
def iter_rows() -> list[dict]:
    rows = []
    for row in load_dataset()["examples"]:
        parsed = parse_task2_row(row["input"])
        if parsed is None:
            continue
        sentence, target = parsed
        if target != "verbs":
            continue

        tokens = tokenize_task2_sentence(sentence)
        protected = has_protected_stuffed_exact(tokens)
        cluster = None if protected else classify_target_cluster(tokens)
        subcluster = None if cluster is None else classify_target_subcluster(tokens, cluster)

        rows.append(
            {
                "id": row["id"],
                "sentence": sentence,
                "tokens": tokens,
                "gold": int(row["output"][0]),
                "surface_count": surface_chain_count(tokens),
                "protected": protected,
                "cluster": cluster,
                "subcluster": subcluster,
            }
        )
    return rows


def bundle_selector(bundle_name: str, row: dict) -> bool:
    key = (row["cluster"], row["subcluster"])
    if bundle_name == "primary_focus_bundle":
        return key in PRIMARY_FOCUS_SUBCLUSTERS
    if bundle_name == "focus_bundle":
        return key in FOCUS_SUBCLUSTERS
    if bundle_name == "defer_bundle":
        return key in DEFER_SUBCLUSTERS
    if bundle_name == "protected_stuffed_exact":
        return row["protected"]
    raise ValueError(f"Unknown bundle: {bundle_name}")


def compute_status(predicted: int, gold: int, surface: int) -> str:
    predicted_error = abs(predicted - gold)
    surface_error = abs(surface - gold)
    if predicted_error < surface_error:
        return "improve"
    if predicted_error > surface_error:
        return "worse"
    return "same"


def summarize_policy_on_rows(rows: list[dict], policy: PolicyConfig) -> dict:
    evaluated_rows = []
    for row in rows:
        predicted = count_verbs_with_policy(row["tokens"], policy)
        evaluated_rows.append(
            row
            | {
                "predicted_count": predicted,
                "predicted_error": abs(predicted - row["gold"]),
                "surface_error": abs(row["surface_count"] - row["gold"]),
                "status_vs_surface": compute_status(predicted, row["gold"], row["surface_count"]),
            }
        )

    exact_count = sum(row["predicted_count"] == row["gold"] for row in evaluated_rows)
    improved = [row for row in evaluated_rows if row["status_vs_surface"] == "improve"]
    worsened = [row for row in evaluated_rows if row["status_vs_surface"] == "worse"]
    unchanged = [row for row in evaluated_rows if row["status_vs_surface"] == "same"]

    return {
        "row_count": len(evaluated_rows),
        "exact_count": exact_count,
        "exact_rate": round(exact_count / len(evaluated_rows), 6) if evaluated_rows else 0.0,
        "improve_count": len(improved),
        "worse_count": len(worsened),
        "same_count": len(unchanged),
        "sample_improvements": improved[:8],
        "sample_worsenings": worsened[:8],
    }


def summarize_subclusters(rows: list[dict], policy: PolicyConfig, allowed_keys: set[tuple[str, str]]) -> dict:
    summary = {}
    for cluster, subcluster in sorted(allowed_keys):
        subset = [row for row in rows if (row["cluster"], row["subcluster"]) == (cluster, subcluster)]
        summary[f"{cluster}:{subcluster}"] = {
            "cluster": cluster,
            "subcluster": subcluster,
            **summarize_policy_on_rows(subset, policy),
        }
    return summary


def rows_changed_vs_anchor(rows: list[dict], candidate: PolicyConfig, anchor: PolicyConfig) -> list[dict]:
    changed = []
    for row in rows:
        candidate_count = count_verbs_with_policy(row["tokens"], candidate)
        anchor_count = count_verbs_with_policy(row["tokens"], anchor)
        if candidate_count == anchor_count:
            continue
        changed.append(
            {
                "id": row["id"],
                "sentence": row["sentence"],
                "gold": row["gold"],
                "anchor_count": anchor_count,
                "candidate_count": candidate_count,
                "anchor_error": abs(anchor_count - row["gold"]),
                "candidate_error": abs(candidate_count - row["gold"]),
                "cluster": row["cluster"],
                "subcluster": row["subcluster"],
            }
        )
    return changed


def build_report() -> dict:
    rows = iter_rows()
    policy_by_name = {policy.name: policy for policy in POLICY_CONFIGS}
    bundle_names = [
        "primary_focus_bundle",
        "focus_bundle",
        "defer_bundle",
        "protected_stuffed_exact",
    ]

    policy_summary = {}
    for policy in POLICY_CONFIGS:
        bundle_summary = {}
        for bundle_name in bundle_names:
            bundle_rows = [row for row in rows if bundle_selector(bundle_name, row)]
            bundle_summary[bundle_name] = summarize_policy_on_rows(bundle_rows, policy)

        policy_summary[policy.name] = {
            "policy": asdict(policy),
            "bundle_summary": bundle_summary,
            "primary_subclusters": summarize_subclusters(rows, policy, PRIMARY_FOCUS_SUBCLUSTERS),
            "defer_subclusters": summarize_subclusters(rows, policy, DEFER_SUBCLUSTERS),
        }

    anchor_policy = policy_by_name[ANCHOR_POLICY]
    candidate_policy = policy_by_name[CANDIDATE_POLICY]

    primary_rows = [row for row in rows if bundle_selector("primary_focus_bundle", row)]
    focus_rows = [row for row in rows if bundle_selector("focus_bundle", row)]
    defer_rows = [row for row in rows if bundle_selector("defer_bundle", row)]
    protected_rows = [row for row in rows if bundle_selector("protected_stuffed_exact", row)]

    changed_primary = rows_changed_vs_anchor(primary_rows, candidate_policy, anchor_policy)
    changed_defer = rows_changed_vs_anchor(defer_rows, candidate_policy, anchor_policy)
    changed_protected = rows_changed_vs_anchor(protected_rows, candidate_policy, anchor_policy)

    anchor_focus = policy_summary[ANCHOR_POLICY]["bundle_summary"]["focus_bundle"]
    candidate_focus = policy_summary[CANDIDATE_POLICY]["bundle_summary"]["focus_bundle"]
    anchor_primary = policy_summary[ANCHOR_POLICY]["bundle_summary"]["primary_focus_bundle"]
    candidate_primary = policy_summary[CANDIDATE_POLICY]["bundle_summary"]["primary_focus_bundle"]

    qualifies_for_isolated_toggle = (
        candidate_focus["exact_rate"] > anchor_focus["exact_rate"]
        and candidate_primary["exact_rate"] > anchor_primary["exact_rate"]
        and candidate_focus["worse_count"] <= anchor_focus["worse_count"]
        and not changed_defer
        and not changed_protected
    )

    return {
        "generated_on": "2026-04-01",
        "purpose": "Second-round task2 dependency-aware policy-table spike that keeps the scope on the focus bundle and tests where compression should continue with one extra lexical count.",
        "current_spike_equivalent": ANCHOR_POLICY,
        "policy_table": [asdict(policy) for policy in POLICY_CONFIGS],
        "bundle_sizes": {
            "primary_focus_bundle": len(primary_rows),
            "focus_bundle": len(focus_rows),
            "defer_bundle": len(defer_rows),
            "protected_stuffed_exact": len(protected_rows),
        },
        "policy_summary": policy_summary,
        "anchor_vs_candidate_gate": {
            "anchor_policy": ANCHOR_POLICY,
            "candidate_policy": CANDIDATE_POLICY,
            "primary_focus_exact_gain": round(
                candidate_primary["exact_rate"] - anchor_primary["exact_rate"], 6
            ),
            "focus_bundle_exact_gain": round(
                candidate_focus["exact_rate"] - anchor_focus["exact_rate"], 6
            ),
            "focus_bundle_worse_delta": candidate_focus["worse_count"] - anchor_focus["worse_count"],
            "changed_defer_rows_vs_anchor": len(changed_defer),
            "changed_protected_rows_vs_anchor": len(changed_protected),
            "qualifies_for_isolated_experiment_toggle": qualifies_for_isolated_toggle,
            "decision": (
                "candidate_ready_for_isolated_experiment_toggle"
                if qualifies_for_isolated_toggle
                else "stop_at_research_conclusion"
            ),
            "sample_primary_recoveries_vs_anchor": changed_primary[:10],
            "sample_defer_changes_vs_anchor": changed_defer[:5],
        },
        "notes": [
            "This round keeps count_to_inf disabled in all promoted policies because to_inf_tail remains a deferred bundle.",
            "The discourse follow-up rule only activates after a compressed be-chain and only looks for finite follow-up predicates after while/as/and.",
            "Serial and secondary VBG rows stay in the focus bundle summary, but the promotion gate is anchored on the narrower primary focus bundle.",
        ],
    }


def render_markdown(report: dict) -> str:
    lines = [
        "# Task2 Dependency Policy-Table Spike",
        "",
        "- This is a second-round offline-only spike focused on the dependency-aware focus bundle.",
        f"- `current_spike_equivalent` is `{report['current_spike_equivalent']}`.",
        "- `count_to_inf` stays off in promoted policies because `to_inf_tail` is still treated as deferred.",
        "",
        "## Policy Table",
        "",
        "| Policy | compress_be_chain | count_trailing_vbg | count_to_inf | count_after_discourse_marker | Notes |",
        "|---|---|---|---|---|---|",
    ]

    for policy in report["policy_table"]:
        lines.append(
            f"| `{policy['name']}` | `{policy['compress_be_chain']}` | `{policy['count_trailing_vbg']}` | "
            f"`{policy['count_to_inf']}` | `{policy['count_after_discourse_marker']}` | {policy['description']} |"
        )

    lines.extend(
        [
            "",
            "## Bundle Summary",
            "",
            "| Policy | Primary focus exact | Focus exact | Focus worsened | Defer exact | Protected changed |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )

    for policy in report["policy_table"]:
        summary = report["policy_summary"][policy["name"]]["bundle_summary"]
        lines.append(
            f"| `{policy['name']}` | `{summary['primary_focus_bundle']['exact_rate']}` | "
            f"`{summary['focus_bundle']['exact_rate']}` | `{summary['focus_bundle']['worse_count']}` | "
            f"`{summary['defer_bundle']['exact_rate']}` | "
            f"`{summary['protected_stuffed_exact']['row_count'] - summary['protected_stuffed_exact']['same_count']}` |"
        )

    lines.extend(
        [
            "",
            "## Primary Focus Subclusters",
            "",
            "| Subcluster | compressed_only | compressed_plus_trailing_vbg | compressed_plus_discourse_followup |",
            "|---|---:|---:|---:|",
        ]
    )

    primary_keys = sorted(
        report["policy_summary"]["compressed_only"]["primary_subclusters"].keys()
    )
    for key in primary_keys:
        compressed_only = report["policy_summary"]["compressed_only"]["primary_subclusters"][key]
        trailing = report["policy_summary"]["compressed_plus_trailing_vbg"]["primary_subclusters"][key]
        discourse = report["policy_summary"]["compressed_plus_discourse_followup"]["primary_subclusters"][key]
        lines.append(
            f"| `{key}` | `{compressed_only['exact_rate']}` | `{trailing['exact_rate']}` | `{discourse['exact_rate']}` |"
        )

    gate = report["anchor_vs_candidate_gate"]
    lines.extend(
        [
            "",
            "## Promotion Gate",
            "",
            f"- Primary focus exact gain vs current spike: `{gate['primary_focus_exact_gain']}`",
            f"- Focus bundle exact gain vs current spike: `{gate['focus_bundle_exact_gain']}`",
            f"- Focus bundle worsened delta vs current spike: `{gate['focus_bundle_worse_delta']}`",
            f"- Changed defer rows vs current spike: `{gate['changed_defer_rows_vs_anchor']}`",
            f"- Changed protected rows vs current spike: `{gate['changed_protected_rows_vs_anchor']}`",
            f"- Decision: `{gate['decision']}`",
            "",
            "## Sample Recoveries",
            "",
        ]
    )

    for row in gate["sample_primary_recoveries_vs_anchor"][:6]:
        lines.append(
            f"- `{row['cluster']}:{row['subcluster']}` gold=`{row['gold']}` anchor=`{row['anchor_count']}` "
            f"candidate=`{row['candidate_count']}` :: {row['sentence']}"
        )

    if not gate["sample_primary_recoveries_vs_anchor"]:
        lines.append("- No primary-focus recoveries found.")

    return "\n".join(lines) + "\n"


def main() -> None:
    report = build_report()
    json_path = WORK_LOGS_DIR / "task2_dependency_policy_table_spike_2026-04-01.json"
    md_path = WORK_LOGS_DIR / "task2_dependency_policy_table_spike_2026-04-01.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
