import json
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
OUTPUTS_DIR = PROJECT_DIR / "outputs"
WORK_LOGS_DIR = OUTPUTS_DIR / "work_logs"

GENERATED_ON = "2026-04-09"
OUTPUT_STEM = "official_score_ledger_2026-04-09"
STABLE_OFFICIAL_BASELINE = "v48"

VERSION_SPECS = [
    {
        "version": "v10",
        "candidate_dir": "final_submission_v10_task5longchat_candidate",
        "base_version": "v7",
        "official_score": 71.08,
        "offline_evidence_files": [
            "outputs/holdout_eval/holdout_task5_task7_focus_seed42.json",
            "outputs/holdout_eval/holdout_task5_task7_focus_seed43.json",
            "outputs/holdout_eval/holdout_task5_task7_focus_seed44.json",
        ],
        "offline_claimed_direction": "positive",
        "notes": [
            "Only the long chat context task5 candidate was promoted into the formal package lineage.",
            "This became the pre-v18 stable base despite later task5 variants not converting online.",
        ],
    },
    {
        "version": "v15",
        "candidate_dir": "final_submission_v15_task6chat_candidate",
        "base_version": "v10",
        "official_score": 70.18,
        "offline_evidence_files": [
            "outputs/holdout_eval/task6_chat_seed123_n40_v15.json",
            "outputs/holdout_eval/task6_chatlong_seed123_n40_v15.json",
        ],
        "offline_claimed_direction": "positive",
        "notes": [
            "Task6 chat looked better locally, but online it underperformed the v10 base badly.",
        ],
    },
    {
        "version": "v17",
        "candidate_dir": "final_submission_v17_task5guided_candidate",
        "base_version": "v10",
        "official_score": 70.98,
        "offline_evidence_files": [
            "outputs/holdout_eval/task5_profiles_seed42_guided.json",
            "outputs/holdout_eval/task5_profiles_seed43_guided.json",
            "outputs/holdout_eval/task5_profiles_seed44_guided.json",
            "outputs/holdout_eval/task5_profiles_seed45_guided.json",
        ],
        "offline_claimed_direction": "positive",
        "notes": [
            "Task5 guided had strong local deltas but still failed to beat v10 online.",
        ],
    },
    {
        "version": "v18",
        "candidate_dir": "final_submission_v18_task2structured_candidate",
        "base_version": "v10",
        "official_score": 72.55,
        "offline_evidence_files": [
            "outputs/work_logs/task2_structured_audit_2026-03-28.json",
            "outputs/holdout_eval/task2_structured_seed42.json",
            "outputs/holdout_eval/task2_structured_seed43.json",
            "outputs/holdout_eval/task2_structured_seed44.json",
            "outputs/holdout_eval/task2_structured_seed123_n40.json",
        ],
        "offline_claimed_direction": "positive",
        "notes": [
            "Task2 structured was the first recent method-level change that clearly transferred online.",
        ],
    },
    {
        "version": "v19",
        "candidate_dir": "final_submission_v19_task7rerank_candidate",
        "base_version": "v18",
        "official_score": 72.70,
        "offline_evidence_files": [
            "outputs/holdout_eval/task7_rerank_seed123_n40_none_n8_t09_metaoff.json",
            "outputs/holdout_eval/task7_rerank_seed2026_n40_none_n8_t09_recheck.json",
        ],
        "offline_claimed_direction": "positive",
        "notes": [
            "Task7 rerank converted online, but only as a modest gain on top of v18.",
        ],
    },
    {
        "version": "v20",
        "candidate_dir": "final_submission_v20_task6structuredhybrid_candidate",
        "base_version": "v19",
        "official_score": 72.53,
        "offline_evidence_files": [
            "outputs/holdout_eval/task6_hybrid_seed123_n40_t02_mx07.json",
            "outputs/holdout_eval/task6_hybrid_seed2026_n40_t02_mx07.json",
            "outputs/holdout_eval/task6_mixture_sweep_v1.json",
        ],
        "offline_claimed_direction": "positive",
        "notes": [
            "Task6 conservative hybrid stayed non-negative on two seeds offline but still lost online.",
        ],
    },
    {
        "version": "v21",
        "candidate_dir": "final_submission_v21_task7u7gate_candidate",
        "base_version": "v19",
        "official_score": 72.60,
        "offline_evidence_files": [
            "outputs/holdout_eval/task7_rerank_failure_report_v1.json",
            "outputs/holdout_eval/task7_rerank_seed123_n40_none_plus_longcontext_uniqueonly_u7_appendunique_j8_n8p4_t09.json",
            "outputs/holdout_eval/task7_rerank_seed2026_n40_none_plus_longcontext_uniqueonly_u7_appendunique_j8_n8p4_t09.json",
        ],
        "offline_claimed_direction": "positive",
        "notes": [
            "Task7 gated secondary had careful offline evidence but still failed to exceed v19 online.",
        ],
    },
    {
        "version": "v22",
        "candidate_dir": "final_submission_v22_task2verbrel_candidate",
        "base_version": "v19",
        "official_score": 72.93,
        "offline_evidence_files": [
            "outputs/work_logs/task2_probe_delta_audit_2026-03-29_verb_rel_or_have.json",
            "outputs/work_logs/task2_structured_audit_2026-03-29_v5.json",
            "outputs/work_logs/task2_rule_probe_2026-03-29_v3.json",
            "outputs/work_logs/task2_rule_probe_2026-03-29_v3_extended.json",
        ],
        "offline_claimed_direction": "positive",
        "notes": [
            "Task2 safer verb_rel_or_have refinement improved online on top of the v19 base.",
            "This was a clean single-task replacement and provides the strongest recent offline-to-online calibration evidence.",
        ],
    },
    {
        "version": "v23",
        "candidate_dir": "final_submission_v23_task2guarded_candidate",
        "base_version": "v22",
        "official_score": 73.00,
        "offline_evidence_files": [
            "outputs/work_logs/task2_probe_delta_audit_2026-03-29_verb_mistagged_guarded.json",
            "outputs/work_logs/task2_structured_audit_2026-03-29_v6_guarded.json",
            "outputs/work_logs/task2_rule_probe_2026-03-29_v5.json",
            "outputs/work_logs/task2_rule_probe_2026-03-29_v5_holdout200.json",
        ],
        "offline_claimed_direction": "positive",
        "notes": [
            "Task2 guarded mistagged-surface refinement improved online on top of the v22 base.",
            "This was another small, single-task replacement with only three changed task2 test rows, all from 0 to 1.",
        ],
    },
    {
        "version": "v25",
        "candidate_dir": "final_submission_v25_task7constraintsecondary_candidate",
        "base_version": "v23",
        "official_score": 73.03,
        "offline_evidence_files": [
            "outputs/final_submission_v25_task7constraintsecondary_candidate/merge_summary.json",
        ],
        "offline_claimed_direction": "positive",
        "notes": [
            "Task7 constraint-secondary beat v23 online by a narrow margin and became the temporary stable base before the later task7 regression.",
        ],
    },
    {
        "version": "v26",
        "candidate_dir": "final_submission_v26_task7typednarrowv3_candidate",
        "base_version": "v25",
        "official_score": 72.93,
        "offline_evidence_files": [
            "docs/phase-summaries/2026-03-30-v26-task7-online-readout.md",
        ],
        "offline_claimed_direction": "positive",
        "notes": [
            "Task7 typed narrow v3 failed to hold the v25 gain online and regressed below the stable base.",
        ],
    },
    {
        "version": "v27",
        "candidate_dir": "final_submission_v27_task2stuffed_on_v25_candidate",
        "base_version": "v25",
        "official_score": 73.05,
        "offline_evidence_files": [
            "outputs/work_logs/task2_probe_delta_audit_2026-03-30_verb_participle_stuffed_exact_vs_v23.json",
            "outputs/work_logs/task2_rule_probe_2026-03-30_stuffed_vs_v23.json",
            "outputs/final_submission_v27_task2stuffed_on_v25_candidate/merge_summary.json",
        ],
        "offline_claimed_direction": "positive",
        "notes": [
            "Task2 stuffed_exact improved online on top of v25.",
            "This was a tiny single-task replacement with one changed task2 test row, and it restored task2 as the latest stable official mainline.",
        ],
    },
    {
        "version": "v28",
        "candidate_dir": "final_submission_v28_task2dependency_discourse_candidate",
        "base_version": "v27",
        "official_score": 71.68,
        "offline_evidence_files": [
            "outputs/work_logs/task2_dependency_policy_table_spike_2026-04-01.md",
            "outputs/work_logs/task2_dependency_v28_failure_audit_2026-04-01.md",
        ],
        "offline_claimed_direction": "positive",
        "notes": [
            "The broad dependency-aware task2 replacement looked positive on the focus bundle offline but failed badly online.",
            "The failure audit showed 93 changed task2 test rows, with 91 outside the intended target family, so the line was closed for promotion.",
        ],
    },
    {
        "version": "v29",
        "candidate_dir": "final_submission_v29_task2dependency_relplain_candidate",
        "base_version": "v27",
        "official_score": 73.03,
        "offline_evidence_files": [
            "outputs/final_submission_v29_task2dependency_relplain_candidate/merge_summary.json",
            "docs/phase-summaries/2026-04-01-task2-dependency-line-closeout.md",
        ],
        "offline_claimed_direction": "positive",
        "notes": [
            "The ultra-narrow dependency-aware recovery candidate only changed one rel_be_vbg:plain task2 row.",
            "It remained near-safe but still underperformed v27, confirming that the dependency-aware line should stay research-only.",
        ],
    },
    {
        "version": "v30",
        "candidate_dir": "final_submission_v30_task8clean2_on_v27_candidate",
        "base_version": "v27",
        "official_score": 73.05,
        "offline_evidence_files": [
            "outputs/work_logs/task8_v30_candidate_build_2026-04-01.md",
            "outputs/final_submission_v30_task8clean2_on_v27_candidate/merge_summary.json",
        ],
        "offline_claimed_direction": "positive",
        "notes": [
            "The v30 candidate carried forward the 2-row audited-clean task8 subset from v14 onto the v27 base.",
            "It matched v27 online, confirming task8 clean maintenance patches can be safe without creating new total-score lift.",
        ],
    },
    {
        "version": "v34",
        "candidate_dir": "final_submission_v34_task7authorproj_stability_on_v30_candidate",
        "base_version": "v30",
        "official_score": 73.18,
        "offline_evidence_files": [
            "outputs/work_logs/task7_author_projection_stability_gate_validation_summary_2026-04-09.json",
            "outputs/work_logs/task7_author_projection_spillover_probe_refresh_default_seed2026_stability_gate_36rows_2026-04-09.json",
            "outputs/work_logs/task7_v34_authorproj_stability_candidate_promotion_readout_2026-04-09.json",
            "outputs/final_submission_v34_task7authorproj_stability_on_v30_candidate/merge_summary.json",
        ],
        "offline_claimed_direction": "positive",
        "notes": [
            "Task7 author projection stability gate converted online on top of v30 and became the new best official package.",
            "The promoted package preserved the known target-side rescues, improved the 36-row calibration spillover signal from 6/7 down to 4, and beat both v27 and v30 online.",
        ],
    },
    {
        "version": "v35",
        "candidate_dir": "final_submission_v35_task7authorproj_revert69_on_v34_candidate",
        "base_version": "v34",
        "official_score": 73.45,
        "offline_evidence_files": [
            "outputs/work_logs/task7_v34_pairwise_revert_audit_full_2026-04-09.json",
            "outputs/work_logs/task7_v35_authorproj_revert69_candidate_build_2026-04-09.json",
            "outputs/final_submission_v35_task7authorproj_revert69_on_v34_candidate/merge_summary.json",
        ],
        "offline_claimed_direction": "positive",
        "notes": [
            "Task7 post-v34 revert69 cleanup improved online on top of v34 and became the new best official package.",
            "The promoted package kept the stability-gate target rescues while reverting only semantic-jump rows whose v30 answer beat the v34 answer 2/2 in pairwise audit.",
        ],
    },
    {
        "version": "v36",
        "candidate_dir": "final_submission_v36_task7manualfix6_on_v35_candidate",
        "base_version": "v35",
        "official_score": 73.60,
        "offline_evidence_files": [
            "outputs/work_logs/task7_v35_manual_review_direct_audit_2026-04-09.json",
            "outputs/work_logs/task7_v36_manualfix6_candidate_build_2026-04-09.json",
            "outputs/final_submission_v36_task7manualfix6_on_v35_candidate/merge_summary.json",
        ],
        "offline_claimed_direction": "positive",
        "notes": [
            "Task7 post-v35 manual fact-fix6 cleanup improved online on top of v35 and became the then-best official package.",
            "The promoted package kept the stability-gate and revert69 gains, then added six ultra-narrow clue-level factual fixes instead of reopening broad semantic churn.",
        ],
    },
    {
        "version": "v37",
        "candidate_dir": "final_submission_v37_task7manualfix4_on_v36_candidate",
        "base_version": "v36",
        "official_score": 73.70,
        "offline_evidence_files": [
            "outputs/work_logs/task7_v37_verified_patch_notes_2026-04-10.json",
            "outputs/work_logs/task7_v37_manualfix4_candidate_build_2026-04-10.json",
            "outputs/final_submission_v37_task7manualfix4_on_v36_candidate/merge_summary.json",
        ],
        "offline_claimed_direction": "positive",
        "notes": [
            "Task7 post-v36 manual fact-fix4 cleanup improved online on top of v36 and became the new best official package.",
            "The promoted package kept the v36 line intact and added four more clue-level factual fixes drawn from the unresolved shortlist.",
        ],
    },
    {
        "version": "v38",
        "candidate_dir": "final_submission_v38_task7manualfix3_on_v37_candidate",
        "base_version": "v37",
        "official_score": 73.78,
        "offline_evidence_files": [
            "outputs/work_logs/task7_v38_verified_patch_notes_2026-04-10.json",
            "outputs/work_logs/task7_v38_manualfix3_candidate_build_2026-04-10.json",
            "outputs/work_logs/task7_v38_submission_ready_check_2026-04-10.json",
            "outputs/final_submission_v38_task7manualfix3_on_v37_candidate/merge_summary.json",
        ],
        "offline_claimed_direction": "positive",
        "notes": [
            "Task7 post-v37 manual fact-fix3 cleanup improved online on top of v37 and reached a new best official score of 73.78.",
            "This package kept the v37 line intact and added three verified clue-level corrections: Louisa, turban, and sodden.",
        ],
    },
    {
        "version": "v39",
        "candidate_dir": "final_submission_v39_task7typofix3_on_v37_candidate",
        "base_version": "v37",
        "official_score": 73.78,
        "offline_evidence_files": [
            "outputs/work_logs/task7_v39_verified_patch_notes_2026-04-10.json",
            "outputs/work_logs/task7_v39_typofix3_candidate_build_2026-04-10.json",
            "outputs/work_logs/task7_v39_candidate_smoke_2026-04-10.json",
            "outputs/final_submission_v39_task7typofix3_on_v37_candidate/merge_summary.json",
        ],
        "offline_claimed_direction": "positive",
        "notes": [
            "Task7 post-v37 typo-fix3 cleanup matched v38 online at 73.78 and tied the best official score.",
            "Because it reaches the same online score with a narrower typo/canonical patch surface, v39 becomes the default stable official baseline for subsequent Task7 branching.",
        ],
    },
    {
        "version": "v40",
        "candidate_dir": "final_submission_v40_task7combo6_on_v39_candidate",
        "base_version": "v39",
        "official_score": 73.85,
        "offline_evidence_files": [
            "outputs/work_logs/task7_v40_combo6_candidate_build_2026-04-10.json",
            "outputs/work_logs/task7_v40_candidate_smoke_2026-04-10.json",
            "outputs/final_submission_v40_task7combo6_on_v39_candidate/merge_summary.json",
        ],
        "offline_claimed_direction": "positive",
        "notes": [
            "Task7 combo6 successfully stacked the two independently positive 73.78 lines and improved online to 73.85.",
            "This package keeps the v39 typo/canonical fixes and reintroduces the three v38 factual fixes, making v40 the new best official package and stable baseline.",
        ],
    },
    {
        "version": "v42",
        "candidate_dir": "final_submission_v42_task7fact4_on_v41_candidate",
        "base_version": "v40",
        "official_score": 73.98,
        "offline_evidence_files": [
            "outputs/work_logs/task7_v42_verified_patch_notes_2026-04-10.json",
            "outputs/work_logs/task7_v42_fact4_candidate_build_2026-04-10.json",
            "outputs/work_logs/task7_v42_candidate_smoke_2026-04-10.json",
            "outputs/final_submission_v42_task7fact4_on_v41_candidate/merge_summary.json",
        ],
        "offline_claimed_direction": "positive",
        "notes": [
            "Task7 fact4 successfully extended the post-v40 narrow factual patch line and improved online to 73.98.",
            "This package kept the entire v40/v41 gain stack intact, added four externally verified factual repairs, and becomes the new best official package and stable baseline.",
        ],
    },
    {
        "version": "v43",
        "candidate_dir": "final_submission_v43_task7fact3_on_v42_candidate",
        "base_version": "v42",
        "official_score": 74.05,
        "offline_evidence_files": [
            "outputs/work_logs/task7_v43_verified_patch_notes_2026-04-10.json",
            "outputs/work_logs/task7_v43_fact3_candidate_build_2026-04-10.json",
            "outputs/work_logs/task7_v43_candidate_smoke_2026-04-10.json",
            "outputs/final_submission_v43_task7fact3_on_v42_candidate/merge_summary.json",
        ],
        "offline_claimed_direction": "positive",
        "notes": [
            "Task7 fact3 successfully extended the post-v42 narrow factual patch line and improved online to 74.05.",
            "This package kept the full v42 gain stack intact, added three externally verified factual repairs, and becomes the new best official package and stable baseline.",
        ],
    },
    {
        "version": "v44",
        "candidate_dir": "final_submission_v44_task7fact3_on_v43_candidate",
        "base_version": "v43",
        "official_score": 74.10,
        "offline_evidence_files": [
            "outputs/work_logs/task7_v44_verified_patch_notes_2026-04-11.json",
            "outputs/work_logs/task7_v44_fact3_candidate_build_2026-04-11.json",
            "outputs/work_logs/task7_v44_candidate_smoke_2026-04-11.json",
            "outputs/final_submission_v44_task7fact3_on_v43_candidate/merge_summary.json",
        ],
        "offline_claimed_direction": "positive",
        "notes": [
            "Task7 fact3 successfully extended the post-v43 short-clue factual patch line and improved online to 74.10.",
            "This package kept the full v43 gain stack intact, added three externally verified short-clue repairs, and becomes the new best official package and stable baseline.",
        ],
    },
    {
        "version": "v45",
        "candidate_dir": "final_submission_v45_task7fact3_on_v44_candidate",
        "base_version": "v44",
        "official_score": 74.15,
        "offline_evidence_files": [
            "outputs/work_logs/task7_v45_verified_patch_notes_2026-04-11.json",
            "outputs/work_logs/task7_v45_fact3_candidate_build_2026-04-11.json",
            "outputs/work_logs/task7_v45_candidate_smoke_2026-04-11.json",
            "outputs/final_submission_v45_task7fact3_on_v44_candidate/merge_summary.json",
        ],
        "offline_claimed_direction": "positive",
        "notes": [
            "Task7 fact3 successfully extended the post-v44 ultra-narrow factual patch line and improved online to 74.15.",
            "This package kept the full v44 gain stack intact, added three externally verified clue-level repairs, and becomes the new best official package and stable baseline.",
        ],
    },
    {
        "version": "v46",
        "candidate_dir": "final_submission_v46_task7fact3_on_v45_candidate",
        "base_version": "v45",
        "official_score": 74.23,
        "offline_evidence_files": [
            "outputs/work_logs/task7_v46_verified_patch_notes_2026-04-11.json",
            "outputs/work_logs/task7_v46_fact3_candidate_build_2026-04-11.json",
            "outputs/work_logs/task7_v46_candidate_smoke_2026-04-11.json",
            "outputs/final_submission_v46_task7fact3_on_v45_candidate/merge_summary.json",
        ],
        "offline_claimed_direction": "positive",
        "notes": [
            "Task7 fact3 successfully extended the post-v45 ultra-narrow factual patch line and improved online to 74.23.",
            "This package kept the full v45 gain stack intact, added three externally verified clue-level repairs, and becomes the new best official package and stable baseline.",
        ],
    },
    {
        "version": "v47",
        "candidate_dir": "final_submission_v47_task7fact3_on_v46_candidate",
        "base_version": "v46",
        "official_score": 74.30,
        "offline_evidence_files": [
            "outputs/work_logs/task7_v47_verified_patch_notes_2026-04-11.json",
            "outputs/work_logs/task7_v47_fact3_candidate_build_2026-04-11.json",
            "outputs/work_logs/task7_v47_candidate_smoke_2026-04-11.json",
            "outputs/final_submission_v47_task7fact3_on_v46_candidate/merge_summary.json",
        ],
        "offline_claimed_direction": "positive",
        "notes": [
            "Task7 fact3 successfully extended the post-v46 ultra-narrow factual patch line and improved online to 74.30.",
            "This package kept the full v46 gain stack intact, added three externally verified clue-level repairs, and becomes the new best official package and stable baseline.",
        ],
    },
    {
        "version": "v48",
        "candidate_dir": "final_submission_v48_task7fact2_on_v47_candidate",
        "base_version": "v47",
        "official_score": 74.35,
        "offline_evidence_files": [
            "outputs/work_logs/task7_v48_verified_patch_notes_2026-04-11.json",
            "outputs/work_logs/task7_v48_fact2_candidate_build_2026-04-11.json",
            "outputs/work_logs/task7_v48_candidate_smoke_2026-04-11.json",
            "outputs/final_submission_v48_task7fact2_on_v47_candidate/merge_summary.json",
        ],
        "offline_claimed_direction": "positive",
        "notes": [
            "Task7 fact2 successfully extended the post-v47 ultra-narrow factual patch line and improved online to 74.35.",
            "This package kept the full v47 gain stack intact, added two externally verified clue-level repairs, and becomes the new best official package and stable baseline.",
        ],
    },
]


def infer_online_direction(delta: float) -> str:
    if delta > 0:
        return "positive"
    if delta < 0:
        return "negative"
    return "neutral"


def main():
    version_to_score = {spec["version"]: spec["official_score"] for spec in VERSION_SPECS}
    ledger = {
        "generated_on": GENERATED_ON,
        "stable_official_baseline": STABLE_OFFICIAL_BASELINE,
        "entries": [],
    }

    for spec in VERSION_SPECS:
        candidate_dir = OUTPUTS_DIR / spec["candidate_dir"]
        merge_summary_path = candidate_dir / "merge_summary.json"
        merge_summary = json.loads(merge_summary_path.read_text(encoding="utf-8")) if merge_summary_path.exists() else {}
        base_version = spec["base_version"]
        base_score = version_to_score.get(base_version)
        delta = round(spec["official_score"] - base_score, 2) if base_score is not None else None
        entry = {
            "version": spec["version"],
            "candidate_dir": str(candidate_dir),
            "merge_summary_path": str(merge_summary_path) if merge_summary_path.exists() else None,
            "base_version": base_version,
            "replaced_tasks": merge_summary.get("replaced_tasks"),
            "official_score": spec["official_score"],
            "official_delta_vs_base": delta,
            "offline_evidence_files": spec["offline_evidence_files"],
            "offline_claimed_direction": spec["offline_claimed_direction"],
            "actual_online_direction": infer_online_direction(delta or 0.0),
            "notes": spec["notes"],
        }
        ledger["entries"].append(entry)

    output_path = WORK_LOGS_DIR / f"{OUTPUT_STEM}.json"
    md_path = WORK_LOGS_DIR / f"{OUTPUT_STEM}.md"
    output_path.write_text(json.dumps(ledger, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(ledger), encoding="utf-8")
    print(json.dumps(ledger, ensure_ascii=False, indent=2))


def render_markdown(ledger: dict) -> str:
    lines = [
        "# Official Score Ledger",
        "",
        f"- Generated on: `{ledger['generated_on']}`",
        f"- Stable official baseline: `{ledger['stable_official_baseline']}`",
        "",
        "## Version Lineage",
        "",
        "| Version | Base | Replaced Tasks | Official Score | Delta vs Base | Online Direction |",
        "|---|---|---|---:|---:|---|",
    ]
    for entry in ledger["entries"]:
        replaced = entry["replaced_tasks"] or {}
        replaced_text = ", ".join(sorted(replaced.keys())) if replaced else "-"
        delta = entry["official_delta_vs_base"]
        delta_text = "-" if delta is None else str(delta)
        lines.append(
            f"| `{entry['version']}` | `{entry['base_version']}` | `{replaced_text}` | "
            f"`{entry['official_score']}` | `{delta_text}` | `{entry['actual_online_direction']}` |"
        )
    lines.extend(["", "## Notes", ""])
    for entry in ledger["entries"]:
        lines.append(f"- `{entry['version']}`: {' '.join(entry['notes'])}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
