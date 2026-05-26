import json
import os
from pathlib import Path

from main import TASK_FILES
from validate_task7_candidate_rerank import (
    build_branch_icl_pool,
    build_task7_prompt,
    collect_branch_choices,
    dedupe_candidates,
    normalize_qa_answer,
)


SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
OUTPUTS_DIR = PROJECT_DIR / "outputs"
WORK_LOGS_DIR = OUTPUTS_DIR / "work_logs"

PAIRWISE_AUDIT_PATH = WORK_LOGS_DIR / "task7_v34_pairwise_revert_audit_full_2026-04-09.json"
OUTPUT_JSON = WORK_LOGS_DIR / "task7_v35_manual_review_direct_audit_2026-04-09.json"
OUTPUT_MD = WORK_LOGS_DIR / "task7_v35_manual_review_direct_audit_2026-04-09.md"

PROFILE_NAME = "frontier_task6_task7"
RETRIEVAL_MODE = "none"
N_CANDIDATES = 8
TEMPERATURE = 0.9
TOP_P = 0.95
SEEDS = ["2027", "2028"]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_report() -> dict:
    task7 = load_json(Path(TASK_FILES[7]))
    task_description = task7["Definition"][0]
    examples = list(task7["examples"])
    test_by_id = {row["id"]: row for row in task7["test_samples"]}
    icl_pool, _ = build_branch_icl_pool(
        examples,
        holdout_ids=set(),
        profile_name=PROFILE_NAME,
        examples_limit=None,
    )

    pairwise_audit = load_json(PAIRWISE_AUDIT_PATH)
    manual_rows = [
        row for row in pairwise_audit["rows"] if row["pairwise"]["recommendation"] == "manual_review"
    ]

    audited_rows = []
    verdict_counts: dict[str, int] = {}
    old_lean_ids = []

    for row in manual_rows:
        sample = test_by_id[row["id"]]
        icl_examples = [example for example in icl_pool if example["id"] != row["id"]]
        prompt = build_task7_prompt(
            task_description,
            sample["input"],
            icl_examples,
            PROFILE_NAME,
            RETRIEVAL_MODE,
        )

        old_norm = normalize_qa_answer(row["old_prediction"])
        new_norm = normalize_qa_answer(row["new_prediction"])
        old_total = 0
        new_total = 0
        seed_reports = []

        for seed in SEEDS:
            os.environ["OPENSEEK_COMPLETION_SEED"] = seed
            _, raw_choices = collect_branch_choices(
                prompt,
                sample["input"],
                N_CANDIDATES,
                TEMPERATURE,
                TOP_P,
                False,
                "off",
            )
            deduped_candidates, counts = dedupe_candidates(raw_choices)
            old_count = counts.get(old_norm, 0)
            new_count = counts.get(new_norm, 0)
            old_total += old_count
            new_total += new_count
            seed_reports.append(
                {
                    "seed": seed,
                    "old_count": old_count,
                    "new_count": new_count,
                    "top_candidates": deduped_candidates[:5],
                }
            )

        verdict = "unclear"
        if new_total > old_total and old_total == 0:
            verdict = "lean_new"
        elif old_total > new_total and new_total == 0:
            verdict = "lean_old"
        elif new_total > old_total:
            verdict = "prefer_new"
        elif old_total > new_total:
            verdict = "prefer_old"

        audited_row = {
            "id": row["id"],
            "input": sample["input"],
            "old_prediction": row["old_prediction"],
            "new_prediction": row["new_prediction"],
            "old_total": old_total,
            "new_total": new_total,
            "verdict": verdict,
            "per_seed": seed_reports,
        }
        audited_rows.append(audited_row)
        verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
        if verdict in {"lean_old", "prefer_old"}:
            old_lean_ids.append(row["id"])

    return {
        "generated_on": "2026-04-09",
        "base_version": "v35",
        "source_pairwise_audit": str(PAIRWISE_AUDIT_PATH),
        "profile": PROFILE_NAME,
        "retrieval_mode": RETRIEVAL_MODE,
        "n_candidates": N_CANDIDATES,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "seeds": SEEDS,
        "manual_review_row_count": len(manual_rows),
        "verdict_counts": verdict_counts,
        "old_lean_ids": old_lean_ids,
        "rows": audited_rows,
        "next_step_hint": (
            "Most pairwise split rows still lean toward the v35 answer under direct generation. "
            "Only a very small old-lean subset should be considered for further factual patching."
        ),
    }


def render_markdown(report: dict) -> str:
    lines = [
        "# Task7 v35 Manual Review Direct Audit",
        "",
        f"- Manual-review row count: `{report['manual_review_row_count']}`",
        f"- Seeds: `{', '.join(report['seeds'])}`",
        "",
        "## Verdict counts",
        "",
    ]
    for key, value in sorted(report["verdict_counts"].items()):
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Old-lean shortlist",
            "",
        ]
    )
    for row in report["rows"]:
        if row["verdict"] not in {"lean_old", "prefer_old"}:
            continue
        lines.append(
            f"- `{row['id']}`: `{row['old_prediction']}` vs `{row['new_prediction']}` -> "
            f"`{row['verdict']}` (old `{row['old_total']}` / new `{row['new_total']}`)"
        )
    lines.extend(["", "## Next-step hint", "", f"- {report['next_step_hint']}", ""])
    return "\n".join(lines)


def main() -> None:
    WORK_LOGS_DIR.mkdir(parents=True, exist_ok=True)
    report = build_report()
    OUTPUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    OUTPUT_MD.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
