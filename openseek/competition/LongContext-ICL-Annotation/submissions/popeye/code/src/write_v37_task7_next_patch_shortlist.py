import json
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
OUTPUTS_DIR = PROJECT_DIR / "outputs"
WORK_LOGS_DIR = OUTPUTS_DIR / "work_logs"
DATA_PATH = PROJECT_DIR / "data" / "openseek-7_jeopardy_answer_generation_all.json"

DIRECT_AUDIT_PATH = WORK_LOGS_DIR / "task7_v35_manual_review_direct_audit_2026-04-09.json"
OUTPUT_JSON = WORK_LOGS_DIR / "task7_v37_next_patch_shortlist_2026-04-10.json"
OUTPUT_MD = WORK_LOGS_DIR / "task7_v37_next_patch_shortlist_2026-04-10.md"

SHORTLIST_SPECS = [
    {
        "id": "openseek-7-15c1b4864ed44c79bb3bd631f660d7f7",
        "priority": 1,
        "reason": "两个候选都像错答，且 direct audit 打平，适合做 clue-level 事实核对。",
    },
    {
        "id": "openseek-7-6630eea0f27445b5b4bd38a0a204c5c2",
        "priority": 1,
        "reason": "两个候选打平且都是高频出现，优先核对真实地名，不建议继续依赖 judge。",
    },
    {
        "id": "openseek-7-4530b749bd964f81ac0d4263fa546da0",
        "priority": 2,
        "reason": "prefer_new 但强度不高，且 clue 指向具体喜剧演员，可能存在第三个更正确答案。",
    },
    {
        "id": "openseek-7-f36b99c93bd949b29e6855412a918f3d",
        "priority": 2,
        "reason": "新答案虽然更常出现，但 clue 表面看仍可能指向更经典答案，值得人工核题。",
    },
    {
        "id": "openseek-7-bc0b7dbb83eb4abcaa447bfc6e13e29c",
        "priority": 3,
        "reason": "偏向 canonicality 选择题，收益可能小于纯 factual patch，放在第二梯队。",
    },
    {
        "id": "openseek-7-8da173db613a4fa895adc022789f5631",
        "priority": 3,
        "reason": "新答案更具体，但是否符合 Jeopardy canonical answer 仍需核对。",
    },
    {
        "id": "openseek-7-34bd86c0bffd49ce86d2e8267b023d92",
        "priority": 3,
        "reason": "新答案优势明显，但如果继续做极窄 patch，可作为低风险补充核对项。",
    },
    {
        "id": "openseek-7-5add39a8fc764f919103e029301a9486",
        "priority": 3,
        "reason": "新答案略占优，且更像常识 canonical form，可留作后手而非首批主攻。",
    },
]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_report() -> dict:
    audit = load_json(DIRECT_AUDIT_PATH)
    task7 = load_json(DATA_PATH)
    rows_by_id = {row["id"]: row for row in task7["test_samples"]}
    audit_rows_by_id = {row["id"]: row for row in audit["rows"]}

    shortlist_rows = []
    for spec in SHORTLIST_SPECS:
        audit_row = audit_rows_by_id[spec["id"]]
        shortlist_rows.append(
            {
                "id": spec["id"],
                "priority": spec["priority"],
                "reason": spec["reason"],
                "input": rows_by_id[spec["id"]]["input"],
                "current_prediction_v36": audit_row["new_prediction"],
                "legacy_prediction_v30": audit_row["old_prediction"],
                "old_total": audit_row["old_total"],
                "new_total": audit_row["new_total"],
                "verdict": audit_row["verdict"],
            }
        )

    shortlist_rows.sort(key=lambda row: (row["priority"], row["id"]))
    return {
        "generated_on": "2026-04-10",
        "base_version": "v36",
        "source_audit": str(DIRECT_AUDIT_PATH),
        "shortlist_count": len(shortlist_rows),
        "rows": shortlist_rows,
        "immediate_action": "先核对 priority=1 的 2 行真实答案；若两行里至少一行可形成高置信修补，再构建 v37 ultra-narrow candidate。",
        "why_this_shortlist": "这些行要么 direct audit 打平，要么虽然偏向现答案但仍像存在第三个更正确答案，最适合继续做小步提分。",
    }


def render_markdown(report: dict) -> str:
    lines = [
        "# Task7 v37 Next Patch Shortlist",
        "",
        f"- Base version: `{report['base_version']}`",
        f"- Shortlist count: `{report['shortlist_count']}`",
        f"- Source audit: `{report['source_audit']}`",
        "",
        "## Rows",
        "",
    ]
    for row in report["rows"]:
        lines.extend(
            [
                f"- Priority `{row['priority']}`: `{row['id']}`",
                f"  Current `v36`: `{row['current_prediction_v36']}` | Legacy `v30`: `{row['legacy_prediction_v30']}` | direct audit `{row['verdict']}` ({row['old_total']} vs {row['new_total']})",
                f"  Reason: {row['reason']}",
                f"  Input: {row['input']}",
            ]
        )
    lines.extend(
        [
            "",
            "## Next Action",
            "",
            f"- {report['immediate_action']}",
            f"- {report['why_this_shortlist']}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    WORK_LOGS_DIR.mkdir(parents=True, exist_ok=True)
    report = build_report()
    OUTPUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    OUTPUT_MD.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
