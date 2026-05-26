import json
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
OUTPUTS_DIR = PROJECT_DIR / "outputs"
WORK_LOGS_DIR = OUTPUTS_DIR / "work_logs"

PACKAGE_DIR = OUTPUTS_DIR / "final_submission_v47_task7fact3_on_v46_candidate"
TASK7_PATH = PACKAGE_DIR / "openseek-7-v1.jsonl"
MERGE_SUMMARY_PATH = PACKAGE_DIR / "merge_summary.json"
OUTPUT_JSON = WORK_LOGS_DIR / "task7_v47_candidate_smoke_2026-04-11.json"
OUTPUT_MD = WORK_LOGS_DIR / "task7_v47_candidate_smoke_2026-04-11.md"

EXPECTED_ROWS = {
    "openseek-7-3deb283e13ec4b5393b1f1d61b426a4e": "Joseph Conrad",
    "openseek-7-88c5e426f6b84e959de64d77e0862297": "John Dos Passos",
    "openseek-7-02e748b0da7f4492864df5ab8b804556": "jack london",
    "openseek-7-220c54dc2fa44178a8b1e881e9ee6fb9": "selenium",
    "openseek-7-63f7eeaee4f4454ea81d62622f2d773a": "petrograd",
    "openseek-7-7b7c13a6f01b4f96997a82f3725258d6": "Victoria",
    "openseek-7-34bd86c0bffd49ce86d2e8267b023d92": "Louisa",
    "openseek-7-8da173db613a4fa895adc022789f5631": "turban",
    "openseek-7-bc0b7dbb83eb4abcaa447bfc6e13e29c": "sodden",
    "openseek-7-01e4fbfca16345b898064b73d9c18659": "Singapore",
    "openseek-7-028b6b04473c44619853d8b5d756fb52": "Franny and Zooey",
    "openseek-7-0486bc59dedb44bf83d1f9ec839f0519": "Ronald Reagan",
    "openseek-7-1559156235ef401eb07e42d29109c354": "Shannon Faulkner",
    "openseek-7-1c33199ee21944bfbfaa8f8dea325a94": "Pathfinder",
    "openseek-7-63dabfe75b1a4025b8f19594752dd535": "Ted Williams",
    "openseek-7-7d754cc1356b49b1ac056a2dbd0a4ea8": "rooster",
    "openseek-7-8095cf4b70cb4ca89fbf42cbe46fc299": "Los Angeles Lakers",
    "openseek-7-7f0fa3773c584ae5b2aeedce59aa9738": "nunnery",
    "openseek-7-842fcd84ee9f4da7ac26c1d1b30fb49b": "It's",
    "openseek-7-fec7d931d0e5439d945d7fdb80a9891f": "E!",
    "openseek-7-15b724b610df44f4a67fb2b2a5ae1ec8": "vanity",
    "openseek-7-905a9dee96e94072b498af8a25e8bbc5": "Tommy",
    "openseek-7-97672493f5d343c48a90deab581b7e20": "A Midsummer Night's Dream",
    "openseek-7-4c2884a96512479faae9bf88c7a5f076": "Manuel Noriega",
    "openseek-7-4f1226451a7e4c2c9c18bb6edd43348b": "Oprah Winfrey",
    "openseek-7-9a88597ca11340669b68dcb70c51f103": "Laurence Olivier",
    "openseek-7-28b5b49b3c87473bb8026d2b71f63ee6": "Toulouse-Lautrec",
    "openseek-7-2c42773c25b64b7d8dccf7cd34082c20": "\"Fast Eddie\" Felson",
    "openseek-7-41898a65a136407eaca7cbd07a83e384": "George C. Scott",
}


def load_jsonl_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_report() -> dict:
    rows = load_jsonl_rows(TASK7_PATH)
    merge_summary = json.loads(MERGE_SUMMARY_PATH.read_text(encoding="utf-8"))
    ids = [row["test_sample_id"] for row in rows]
    predictions = {row["test_sample_id"]: row["prediction"] for row in rows}
    return {
        "generated_on": "2026-04-11",
        "package_dir": str(PACKAGE_DIR),
        "zip_path": str(PACKAGE_DIR / "final_submission_v47_task7fact3_on_v46_candidate.zip"),
        "line_count": len(rows),
        "unique_id_count": len(set(ids)),
        "duplicate_ids": sorted({sample_id for sample_id in ids if ids.count(sample_id) > 1}),
        "spot_checks": {
            sample_id: {
                "expected": expected,
                "actual": predictions.get(sample_id),
                "matched": predictions.get(sample_id) == expected,
            }
            for sample_id, expected in EXPECTED_ROWS.items()
        },
        "merge_summary": {
            "base": merge_summary.get("base"),
            "replaced_tasks": merge_summary.get("replaced_tasks"),
            "task7_test_change_count_vs_v46": merge_summary.get("task7_test_change_count_vs_v46"),
        },
        "ready_for_external_scoring": (
            len(rows) == 500
            and len(set(ids)) == 500
            and all(predictions.get(sample_id) == expected for sample_id, expected in EXPECTED_ROWS.items())
            and merge_summary.get("base") == "v46"
            and merge_summary.get("replaced_tasks")
            == {"7": "outputs/task7_v47_fact3_candidate/openseek-7-v1.jsonl"}
        ),
    }


def render_markdown(report: dict) -> str:
    lines = [
        "# Task7 v47 Candidate Smoke",
        "",
        f"- Package dir: `{report['package_dir']}`",
        f"- Zip path: `{report['zip_path']}`",
        f"- Task7 line count: `{report['line_count']}`",
        f"- Task7 unique id count: `{report['unique_id_count']}`",
        f"- Ready for external scoring: `{report['ready_for_external_scoring']}`",
        "",
        "## Spot checks",
        "",
    ]
    for sample_id, check in report["spot_checks"].items():
        lines.append(
            f"- `{sample_id}`: expected `{check['expected']}` | actual `{check['actual']}` | matched `{check['matched']}`"
        )
    lines.extend(
        [
            "",
            "## Package checks",
            "",
            f"- merge base: `{report['merge_summary']['base']}`",
            f"- replaced tasks: `{report['merge_summary']['replaced_tasks']}`",
            f"- changed rows vs v46: `{report['merge_summary']['task7_test_change_count_vs_v46']}`",
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
