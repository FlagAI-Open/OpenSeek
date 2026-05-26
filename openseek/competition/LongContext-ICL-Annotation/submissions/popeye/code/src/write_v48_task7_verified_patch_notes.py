import json
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
OUTPUTS_DIR = PROJECT_DIR / "outputs"
WORK_LOGS_DIR = OUTPUTS_DIR / "work_logs"

OUTPUT_JSON = WORK_LOGS_DIR / "task7_v48_verified_patch_notes_2026-04-11.json"
OUTPUT_MD = WORK_LOGS_DIR / "task7_v48_verified_patch_notes_2026-04-11.md"


def build_report() -> dict:
    return {
        "generated_on": "2026-04-11",
        "base_version": "v47",
        "selection_rule": (
            "Stay on the official v47 baseline and add only a tiny set of externally "
            "verifiable clue-level repairs where the current v47 answer is still clearly off-target."
        ),
        "verified_rows": [
            {
                "id": "openseek-7-2a9b264297094737b16bb05fe2525040",
                "current_prediction_v47": "thurgood marshall",
                "patched_prediction_v48": "Alan Dershowitz",
                "confidence": "high",
                "support": [
                    {
                        "type": "britannica",
                        "source": "Britannica: Alan Dershowitz",
                        "url": "https://www.britannica.com/biography/Alan-Dershowitz",
                        "note": "Britannica identifies Alan Dershowitz as a lawyer and author who clerked for Supreme Court Justice Arthur J. Goldberg.",
                    },
                    {
                        "type": "wikipedia",
                        "source": "Wikipedia: Alan Dershowitz",
                        "url": "https://en.wikipedia.org/wiki/Alan_Dershowitz",
                        "note": "The biography states that Dershowitz clerked for Justice Arthur Goldberg from 1963 to 1964.",
                    },
                ],
            },
            {
                "id": "openseek-7-2ab281528bdb4f488c8861ce13ac0ca7",
                "current_prediction_v47": "the bobby parrish show",
                "patched_prediction_v48": "American Dreams",
                "confidence": "high",
                "support": [
                    {
                        "type": "reference",
                        "source": "Wikipedia: American Dreams",
                        "url": "https://en.wikipedia.org/wiki/American_Dreams",
                        "note": "The series centers on Meg Pryor, who dances on Dick Clark's American Bandstand.",
                    },
                    {
                        "type": "nbc_reference",
                        "source": "NBC Insider: Sarah Ramos Is Joining the Cast of Chicago Med",
                        "url": "https://www.nbc.com/nbc-insider/sarah-ramos-joining-chicago-med",
                        "note": "NBC describes American Dreams as the NBC drama built around the Pryor family, including Meg Pryor's Bandstand storyline.",
                    },
                ],
            },
        ],
        "next_action": "Build and smoke-check a v48 fact2 candidate on top of v47.",
    }


def render_markdown(report: dict) -> str:
    lines = [
        "# Task7 v48 Verified Patch Notes",
        "",
        f"- Base version: `{report['base_version']}`",
        f"- Selection rule: {report['selection_rule']}",
        "",
        "## Verified rows",
        "",
    ]
    for row in report["verified_rows"]:
        lines.append(
            f"- `{row['id']}`: `{row['current_prediction_v47']}` -> "
            f"`{row['patched_prediction_v48']}` (`{row['confidence']}`)"
        )
        for support in row["support"]:
            lines.append(f"  - {support['source']}: {support['note']} ({support['url']})")
    lines.extend(["", "## Next action", "", f"- {report['next_action']}", ""])
    return "\n".join(lines)


def main() -> None:
    WORK_LOGS_DIR.mkdir(parents=True, exist_ok=True)
    report = build_report()
    OUTPUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    OUTPUT_MD.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
