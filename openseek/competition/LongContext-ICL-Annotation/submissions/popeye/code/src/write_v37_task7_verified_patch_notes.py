import json
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
OUTPUTS_DIR = PROJECT_DIR / "outputs"
WORK_LOGS_DIR = OUTPUTS_DIR / "work_logs"

OUTPUT_JSON = WORK_LOGS_DIR / "task7_v37_verified_patch_notes_2026-04-10.json"
OUTPUT_MD = WORK_LOGS_DIR / "task7_v37_verified_patch_notes_2026-04-10.md"


def build_report() -> dict:
    return {
        "generated_on": "2026-04-10",
        "base_version": "v36",
        "verified_rows": [
            {
                "id": "openseek-7-15c1b4864ed44c79bb3bd631f660d7f7",
                "current_prediction_v36": "alex garland",
                "patched_prediction_v37": "Pedro Almodovar",
                "confidence": "high",
                "support": [
                    {
                        "type": "official_awards_reference",
                        "source": "Academy Awards database / Oscars.org",
                        "url": "https://awardsdatabase.oscars.org/Help/Statistics?file=For-OtherCats.pdf",
                        "note": "Talk to Her is listed with Original Screenplay written by Pedro Almodovar.",
                    }
                ],
            },
            {
                "id": "openseek-7-6630eea0f27445b5b4bd38a0a204c5c2",
                "current_prediction_v36": "spokane",
                "patched_prediction_v37": "Walla Walla",
                "confidence": "high_inference",
                "support": [
                    {
                        "type": "city_history",
                        "source": "City of Walla Walla",
                        "url": "https://www.wallawallawa.gov/our-city/history",
                        "note": "Walla Walla history references Fort Walla Walla and the city's southeast Washington location.",
                    },
                    {
                        "type": "history_reference",
                        "source": "HistoryLink",
                        "url": "https://www.historylink.org/File/9649",
                        "note": "Fort Walla Walla was established in 1856.",
                    },
                    {
                        "type": "category_inference",
                        "source": "Task clue structure",
                        "url": None,
                        "note": "DOUBLE TALK strongly implies a repeated-word city name; among southeast Washington cities, Walla Walla is the obvious fit.",
                    },
                ],
            },
            {
                "id": "openseek-7-4530b749bd964f81ac0d4263fa546da0",
                "current_prediction_v36": "steve martin",
                "patched_prediction_v37": "Foster Brooks",
                "confidence": "high",
                "support": [
                    {
                        "type": "biographical_reference",
                        "source": "Washington Post obituary",
                        "url": "https://www.washingtonpost.com/archive/local/2001/12/23/comedian-foster-brooks/2558610c-0c88-4bc6-9535-f711ef9523fa/",
                        "note": "Identifies Foster Brooks as a bearded comic known for the Lovable Lush act.",
                    },
                    {
                        "type": "biographical_reference",
                        "source": "Wikipedia",
                        "url": "https://en.wikipedia.org/wiki/Foster_Brooks",
                        "note": "States his signature routine was Foster Brooks, The Lovable Lush.",
                    },
                ],
            },
            {
                "id": "openseek-7-f36b99c93bd949b29e6855412a918f3d",
                "current_prediction_v36": "monopoly",
                "patched_prediction_v37": "dominoes",
                "confidence": "high",
                "support": [
                    {
                        "type": "clue_semantics",
                        "source": "Task clue structure",
                        "url": None,
                        "note": "In common game terminology, bones drawn from the boneyard are dominoes.",
                    }
                ],
            },
        ],
        "next_action": "Build and smoke-check a v37 ultra-narrow candidate on top of v36 using only these four verified rows.",
    }


def render_markdown(report: dict) -> str:
    lines = [
        "# Task7 v37 Verified Patch Notes",
        "",
        f"- Base version: `{report['base_version']}`",
        "",
        "## Verified rows",
        "",
    ]
    for row in report["verified_rows"]:
        lines.append(
            f"- `{row['id']}`: `{row['current_prediction_v36']}` -> `{row['patched_prediction_v37']}` "
            f"(`{row['confidence']}`)"
        )
        for support in row["support"]:
            if support["url"]:
                lines.append(
                    f"  - {support['source']}: {support['note']} ({support['url']})"
                )
            else:
                lines.append(f"  - {support['source']}: {support['note']}")
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
