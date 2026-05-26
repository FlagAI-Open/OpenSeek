import json
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
OUTPUTS_DIR = PROJECT_DIR / "outputs"
WORK_LOGS_DIR = OUTPUTS_DIR / "work_logs"

OUTPUT_JSON = WORK_LOGS_DIR / "task7_v42_verified_patch_notes_2026-04-10.json"
OUTPUT_MD = WORK_LOGS_DIR / "task7_v42_verified_patch_notes_2026-04-10.md"


def build_report() -> dict:
    return {
        "generated_on": "2026-04-10",
        "base_version": "v41",
        "selection_rule": (
            "Stay on the already smoke-checked v41 branch and add only a very small set of "
            "externally verifiable factual repairs whose current v41 answers are still clearly wrong."
        ),
        "verified_rows": [
            {
                "id": "openseek-7-028b6b04473c44619853d8b5d756fb52",
                "current_prediction_v41": "franny and bernie",
                "patched_prediction_v42": "Franny and Zooey",
                "confidence": "high",
                "support": [
                    {
                        "type": "magazine_archive",
                        "source": "The New Yorker: Story Pairs",
                        "url": "https://www.newyorker.com/books/double-take/story-pairs-matching-the-summer-fiction-issue",
                        "note": "The New Yorker notes that Salinger's 'Franny' and 'Zooey' were originally published separately in the magazine.",
                    },
                    {
                        "type": "book_reference",
                        "source": "Encyclopaedia Britannica: Franny and Zooey",
                        "url": "https://www.britannica.com/topic/Franny-and-Zooey",
                        "note": "Britannica identifies Franny and Zooey as Salinger's 1961 work built from the paired stories.",
                    },
                ],
            },
            {
                "id": "openseek-7-0486bc59dedb44bf83d1f9ec839f0519",
                "current_prediction_v41": "it's morning again in america",
                "patched_prediction_v42": "Ronald Reagan",
                "confidence": "high",
                "support": [
                    {
                        "type": "historical_reference",
                        "source": "Hoover Institution: Morning Again in America",
                        "url": "https://www.hoover.org/research/morning-again-america",
                        "note": "Hoover identifies 'Morning Again in America' as Reagan's 1984 re-election campaign slogan.",
                    },
                    {
                        "type": "media_archive",
                        "source": "PBS American Experience: Reagan Chapter 20",
                        "url": "https://panhandlepbs.org/wgbh/amex/presidents/video/reagan_20.html",
                        "note": "PBS's campaign-ad segment ties the line directly to Ronald Reagan's re-election messaging.",
                    },
                ],
            },
            {
                "id": "openseek-7-1559156235ef401eb07e42d29109c354",
                "current_prediction_v41": "ann mcginnis",
                "patched_prediction_v42": "Shannon Faulkner",
                "confidence": "high",
                "support": [
                    {
                        "type": "institutional_history",
                        "source": "The Citadel: First women at The Citadel",
                        "url": "https://today.citadel.edu/first-women-at-the-citadel-and-in-the-south-carolina-corps-of-cadets/",
                        "note": "The Citadel's own history identifies Shannon Faulkner as the woman whose admission fight preceded women entering the Corps of Cadets.",
                    },
                    {
                        "type": "legal_history",
                        "source": "FindLaw: The Citadel and Shannon Faulkner 20 years later",
                        "url": "https://www.findlaw.com/legalblogs/legally-weird/the-citadel-and-shannon-faulkner-20-years-later/",
                        "note": "FindLaw recounts the litigation over Faulkner's admission to The Citadel.",
                    },
                ],
            },
            {
                "id": "openseek-7-1c33199ee21944bfbfaa8f8dea325a94",
                "current_prediction_v41": "sojourner",
                "patched_prediction_v42": "Pathfinder",
                "confidence": "high",
                "support": [
                    {
                        "type": "space_mission_reference",
                        "source": "NASA Science: Mars Pathfinder",
                        "url": "https://science.nasa.gov/mission/mars-pathfinder",
                        "note": "NASA identifies Mars Pathfinder as the 1997 spacecraft/mission that landed and delivered the Sojourner rover.",
                    },
                    {
                        "type": "space_history",
                        "source": "NASA History: 25 Years of Continuous Robotic Mars Exploration",
                        "url": "https://www.nasa.gov/history/25-years-of-continuous-robotic-mars-exploration-from-pathfinder-to-perseverance/",
                        "note": "NASA history describes Pathfinder as the first Mars landing in 21 years, matching the clue wording.",
                    },
                ],
            },
        ],
        "next_action": "Build and smoke-check a v42 fact4 candidate on top of v41.",
    }


def render_markdown(report: dict) -> str:
    lines = [
        "# Task7 v42 Verified Patch Notes",
        "",
        f"- Base version: `{report['base_version']}`",
        f"- Selection rule: {report['selection_rule']}",
        "",
        "## Verified rows",
        "",
    ]
    for row in report["verified_rows"]:
        lines.append(
            f"- `{row['id']}`: `{row['current_prediction_v41']}` -> "
            f"`{row['patched_prediction_v42']}` (`{row['confidence']}`)"
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
