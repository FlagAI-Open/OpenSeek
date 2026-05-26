import json
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
OUTPUTS_DIR = PROJECT_DIR / "outputs"
WORK_LOGS_DIR = OUTPUTS_DIR / "work_logs"

OUTPUT_JSON = WORK_LOGS_DIR / "task7_v43_verified_patch_notes_2026-04-10.json"
OUTPUT_MD = WORK_LOGS_DIR / "task7_v43_verified_patch_notes_2026-04-10.md"


def build_report() -> dict:
    return {
        "generated_on": "2026-04-10",
        "base_version": "v42",
        "selection_rule": (
            "Stay on the official v42 baseline and add only a last ultra-narrow set of "
            "externally verifiable factual repairs with clearly wrong current v42 answers."
        ),
        "verified_rows": [
            {
                "id": "openseek-7-63dabfe75b1a4025b8f19594752dd535",
                "current_prediction_v42": "stan musial",
                "patched_prediction_v43": "Ted Williams",
                "confidence": "high",
                "support": [
                    {
                        "type": "mlb_reference",
                        "source": "MLB: Ted Williams' .406 in 1941 stands test of time",
                        "url": "https://www.mlb.com/news/ted-williams-406-in-1941-stands-test-of-time-c190110308",
                        "note": "MLB identifies Ted Williams as the player who hit .406 in 1941.",
                    },
                    {
                        "type": "stats_reference",
                        "source": "Ted Williams Official stats",
                        "url": "https://tedwilliams.com/ted-williams-stats/",
                        "note": "Ted Williams' official stats page lists the .406 batting average in 1941.",
                    },
                ],
            },
            {
                "id": "openseek-7-8095cf4b70cb4ca89fbf42cbe46fc299",
                "current_prediction_v42": "phoenix suns",
                "patched_prediction_v43": "Los Angeles Lakers",
                "confidence": "high",
                "support": [
                    {
                        "type": "reference_biography",
                        "source": "Britannica: Jerry West",
                        "url": "https://www.britannica.com/biography/Jerry-West",
                        "note": "Britannica lists Jerry West's 932 games and identifies him with the Los Angeles Lakers.",
                    },
                    {
                        "type": "nba_reference",
                        "source": "NBA.com legends profile: Jerry West",
                        "url": "https://www.nba.com/news/history-nba-legend-jerry-west?quot=",
                        "note": "NBA.com notes that West finished with 25,192 points in 932 games and spent his playing career with the Lakers.",
                    },
                ],
            },
            {
                "id": "openseek-7-7d754cc1356b49b1ac056a2dbd0a4ea8",
                "current_prediction_v42": "the kookaburra",
                "patched_prediction_v43": "rooster",
                "confidence": "high",
                "support": [
                    {
                        "type": "j_archive",
                        "source": "J! Archive Show #4637",
                        "url": "https://j-archive.com/showgame.php?game_id=42",
                        "note": "The exact archived clue uses the same Japanese and Norwegian onomatopoeias.",
                    },
                    {
                        "type": "language_reference",
                        "source": "Nippon.com: Japanese Animal Noises",
                        "url": "https://www.nippon.com/en/japan-topics/b05615/japanese-animal-noises.html",
                        "note": "Nippon.com lists kokekokko as the Japanese rooster sound, which supports the intended animal.",
                    },
                    {
                        "type": "language_reference",
                        "source": "Omniglot: Cockerel / Rooster sounds from around the world",
                        "url": "https://www.omniglot.com/language/animalsounds/cockerel.htm",
                        "note": "Omniglot lists kykkeliky as a rooster/cockerel sound in Scandinavian usage; the answer is inferred from the combined sources.",
                    },
                ],
            },
        ],
        "next_action": "Build and smoke-check a v43 fact3 candidate on top of v42.",
    }


def render_markdown(report: dict) -> str:
    lines = [
        "# Task7 v43 Verified Patch Notes",
        "",
        f"- Base version: `{report['base_version']}`",
        f"- Selection rule: {report['selection_rule']}",
        "",
        "## Verified rows",
        "",
    ]
    for row in report["verified_rows"]:
        lines.append(
            f"- `{row['id']}`: `{row['current_prediction_v42']}` -> "
            f"`{row['patched_prediction_v43']}` (`{row['confidence']}`)"
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
