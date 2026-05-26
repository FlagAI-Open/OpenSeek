import json
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
OUTPUTS_DIR = PROJECT_DIR / "outputs"
WORK_LOGS_DIR = OUTPUTS_DIR / "work_logs"

OUTPUT_JSON = WORK_LOGS_DIR / "task7_v38_verified_patch_notes_2026-04-10.json"
OUTPUT_MD = WORK_LOGS_DIR / "task7_v38_verified_patch_notes_2026-04-10.md"


def build_report() -> dict:
    return {
        "generated_on": "2026-04-10",
        "base_version": "v37",
        "verified_rows": [
            {
                "id": "openseek-7-34bd86c0bffd49ce86d2e8267b023d92",
                "current_prediction_v37": "abigail",
                "patched_prediction_v38": "Louisa",
                "confidence": "high",
                "support": [
                    {
                        "type": "j_archive",
                        "source": "J! Archive Show #5508",
                        "url": "https://j-archive.com/showgame.php?game_id=2561",
                        "note": "The clue is archived with the response Louisa.",
                    }
                ],
            },
            {
                "id": "openseek-7-8da173db613a4fa895adc022789f5631",
                "current_prediction_v37": "pagri",
                "patched_prediction_v38": "turban",
                "confidence": "medium_high",
                "support": [
                    {
                        "type": "trivia_transcript",
                        "source": "Instant Trivia transcript",
                        "url": "https://music.amazon.in/podcasts/a23f0013-d20d-4320-b7dc-4a00c4072546/episodes/66e0be21-8464-4580-8b1d-9675ade6a207/instant-trivia-episode-734---great-things---what%27s-the-pitch---whatsits-doohickeys-thingamabobs---50-50---restaurants-episode-734---everyday-inventions---twin-peaks---peter-paul-or-mary---crossword-clues-e---th",
                        "note": "The archived clue transcript gives the answer as a turban.",
                    },
                    {
                        "type": "supporting_reference",
                        "source": "U.S. Department of Justice Sikh head covering guide",
                        "url": "https://www.justice.gov/sites/default/files/crt/legacy/2008/10/21/sikh_poster.pdf",
                        "note": "The guide describes pagri as a type of turban, which supports the broader canonical Jeopardy answer.",
                    }
                ],
            },
            {
                "id": "openseek-7-bc0b7dbb83eb4abcaa447bfc6e13e29c",
                "current_prediction_v37": "soaked",
                "patched_prediction_v38": "sodden",
                "confidence": "high",
                "support": [
                    {
                        "type": "j_archive",
                        "source": "J! Archive Show #5233",
                        "url": "https://j-archive.com/showgame.php?game_id=1860",
                        "note": "The clue is archived with the response sodden.",
                    }
                ],
            },
        ],
        "next_action": "Build and smoke-check a v38 ultra-narrow candidate on top of v37 using only these three verified rows.",
    }


def render_markdown(report: dict) -> str:
    lines = [
        "# Task7 v38 Verified Patch Notes",
        "",
        f"- Base version: `{report['base_version']}`",
        "",
        "## Verified rows",
        "",
    ]
    for row in report["verified_rows"]:
        lines.append(
            f"- `{row['id']}`: `{row['current_prediction_v37']}` -> `{row['patched_prediction_v38']}` "
            f"(`{row['confidence']}`)"
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
