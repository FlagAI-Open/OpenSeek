import json
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
OUTPUTS_DIR = PROJECT_DIR / "outputs"
WORK_LOGS_DIR = OUTPUTS_DIR / "work_logs"

OUTPUT_JSON = WORK_LOGS_DIR / "task7_v41_verified_patch_notes_2026-04-10.json"
OUTPUT_MD = WORK_LOGS_DIR / "task7_v41_verified_patch_notes_2026-04-10.md"


def build_report() -> dict:
    return {
        "generated_on": "2026-04-10",
        "base_version": "v40",
        "selection_rule": (
            "Stay on the official v40 baseline and apply only one additional road-sticker "
            "canonicalization patch: SGP -> Singapore."
        ),
        "verified_rows": [
            {
                "id": "openseek-7-01e4fbfca16345b898064b73d9c18659",
                "current_prediction_v40": "sgp",
                "patched_prediction_v41": "Singapore",
                "confidence": "high",
                "support": [
                    {
                        "type": "unece_reference",
                        "source": "UNECE road traffic distinguishing sign list",
                        "url": "https://unece.org/sites/default/files/2025-06/ECE-TRANS-WP.1-S-June-13e.pdf",
                        "note": "The UNECE distinguishing-sign list explicitly maps SGP to Singapore for international road traffic.",
                    },
                    {
                        "type": "clue_semantics",
                        "source": "Task clue structure",
                        "url": None,
                        "note": "The clue is the vehicle sticker code itself, so the expected answer is the country name rather than the code token.",
                    },
                ],
            }
        ],
        "next_action": "Build and smoke-check a v41 one-row candidate on top of v40.",
    }


def render_markdown(report: dict) -> str:
    lines = [
        "# Task7 v41 Verified Patch Notes",
        "",
        f"- Base version: `{report['base_version']}`",
        f"- Selection rule: {report['selection_rule']}",
        "",
        "## Verified rows",
        "",
    ]
    for row in report["verified_rows"]:
        lines.append(
            f"- `{row['id']}`: `{row['current_prediction_v40']}` -> "
            f"`{row['patched_prediction_v41']}` (`{row['confidence']}`)"
        )
        for support in row["support"]:
            if support["url"]:
                lines.append(f"  - {support['source']}: {support['note']} ({support['url']})")
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
