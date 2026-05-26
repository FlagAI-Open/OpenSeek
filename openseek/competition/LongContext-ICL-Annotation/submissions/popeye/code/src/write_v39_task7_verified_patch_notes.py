import json
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
OUTPUTS_DIR = PROJECT_DIR / "outputs"
WORK_LOGS_DIR = OUTPUTS_DIR / "work_logs"

OUTPUT_JSON = WORK_LOGS_DIR / "task7_v39_verified_patch_notes_2026-04-10.json"
OUTPUT_MD = WORK_LOGS_DIR / "task7_v39_verified_patch_notes_2026-04-10.md"


def build_report() -> dict:
    return {
        "generated_on": "2026-04-10",
        "base_version": "v37",
        "selection_rule": (
            "Stay on the current formal baseline v37 and apply only typo/canonical-form "
            "repairs that are directly supported by the clue text plus external references."
        ),
        "carry_forward_note": (
            "The earlier v37 shortlist has effectively been consumed by v37/v38, so v39 "
            "switches to ultra-narrow typo/canonical cleanup rather than replaying the old shortlist."
        ),
        "verified_rows": [
            {
                "id": "openseek-7-220c54dc2fa44178a8b1e881e9ee6fb9",
                "current_prediction_v37": "seleium",
                "patched_prediction_v39": "selenium",
                "confidence": "high",
                "support": [
                    {
                        "type": "encyclopedia_reference",
                        "source": "Britannica: Selenium",
                        "url": "https://www.britannica.com/science/selenium",
                        "note": "Selenium is the chemical element with symbol Se, which exactly matches the clue.",
                    },
                    {
                        "type": "baseline_crosscheck",
                        "source": "v30 task7 output",
                        "url": None,
                        "note": "The pre-author-projection baseline already carried the near-correct form `seLENium`, which supports a typo-only correction.",
                    },
                ],
            },
            {
                "id": "openseek-7-63f7eeaee4f4454ea81d62622f2d773a",
                "current_prediction_v37": "petrgrad",
                "patched_prediction_v39": "petrograd",
                "confidence": "high",
                "support": [
                    {
                        "type": "encyclopedia_reference",
                        "source": "Britannica: Saint Petersburg summary",
                        "url": "https://www.britannica.com/summary/St-Petersburg-Russia",
                        "note": "Britannica lists the former 1914-24 name of Saint Petersburg as Petrograd.",
                    },
                    {
                        "type": "baseline_crosscheck",
                        "source": "v30 task7 output",
                        "url": None,
                        "note": "The v30 line already used the correctly spelled form `petrograd`, so this is a typo repair rather than a semantic rewrite.",
                    },
                ],
            },
            {
                "id": "openseek-7-7b7c13a6f01b4f96997a82f3725258d6",
                "current_prediction_v37": "victonia",
                "patched_prediction_v39": "Victoria",
                "confidence": "high",
                "support": [
                    {
                        "type": "encyclopedia_reference",
                        "source": "Britannica: Bass Strait",
                        "url": "https://www.britannica.com/place/Bass-Strait",
                        "note": "Bass Strait separates Victoria from Tasmania, which directly resolves the clue.",
                    },
                    {
                        "type": "clue_semantics",
                        "source": "Task clue structure",
                        "url": None,
                        "note": "The clue asks for an Australian state, so the fully spelled proper noun Victoria is the canonical form.",
                    },
                ],
            },
        ],
        "watchlist_rows": [
            {
                "id": "openseek-7-01e4fbfca16345b898064b73d9c18659",
                "current_prediction_v37": "sgp",
                "candidate_answer": "Singapore",
                "status": "hold_for_next_round",
                "reason": "High-potential semantic cleanup, but keep v39 at three rows and leave the road-sticker expansion for the next package if needed.",
            }
        ],
        "next_action": (
            "Build and smoke-check a v39 ultra-narrow candidate on top of v37 using only "
            "these three typo/canonical-form fixes."
        ),
    }


def render_markdown(report: dict) -> str:
    lines = [
        "# Task7 v39 Verified Patch Notes",
        "",
        f"- Base version: `{report['base_version']}`",
        f"- Selection rule: {report['selection_rule']}",
        f"- Carry-forward note: {report['carry_forward_note']}",
        "",
        "## Verified rows",
        "",
    ]
    for row in report["verified_rows"]:
        lines.append(
            f"- `{row['id']}`: `{row['current_prediction_v37']}` -> "
            f"`{row['patched_prediction_v39']}` (`{row['confidence']}`)"
        )
        for support in row["support"]:
            if support["url"]:
                lines.append(
                    f"  - {support['source']}: {support['note']} ({support['url']})"
                )
            else:
                lines.append(f"  - {support['source']}: {support['note']}")
    lines.extend(["", "## Watchlist", ""])
    for row in report["watchlist_rows"]:
        lines.append(
            f"- `{row['id']}`: keep `{row['current_prediction_v37']}` for now; "
            f"candidate next answer `{row['candidate_answer']}`. {row['reason']}"
        )
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
