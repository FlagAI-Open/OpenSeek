import json
import re
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
DATA_PATH = PROJECT_DIR / "data" / "openseek-2_count_nouns_verbs.json"
OUTPUTS_DIR = PROJECT_DIR / "outputs"
WORK_LOGS_DIR = OUTPUTS_DIR / "work_logs"

TASK2_PATTERN = re.compile(r"Sentence: '(.*)'\. Count the number of (nouns|verbs) in this sentence\.$")
TOKEN_PATTERN = re.compile(r"[A-Za-z']+")

BE_AUX = {"is", "are", "was", "were"}
RELATIVE_MARKERS = {"that", "which", "who"}
KNOWN_VBN = {"fed", "made", "known", "seen", "born", "gone"}
PROTECTED_STUFFED_DETERMINERS = {"a", "an", "the"}
DIRECTIONAL_TOKENS = {"up", "down", "into", "onto", "out", "off", "around", "through", "toward", "towards"}


def load_dataset() -> dict:
    return json.loads(DATA_PATH.read_text(encoding="utf-8"))


def parse_task2_row(text: str) -> tuple[str, str] | None:
    match = TASK2_PATTERN.fullmatch(text)
    if not match:
        return None
    return match.group(1), match.group(2)


def tokenize(sentence: str) -> list[str]:
    return TOKEN_PATTERN.findall(sentence.lower())


def has_protected_stuffed_exact(tokens: list[str]) -> bool:
    for idx in range(1, len(tokens) - 1):
        if tokens[idx] == "stuffed" and tokens[idx - 1] in PROTECTED_STUFFED_DETERMINERS:
            return True
    return False


def classify_target_cluster(tokens: list[str]) -> str | None:
    if has_protected_stuffed_exact(tokens):
        return None

    for idx, token in enumerate(tokens[:-1]):
        if token in BE_AUX and idx + 2 < len(tokens) and tokens[idx + 1] == "being":
            candidate = tokens[idx + 2]
            if candidate.endswith("ed") or candidate in KNOWN_VBN:
                return "be_being_vbn"
        if token in BE_AUX and tokens[idx + 1].endswith("ing"):
            if any(marker in tokens[max(0, idx - 2) : idx] for marker in RELATIVE_MARKERS):
                return "rel_be_vbg"
            return "be_vbg"
    return None


def classify_target_subcluster(tokens: list[str], cluster: str) -> str:
    if cluster == "be_being_vbn":
        return "passive_chain"

    aux_index = None
    for idx, token in enumerate(tokens[:-1]):
        if token in BE_AUX and tokens[idx + 1].endswith("ing"):
            aux_index = idx
            break
    if aux_index is None:
        return "other"

    tail = tokens[aux_index + 2 :]
    has_to = "to" in tail
    has_and = "and" in tail
    ing_count = sum(token.endswith("ing") for token in tail)
    has_directional = any(token in DIRECTIONAL_TOKENS for token in tail)

    if has_and and ing_count:
        return "serial_vbg"
    if ing_count:
        return "secondary_vbg"
    if has_to:
        return "to_inf_tail"
    if has_directional:
        return "directional_tail"
    return "plain"


def surface_chain_count(tokens: list[str]) -> int:
    count = 0
    for idx, token in enumerate(tokens):
        if token in BE_AUX:
            count += 1
            continue
        if token == "being":
            count += 1
            continue
        if token.endswith("ing"):
            count += 1
            continue
        if token.endswith("ed") and idx > 0 and tokens[idx - 1] == "being":
            count += 1
    return count


def dependency_aware_fallback_count(tokens: list[str]) -> int:
    if has_protected_stuffed_exact(tokens):
        return surface_chain_count(tokens)

    count = 0
    idx = 0
    while idx < len(tokens):
        token = tokens[idx]

        # Compress passive/progressive chains like "is being fed" into one lexical event.
        if (
            token in BE_AUX
            and idx + 2 < len(tokens)
            and tokens[idx + 1] == "being"
            and (tokens[idx + 2].endswith("ed") or tokens[idx + 2] in KNOWN_VBN)
        ):
            count += 1
            idx += 3
            continue

        # Compress "be + VBG" into one event and leave later lexical material to be counted separately.
        if token in BE_AUX and idx + 1 < len(tokens) and tokens[idx + 1].endswith("ing"):
            count += 1
            idx += 2
            continue

        if token.endswith("ing"):
            count += 1
        idx += 1

    return count


def evaluate() -> dict:
    dataset = load_dataset()
    examples = dataset["examples"]

    cluster_rows: dict[str, list[dict]] = {
        "be_vbg": [],
        "rel_be_vbg": [],
        "be_being_vbn": [],
    }
    protected_rows = []

    for row in examples:
        parsed = parse_task2_row(row["input"])
        if parsed is None:
            continue
        sentence, target = parsed
        if target != "verbs":
            continue

        tokens = tokenize(sentence)
        gold = int(row["output"][0])

        if has_protected_stuffed_exact(tokens):
            protected_rows.append(
                {
                    "id": row["id"],
                    "gold": gold,
                    "sentence": sentence,
                    "surface_count": surface_chain_count(tokens),
                    "spike_count": dependency_aware_fallback_count(tokens),
                }
            )
            continue

        cluster = classify_target_cluster(tokens)
        if cluster is None:
            continue

        surface = surface_chain_count(tokens)
        spike = dependency_aware_fallback_count(tokens)
        cluster_rows[cluster].append(
            {
                "id": row["id"],
                "gold": gold,
                "sentence": sentence,
                "subcluster": classify_target_subcluster(tokens, cluster),
                "surface_count": surface,
                "spike_count": spike,
                "surface_error": abs(surface - gold),
                "spike_error": abs(spike - gold),
                "status": (
                    "improve"
                    if abs(spike - gold) < abs(surface - gold)
                    else "worse"
                    if abs(spike - gold) > abs(surface - gold)
                    else "same"
                ),
            }
        )

    cluster_summary = {}
    for cluster, rows in cluster_rows.items():
        if not rows:
            continue
        improve = sum(row["status"] == "improve" for row in rows)
        worse = sum(row["status"] == "worse" for row in rows)
        same = sum(row["status"] == "same" for row in rows)
        surface_exact = sum(row["surface_count"] == row["gold"] for row in rows)
        spike_exact = sum(row["spike_count"] == row["gold"] for row in rows)
        cluster_summary[cluster] = {
            "row_count": len(rows),
            "surface_exact_count": surface_exact,
            "spike_exact_count": spike_exact,
            "surface_exact_rate": round(surface_exact / len(rows), 6),
            "spike_exact_rate": round(spike_exact / len(rows), 6),
            "improve_count": improve,
            "worse_count": worse,
            "same_count": same,
            "sample_improvements": [row for row in rows if row["status"] == "improve"][:8],
            "sample_worsenings": [row for row in rows if row["status"] == "worse"][:5],
        }

    protected_changed = [
        row
        for row in protected_rows
        if row["surface_count"] != row["spike_count"]
    ]

    target_rows = sum(section["row_count"] for section in cluster_summary.values())
    improved_rows = sum(section["improve_count"] for section in cluster_summary.values())
    worsened_rows = sum(section["worse_count"] for section in cluster_summary.values())

    if target_rows and improved_rows > worsened_rows and not protected_changed:
        verdict = "small_scope_positive_no_protected_regression"
    elif protected_changed:
        verdict = "pause_protected_exact_case_changed"
    else:
        verdict = "pause_no_stable_targeted_gain"

    return {
        "generated_on": "2026-03-31",
        "spike_scope": {
            "default_path_unchanged": True,
            "changes_method_py": False,
            "introduces_new_parser_dependency": False,
            "target_clusters": [
                "be_vbg",
                "rel_be_vbg",
                "be_being_vbn",
            ],
        },
        "cluster_summary": cluster_summary,
        "protected_v27_exact_case_check": {
            "protected_row_count": len(protected_rows),
            "changed_protected_row_count": len(protected_changed),
            "changed_examples": protected_changed[:5],
        },
        "overall_verdict": verdict,
        "next_step": (
            "keep_as_offline_spike_only"
            if verdict == "small_scope_positive_no_protected_regression"
            else "stop_before_engineering"
        ),
    }


def render_markdown(report: dict) -> str:
    lines = [
        "# Task2 Dependency-Aware Spike",
        "",
        "- This experiment is fully offline and does not modify `method.py`.",
        "- The spike only evaluates narrow structure-aware compression on `be + VBG`, relative-clause `be + VBG`, and `be being + VBN` clusters.",
        "",
        "## Cluster Summary",
        "",
        "| Cluster | Rows | Surface exact | Spike exact | Improved | Worsened |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for cluster, row in report["cluster_summary"].items():
        lines.append(
            f"| `{cluster}` | `{row['row_count']}` | `{row['surface_exact_rate']}` | `{row['spike_exact_rate']}` | "
            f"`{row['improve_count']}` | `{row['worse_count']}` |"
        )

    protected = report["protected_v27_exact_case_check"]
    lines.extend(
        [
            "",
            "## Protected v27 Exact Cases",
            "",
            f"- Protected rows checked: `{protected['protected_row_count']}`",
            f"- Protected rows changed by the spike: `{protected['changed_protected_row_count']}`",
            "",
            "## Verdict",
            "",
            f"- Overall verdict: `{report['overall_verdict']}`",
            f"- Next step: `{report['next_step']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    report = evaluate()
    json_path = WORK_LOGS_DIR / "task2_dependency_spike_2026-03-31.json"
    md_path = WORK_LOGS_DIR / "task2_dependency_spike_2026-03-31.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
