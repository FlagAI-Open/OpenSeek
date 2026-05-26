import json
import shutil
import zipfile
from pathlib import Path

from method import parse_task2_fields
from spike_task2_dependency_aware import (
    classify_target_cluster,
    classify_target_subcluster,
    has_protected_stuffed_exact,
    tokenize,
)
from task2_dependency_policy import count_verbs_with_policy, get_policy_config


SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
OUTPUTS_DIR = PROJECT_DIR / "outputs"
WORK_LOGS_DIR = OUTPUTS_DIR / "work_logs"
DATA_PATH = PROJECT_DIR / "data" / "openseek-2_count_nouns_verbs.json"

BASE_VERSION = "v27"
BASE_DIR = OUTPUTS_DIR / "final_submission_v27_task2stuffed_on_v25_candidate"
BASE_TASK2_PATH = BASE_DIR / "openseek-2-v1.jsonl"

POLICY_NAME = "compressed_plus_discourse_followup"
ALLOWLIST = {
    ("be_being_vbn", "passive_chain"),
    ("rel_be_vbg", "directional_tail"),
    ("rel_be_vbg", "plain"),
}

TASK2_OUT_DIR = OUTPUTS_DIR / "task2_dependency_relplus_candidate"
TASK2_OUTPUT_PATH = TASK2_OUT_DIR / "openseek-2-v1.jsonl"
OUT_DIR = OUTPUTS_DIR / "final_submission_v32_task2dependency_relplus_candidate"
ZIP_NAME = "final_submission_v32_task2dependency_relplus_candidate.zip"
REPORT_PATH = WORK_LOGS_DIR / "task2_dependency_v32_candidate_build_2026-04-06.json"


def load_jsonl_rows(path: Path) -> dict[str, dict]:
    return {
        row["test_sample_id"]: row
        for row in (
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }


def build_task2_file() -> dict:
    policy = get_policy_config(POLICY_NAME)
    if policy is None:
        raise ValueError(f"Unknown policy: {POLICY_NAME}")

    task_dict = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    test_samples = task_dict["test_samples"]
    base_rows = load_jsonl_rows(BASE_TASK2_PATH)

    TASK2_OUT_DIR.mkdir(parents=True, exist_ok=True)

    changed_rows = []
    output_rows = []

    for sample in test_samples:
        sample_id = sample["id"]
        sentence, target = parse_task2_fields(sample["input"])
        row = dict(base_rows[sample_id])

        if target == "verbs":
            tokens = tokenize(sentence)
            if not has_protected_stuffed_exact(tokens):
                cluster = classify_target_cluster(tokens)
                subcluster = classify_target_subcluster(tokens, cluster) if cluster is not None else None
                if (cluster, subcluster) in ALLOWLIST:
                    updated_prediction = str(count_verbs_with_policy(tokens, policy))
                    if updated_prediction != row["prediction"]:
                        changed_rows.append(
                            {
                                "id": sample_id,
                                "sentence": sentence,
                                "cluster": cluster,
                                "subcluster": subcluster,
                                "old": row["prediction"],
                                "new": updated_prediction,
                            }
                        )
                        row["prediction"] = updated_prediction

        output_rows.append(row)

    TASK2_OUTPUT_PATH.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output_rows),
        encoding="utf-8",
    )

    summary = {
        "base_version": BASE_VERSION,
        "policy_name": POLICY_NAME,
        "allowlist": [f"{cluster}:{subcluster}" for cluster, subcluster in sorted(ALLOWLIST)],
        "task2_output_path": str(TASK2_OUTPUT_PATH),
        "changed_count_vs_base": len(changed_rows),
        "changed_rows": changed_rows,
    }
    (TASK2_OUT_DIR / "task2_change_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def build_package(task2_summary: dict) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for task_id in range(1, 9):
        source = BASE_DIR / f"openseek-{task_id}-v1.jsonl"
        destination = OUT_DIR / f"openseek-{task_id}-v1.jsonl"
        if task_id == 2:
            shutil.copy2(Path(task2_summary["task2_output_path"]), destination)
        else:
            shutil.copy2(source, destination)

    zip_path = OUT_DIR / ZIP_NAME
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for task_id in range(1, 9):
            file_path = OUT_DIR / f"openseek-{task_id}-v1.jsonl"
            zf.write(file_path, arcname=file_path.name)

    merge_summary = {
        "files": [f"openseek-{task_id}-v1.jsonl" for task_id in range(1, 9)],
        "zip_path": str(zip_path),
        "base": BASE_VERSION,
        "replaced_tasks": {
            "2": "outputs/task2_dependency_relplus_candidate/openseek-2-v1.jsonl"
        },
        "notes": [
            "Only task2 was replaced relative to v27.",
            "Task2 applies the dependency-aware policy only on the rel_be_vbg:plain, rel_be_vbg:directional_tail, and be_being_vbn:passive_chain allowlist.",
            "This candidate stays ultra-narrow after the v28 failure audit showed broad scope leakage outside the target family.",
            "The rel_be_vbg:plain and rel_be_vbg:directional_tail subclusters were the cleanest next focus in the offline dependency audit, with be_being_vbn:passive_chain added as a tiny perfect-hit extension.",
        ],
        "task2_test_change_count": task2_summary["changed_count_vs_base"],
        "task2_test_changes": task2_summary["changed_rows"],
    }
    (OUT_DIR / "merge_summary.json").write_text(
        json.dumps(merge_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "out_dir": str(OUT_DIR),
        "zip_path": str(zip_path),
        "task2_test_change_count": task2_summary["changed_count_vs_base"],
    }


def main() -> None:
    task2_summary = build_task2_file()
    package_summary = build_package(task2_summary)
    report = {
        "task2_summary": task2_summary,
        "package_summary": package_summary,
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
