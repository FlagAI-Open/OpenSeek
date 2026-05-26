import argparse
import json
from pathlib import Path

from task8_semantics import audit_task8_code


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_file", type=str, required=True)
    parser.add_argument("--candidate_file", type=str, required=True)
    parser.add_argument("--task8_data", type=str, default="data/openseek-8_kernel_generation.json")
    parser.add_argument("--only_changed", action="store_true")
    parser.add_argument("--output_path", type=str, default=None)
    return parser.parse_args()


def load_jsonl(path: Path) -> dict[str, dict]:
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row["test_sample_id"]] = row
    return rows


def main():
    args = parse_args()
    base_rows = load_jsonl(Path(args.base_file))
    candidate_rows = load_jsonl(Path(args.candidate_file))
    task8 = json.loads(Path(args.task8_data).read_text(encoding="utf-8"))
    test_map = {sample["id"]: sample for sample in task8["test_samples"]}

    audited = []
    for sample_id, row in candidate_rows.items():
        base_prediction = (base_rows.get(sample_id) or {}).get("prediction")
        candidate_prediction = row.get("prediction") or ""
        if args.only_changed and candidate_prediction == (base_prediction or ""):
            continue

        audit = audit_task8_code(test_map[sample_id]["input"], candidate_prediction)
        audit["test_sample_id"] = sample_id
        audit["changed_from_base"] = candidate_prediction != (base_prediction or "")
        audit["candidate_length"] = len(candidate_prediction)
        audited.append(audit)

    report = {
        "base_file": str(args.base_file),
        "candidate_file": str(args.candidate_file),
        "only_changed": args.only_changed,
        "audited_count": len(audited),
        "rows": audited,
    }

    if args.output_path:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
