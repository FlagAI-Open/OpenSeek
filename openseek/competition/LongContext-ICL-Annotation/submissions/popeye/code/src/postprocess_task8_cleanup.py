import argparse
import ast
import json
from pathlib import Path


NOISE_MARKERS = (
    "placeholder",
    "illustrative purposes only",
    "not tested and may contain errors",
    "may need to be adjusted",
    "the code is not complete",
    "provided as an example",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="Task 8 jsonl file to clean.")
    parser.add_argument("--output_file", type=str, required=True, help="Output jsonl path.")
    return parser.parse_args()


def clean_code(code: str) -> str:
    if not code:
        return code

    lines = code.splitlines()
    cleaned = []
    previous = None

    for line in lines:
        stripped = line.strip()
        lowered = stripped.lower()

        if stripped.startswith("#") and any(marker in lowered for marker in NOISE_MARKERS):
            continue

        if previous == stripped and stripped.startswith("#"):
            continue

        cleaned.append(line)
        previous = stripped

    result = "\n".join(cleaned).strip() + "\n"
    try:
        ast.parse(result)
        return result
    except SyntaxError:
        return code


def main():
    args = parse_args()
    input_file = Path(args.input_file)
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(line) for line in input_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    modified = 0
    total_chars_before = 0
    total_chars_after = 0

    for row in rows:
        code = row.get("prediction") or ""
        total_chars_before += len(code)
        new_code = clean_code(code)
        total_chars_after += len(new_code)
        if new_code != code:
            row["prediction"] = new_code
            modified += 1

    with output_file.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "input_file": str(input_file),
                "output_file": str(output_file),
                "rows": len(rows),
                "modified_rows": modified,
                "avg_chars_before": round(total_chars_before / max(len(rows), 1), 2),
                "avg_chars_after": round(total_chars_after / max(len(rows), 1), 2),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
