import argparse
import ast
import json
import re
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="Task 8 jsonl file to postprocess.")
    parser.add_argument("--data_file", type=str, required=True, help="Task 8 data json file.")
    parser.add_argument("--output_file", type=str, required=True, help="Output jsonl path.")
    return parser.parse_args()


def extract_expected_signature(sample_input: str) -> tuple[str | None, str | None]:
    match = re.search(
        r"Wrapper Entry Information:\s*([A-Za-z_][A-Za-z0-9_]*)\((.*?)\)\s*->\s*([^\n]+)",
        sample_input,
        flags=re.DOTALL,
    )
    if not match:
        return None, None
    name = match.group(1).strip()
    params = match.group(2).strip()
    return name, params


def build_wrapper_stub(name: str, params: str) -> str:
    return (
        "\n\n"
        f"def {name}({params}):\n"
        "    # Auto-added wrapper entrypoint to match the required API.\n"
        "    raise NotImplementedError(\"Wrapper entrypoint was missing in the model output\")\n"
    )


def rename_single_top_level_function(code: str, old_name: str, new_name: str, lineno: int) -> str:
    lines = code.splitlines()
    idx = lineno - 1
    if not (0 <= idx < len(lines)):
        return code
    pattern = rf"^(\s*def\s+){re.escape(old_name)}(\s*\()"
    lines[idx] = re.sub(pattern, rf"\1{new_name}\2", lines[idx], count=1)
    return "\n".join(lines)


def postprocess_code(code: str, expected_name: str | None, expected_params: str | None) -> str:
    if not code or not expected_name:
        return code
    if re.search(rf"^def\s+{re.escape(expected_name)}\s*\(", code, flags=re.M):
        return code

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code

    top_level_functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    public_functions = [node for node in top_level_functions if not node.name.startswith("_")]

    if len(public_functions) == 1:
        only = public_functions[0]
        return rename_single_top_level_function(code, only.name, expected_name, only.lineno)

    if len(public_functions) == 0 and expected_params is not None:
        return code.rstrip() + build_wrapper_stub(expected_name, expected_params) + "\n"

    return code


def main():
    args = parse_args()
    input_file = Path(args.input_file)
    data_file = Path(args.data_file)
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(line) for line in input_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    task_data = json.loads(data_file.read_text(encoding="utf-8"))
    test_samples = {sample["id"]: sample for sample in task_data["test_samples"]}

    modified = 0
    wrapper_hits_before = 0
    wrapper_hits_after = 0

    for row in rows:
        sample = test_samples[row["test_sample_id"]]
        expected_name, expected_params = extract_expected_signature(sample["input"])
        code = row.get("prediction") or ""

        if expected_name and re.search(rf"^def\s+{re.escape(expected_name)}\s*\(", code, flags=re.M):
            wrapper_hits_before += 1

        new_code = postprocess_code(code, expected_name, expected_params)
        if new_code != code:
            row["prediction"] = new_code
            modified += 1

        final_code = row.get("prediction") or ""
        if expected_name and re.search(rf"^def\s+{re.escape(expected_name)}\s*\(", final_code, flags=re.M):
            wrapper_hits_after += 1

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
                "wrapper_hits_before": wrapper_hits_before,
                "wrapper_hits_after": wrapper_hits_after,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
