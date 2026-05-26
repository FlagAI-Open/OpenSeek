import argparse
import json
from pathlib import Path

from main import TASK_FILES
from method import TASK_SAMPLE_ATTEMPTS, build_prompt, select_examples, solve_task_locally
from method import annotate_nvidia as annotate


BAD_VALUES = {"", "...", "None", "null"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_id", type=int, required=True, help="Task id to repair.")
    parser.add_argument("--input_file", type=str, required=True, help="Existing jsonl file to repair.")
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where the repaired jsonl file will be written.",
    )
    parser.add_argument(
        "--examples_limit",
        type=int,
        default=24,
        help="Maximum number of ICL examples to consider before selection.",
    )
    parser.add_argument(
        "--max_input_length",
        type=int,
        default=128000,
        help="Maximum prompt token length allowed before skipping a sample.",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="Optional local tokenizer/model path for prompt length checks.",
    )
    parser.add_argument(
        "--only_bad_ids",
        type=str,
        nargs="*",
        default=None,
        help="Optional explicit test_sample_id list to repair.",
    )
    return parser.parse_args()


def is_bad_prediction(prediction) -> bool:
    if prediction is None:
        return True
    return str(prediction).strip() in BAD_VALUES


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    args = parse_args()
    task_file = TASK_FILES[args.task_id]
    task_dict = json.loads(task_file.read_text(encoding="utf-8"))
    icl_examples = task_dict["examples"][: args.examples_limit]
    test_samples = {sample["id"]: sample for sample in task_dict["test_samples"]}

    rows = load_rows(Path(args.input_file))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / Path(args.input_file).name

    tokenizer = None
    if args.tokenizer_path:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)

    examples_str = None
    repaired = 0
    unresolved = 0

    for row in rows:
        sample_id = row["test_sample_id"]
        if args.only_bad_ids and sample_id not in set(args.only_bad_ids):
            continue
        if not is_bad_prediction(row.get("prediction")):
            continue

        sample = test_samples[sample_id]
        text2annotate = sample["input"]
        prediction = solve_task_locally(args.task_id, text2annotate)

        if prediction is None:
            prompt = build_prompt(task_dict["Definition"][0], text2annotate, task_id=args.task_id)
            if examples_str is None:
                examples_str = select_examples(
                    icl_examples,
                    task_dict["Definition"][0],
                    text2annotate,
                    task_id=args.task_id,
                )
            input_prompt = prompt.replace("[[EXAMPLES]]\n\n", examples_str + "\n\n")

            if tokenizer is not None:
                tokenized_input = tokenizer(input_prompt, return_tensors="pt")
                if tokenized_input["input_ids"].shape[1] > args.max_input_length:
                    prediction = None

            if prediction is None:
                for _ in range(TASK_SAMPLE_ATTEMPTS.get(args.task_id, 1)):
                    prediction = annotate(input_prompt, task_id=args.task_id)
                    if prediction is not None:
                        break

        row["prediction"] = prediction
        repaired += 1
        if is_bad_prediction(prediction):
            unresolved += 1

    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "task_id": args.task_id,
        "input_file": str(args.input_file),
        "output_file": str(output_path),
        "rows": len(rows),
        "repaired_candidates": repaired,
        "unresolved_after_repair": unresolved,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
