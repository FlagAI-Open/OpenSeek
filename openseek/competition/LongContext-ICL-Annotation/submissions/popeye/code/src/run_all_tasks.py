import argparse
import json
import zipfile
from pathlib import Path

from main import OUTPUTS_DIR, evaluate


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tasks",
        type=int,
        nargs="*",
        default=[1, 2, 3, 4, 5, 6, 7, 8],
        help="Task ids to run. Defaults to all eight competition tasks.",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="Optional local tokenizer/model path for prompt length checks.",
    )
    parser.add_argument(
        "--max_input_length",
        type=int,
        default=128000,
        help="Maximum prompt token length allowed before skipping a sample.",
    )
    parser.add_argument(
        "--examples_limit",
        type=int,
        default=100,
        help="Maximum number of ICL examples to consider before selection.",
    )
    parser.add_argument(
        "--sample_limit",
        type=int,
        default=None,
        help="Optional per-task sample cap for smoke tests.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(OUTPUTS_DIR / "baseline_submission"),
        help="Directory to store generated jsonl files.",
    )
    parser.add_argument(
        "--zip_name",
        type=str,
        default="baseline_submission.zip",
        help="Name of the final zip package.",
    )
    return parser.parse_args()


def package_submission(output_dir: Path, zip_name: str, files: list[str]) -> Path:
    zip_path = output_dir / zip_name
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in files:
            fp = Path(file_path)
            zf.write(fp, arcname=fp.name)
    return zip_path


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = None
    if args.tokenizer_path:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    generated_files = []
    for task_id in args.tasks:
        generated_files.append(
            evaluate(
                task_id=task_id,
                qwen_tokenizer=tokenizer,
                max_input_length=args.max_input_length,
                log_path_prefix=str(output_dir),
                examples_limit=args.examples_limit,
                sample_limit=args.sample_limit,
            )
        )

    zip_path = package_submission(output_dir, args.zip_name, generated_files)
    summary_path = output_dir / "run_summary.json"
    summary = {
        "tasks": args.tasks,
        "generated_files": [Path(path).name for path in generated_files],
        "zip_path": str(zip_path),
        "sample_limit": args.sample_limit,
        "examples_limit": args.examples_limit,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
