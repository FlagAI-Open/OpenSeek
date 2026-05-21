from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .data import read_jsonl, write_jsonl
from .flagscale_adapter import describe_runtime
from .pipeline import AnnotationPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FlagOS Track 3 long-context ICL annotation runner")
    parser.add_argument("--config", default="configs/baseline.yaml", help="Path to YAML config")
    parser.add_argument("--train", required=True, help="Training JSONL with labels")
    parser.add_argument("--input", required=True, help="Evaluation JSONL without labels")
    parser.add_argument("--output", default="outputs/predictions.jsonl", help="Prediction JSONL path")
    parser.add_argument("--mock", action="store_true", help="Use mock model provider for a dry run")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    if args.mock:
        config.model.provider = "mock"
    runtime = describe_runtime(config)
    print(
        "Runtime: "
        f"framework={runtime['framework']}, model={runtime['model_name']}, "
        f"flagscale_available={runtime['flagscale_available']}"
    )
    if config.model.framework == "flagscale" and runtime["flagscale_available"] != "True":
        print(runtime["message"])

    train = read_jsonl(args.train, config.pipeline.id_field, config.pipeline.text_field, config.pipeline.label_field)
    inputs = read_jsonl(args.input, config.pipeline.id_field, config.pipeline.text_field, config.pipeline.label_field)
    print(f"Loaded {len(train)} train records and {len(inputs)} input records.")

    pipeline = AnnotationPipeline(config, train)
    rows = pipeline.annotate(inputs)
    write_jsonl(Path(args.output), rows)
    print(f"Wrote predictions to {args.output}")


if __name__ == "__main__":
    main()
