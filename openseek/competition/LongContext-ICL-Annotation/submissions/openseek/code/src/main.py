import argparse

from src.ai_lab.pipeline import run_evaluation, run_packaging, run_prediction


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AI-LAB competition scaffold")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name in ("predict", "evaluate", "package"):
        subparser = subparsers.add_parser(name)
        subparser.add_argument("--config", required=True, help="Path to yaml config")

    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "predict":
        run_prediction(args.config)
    elif args.command == "evaluate":
        run_evaluation(args.config)
    elif args.command == "package":
        run_packaging(args.config)


if __name__ == "__main__":
    main()
