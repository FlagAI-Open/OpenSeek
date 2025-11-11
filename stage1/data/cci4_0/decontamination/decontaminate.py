import argparse
import difflib
import re
import unicodedata
from pathlib import Path

import jieba_fast
from tqdm.auto import tqdm
from datasets import load_dataset, Dataset


def tokenize(text):
    """Normalize text by removing diacritics and tokenize."""
    text = "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")
    tokens = re.findall("\w+", text.lower())

    """tokenize zh tokens by jieba."""
    tokens = [word for text in tokens for word in jieba_fast.lcut(text, cut_all=False)]
    return tokens


def get_ngrams(tokens, n):
    """Generate n-grams from tokens."""
    ngrams = set(zip(*[tokens[i:] for i in range(n)]))
    return ngrams


def retrieve_ngrams_batch(batch, eval_ngrams, eval_datasets, eval_texts, ngram_len):
    """Find contaminated samples based on n-grams."""
    new_batch = {"text": [], "ngram": [], "bench_name": [], "bench_text": []}
    for completion in batch["text"]:
        tokens = tokenize(completion)
        ngrams = get_ngrams(tokens, ngram_len)
        for ngram in ngrams:
            if ngram in eval_ngrams:
                idx = eval_ngrams[ngram]
                new_batch["text"].append(completion)
                new_batch["ngram"].append(ngram)
                new_batch["bench_name"].append(eval_datasets[idx])
                new_batch["bench_text"].append(eval_texts[idx])
                break
    return new_batch


def diff_strings(string1, string2):
    """Find matching parts between two strings."""
    matcher = difflib.SequenceMatcher(None, string1.lower(), string2.lower(), autojunk=False)
    matching_blocks = matcher.get_matching_blocks()
    matches = []
    for block in matching_blocks:
        start_a, start_b, length = block
        if length > 5:
            match = string1[start_a:start_a + length]
            matches.append(match)
    return matches


def add_match_stats(example):
    gen_text = " ".join(tokenize(example["text"]))
    bench_text = " ".join(tokenize(example["bench_text"]))
    matching_parts = diff_strings(gen_text, bench_text)
    match = " ".join("".join(matching_parts).split())
    example["diff"] = matching_parts
    example["diff_ratio"] = len(match) / len(bench_text) if len(bench_text) > 0 else 0
    example["diff_length"] = len(match)
    example["longest_diff_part"] = max(matching_parts, key=len, default="")
    example["longest_diff_part_length"] = len(example["longest_diff_part"])
    return example


def main(args):
    # Load the evaluation data to build n-grams index
    eval_ngrams, eval_datasets, eval_texts = {}, [], []
    # eval_data = load_dataset(args.eval_dataset, split="train", num_proc=args.num_proc)
    eval_data = Dataset.from_json(path_or_paths=args.eval_dataset, num_proc=args.num_proc)
    for example in tqdm(eval_data):
        tokens = tokenize(example["text"])
        ngrams = get_ngrams(tokens, args.ngram_length)
        if ngrams:
            idx = len(eval_texts)
            eval_ngrams.update(zip(ngrams, [idx] * len(ngrams)))
            eval_datasets.append(example.get("task_name", "unknown"))
            eval_texts.append(example["text"])

    train_dataset_path = Path(args.train_dataset)
    if train_dataset_path.exists() and train_dataset_path.suffix in [".json", ".jsonl", ".csv"]:
        if train_dataset_path.suffix == ".json" or train_dataset_path.suffix == ".jsonl":
            train_data = Dataset.from_json(args.train_dataset)
        elif train_dataset_path.suffix == ".csv":
            train_data = Dataset.from_csv(args.train_dataset)
    else:
        train_data = load_dataset(args.train_dataset, split="train", num_proc=args.num_proc)

    contamination_report = train_data.map(
        lambda batch: retrieve_ngrams_batch(batch, eval_ngrams, eval_datasets, eval_texts, args.ngram_length),
        batched=True, batch_size=1000, num_proc=args.num_proc, remove_columns=train_data.column_names
    )

    contamination_report = contamination_report.map(
        lambda example: add_match_stats(example), num_proc=args.num_proc
    )

    if args.report_dataset_name:
        contamination_report_df = contamination_report.to_pandas()
        contamination_report_df.to_json(path_or_buf=args.report_dataset_name, orient="records", force_ascii=False, lines=True)

    contamination_report = contamination_report.filter(lambda x: x["diff_ratio"] > args.diff_threshold)

    if args.save_decontaminated:
        contaminated_completions = set(contamination_report["text"])
        filtered_data = train_data.filter(lambda x: x["text"] not in contaminated_completions)
        filtered_data_df = filtered_data.to_pandas()
        filtered_data_df.to_json(path_or_buf=args.decontaminated_dataset_name, orient="records", force_ascii=False, lines=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a decontamination report.jsonl for a dataset.")
    parser.add_argument("--eval_dataset", type=str,
                        default="eval.jsonl",
                        help="Name of the dataset with benchmark samples to use for decontamination.")
    parser.add_argument("--train_dataset", type=str, required=True,
                        help="Path or name of the training dataset to process.")
    parser.add_argument("--report_dataset_name", type=str,
                        help="Path of the output dataset with decontamination report.jsonl.")
    parser.add_argument("--decontaminated_dataset_name", type=str, help="Path of the decontaminated dataset.")
    parser.add_argument("--ngram_length", type=int, default=10, help="Length of the n-grams to consider.")
    parser.add_argument("--diff_threshold", type=float, default=0.85,
                        help="Threshold for filtering based on difference ratio.")
    parser.add_argument("--num_proc", type=int, default=16, help="Number of processes to use for map operations.")
    parser.add_argument("--save_decontaminated", action='store_true',
                        help="Whether to save the decontaminated dataset.")

    args = parser.parse_args()
    main(args)
