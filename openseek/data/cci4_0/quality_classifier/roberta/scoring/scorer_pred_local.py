# !/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import time
import torch

from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scorer-model-path", type=str, default="", help="file path", required=True
    )
    parser.add_argument(
        "--input-file-path", type=str, default="", help="file path", required=True
    )
    parser.add_argument(
        "--output-file-path", type=str, default="", help="file path", required=True
    )
    parser.add_argument(
        "--score-thres", type=float, default=3.0, help="score thres", required=False
    )
    parser.add_argument(
        "--text-key", type=str, default="text", help="file path", required=False
    )
    parser.add_argument(
        "--output-key", type=str, default="score", help="file path", required=False
    )
    parser.add_argument(
        "--do-score-filter",
        action="store_true",
        default=False,
        help="do score filter or not",
        dest="do_score_filter",
    )
    args = parser.parse_args()

    model_dir = args.scorer_model_path
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        trust_remote_code=False,
        ignore_mismatched_sizes=False,
    )
    model.cuda()
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        use_fast=True,
        token=None,
        trust_remote_code=False,
    )

    max_length = 2048

    import jsonlines

    file_path = args.input_file_path
    output_file_path = args.output_file_path
    writer = jsonlines.open(output_file_path, mode="w")

    dir_path = None
    if os.path.isdir(file_path):
        dir_path = os.listdir(file_path)
    else:
        dir_path = [file_path]

    lines = 0
    filtered = 0
    start_time = time.time()

    for file_path in dir_path:
        input_file = os.path.join(args.input_file_path, file_path)
        with jsonlines.open(input_file) as reader:
            for line in reader:
                lines += 1
                if lines % 1000 == 0:
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    samples_per_second = lines / elapsed_time
                    print(
                        f"Processed {lines} lines in {elapsed_time:.2f} seconds.",
                        flush=True,
                    )
                    print(f"Samples per second: {samples_per_second:.2f}.", flush=True)

                if args.text_key not in line:
                    filtered += 1
                    continue
                sentecnce = line[args.text_key]
                result = tokenizer(
                    [sentecnce],
                    padding=False,
                    max_length=max_length,
                    truncation=True,
                    return_tensors="pt",
                ).to("cuda")
                for key in result:
                    result[key] = torch.tensor(result[key])

                model_out = model(**result)
                score = float(model_out.logits.tolist()[0][0])
                if args.do_score_filter and score < args.score_thres:
                    filtered += 1
                    continue

                line[args.output_key] = score
                writer.write(line)

        end_time = time.time()
        elapsed_time = end_time - start_time
        samples_per_second = lines / elapsed_time
        print(
            f"Processed {lines} lines in {elapsed_time:.2f} seconds, Filtered {filtered} samples.",
            flush=True,
        )
        print(f"Samples per second: {samples_per_second:.2f}.", flush=True)