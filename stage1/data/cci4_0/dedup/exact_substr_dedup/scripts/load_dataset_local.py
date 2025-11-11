import os
import argparse
from datatrove.executor.base import PipelineExecutor
from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.dedup import ESDatasetToSequence, ESMergeSequences, ESRangeRemover
from datatrove.pipeline.readers import ParquetReader, JsonlReader
from datatrove.pipeline.writers.jsonl import JsonlWriter


"""
example on how to run exact-substring deduplication. It also requires using
https://github.com/google-research/deduplicate-text-datasets after stage 1, 2
1) ESDatasetToSequence maps 1 file into a sequence S. With unique separators at the beginning of each document. It also
    saves the bytes offset of where each individual document begins.
2) ESMergeSequences merges all sequences into a big single sequence. It also saves the bytes offset per file.

---
after stage two you should use deduplicate-text-datasets scripts to create the suffix array and find all the
duplicates. The final output of these scripts should be a .bytearange file with the ranges in bytes wrt the big
sequence
---

"""

def count_files_with_suffix(directory, suffix):
    count = 0
    for filename in os.listdir(directory):
        if filename.endswith(suffix):
            count += 1
    return count


def load_dataset(
    data_folder, data_file_type="jsonl", workers=8,
    tasks=8, tokenizer="gpt2", intermediate="intermediate",
    es="es", samples=-1,
):
    files_num = count_files_with_suffix(data_folder, f".{data_file_type}")
    if data_file_type == "jsonl":
        reader = JsonlReader(
            data_folder,
            limit=samples,
            shuffle_files=(samples > 0)
        )
    elif data_file_type == "parquet":
        reader = ParquetReader(
            data_folder,
            limit=samples,
            shuffle_files=(samples > 0)
        )
    else:
        raise Exception(f"{data_file_type} type dataset is not support, using jsonl or parquet file.")

    tasks_stage_1 = min(tasks, files_num)

    pipeline_1 = [
        reader,
        JsonlWriter(
            intermediate,
        ),
        ESDatasetToSequence(
            output_folder=es,
            tokenizer_name_or_path=tokenizer,
        ),
    ]

    pipeline_2 = [
        ESMergeSequences(
            data_folder=es,
            tasks_stage_1=tasks_stage_1,
        )
    ]

    executor_1: PipelineExecutor = LocalPipelineExecutor(pipeline=pipeline_1, workers=workers, tasks=tasks_stage_1)

    executor_2: PipelineExecutor = LocalPipelineExecutor(pipeline=pipeline_2, workers=1, tasks=1)

    print(executor_1.run())
    print(executor_2.run())


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--data-folder", type=str, required=True)
    parser.add_argument("--working-folder", type=str, default="./")
    parser.add_argument("--file-type", type=str, default="jsonl")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--tasks", type=int, default=8)

    args = parser.parse_args()

    load_dataset(
        data_folder=args.data_folder,
        data_file_type=args.file_type,
        workers=args.workers,
        tasks=args.tasks,
        intermediate=os.path.join(args.working_folder, "intermediate"),
        es=os.path.join(args.working_folder, "es")
    )
    