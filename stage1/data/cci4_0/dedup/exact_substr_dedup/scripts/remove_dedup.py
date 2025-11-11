import os
import argparse

from datatrove.executor.base import PipelineExecutor
from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.dedup import (
    ESRangeRemover,
)
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.writers.jsonl import JsonlWriter
from datatrove.utils.typeshelper import Languages


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

3) ESRangeRemover reads from DocumentsPipeline and duplicates ranges at the same time removing the duplicates ranges.


to run stage 1,2 call run_stage_1_2, after you have followed deduplicate-text-datasets instructions in the README you
can call stage 3 with run_stage_3.

N.B
The steps

"""

def count_files_with_suffix(directory, suffix):
    count = 0
    for filename in os.listdir(directory):
        if filename.endswith(suffix):
            count += 1
    return count


def remove_dedup(
    output_dir, min_doc_words=20, workers=8, tasks=8, 
    tokenizer="gpt2", language="en", intermediate="intermediate",
    es="es", max_file_size=1<<31, compression=None
):
    if language in ["cn", "zh", "Chinese", "chinese"]:
        language = Languages.chinese
    elif language in ["en", "english", "English"]:
        language = Languages.english
    else:
        print(f"language not support. use default value: en")
        language = Languages.english

    intermediate_num = count_files_with_suffix(intermediate, ".jsonl.gz")
    tasks = min(tasks, intermediate_num)

    pipeline_3 = [
        JsonlReader(
            intermediate
        ),  # must be the same data that was passed to DatasetToSequence
        ESRangeRemover(
            # sequence_folder=f"{os.getcwd()}/es/",
            sequence_folder=es,
            min_doc_words=min_doc_words,
            language=language,
            tokenizer_name_or_path=tokenizer,
        ),
        JsonlWriter(
            output_dir,
            compression="gzip" if compression else None,
            max_file_size=max_file_size,
        ),
    ]

    executor_3: PipelineExecutor = LocalPipelineExecutor(
        pipeline=pipeline_3, workers=workers, tasks=tasks
    )

    print(executor_3.run())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--working-folder", type=str, default="./")
    parser.add_argument("--min-doc-words", type=int, default=50)
    parser.add_argument("--tasks", default=16, type=int)
    parser.add_argument("--threads", type=int, default=8)

    args = parser.parse_args()

    remove_dedup(
        output_dir=os.path.join(args.working_folder, "final-deduped-data"),
        min_doc_words=args.min_doc_words,
        workers=args.threads,
        tasks=args.tasks,
        intermediate=os.path.join(args.working_folder, "intermediate"),
        es=os.path.join(args.working_folder, "es"),
    )

