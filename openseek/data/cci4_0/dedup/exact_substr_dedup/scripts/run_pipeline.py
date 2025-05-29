import os
import argparse
import shutil
from loguru import logger
from datetime import datetime
from load_dataset_local import load_dataset
from make_suffix_array import make_suffix_array
from remove_dedup import remove_dedup
from utils import human_readable_size, get_size, assert_files_exit, Timer
from diff import get_line_counts_for_folders, get_total_disk_usage_for_folders

"""
**Date:**  2025-2-15
**Last Modified:** 2025-2-20

This script performs exact-substring deduplication on a text dataset.

It utilizes the datatrove library for dataset loading and processing, 
and Google's deduplicate-text-datasets library for the core exact-substring 
deduplication algorithm.  The script proceeds in several stages:

1. **Loading and Preprocessing:**  The input dataset is loaded and preprocessed 
   using datatrove. This includes tokenization and conversion to a binary format.
2. **Suffix Array Construction:** A suffix array is built for the concatenated 
   dataset. This data structure enables efficient identification of duplicate 
   substrings.
3. **Self-Similarity Detection:** The `dedup_dataset` executable (part of 
   deduplicate-text-datasets) is used to find self-similar (duplicate) 
   substrings within the dataset, using the suffix array.
4. **Duplicate Range Collection:** The script collects the byte ranges 
   corresponding to the identified duplicate substrings.
5. **Deduplication:** The datatrove library is used again to remove the 
   identified duplicate sections from the original dataset.
6. **Post-processing:** (Optional) The script can compress the deduplicated 
   output, merge multiple output files, and remove intermediate cache directories.

**Libraries Used:**

*   **datatrove:**  A library for efficient dataset loading and processing.
    (`https://github.com/huggingface/datatrove`).
*   **deduplicate-text-datasets:**  Google's library for exact-substring
    deduplication.  
    (`https://github.com/google-research/deduplicate-text-datasets`)

**Prerequisites:**

*   The `dedup_dataset` executable must be built and located at 
    `./target/debug/dedup_dataset`.  This typically requires a Rust environment
    and running `cargo build` in the appropriate directory of the 
    `deduplicate-text-datasets` repository.
*   Sufficient disk space is needed for both the original dataset and the
    intermediate files (suffix array, duplicate ranges, etc.). The cache
    directories can consume significant space.
*   Increase the open files limit with 'ulimit -Sn 1000000'.

**Example Usage:**
python scripts/run_pipeline.py --input-dir /path/to/input --output-dir /path/to/output
"""


def self_similar(data_file, length_threshold, cache_dir, num_threads):
    os.makedirs(cache_dir, exist_ok=True)
    logger.info(f"start to run dedup_dataset self_similar")
    cmd = f"./target/debug/dedup_dataset self-similar --data-file {data_file} --length-threshold {length_threshold} --cache-dir {cache_dir} --num-threads {num_threads} 2>&1"
    print(cmd)
    output = os.popen(cmd).read()
    logger.info(f"output of dedup_dataset self-similar is as following.")
    print(output)


def collect(data_file, length_threshold, cache_dir, output_path):
    os.makedirs(cache_dir, exist_ok=True)
    logger.info(f"start to run dedup_dataset collect")
    cmd = f"./target/debug/dedup_dataset collect --data-file {data_file} --cache-dir {cache_dir} --length-threshold {length_threshold} > {output_path}  2>&1"
    print(cmd)
    output = os.popen(cmd).read()
    logger.info(f"output of dedup_dataset collect is as following.")
    print(output)


def main(args):
    working_dir = args.working_dir
    os.makedirs(os.path.join(working_dir, "log"), exist_ok=True)
    logger.add(sink=os.path.join(working_dir, f"log/{datetime.now().strftime(r'%Y-%m-%d_%H-%M-%S')}.log"), level="INFO")
    logger.warning("Make sure running `ulimit -Sn 1000000` to \"fix\" the error that you have too many open files next step.")

    logger.info(f"mission config: {args}")
    if not os.path.exists("./target/debug/dedup_dataset"):
        logger.critical("Make sure your corrent directory is deduplicate-text-datasets and you have run cargo build successfully.")
        raise FileNotFoundError("./target/debug/dedup_dataset not exists")
    
    os.makedirs(working_dir, exist_ok=True)
    cache_dirs = [
        "cache", "es", "intermediate", "tmp"
    ]
    output_dir = os.path.join(working_dir, "final-deduped-data")

    # Remove exists cache dir
    cache_dirs = {
        cache: os.path.join(working_dir, cache)
        for cache in cache_dirs
    }
    
    for cache in cache_dirs:
        if os.path.exists(cache):
            print(f"clear cache dir {cache}.")
            shutil.rmtree(cache)
    
    timer = Timer()
    # Step 1: load dataset
    load_dataset(
        args.input_dir, args.file_type, args.threads, args.tasks, args.tokenizer,
        cache_dirs["intermediate"], cache_dirs["es"], args.samples // args.tasks,
    )
    timer.stop("Step 1: load dataset")

    assert os.path.exists(os.path.join(cache_dirs['es'], 'bytes_offsets.info'))
    assert os.path.exists(os.path.join(cache_dirs['es'], 'dataset.big_sequence'))

    timer.start()
    # Step 2: make suffix array
    make_suffix_array(
        os.path.join(cache_dirs["es"], "dataset.big_sequence"),
        cache_dirs['tmp']    
    )
    timer.stop("Step 2: make suffix array")

    assert os.path.exists(os.path.join(cache_dirs['es'], 'dataset.big_sequence.table.bin'))

    timer.start()
    # Step 3: run self similar
    self_similar(
        os.path.join(cache_dirs["es"], "dataset.big_sequence"),
        args.length_threshold,
        os.path.join(cache_dirs["cache"]),
        args.threads,
    )
    timer.stop("Step 3: run self similar")

    assert_files_exit(
        cache_dirs["cache"], 
        ["dups_dataset.big_sequence*", "sizes_dataset.big_sequence*"]
    )

    timer.start()
    # Step 4: run collect
    collect(
        os.path.join(cache_dirs["es"], "dataset.big_sequence"),
        args.length_threshold,
        os.path.join(cache_dirs["cache"]),
        os.path.join(cache_dirs["es"], "dataset.big_sequence.remove.bytearange"),
    )
    timer.stop("Step 4: run collect")

    assert os.path.exists(os.path.join(cache_dirs["es"], "dataset.big_sequence.remove.bytearange")) 

    timer.start()
    # Step 5: remove deduplicates
    remove_dedup(
        output_dir, args.min_doc_words, args.threads, args.tasks, args.tokenizer, 
        args.language, cache_dirs["intermediate"], cache_dirs["es"],
        args.max_file_size, args.compress
    )
    timer.stop("Step 5: remove deduplicates")
    
    record = []
    if not args.compress and args.samples > 0:
        os.popen(f"gunzip {cache_dirs['intermediate']}/*").read()
        disk_usage_before, disk_usage_after = get_total_disk_usage_for_folders(cache_dirs['intermediate'], output_dir)
        record.append(disk_usage_before)
        record.append(disk_usage_after )
        assert disk_usage_before> 0
        lines_before, lines_after = get_line_counts_for_folders(cache_dirs['intermediate'], output_dir)
        record.append(lines_before)
        record.append(lines_after)
        assert lines_before > 0
        logger.success(f"Disk usage before deduplicate: {human_readable_size(disk_usage_before)}, disk usage after deduplicate: {human_readable_size(disk_usage_after )}, deduplicate ratio: {1-disk_usage_after/disk_usage_before}")
        logger.success(f"Total lines before deduplicate: {lines_before}, total lines after deduplicate {lines_after}, deduplicate ratio: {1-lines_after/lines_before}")

    elif not args.compress and args.file_type == "jsonl":
        disk_usage_before, disk_usage_after = get_total_disk_usage_for_folders(args.input_dir, output_dir)
        record.append(disk_usage_before)
        record.append(disk_usage_after )
        assert disk_usage_before> 0
        lines_before, lines_after = get_line_counts_for_folders(args.input_dir, output_dir)
        record.append(lines_before)
        record.append(lines_after)
        assert lines_before > 0
        logger.success(f"Disk usage before deduplicate: {human_readable_size(disk_usage_before)}, disk usage after deduplicate: {human_readable_size(disk_usage_after )}, deduplicate ratio: {1-disk_usage_after/disk_usage_before}")
        logger.success(f"Total lines before deduplicate: {lines_before}, total lines after deduplicate {lines_after}, deduplicate ratio: {1-lines_after/lines_before}")

    if args.remove_cache:
        for cache in cache_dirs.values():
            shutil.rmtree(cache)
        print("delete cache dirs {cache, es, intermediate, tmp}.")
        os.popen(f"mv {output_dir}/* {working_dir}/").read()
        os.popen(f"rmdir {output_dir}").read()
        output_dir = working_dir

    timer.pretty_print()
    print("done.")
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="run_pipeline", description="This script performs exact-substring deduplication on a text dataset.")

    parser.add_argument(
        "--input-dir", type=str, required=True,
        help="Path to the directory containing the input dataset."
    )
    parser.add_argument(
        "--working-dir", type=str, default=f"./results/output-{datetime.now().strftime(r'%Y-%m-%d_%H-%M-%S')}", help="Path to the directory where the final deduped data and cache will be saved."
    )
    parser.add_argument(
        "--max-file-size", type=int, default=1<<31,
        help="Maximum file size of a single output file. Default to 2GB."
    )
    parser.add_argument(
        "--file-type", type=str, default="jsonl",
        help="Type of the input dataset files ('jsonl' or 'parquet'). Defaults to 'jsonl'."
    )
    parser.add_argument(
        "--language", type=str, default="cn",
        help="Language code to use for the tokenizer (e.g., 'en' for English, 'cn' for Chinese). Defaults to 'cn'."
    )
    parser.add_argument(
        "--tasks", type=int, default=32,
        help="Maximum number of tasks to use for loading the dataset. Defaults to 8."
    )
    parser.add_argument(
        "--tokenizer", type=str, default="gpt2",
        help="Name of the tokenizer to use ('gpt2' or a path to a Hugging Face tokenizer). Defaults to 'gpt2'."
    )
    parser.add_argument(
        "--length-threshold", type=int, default=800, 
        help="Length threshold for duplicate substrings. Substrings longer than this will be considered for removal. Defaults to 100."
    )
    parser.add_argument(
        "--threads", type=int, default=128,
        help="Number of threads to use for the duplicate substring search step. Defaults to 8."
    )
    parser.add_argument(
        "--min-doc-words", type=int, default=35,
        help="Minimum number of words a document must have after deduplication to be kept. Defaults to 20."
    )
    parser.add_argument(
        "--remove-cache", action="store_true",
        help="Remove intermediate cache directories after deduplication is complete."
    )
    parser.add_argument(
        "--compress", action="store_true",
        help="Compress the output dataset files using gzip."
    )
    parser.add_argument(
        "--samples", type=int, default=-1,
        help="Limit input file lines. -1 for infinite."
    )


    args = parser.parse_args()
    main(args)
