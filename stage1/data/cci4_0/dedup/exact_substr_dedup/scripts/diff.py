from rich.console import Console
from rich.text import Text
import json
import random
import json
import argparse
import subprocess
import concurrent.futures
from pathlib import Path
from utils import human_readable_size

"""

**Date:**  2025-2-20
**Last Modified:** 2025-2-23

This script provides utilities for comparing original and deduplicated JSONL datasets.

It offers two main functionalities:

1.  **Line Count (-l or --line):**  Calculates and displays the total number of lines
    in JSONL files within specified folders.  This is useful for quickly assessing
    the overall size of datasets before and after deduplication. It utilizes a
    thread pool to efficiently count lines in multiple files concurrently.

2.  **Document Comparison (-c or --compare):**  Compares individual documents
    between two JSONL files, highlighting the differences.  It identifies and
    displays removed substrings within documents, using the 'rich' library for
    colored output in the terminal.  This allows for a detailed examination of
    the specific changes made during deduplication.  The comparison is done
    on a document-by-document basis, based on matching 'id' fields. It can
    also handle cases where entire documents have been removed.

Example:
    # To get line count
    python diff.py line original_data deduplicated_data
    # output example:
    # File original_data/shard_0.jsonl has 1000 lines.
    # File original_data/shard_1.jsonl has 1500 lines.
    # File deduplicated_data/shard_0.jsonl has 800 lines.
    # File deduplicated_data/shard_1.jsonl has 1200 lines.
    # File deduplicated_data/shard_2.jsonl has 500 lines.
    # Total lines in original folder: 2500
    # Total lines in deduplicated folder: 2500

    # To compare doc
    python diff.py compare original.jsonl deduplicated.jsonl
    # output example:
    # ID: 1
    # ====================
    # This is the first document. It has some repeated content. This is the first document.
    # Press 'n' to see the next comparison (or any other key to exit)...

    # ID: 2
    # ====================
    # This is the second document.  It is unique.

    # Press 'n' to see the next comparison (or any other key to exit)...
    # ID: 4
    # ====================
    # This doc has AAA inside. More AAA.
    # Press 'n' to see the next comparison (or any other key to exit)...n
"""


def count_lines_in_file(file_path):
    """
    Calculate the number of lines in a single JSONL file using the `wc -l` command.

    Args:

        file_path: The path to the JSONL file.

    Returns:

        The number of lines in the file (integer).
        If an error occurs, return -1.
    """
    # try:
    result = subprocess.check_output(['wc', '-l', file_path], text=True)
    line_count = int(result.split()[0])
    return line_count
    # except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
    #     return -1


def count_disk_usage_in_file(file_path):
    result = subprocess.check_output(['du', '-s', file_path], text=True)
    disk_usage = int(result.split()[0]) * 1024
    return disk_usage


def count_lines_in_jsonl_files(folder_path, num_workers=None):
    """
    Calculate the total number of lines in all JSONL files within a folder 
    using a thread pool.

    Args:

        folder_path: The path to the folder containing the JSONL files.

        num_workers: The number of worker threads in the thread pool. 
        If None, os.cpu_count() is used.

    Returns:

        The total number of lines in all JSONL files within the folder.
    """

    total_lines = 0
    jsonl_files = list(Path(folder_path).glob("*.jsonl"))

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_file = {executor.submit(count_lines_in_file, file_path): file_path for file_path in jsonl_files}

        for future in concurrent.futures.as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                line_count = future.result()
                total_lines += line_count
                print(f"File {file_path} has {line_count} lines.") 
            except Exception as exc:
                print(f"File {file_path} generated an exception: {exc}")

    return total_lines


def count_disk_usage_in_jsonl_files(folder_path, num_workers=None):
    """
    """

    total_disk_usage = 0
    jsonl_files = list(Path(folder_path).glob("*.jsonl"))

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_file = {executor.submit(count_disk_usage_in_file, file_path): file_path for file_path in jsonl_files}

        for future in concurrent.futures.as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                disk_usage = future.result()
                total_disk_usage += disk_usage
                print(f"File {file_path} size: {human_readable_size(disk_usage)}.") 
            except Exception as exc:
                print(f"File {file_path} generated an exception: {exc}")

    return total_disk_usage


def get_line_counts_for_folders(original_folder, deduplicated_folder):
    """
    Calculate the total number of lines in all JSONL files in the two 
    folders before and after deduplication.

    Args:

        original_folder: The path to the folder containing the original 
        JSONL files.

        deduplicated_folder: The path to the folder containing the 
        deduplicated JSONL files.

    Returns:

        A tuple containing the total line counts of the two folders 
        (original_total_lines, deduplicated_total_lines).
    """

    original_total_lines = count_lines_in_jsonl_files(original_folder)
    deduplicated_total_lines = count_lines_in_jsonl_files(deduplicated_folder)

    return original_total_lines, deduplicated_total_lines


def get_total_disk_usage_for_folders(original_folder, deduplicated_folder):
    original_disk_usage = count_disk_usage_in_jsonl_files(original_folder)
    deduplicated_disk_usage = count_disk_usage_in_jsonl_files(deduplicated_folder)

    return original_disk_usage, deduplicated_disk_usage


def highlight_removed_substrings(original_doc, removed_indices):
    """
    Use the rich library to highlight the deduplicated parts of 
    the original document in red in the terminal.

    Args:

        original_doc: The original document (string).

        removed_indices: The list of indices of the removed substrings 
        returned by the find_removed_substrings function.
    """

    console = Console()
    text = Text()

    last_index = 0  # 记录上一次处理到的位置
    for start, end in removed_indices:
        # 添加未被移除的部分（普通样式）
        text.append(original_doc[last_index:start])
        # 添加被移除的部分（红色样式）
        text.append(original_doc[start:end], style="red")
        last_index = end

    # 添加最后剩余的未被移除部分
    text.append(original_doc[last_index:])

    console.print(text)


def find_removed_substrings(original_doc, deduplicated_doc):
    """
    Compare the document before and after deduplication, and find 
    the index positions of the substrings removed in the original document.

    Args:

        original_doc: The document before deduplication (string).

        deduplicated_doc: The document after deduplication (string).

    Returns:

        A list containing tuples of the start and end indices (left-closed, 
        right-open interval) of the removed substrings.

    For example: [(0, 12), (50, 67)] indicates that the substrings from 
    index 0 to 12 (not inclusive) and 50 to 67 (not inclusive) in the 
    original document have been removed.
    """

    removed_indices = []
    original_index = 0
    deduplicated_index = 0

    while original_index < len(original_doc) and deduplicated_index < len(deduplicated_doc):
        if original_doc[original_index] == deduplicated_doc[deduplicated_index]:
            original_index += 1
            deduplicated_index += 1
        else:
            start_index = original_index
            while original_index < len(original_doc) and (deduplicated_index >= len(deduplicated_doc) or original_doc[original_index] != deduplicated_doc[deduplicated_index] ):
                original_index +=1

            # Only if confirmed to be removed rather than modified 
            # will it be added to removed_indices
            if deduplicated_index < len(deduplicated_doc) or original_index == len(original_doc):
                removed_indices.append((start_index, original_index))
            
    # The deduplicated_doc has been traversed, but there are still 
    # remaining cases for the original_doc.
    if original_index < len(original_doc):
        removed_indices.append((original_index, len(original_doc)))
        

    return removed_indices


def compare_docs_from_jsonl(original_file_path, deduplicated_file_path, doc_field, num_comparisons):
    """
    Read the documents before and after deduplication from two JSONL files, 
    compare them item by item, and highlight the differences in the terminal.

    Args:

        original_file_path: The path to the original document's JSONL file.

        deduplicated_file_path: The path to the deduplicated document's JSONL file.

        doc_field: The field name containing the document content.

        num_comparisons: The number of documents to compare (randomly selected).
    """

    console = Console()

    with open(original_file_path, 'r', encoding='utf-8') as original_file, \
            open(deduplicated_file_path, 'r', encoding='utf-8') as deduplicated_file:

        original_data = {}  # Use a dictionary to store original data, 
                            # indexed by id.
        for line in original_file:
            item = json.loads(line)
            original_data[item['id']] = item[doc_field]

        deduplicated_data = {}  # Use a dictionary to store the deduplicated data, indexed by id.
        for line in deduplicated_file:
            item = json.loads(line)
            deduplicated_data[item['id']] = item[doc_field]

        # Retrieve the IDs of all original data and shuffle them randomly.
        original_ids = list(original_data.keys())
        random.shuffle(original_ids)

        comparisons_count = 0
        for original_id in original_ids:
            if comparisons_count >= num_comparisons:
                break

            original_doc = original_data[original_id]

            if original_id in deduplicated_data:
                deduplicated_doc = deduplicated_data[original_id]
                removed_indices = find_removed_substrings(original_doc, deduplicated_doc)

                console.print(f"ID: {original_id}")
                console.print("=" * 20)
                highlight_removed_substrings(original_doc, removed_indices)

            else:
                # If the original ID does not exist in the deduplicated 
                # data, it indicates that the entire record has been deleted.
                console.print(f"ID: {original_id} (Completely Removed)")
                console.print("=" * 20)
                console.print(Text(original_doc, style="red"))

            console.print("\n")
            comparisons_count += 1
            if comparisons_count < num_comparisons:
                if input("Press 'n' to see the next comparison (or any other key to exit)...").lower() != 'n':
                    break

def main():
    parser = argparse.ArgumentParser(description="Compare original and deduplicated JSONL datasets.")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # 'line' subcommand
    line_parser = subparsers.add_parser("line", aliases=["l"], help="Count lines in JSONL files.")
    line_parser.add_argument("original_folder", help="Path to the folder containing original JSONL files.")
    line_parser.add_argument("deduplicated_folder", help="Path to the folder containing deduplicated JSONL files.")

    # 'compare' subcommand
    compare_parser = subparsers.add_parser("compare", aliases=["c"], help="Compare documents in JSONL files.")
    compare_parser.add_argument("original_file", help="Path to the original JSONL file.")
    compare_parser.add_argument("deduplicated_file", help="Path to the deduplicated JSONL file.")
    compare_parser.add_argument("-n", "--num_comparisons", type=int, default=200, help="Number of comparisons to make (default: 200).")
    compare_parser.add_argument("-f", "--doc_field", default="text", help="Name of the field containing the document (default: 'text').")
    
    # 'disk' subcommand
    disk_parser = subparsers.add_parser("disk", aliases=["d"], help="Count disk usage in JSONL files.")
    disk_parser.add_argument("original_folder", help="Path to the folder containing original JSONL files.")
    disk_parser.add_argument("deduplicated_folder", help="Path to the folder containing deduplicated JSONL files.")

    args = parser.parse_args()

    if args.command == "line" or args.command == 'l':
        original_lines, deduplicated_lines = get_line_counts_for_folders(args.original_folder, args.deduplicated_folder)
        print(f"Total lines in original folder: {original_lines}")
        print(f"Total lines in deduplicated folder: {deduplicated_lines}")

    elif args.command == "compare" or args.command == 'c':
        compare_docs_from_jsonl(args.original_file, args.deduplicated_file, args.doc_field, args.num_comparisons)
    elif args.command == "disk" or args.command == 'd':
        original_disk_usage, deduplicated_disk_usage = get_total_disk_usage_for_folders(args.original_folder, args.deduplicated_folder)
        print(f"Total disk usage in original folder: {human_readable_size(original_disk_usage)}")
        print(f"Total disk usage in deduplicated folder: {human_readable_size(deduplicated_disk_usage)}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
