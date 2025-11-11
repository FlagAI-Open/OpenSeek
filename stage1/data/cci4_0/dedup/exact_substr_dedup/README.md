# Exact Substring Deduplication Pipeline

We provide a pipeline for performing exact substring deduplication on text datasets. It integrates the Rust-based tool [`deduplicate-text-datasets`](https://github.com/google-research/deduplicate-text-datasets) with a set of Python scripts to efficiently remove duplicate substrings, making it ideal for cleaning large datasets used in natural language processing tasks.

## Introduction

Exact substring deduplication identifies and removes identical substrings within a text dataset, reducing redundancy and improving data quality. This pipeline leverages `deduplicate-text-datasets` for core deduplication algorithms and enhances it with Python scripts for dataset handling and processing.

## Prerequisites

- **Rust and Cargo**: Required to compile `deduplicate-text-datasets`. Install from [rust-lang.org](https://www.rust-lang.org/).

- **Python 3.10**: Ensure Python is installed on your system.

- **Python Libraries**:

  - [`datatrove`](https://github.com/huggingface/datatrove): For loading datasets and performing deduplication steps.
  - [`rich`](https://github.com/Textualize/rich): For displaying tables and comparing dataset changes.
  - [`loguru`](https://github.com/Delgan/loguru): For logging pipeline activities.

  Install these libraries using pip:

  ```bash
  pip install datatrove rich loguru
  ```

## Installation

1. **Clone and Compile `deduplicate-text-datasets`**:

   ```bash
   git clone https://github.com/google-research/deduplicate-text-datasets.git
   cd deduplicate-text-datasets
   cargo build --release
   ```

   This compiles the `dedup_dataset` executable, located at `./target/release/dedup_dataset`.

2. **Integrate Scripts**:
   Copy all scripts from the `scripts` directory to `deduplicate-text-datasets/scripts`. If prompted, overwrite any existing files with the same names:

   ```bash
   cp -r ./scripts/* ./deduplicate-text-datasets/scripts/
   ```

## Usage

You can perform deduplication using two approaches: a single-script execution for simplicity or a step-by-step process for greater control.

### Single Script Execution

The easiest way to deduplicate a dataset is by running the `run_pipeline.py` script, which automates the entire process.

#### Example

```bash
python scripts/run_pipeline.py --input-dir ./data/input --working-dir ./results --threads 16 --tasks 32 --length-threshold 100 --min-doc-words 50
```

- **`--input-dir`**: Path to the input dataset directory.

- **`--working-dir`**: Directory where intermediate files and final deduplicated data (in `final-deduped-data`) are saved.

- **Other Options**: See all available parameters with:

  ```bash
  python scripts/run_pipeline.py --help
  ```

### Step-by-Step Deduplication

For more flexibility, you can execute each deduplication step individually. This process mirrors the workflow in `run_pipeline.sh`.

1. **Load Dataset**:
   Convert the input dataset into a format suitable for deduplication.

   ```bash
   python scripts/load_dataset_local.py --data-folder ./data/input --working-folder ./working --file-type jsonl --threads 16 --tasks 32
   ```

2. **Generate Suffix Array**:
   Build a suffix array to enable efficient substring matching.

   ```bash
   python scripts/make_suffix_array.py --working-folder ./working
   ```

3. **Find Self-Similar Substrings**:
   Use `deduplicate-text-datasets` to identify duplicate substrings.

   ```bash
   ./target/release/dedup_dataset self-similar --data-file ./working/es/dataset.big_sequence --length-threshold 100 --cache-dir ./working/cache --num-threads 16
   ```

4. **Collect Duplicate Ranges**:
   Gather byte ranges of duplicates for removal.

   ```bash
   ./target/release/dedup_dataset collect --data-file ./working/es/dataset.big_sequence --cache-dir ./working/cache --length-threshold 100 > ./working/es/dataset.big_sequence.remove.bytearange
   ```

5. **Remove Duplicates**:
   Apply the deduplication based on collected ranges.

   ```bash
   python scripts/remove_dedup.py --working-folder ./working --min-doc-words 50 --threads 16 --tasks 32
   ```

   The deduplicated data will be saved in `./working/final-deduped-data`.

## Important Notes

- **Large Datasets**: Before processing large datasets with `run_pipeline.py` or the step-by-step method, increase the open file limit to avoid errors during suffix array construction:

  ```bash
  ulimit -Sn 1000000
  ```

  Run this command in your terminal prior to executing any scripts.

- **Memory Usage**: This deduplication method is memory-intensive, especially during suffix array construction and substring matching. For very large datasets, consider splitting the dataset into smaller chunks and processing them separately to avoid memory exhaustion. 
- **Disk Space**: Ensure sufficient disk space is available, as intermediate files (e.g., suffix arrays, caches) can be substantial.

## Examples

### Single Script Execution

Deduplicate a dataset with custom parameters:

```bash
python scripts/run_pipeline.py --input-dir ./data/input --working-dir ./results/output-2025-01-01 --threads 16 --tasks 32 --length-threshold 100 --min-doc-words 50 --file-type jsonl
```

Output is saved in `./results/output-2025-01-01/final-deduped-data`.

### Step-by-Step Execution

Process a dataset manually:

```bash
# Step 1: Load dataset
python scripts/load_dataset_local.py --data-folder ./data/input --working-folder ./working --file-type jsonl --threads 16 --tasks 32

# Step 2: Generate suffix array
python scripts/make_suffix_array.py --working-folder ./working

# Step 3: Find self-similar substrings
./target/release/dedup_dataset self-similar --data-file ./working/es/dataset.big_sequence --length-threshold 100 --cache-dir ./working/cache --num-threads 16

# Step 4: Collect duplicate ranges
./target/release/dedup_dataset collect --data-file ./working/es/dataset.big_sequence --cache-dir ./working/cache --length-threshold 100 > ./working/es/dataset.big_sequence.remove.bytearange

# Step 5: Remove duplicates
python scripts/remove_dedup.py --working-folder ./working --min-doc-words 50 --threads 16 --tasks 32
```

Final output is in `./working/final-deduped-data`.

## Additional Utilities

- **Compare Datasets**: Use `diff.py` to analyze differences between original and deduplicated datasets:

  - Line count:

    ```bash
    python scripts/diff.py line ./data/input ./working/final-deduped-data
    ```

  - Document comparison: Compare specific documents before and after deduplication. The two input files must come from the `intermediate` and `final-deduped-data` folders within the same working directory, and their file numbers must match (e.g., `shard_0.jsonl` from both folders) to ensure they contain the same batch of data pre- and post-deduplication:

    ```bash
    python scripts/diff.py compare ./working/intermediate/shard_0.jsonl ./working/final-deduped-data/shard_0.jsonl
    ```

