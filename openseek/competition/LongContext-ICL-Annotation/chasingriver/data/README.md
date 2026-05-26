# Datasets

This repository provides the official datasets for the **LLM Automatic Data Annotation**.  
The datasets are specifically designed to evaluate the capability of Large Language Models (LLMs) to perform **automatic data annotation under ultra-long context settings** using the In-context Learning (ICL) paradigm.


## Overview

- Most tasks require a **minimum ICL context length of 30K tokens**, deliberately exceeding standard context limits to evaluate long-context understanding, prompt engineering, and example selection strategies.
- Task **openseek-8** is configured with a **shorter minimum context length (15K tokens)** and a **smaller test set**, reflecting the unique challenges of **kernel generation**.
- All datasets are released with **fixed and standardized test splits** to ensure fair comparison and reproducibility across submissions.
- The task suite covers a **diverse range of domains and reasoning types**, including symbolic reasoning, linguistic analysis, natural language inference, code-related tasks, and open-ended generation.


| Task ID | task name | Minimum ICL context | Test sample number | 
| --- | --- | --- | --- | 
| openseek-1 | closest_integers |  30K  | 500 | 
| openseek-2 | count_nouns_verbs |  30K  | 500 | 
| openseek-3 | collatz_conjecture |  30K  | 500 | 
| openseek-4 | conala_concat_strings |  30K  | 500 | 
| openseek-5 | semeval_2018_task1_tweet_sadness_detection |  30K  | 500 | 
| openseek-6 | mnli_same_genre_classification |  30K  | 500 | 
| openseek-7 | jeopardy_answer_generation_all |  30K  | 500 | 
| openseek-8 | kernel_genernation |  16K  | 166 | 


## Data Structure
The datasets are organized in JSON format, with each task having its own json file. Here's a brief overview of the data structure:

- `task_id`: A unique identifier for the task.
- "task_name": A short human-readable name of the task.
- `Definition`: A detailed description of what the model should do.
- `examples`: Demonstration samples intended for understanding the task format (not necessarily used for scoring). Each example typically includes: `id`, `input` and `output`.
- `test_samples`: The samples to be predicted by participants. Labels/ground truth is hidden. Each test sample typically includes: `id` and `input`.
- `License`: The dataset license name and/or a URL to the license text, describing allowed use and redistribution.


## Usage Notes

- Participants must use the **official datasets as provided**, without altering test splits or labels, for leaderboard evaluation.
- Any preprocessing steps, context construction strategies, or example selection mechanisms should be clearly described in the accompanying technical report.
- All experimental results must be **fully reproducible** using the datasets in this repository.