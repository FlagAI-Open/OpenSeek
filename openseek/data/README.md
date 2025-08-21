# Data

## Overview

This directory introduces two significant contributions to the OpenSeek:

- CCI4.0: A large-scale, high-quality bilingual (Chinese-English) pre-training dataset designed to enhance LLM reasoning capabilities.

- Data Mixture Experiments: A comparative study evaluating the impact of different data mixture strategies on LLM performance.

### CCI4.0

CCI4.0 is an extensive bilingual pre-training dataset, occupying approximately 35 TB of disk space. It is engineered with a focus on superior data quality and fostering diverse, human-like reasoning trajectories in LLMs. The dataset is divided into two main components:

- CCI4.0-M2-Base: This sub-dataset forms the foundational layer of CCI4.0.

- CCI4.0-M2-CoT: This sub-dataset consists of a massive collection of Chain-of-Thought (CoT) templates.

The CCI4.0-M2-Base sub-dataset amalgamates diverse data sources:

- A 5.2 TB meticulously curated Chinese web corpus.

- A 22.5 TB English subset derived from Nemotron-CC.

- Varied content from specialized domains including mathematics, Wikipedia, arXiv preprints, and code repositories.

Recognizing that data quality standards are dynamic and domain-specific, we developed a novel model-centric data processing pipeline. This pipeline ensures high-quality data through:

- Two-Stage Deduplication: Efficiently removes redundant data.

- Multiclassifier Quality Scoring: Employs multiple classifiers to assess and score data quality.

- Domain-Aware Fluency Filtering: Filters data based on fluency, tailored to the specific characteristics of each domain.

This approach moves beyond traditional methods that rely heavily on expert experience and manual labor, offering a more scalable and automated solution to data curation.

A key innovation within CCI4.0 is the CCI4.0-M2-CoT sub-dataset, which comprises 4.5 billion Chain-of-Thought (CoT) templates. Unlike methods that distill CoT from larger, pre-existing models, our staged CoT extraction process is designed to:

- Exemplify a wider variety of reasoning patterns.

- Significantly reduce the likelihood of generating "hallucinated" or factually incorrect reasoning steps.

Empirical evaluations consistently demonstrate that LLMs pre-trained on the CCI4.0 dataset benefit significantly from its cleaner and more reliable training signals. This translates to notable improvements in performance across various downstream tasks, with particularly strong gains observed in mathematics and code reflection tasks.

Our findings highlight the critical importance of rigorous data curation and the incorporation of human-like thinking patterns (via CoT templates) in advancing LLM capabilities. This work also offers insights into automating the processing of large-scale pre-training corpora.

More information can be found in [huggingface](https://huggingface.co/datasets/BAAI/CCI4.0-M2-Base-v1) or [readme](./cci4_0/README.md).

### Data Mixture Experiment

To investigate the impact of different data mixture strategies on LLM performance, we conducted experiments comparing two main configurations. Both configurations were trained for 100 billion tokens to ensure a fair comparison.

1. No-Sampling (Baseline):
    - The dataset was used without any specific sampling applied to its sub-domains. This serves as our control group.

2. Phi4-Sampling:
    - Inspired by the methodology presented in the Phi-4 paper, this configuration involves upsampling synthetic and rewritten sub-domain datasets.

We evaluated the models trained with these configurations across a range of benchmarks covering English commonsense reasoning, English problem-solving, English mathematics, and Chinese language tasks.

| Category                    | Metrics (shots)      | No-Sampling (100B) | Phi4-Sampling (100B) |
|----------------------------|----------------------|----------------------------|------------------------------|
| **English-Commonsense Reasoning** | HellaSwag (5-shot)       | 0.4722                     | 0.4827                       |
|                            | TruthfulQA (0-shot)     | 0.4114                     | 0.4016                       |
|                            | Winogrande (5-shot)     | 0.5975                     | 0.6117                       |
|                            | CommonsenseQA (5-shot)  | 0.2023                     | 0.1933                       |
|                            | PIQA (5-shot)           | 0.7612                     | 0.7644                       |
|                            | OpenBookQA (5-shot)     | 0.2640                     | 0.3040                       |
|                            | BoolQ (5-shot)          | 0.5615                     | 0.6621                       |
| **English-Problem-Solving**| ARC Easy (5-shot)       | 0.7222                     | 0.7487                       |
|                            | ARC Challenge (5-shot)  | 0.3703                     | 0.3993                       |
|                            | MMLU (5-shot)           | 0.2707                     | 0.2652                       |
| **English-Mathematics**    | GSM8K (5-shot)          | 0.0349                     | 0.1039                       |
|                            | Minerva Math (4-shot)   | 0.0118                     | 0.0248                       |
| **Chinese**                | CEval (5-shot)          | 0.2481                     | 0.2281                       |
|                            | CMMLU (5-shot)          | 0.2469                     | 0.2538                       |
| **Average Metrics**        | Average-English (w/o Math) | 0.4633                 | 0.4833                       |
|                            | Average-English         | 0.3900                     | 0.3992                       |
|                            | Average-Chinese         | 0.2475                     | 0.2538                       |
|                            | Average                 | 0.3696                     | 0.3888                       |
|                            | Average (w/o Math)      | 0.4274                     | 0.4429                       |

More information can be found [here](./data_mix_exp/README.md).
