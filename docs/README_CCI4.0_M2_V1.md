# CCI4.0-M2 v1 Dataset Documentation

## Overview
CCI4.0-M2 v1 is a comprehensive dataset collection consisting of three specialized subsets designed for different aspects of language model training. This document provides detailed specifications for each subset.
| Dataset | Download Link |
| --- | --- |
| CCI4.0-M2-Base v1 | [hf](https://huggingface.co/datasets/BAAI/CCI4.0-M2-Base-v1) |
| CCI4.0-M2-CoT v1 | [hf](https://huggingface.co/datasets/BAAI/CCI4.0-M2-CoT-v1) |
| CCI4.0-M2-Extra v1 | [hf](https://huggingface.co/datasets/BAAI/CCI4.0-M2-Extra-v1) |


## Subset Specifications

### CCI4.0-M2-Base v1
- **Purpose**: Core pretraining data for general language understanding
- **Data Composition**:
  - Chinese: 30% (Including data from cooperation projects and open-source projects)
  - English: 70%, primarily sourced from Nemotron-CC
- **Source Distribution**:
  - Webpages: 100%
- **Total Volume**: 3500GB
- **License**: Alpache 2.0
- **Processing**:
  - Document-level and phrase-level deduplication
  - Rigorous quality filtering through the integration of three quality scores
  - Knowledge enhancement via LLM rewriting and generation
  - Filtering based on LLM Loss grouped by domain

Data sources for the first batch of open-source CCI4.0-M2-Base v1
| Serial Number | Data Source | Open Source License |
| - | --- | --- |
| 1 | ChineseWebText2.0 | apache-2.0 |
| 2 | HPLT2.0_cleaned/zho_Hans | cc0-1.0 |
| 3 | TeleChat-PTD | apache-2.0 |
| 4 | data from cooperation projects  | apache-2.0 |
| 5 | Nemotron-CC | Common Crawl License |

*Nemotron-CC is likely subject to the Common Crawl License. Therefore, we will only release the metadata of Nemotron-CC along with our processed scores.*

### CCI4.0-M2-CoT v1 
- **Purpose**: Chain-of-Thought reasoning enhancement
- **Data Composition**:
  - English: 40%
- **Total Volume**: 4200GB
- **Special Features**:
  - Step-by-step CoT trajactorys
  - Detailed question generation
  - Multiple domain coverage(e.g., math, code, webpages)
#### Introduction and Demonstration of Synthesized Chain-of-Thought (CoT)

The Chain-of-Thought (CoT) in the CCI4.0-M2-CoT v1 subset is synthesized to enhance the reasoning capabilities of language models. This synthesis process involves generating step-by-step reasoning trajectories based on various data sources

The following image illustrates the CoT synthesis pipeline:

![CoT Pipeline](CoT_Pipeline.png)

This pipeline showcases how the raw data from different sources is processed and transformed into structured CoT data, which can be used for training language models to perform complex reasoning tasks.

| Data Source | License |
| --- | --- |
| KodCode/KodCode-V1 | cc-by-nc-4.0 |
| facebook/natural_reasoning | cc-by-nc-4.0 |
| allenai/dolma | odc-by |
| allenai/dolmino-mix-1124 | odc-by |
| HuggingFaceTB/finemath | odc-by |
| open-web-math/open-web-math | ODC-By 1.0 |
| allenai/dolmino-mix-1124 | odc-by |

*Based on the data from these sources, CoT synthesis and instruction synthesis of reverse thinking have been carried out. Due to license considerations, a separate directory will be created to open-source these data.*


### CCI4.0-M2-Extra v1
- **Purpose**: Supplemental domain-specific knowledge
- **Total Volume**: to be determined
- **License**: license to be determined

| Data Source | License |
| --- | --- |
| MAP-CC | CC-BY-NC-ND-4.0 |
| fineweb-2 | ODC-BY |
| wanjuan/data/raw/nlp/CN | CC-BY-4.0 |
| starcoder | Multiple Licenses, see https://huggingface.co/datasets/bigcode/the-stack-dedup/blob/main/licenses.json |
| opc-annealing-corpus | Multiple agreements. Some corpora are from the-stack-v2. See agreements at: https://huggingface.co/datasets/bigcode/the-stack-v2/blob/main/license_stats.csv |
| smollm-corpu | Multiple agreements. Some corpora are from the-stack-v2. See agreements at: https://huggingface.co/datasets/bigcode/the-stack-v2/blob/main/license_stats.csv |
| dolma_pes2o_v2 | ODC-BY |
| pes2o | ODC-BY |
| dolma | ODC-BY |
| opc-fineweb-math-corpus | ODC-BY |
| proof-pile-2 | MIT, BSD, or Apache, ODC-By 1.0 license, etc. |

*These datasets will be separately open-sourced under a single project due to their license agreements.*

## Usage Agreement
Users need to comply with the usage agreement of the CCI 3.0 HQ dataset. You can view the agreement by clicking on the following link: （[View Usage Agreement](https://data.baai.ac.cn/resources/agreement/cci_usage_aggrement.pdf)）.

## Citation
Please cite using:
```
@dataset{cci4_m2_v1,
  title={CCI4.0-M2 v1 Dataset Collection},
  author={OpenSeek Team},
  year={2025},
  publisher={Beijing Academy of Artificial Intelligence}
}
```
