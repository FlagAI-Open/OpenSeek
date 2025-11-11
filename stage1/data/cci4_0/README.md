Data Processing Pipeline in CCI4.0
---
### Overview
The data processing pipeline in CCI4.0 is meticulously designed to yield a high-quality, diverse, and robust dataset, comprising principal stages: Deduplication, Multi-classifier Quality Scoring, Fluency Filtering, Data Synthesis (including CoT synthesis), and comprehensive Privacy and Toxicity Handling. 
- **Deduplication**. Remove redundancy and operate at both a global document level and a finer-grained string level.
- **Multi-classifier Quality Scoring**. Evaluate data integrity and relevance across various dimensions.
- **Fluency Filtering**. Applied independently to each domain to effectively remove samples with notably poor linguistic flow.
- **Data Synthesis**. Leverage filtered high-quality samples as seeds with large models to generate novel data instances in diverse formats. 
- **Privacy and Toxicity Handling**. Incorporate essential safety and privacy measures.
---
### Experiments
We conduct ablation studies to verify the effectiveness of the operations in our data processing pipeline. Specifically, we provide ablations related to operations specially proposed in CCI4.0, *i.e.*, Multi-classifier Quality Scoring and Fluency Filtering. 

We used the LightEval library for model evaluation. All evaluations were conducted in a zero-shot setting. To directly compare the performance across different datasets, we use Average, which refers to the overall average score across all Chinese and English benchmarks. The evaluation metrics include:
- Chinese benchmarks: CEval and CMMLU. 
- English benchmarks: ARC-C, ARC-E, HellaSwag, Winograd, MMLU, OpenbookQA, PIQA and SIQA.

#### Multi-classifier Quality Scoring
To ensure the high quality of our processed datasets, a multi-faceted quality classification approach was employed, tailored to the characteristics of both English and Chinese corpora. For the English web data, primarily sourced from Nemotron-CC, three independent quality classifiers were utilized to score each document, as utilized in Nemotron-CC. For the Chinese dataset, three specialized Chinese quality classifiers are designed and trained for the Chinese dataset prepocessing. 

To validate the effectiveness of our Chinese data processing pipeline, we compare it against the Chinese web corpus used in CCI 3.0, which serves as the baseline. The following figure illustrates the model average performance across downstream evaluation tasks, trained with different high-quality (HQ) Chinese corpora in CCI3.0 and CCI4.0. The result demonstrates that, compared to CCI 3.0, the Chinese web data in CCI 4.0 significantly improves training efficiency. 

![image](/figs/CN_ab.png)

#### Fluency Filtering

Considering the significant variations of the loss distributions aross different domains, we employed a multilingual domain classifier to categorize all raw corpus data into distinct domains, then computed the Perplexity Loss for all samples within each domain to do the domain-specific fluency filtering. To mitigate the influence of extreme outliers within each domain, we established a filtering criterion based on the calculated loss distributions. Specifically, samples exceeding the 99.5th percentile of the loss value within their respective domains were systematically removed.

To assess the effectiveness of Fluency Filtering, we compare models trained on English corpora before and after the Fluency Filtering. The following figure presents the average performance during training, where filtering based on Perplexity loss improves training efficiency throughout the learning process. 

![image](/figs/EN_ab.png)
