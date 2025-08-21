# Overview
- No-Sampling：No sampling is applied to the dataset, serving as the baseline.
- Phi4-Sampling：Following the Phi-4 paper, synthetic and rewritten sub-domain datasets are upsampled.
  
Both configs were trained with 100B tokens for performance comparison.

# Evaluation
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
