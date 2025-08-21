# Experiment Overview
- Default: Using the default training configuration, InitStd = 0.02, RouterDtype = bfloat16, AuxCoeff = 0.02.
- InitStd-RouterDtype-AuxCoeff: InitStd = 6e-3, RouterDtype = float32, AuxCoeff = 0.0001.
- Add-MTP: Based on the above InitStd-RouterDtype-AuxCoeff, MTP module was added zai, with an auxiliary loss weight of 0.3.

All configs were trained with 100B tokens for performance comparison.

# Evaluation
| Category                    | Metrics (shots)         | Default (100B) | InitStd-RouterDtype-AuxCoeff (100B) | Add-MTP (100B) |
|----------------------------|-------------------------|----------------|--------------------------------------|---------------|
| English-Commonsense Reasoning | HellaSwag (5-shot)       | 0.4414         | 0.4544                               | 0.4568        |
|                            | TruthfulQA (0-shot)     | 0.3735         | 0.3707                               | 0.3438        |
|                            | Winogrande (5-shot)     | 0.5927         | 0.5777                               | 0.6062        |
|                            | CommonsenseQA (5-shot)  | 0.2056         | 0.1966                               | 0.2531        |
|                            | PIQA (5-shot)           | 0.7274         | 0.7476                               | 0.7454        |
|                            | OpenBookQA (5-shot)     | 0.2760         | 0.3040                               | 0.3180        |
|                            | BoolQ (5-shot)          | 0.6294         | 0.6465                               | 0.6471        |
| English-Problem-Solving    | ARC Easy (5-shot)       | 0.7029         | 0.7353                               | 0.7264        |
|                            | ARC Challenge (5-shot)  | 0.3703         | 0.3993                               | 0.4053        |
|                            | MMLU (5-shot)           | 0.2602         | 0.2671                               | 0.3397        |
| English-Mathematics        | GSM8K (5-shot)          | 0.0182         | 0.0265                               | 0.0136        |
|                            | Minerva Math (4-shot)   | 0.0094         | 0.0098                               | 0.0080        |
| Chinese                    | CEval (5-shot)          | 0.2645         | 0.2600                               | 0.3076        |
|                            | CMMLU (5-shot)          | 0.2455         | 0.2475                               | 0.2856        |
| **Average Metrics**        | Average-English(w/o Math)| 0.4579         | 0.4699                               | 0.4842        |
|                            | Average-English         | 0.3839         | 0.3946                               | 0.4053        |
|                            | Average-Chinese         | 0.2550         | 0.2538                               | 0.2966        |
|                            | Average                 | 0.3655         | 0.3745                               | 0.3897        |
|                            | Average(w/o Math)       | 0.4241         | 0.4339                               | 0.4529        |
