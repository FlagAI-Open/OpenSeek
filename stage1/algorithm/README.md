# Algorithm

## Overview

This document details two key areas of algorithmic experimentation conducted to enhance model performance:

- Hyperparameter Tuning Experiments: Focused on optimizing critical system and model parameters.

- Multiple Token Prediction (MTP) Experiments: Aimed at improving the model's ability to predict sequences of tokens.

All experimental configurations were trained for 100 billion tokens to ensure a consistent basis for comparison. You can find detailed instructions about the algorithm experiment [here](../../docs/algorithm_exp.md#algorithm-experiment).

## Hyperparameter experiment

The primary goal of the hyperparameter experiments was to identify optimal settings for key parameters that significantly influence model behavior and training efficiency. The parameters investigated include those related to weight initialization, Mixture of Experts (MoE) router precision, and MoE auxiliary loss.


Two main configurations were compared:

1. Default:

    - init_method_std: 0.02

    - moe_router_dtype: bfloat16

    - moe_aux_loss_coeff: 0.02

2. InitStd-RouterDtype-AuxCoeff (Tuned):

    - init_method_std: 6e-3 (0.006)

    - moe_router_dtype: float32

    - moe_aux_loss_coeff: 0.0001

The "InitStd-RouterDtype-AuxCoeff" configuration, with tuned hyperparameters, generally showed improved performance over the "Default" settings across several benchmarks.

- English Performance: The tuned configuration demonstrated gains in average English performance (0.3946 vs. 0.3839 overall, and 0.4699 vs. 0.4579 excluding mathematics). Specific improvements were seen in HellaSwag, PIQA, OpenBookQA, BoolQ, ARC (Easy and Challenge), and MMLU.

- Mathematics: English mathematics tasks like GSM8K saw an increase from 0.0182 to 0.0265.

- Chinese Performance: Results on Chinese benchmarks were mixed, with a slight decrease in CEval but a small improvement in CMMLU.

- Overall Average: The overall average score improved from 0.3655 to 0.3745 with the tuned parameters.

These results suggest that careful tuning of initialization methods, router data types, and auxiliary loss coefficients can lead to tangible benefits in model performance.

## MTP experiment

The MTP experiments were designed to evaluate the impact of incorporating a Multiple Token Prediction module. This module aims to enhance the model's capability to predict subsequent tokens in a sequence more effectively.

The MTP experiment built upon the optimized hyperparameter settings:

- Add-MTP: This configuration used the "InitStd-RouterDtype-AuxCoeff" settings as a base and introduced an MTP module.

    - num_mtp_predictor: 1 (indicating prediction of one future token beyond the immediate next token)

    - mtp_loss_coeff: 0.3 (weight for the MTP auxiliary loss)

Adding the MTP module (Add-MTP) generally led to further improvements across a majority of the evaluated benchmarks compared to both the "Default" and the "InitStd-RouterDtype-AuxCoeff" configurations.

- Broad English Improvements: The "Add-MTP" configuration achieved the highest average English scores (0.4053 overall, and 0.4842 excluding mathematics). Notable gains were seen in Winogrande, CommonsenseQA, OpenBookQA, ARC Challenge, and particularly MMLU (0.3397 compared to 0.2671 and 0.2602).

- Chinese Performance Boost: Significant improvements were observed in Chinese tasks, with CEval increasing to 0.3076 and CMMLU to 0.2856, resulting in a higher average Chinese score (0.2966).

- Mathematics (Mixed): Performance on English mathematics tasks was mixed for the "Add-MTP" setup. GSM8K saw a decrease compared to the tuned hyperparameter set (0.0136 vs. 0.0265), though still an improvement over the default for Minerva Math. This suggests that while MTP can be broadly beneficial, its interaction with highly specialized tasks like mathematics may require further targeted tuning.

- Highest Overall Average: The "Add-MTP" configuration achieved the highest overall average score of 0.3897.

These findings indicate that incorporating an MTP module, especially when combined with optimized hyperparameters, can substantially enhance a model's predictive capabilities and overall performance on a diverse set of language tasks.

## Evaluation

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
