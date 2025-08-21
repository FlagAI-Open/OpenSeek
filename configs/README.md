# configs directory

This directory contains configuration files for different models and training setups.

## Experiment-Level YAML
This YAML defines the experiment directory, backend engine, task type, and environment configuration. Generally, these files require minimal modification.

Example: **configs/OpenSeek-Small-v1-Baseline/config_deepseek_v3_1_4b.yaml**

## Task-Level YAML
This YAML specifies model parameters, dataset configurations, and training-specific settings. Within data.data_path, you can configure the data ratio, and in model, you can set up the model configuration.

Example: **configs/OpenSeek-Small-v1-Baseline/train_deepseek_v3_1_4b.yaml**

The subdirectories:

- `OpenSeek-Small-v1`: Configuration for the OpenSeek-Small-v1 model (3B parameters).
  - `config_deepseek_v3_3b_1330B.yaml`
  - `train_deepseek_v3_3b_1330B.yaml`

- `OpenSeek-Small-v1-Baseline`: Configuration for the OpenSeek-Small-v1-Baseline model (1.4B parameters).
  - `config_deepseek_v3_1_4b.yaml`
  - `train_deepseek_v3_1_4b.yaml`

> Note: You should modify the `config_*.yaml` files to set the `dataset_base_dir`, `nnodes`, `nproc_per_node`, and `hostfile` for your training environment. The `train_*.yaml` files should be modified to set the `tokenizer_path`.
