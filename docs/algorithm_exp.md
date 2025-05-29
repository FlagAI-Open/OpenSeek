# Algorithm Experiment

This document outlines the process for conducting algorithm experiments, including Hyper-parameter tuning and Multiple Token Prediction (MTP) experiments.

## Pipeline

The basic steps for conducting an algorithm experiment are as follows:

1. Ensure you have installed the necessary tools and environment by following the general [setup instructions](../README.md#-getting-started).

2. Modify the experiment configuration files to set up your specific algorithm experiment. Details for [Hyper-parameter](#hyper-parameter-experiment) and [MTP experiments](#mtp-experimemt) are provided below.

3. Start the experiment using the provided [script](openseek/algorithm/run_exp.sh).

4. View the experiment results and logs to analyze performance.

## Configuration

To set up your algorithm experiment, you'll typically modify the configuration by referring to the baseline experiment configuration which is located in the `openseek/algorithm` directory, including:

- `config_deepseek_v3_16b.yaml`: This is the experiment configuration file, defining the experiment directory, backend engine, task type, and environment settings.

- `train_deepseek_v3_16b.yaml`: This is the job configuration (job config) file, specifying model parameters, dataset configurations, and training-specific settings.

It's recommended to copy the existing baseline configuration directory (configs/OpenSeek-Small-v1-Baseline) and rename it for your new experiment (e.g., configs/MyAlgorithmExp-Hyperparam-Variant1).

Then modify the `experiment.exp_name` field in your experiment configuration file. The experiment output path will be created under a directory with this new name.

```yaml
# ...
experiment:
  exp_name: MyAlgorithmExp-VariantX # Change this to your specific experiment name
# ...
```

Adjust parameters in the job configuration file as detailed in the sections below for Hyper-parameter and MTP experiments.

## Hyper-parameter Experiment

Hyper-parameter experiments involve tuning specific system or model parameters to optimize performance. For the models being used, parameters related to initialization, Mixture of Experts (MoE) router precision, and MoE auxiliary loss are often critical.

```yaml
# ...
system:
  ## override configs for hyper-parameter tuning
  init_method_std: 6e-3     # Adjust: Standard deviation for weight initialization.
                            # Experiment with different small values (e.g., 5e-3, 7e-3).
  moe_router_dtype: fp32    # Adjust: Data type for the MoE router.
                            # Options might include fp32, bf16, fp16. Consider computational cost vs. precision.
  moe_aux_loss_coeff: 0.0001 # Adjust: Coefficient for the MoE auxiliary loss.
                            # This balances expert utilization. Try values like 0.001, 0.00005, etc.
  # ... other system parameters
# ...
```

For each hyper-parameter variant, create a new experiment configuration or systematically vary these values.

## MTP Experimemt

Multiple Token Prediction (MTP) experiments focus on evaluating or enhancing the model's capability to predict sequences of tokens effectively. This often involves adding specific prediction heads or modifying the loss function to account for multiple future tokens.

```yaml
# ...
system:
# using MTP
  num_mtp_predictor: 1  # Adjust: Number of future tokens the MTP head is designed to predict.
                      # Experiment with different values (e.g., 1, 2, 3) depending on the task.
  mtp_loss_coeff: 0.3   # Adjust: Coefficient for the MTP loss.
                      # This balances the MTP objective with the primary language modeling loss.
                      # Try values like 0.1, 0.3, 0.5, etc.

# ... other model or system parameters
# ...

```

## Results and Log

Experiment results will be located in a folder named after the exp_name you specified in your experiment configuration file.

Logs will be located in the logs subdirectory within that folder. The structure is typically:

```
<your_exp_name>/logs/details/host_<id>_<ip>/<timestamp>/<attempt_info>/<gpu_id>
```

- Experiment Startup Log: Usually found in the path corresponding to the first GPU on the first host.

- Training Loss Log (and other metrics): Often found in the path corresponding to the last GPU on the last host (or aggregated, depending on the logging setup).

Check your specific logging implementation for exact paths and details.

For more detailed information on general training arguments and configurations, you may refer to [Megatron-LM](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/training/arguments.py).