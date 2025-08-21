# Troubleshooting Guide for Training Issues

## 1. Setting the Path for `run.py`

**Issue:** The path for `run.py` needs to be set correctly to ensure the script can be executed.

**Solution:**
Ensure that the working directory is set to the location of `run.py`, or provide the full path to `run.py` in the command. For example:  

```bash
python /path/to/run.py
```
> NOTE: run.py is from the flagscale repository, so ensure you have cloned the repository. Then you will find `run.py` in the `flagscale` directory maybe with a path: `run.py` or `flagscale/run.py`.
## 2. Error: `unrecognized arguments: --moe-router-dtype fp32`

Please update to the latest version of [flagscale](https://github.com/FlagOpen/FlagScale).

## 3. RuntimeError: `CUDA_DEVICE_MAX_CONNECTIONS` Environment Variable

**Issue:** The following error occurs during execution:
`RuntimeError: Using async gradient all reduce requires setting the environment variable CUDA_DEVICE_MAX_CONNECTIONS to 1`

**Solution:**
Set the environment variable `CUDA_DEVICE_MAX_CONNECTIONS` to `1` before running the command. Modify the command as follows:  

```bash
CUDA_DEVICE_MAX_CONNECTIONS=1 bash run_exp.sh
```

This ensures that the environment variable is set for the duration of the script execution.

## 4. OSError: Incorrect `path_or_model_id` for Tokenizer

**Issue:** The following error occurs when trying to load a tokenizer:
`OSError: Incorrect path_or_model_id: 'examples/aquila/qwentokenizer'. Please provide either the path to a local folder or the repo_id of a model on the Hub.`
Additionally, the process fails with:
`E0526 18:19:40.893000 583409 site-packages/torch/distributed/elastic/multiprocessing/api.py:869] failed (exitcode: 1) local_rank: 0 (pid: 583474) of binary: /root/miniconda3/envs/flagscale/bin/python`

**Solution:**
Update the `tokenizer_path` in the configuration file such as `configs/OpenSeek-Small-v1-Baseline/train/train_deepseek_v3_1_4b.yaml` to a valid path from the . Set it to:  

```yaml
tokenizer_path: hf_openseek/tokenizer
```

This points to a valid tokenizer model from `OpenSeek` project (we have already built this tokenizer in this repo), resolving the issue.

## 5. Update Training Data Path in `train_deepseek_v3_1_4b.yaml`

**Issue:** The training data path in `train_deepseek_v3_1_4b.yaml` needs to be updated.

**Solution:**
Edit the `config_deepseek_v3_1_4b.yaml` file and update the `dataset_base_dir` to the correct path for your training data. For example, you can download the [OpenSeek-Pretrain-100B](https://huggingface.co/datasets/BAAI/OpenSeek-Pretrain-100B/tree/main) dataset and save to a `/root/dataset/OpenSeek-Pretrain-100B`, then you can set the:

```yaml
dataset_base_dir: /root/dataset/OpenSeek-Pretrain-100B
```

