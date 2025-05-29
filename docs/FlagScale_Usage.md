## Setup

We recommend using the latest release of [NGC's PyTorch container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch) for setup.

1. Clone the repository:
```shell
git clone https://github.com/FlagOpen/FlagScale.git
```

2. Install the requirements:
```shell
cd FlagScale/install
./install-requirements.sh --env train
./install-requirements.sh --env inference
```
The above instructions create two conda environments: `flagscale-train` and `flagscale-inference`, which contain the dependency environments for training and inference, respectively.

3. Install the packages with customized extensions:
```shell
cd vllm
pip install .

pip install -e ./megatron-energon
cp -r megatron-energon/src/megatron/energon megatron/megatron
```


## Run a task
FlagScale provides a unified runner for various tasks, including training, inference, and serving. Simply specify the configuration file to run the task with a single command.

**Start the distributed training job**
```shell
python run.py --config-path=examples/deepseek_v3/conf --config-name=config_deepseek_v3.yaml action=run
```

**Stop the distributed training job**
```shell
python run.py --config-path=examples/deepseek_v3/conf --config-name=config_deepseek_v3.yaml action=stop
```
**YAML Configuration**
FlagScale leverages [Hydra](https://github.com/facebookresearch/hydra) for configuration management, which is organized into two levels: an outer experiment-level YAML file and an inner task-level YAML file.

In the OpenSeek project, we have open-sourced a DeepSeek model with a total parameter count of 16B and an activated parameter count of 2.4B. This model has been thoroughly validated on real-world datasets, and the loss curve will be released shortly.

1. **Experiment-level YAML**: The experiment-level YAML file defines the experiment directory, backend engine, task type, and other related environmental configurations. [config_deepseek_v3.yaml](https://github.com/FlagOpen/FlagScale/blob/main/examples/deepseek_v3/conf/config_deepseek_v3.yaml)

2. **Task-level YAML**: The task-level YAML file specifies the model, dataset, and parameters for specific tasks such as training or inference. [train_deepseek_v3.yaml](https://github.com/FlagOpen/FlagScale/blob/main/examples/deepseek_v3/conf/train/train_deepseek_v3.yaml)

## Model Checkpoint conversion

### 1. HuggingFace --> Megatron
- FlagScale supports the conversion of open-source models and checkpoints (CKPT) from HuggingFace to the Megatron format. Once the conversion is completed, the CKPT can be loaded, and distributed training can be initiated using FlagScale.
- For instance, the [DeepSeek-V2-Lite](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite) 16B model, which is openly available on HuggingFace, can be converted into a CKPT format supported by FlagScale. Subsequently, the model can be directly warm-started by configuring the load option in `config_deepseek_v3.yaml`.
- FlagScale conversion supports tensor model parallelism, expert model parallelism, and pipeline model parallelism with even & uneven partitioning of pipeline stages during the checkpoint (CKPT) conversion.

#### CKPT conversion script
```shell
cd FlagScale/tools
python convert.py \
    --model-type deepseek_v3 \
    --loader transformers \
    --saver mcore \
    --load-dir DeepSeek-V2-Lite \
    --save-dir converted_mcore_bf16_model \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 2 \
    --target-decoder-first-pipeline-num-layers 13 \
    --target-expert-parallel-size 4 \
    --target-params-dtype bf16 \
    --true-vocab-size 151851
```

#### Modify task yaml
Set the **load** field in the YAML file to the path of the converted checkpoint
```yaml
system:
  tensor_model_parallel_size: 2
  pipeline_model_parallel_size: 2
  expert_model_parallel_size: 2
  context_parallel_size: 1
  sequence_parallel: true
  use_distributed_optimizer: true
  ...
  checkpoint:
    save_interval: 10000
    load: converted_mcore_bf16_model # the save_dir after conversion
    ckpt_format: torch
```

#### Start training
```shell
python run.py --config-path=examples/deepseek_v3/conf --config-name=config_deepseek_v3.yaml action=run
```


### 2. Megatron --> HuggingFace
FlagScale also supports the conversion of model checkpoints (CKPT) trained on FlagScale into the HuggingFace format, facilitating model release and evaluation.

#### CKPT conversion script
```shell
python convert.py \
    --model-type deepseek_v3 \
    --loader mcore \
    --saver transformers \
    --load-dir bf16_model \
    --save-dir converted_huggingface_model \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 1 \
    --target-expert-parallel-size 1 \
    --target-params-dtype bf16 \
    --true-vocab-size 151851 \
```

## Contribute to FlagScale
Currently, we have preliminarily reproduced the DeepSeek V3 pre-training code with the following features:
- [x] Support for MLA and MTP structures: shared embedding.
- [x] Support for DeepSeekMoE structure: shared expert, loss-free, etc.
- [x] Support checkpoint conversion with the Hugging Face.
- [x] Support for hybrid parallelism of TP (Tensor Parallelism), PP (Pipeline Parallelism), DP (Data Parallelism), and EP (Expert Parallelism).

The framework system side still has the following tasks. Everyone is welcome to participate and contribute. See the [FlagAI-Open OpenSeek](https://github.com/FlagAI-Open/OpenSeek) for a full list of proposed features .

### Roadmap
#### Basic
- [ ] Enhance the distributed training documentation
- [ ] Improve the installation and usage
- [ ] Conversion ckpt between FlagScale and Huggingface parameters
- [ ] Research and design a solution can be easily implemented in FlagScale

#### Intermediate
- [ ] Implement a distributed log consolidation mechanism
- [ ] Improve the monitoring system of distributed training
- [ ] Performance analysis of current long sequence handling
- [ ] Performance analysis of the current DeepSeekMoE distributed training implementation
- [ ] Support for DeepSeek NAS or Kimi MoBA etc
- [ ] Integration of the FlagGems Triton operator library and corresponding training accuracy validation
- [ ] Implementation of the FP8 operators required in DeepSeek V3, with support for validation during the training process
- [ ] Implementation of a distributed reinforcement learning system to support efficient DeepSeek R1
- [ ] Develop tools for detecting slow nodes, faulty nodes, and NCCL errors in large-scale clusters
- [ ] Visualization of the communication flows and scheduling relationships in complex large-scale distributed clusters

#### Advanced
- [ ] Support for DualPipe pipeline parallelism
- [ ] Achieve more efficient pipeline parallelism
- [ ] Improve communication algorithms to achieve more efficient MoE parallelism optimization
- [ ] Collaborate with algorithm teams to achieve more efficient long sequence optimization
- [ ] Implement customized parallel and optimization strategies for specific chips
- [ ] Implement more innovative FP8 training solutions


### How to contribute
We warmly welcome contributions to the FlagScale project! If you would like to contribute, please follow these steps:

1. **Fork** [FlagScale](https://github.com/FlagOpen/FlagScale) to your own github repo
2. Create a copy of the FlagScale repo under your account, with a URL like https://github.com/your-own-id/FlagScale
3. **Clone** the forked repository to your local machine and navigate into the local FlagScale directory
    ```shell
    git clone https://github.com/your-own-id/FlagScale.git
    cd FlagScale
    git config --global user.name XXX
    git config --global user.email XXX
    pre-commit install
    ```
4. **Add** the upstream repository to keep your fork updated with changes from the original FlagScale repository
    ```shell
    git remote add upstream https://github.com/FlagOpen/FlagScale.git
    ```
5. **Sync** updates from the upstream FlagScale repository
    ```shell
    git pull upstream main:main
    ```
6. **Create** a new branch and start your development
    ```shell
    git checkout -b feature/my-new-feature
    ```
7. **Commit** your changes
    ```shell
    git add .
    git commit -m "Add my new feature"
    ```
8. **Push** your new branch to your GitHub repository
    ```shell
    git push origin feature/my-new-feature
    ```
9. **Create a pull request (PR)** for FlagScale
  - Open your GitHub repository page (`https://github.com/your-own-id/FlagScale`)
  - You will see a prompt with a **compare & pull request** button for your newly pushed branch
  - Please provide a title and a description for your pull request that succinctly describes the modifications you have made
  - Click this button to proceed to the Pull Request page
10. **Wait** for review and merge