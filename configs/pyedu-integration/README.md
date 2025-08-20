# PyEdu Dataset Integration Configuration

This directory contains configuration files for integrating the pyedu dataset into OpenSeek training pipelines.

## About PyEdu Dataset

PyEdu is a high-quality educational Python code dataset that is a subset of the "stack-edu" subset from smollm-corpus. Key characteristics:

- **Source**: https://huggingface.co/datasets/Leon-Leee/unofficial-pyedu
- **Size**: ~6GB
- **Quality**: High-quality according to the smollm-v2 tech report
- **Content**: Educational Python code examples
- **Use Cases**: Further training, annealing, or synthesizing datasets

## Configuration Files

- `config_pyedu_integration.yaml`: Experiment-level configuration for pyedu integration
- `train/train_pyedu_integration.yaml`: Task-level configuration with pyedu dataset included

## Usage

To use these configurations:

1. Ensure the pyedu dataset is downloaded and preprocessed
2. Update the `dataset_base_dir` in the config file to point to your data directory
3. Adjust the data mixture ratios as needed for your specific training requirements
4. Run training with the provided configuration files

## Data Mixture Strategy

The pyedu dataset can be integrated into existing training pipelines in several ways:

1. **Annealing**: Use pyedu for final training phases to improve code understanding
2. **Synthesis**: Use pyedu as source material for generating additional training data
3. **Mixed Training**: Include pyedu as part of the regular training data mixture

The configuration provided uses a balanced approach, incorporating pyedu alongside existing code datasets.