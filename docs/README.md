# OpenSeek Documentation

Welcome to the OpenSeek documentation! This directory contains comprehensive guides, tutorials, and reference materials for the OpenSeek project.

## 📚 Documentation Index

### Getting Started
- [Getting Started Guide](../README.md#-getting-started) - Quick start with OpenSeek
- [FlagScale Usage Guide](FlagScale_Usage.md) - How to use FlagScale framework
- [FAQ](faq.md) - Frequently asked questions

### Datasets
- [CCI4.0-M2 V1 Dataset](README_CCI4.0_M2_V1.md) - Large-scale bilingual pre-training dataset
- [OpenSeek-Pretrain-100B Pipeline](100B_pipeline.md) - 100B token dataset pipeline
- [OpenSeek-Pretrain-100B Pipeline (中文)](100B_pipeline-zh.md) - 100B token dataset pipeline (Chinese)
- [Data Processing Guide](Data.md) - Data sources and processing methods

### Models
- [OpenSeek-Small v1 Model](README_OPENSEEK_SMALL_V1.md) - Model documentation and evaluation results
- [OpenSeek-Small V1 Download Link](OpenSeek-Small_V1_download_link) - Model download information

### Training
- [Distributed Training Guide](distributed_training.md) - How to run distributed training
- [Baseline Training](../examples/baseline/README.md) - Baseline model training scripts

### Experiments
- [Algorithm Experiments](algorithm_exp.md) - Algorithm experiment guide
- [Algorithm Experiment Results](algorithm_exp_results.md) - Results and analysis
- [Data Mixture Experiments](data_mixture_exp.md) - Data mixture experiment guide
- [Data Mixture Experiment Results](data_mixture_exp_results.md) - Results and analysis
- [System Experiments](system_exp.md) - System optimization experiments

### Project Information
- [Roadmap](roadmap.md) - Project development roadmap
- [Contributing Guide](../CONTRIBUTING.md) - How to contribute to OpenSeek

## 📁 Documentation Structure

```
docs/
├── README.md                    # This file - documentation index
├── getting-started/             # Getting started guides
├── datasets/                    # Dataset documentation
│   ├── README_CCI4.0_M2_V1.md
│   └── 100B_pipeline.md
├── models/                      # Model documentation
│   └── README_OPENSEEK_SMALL_V1.md
├── training/                    # Training guides
│   ├── distributed_training.md
│   └── FlagScale_Usage.md
├── experiments/                 # Experiment documentation
│   ├── algorithm_exp.md
│   ├── data_mixture_exp.md
│   └── system_exp.md
└── imgs/                        # Documentation images
    ├── CCI4.0_M2_v1_Ablation.jpeg
    ├── CoT_Pipeline.png
    └── dataset_dist_reference_phi4.jpeg
```

## 🔍 Quick Links

### For New Users
1. Start with the [Getting Started Guide](../README.md#-getting-started)
2. Read the [FAQ](faq.md) for common questions
3. Check out [Baseline Training](../examples/baseline/README.md) to run your first experiment

### For Data Scientists
- [CCI4.0-M2 Dataset](README_CCI4.0_M2_V1.md)
- [Data Processing Guide](Data.md)
- [Data Mixture Experiments](data_mixture_exp.md)

### For ML Engineers
- [Distributed Training](distributed_training.md)
- [Algorithm Experiments](algorithm_exp.md)
- [System Experiments](system_exp.md)

### For Researchers
- [Model Documentation](README_OPENSEEK_SMALL_V1.md)
- [Experiment Results](algorithm_exp_results.md)
- [Roadmap](roadmap.md)

## 📝 Contributing to Documentation

If you find any issues with the documentation or want to contribute improvements:

1. Check the [Contributing Guide](../CONTRIBUTING.md)
2. Submit a pull request with your changes
3. Ensure all links are working and images are properly referenced

## 🖼️ Images

Documentation images are stored in the `imgs/` subdirectory. When referencing images in markdown files, use:

```markdown
![Image description](imgs/image_name.png)
```

## 📞 Need Help?

- Check the [FAQ](faq.md) first
- Open an issue on [GitHub](https://github.com/FlagAI-Open/OpenSeek/issues)
- Join the [Discord community](https://discord.gg/dPKWUC7ZP5)

