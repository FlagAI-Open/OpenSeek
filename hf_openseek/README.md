# 🔧 Model Components Overview

This repository contains model code and data used in our Hugging Face workflows. It is organized into the following three directories:

- **`aquila/`**  
  Contains the model, configuration, and tokenizer code for the dense model used in evaluation experiments during dataset construction.

- **`deepseek_v3/`**  
  Contains the model, configuration, and tokenizer code for the sparse expert (MoE) model, which serves as the main architecture for training and strategy optimization.

- **`tokenizer/`**  
  Includes tokenizer data files shared across both model architectures.

Each component is designed to support high-quality pretraining and evaluation workflows.
