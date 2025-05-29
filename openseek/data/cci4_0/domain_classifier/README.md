# Domain Classifier

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-orange.svg)
![Transformers](https://img.shields.io/badge/Transformers-4.49.0-green.svg)

This script implements a domain classification script based on PyTorch and Hugging Face's Transformers library. The model is based on [nvidia/multilingual-domain-classifier](https://huggingface.co/nvidia/multilingual-domain-classifier).

## Requirements

Install dependencies via pip using the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```

### requirements.txt

```
huggingface_hub==0.28.1
torch==2.5.1
transformers==4.49.0
```

## Usage

### Running the Script

The script can be executed directly from the command line with optional arguments.

#### Command-Line Arguments

- `--deberta-model-path`: Path to the pre-trained domain classifier model (default: `nvidia/multilingual-domain-classifier`).
- `--deberta-base-model-path`: Local path to the base model (e.g., `microsoft/mdeberta-v3-base`). Optional; leave empty to use the default base model specified in the config.

#### Example

```bash
python domain_classifier.py --deberta-model-path "nvidia/multilingual-domain-classifier" --deberta-base-model-path "/path/to/microsoft/mdeberta-v3-base"
```

This runs the script with the default test data:

- Input: `["Los deportes son un dominio popular", "La política es un dominio popular"]`
- Output: `['Sports', 'News']` 