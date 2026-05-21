# Models Directory

This directory stores the model weights used for inference. It is intentionally left empty in the repository.

## Expected Layout

After downloading the model weights, the directory should look like:

```
models/
└── Qwen/
    └── Qwen3-4B/
        ├── config.json
        ├── tokenizer.json
        ├── ... (other model files)
```

## How to Download

From the `COM/` directory, run:

```bash
modelscope download --model Qwen/Qwen3-4B --cache_dir ./models
```

This will place the weights at `COM/models/Qwen/Qwen3-4B/`, which is the path referenced by `MODEL_DIR` in `src/common/paths.py`.

For full reproduction steps, see the top-level [README.md](../README.md).
