# OpenSeek Legacy Package

This directory now serves as a compatibility landing page for the historical `openseek` Python package structure. All actively maintained source code, experiments, and datasets have been reorganized into the `stage1/` module tree.

## Migration Guide

- **Algorithms & Experiments** → `stage1/algorithm/`
- **Data Pipelines** → `stage1/data/`
- **System & Distributed Training** → `stage1/system/`
- **Baseline Examples** → `examples/baseline/`

Please update your imports, scripts, and documentation references to the new locations. The reorganization consolidates duplicated content, aligns documentation with execution scripts, and makes future multi-stage releases easier to manage.

For further details, consult the top-level [`README.md`](../README.md) and the documentation index in [`docs/README.md`](../docs/README.md). 