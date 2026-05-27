# OpenSeek Submission

This directory contains the final open-source submission for the
`LongContext-ICL-Annotation` track.

## Included Files

- `技术报告-OpenSeek.pdf`
  - Final technical report PDF used for the competition submission.
- `submission.zip`
  - Exact 8-task prediction archive submitted on the platform.
- `源代码-OpenSeek.zip`
  - Exact source-code archive included in the final competition submission.
- `code/`
  - Extracted source tree for direct review, including configs, prompts,
    scripts, tests, and usage documentation.

## Method Summary

The solution uses `Qwen3-4B` as the only large language model and uses
`FlagScale` as the required runtime framework. The pipeline combines:

- long-context example retrieval and compression
- front-back evidence reordering
- multi-protocol first-pass inference
- confidence-based recheck and adjudication
- deterministic post-processing for selected tasks
- submission validation and packaging

The final confirmed platform score for this submission line is `87.35`.

## Reproduction

See [code/README.md](code/README.md) for environment setup, data preparation,
FlagScale deployment, inference, evaluation, packaging, and validation.

## Scope

This submission only adds files under:

`openseek/competition/LongContext-ICL-Annotation/submissions/OpenSeek/`
