#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
python3 -m src.main evaluate --config "${1:-configs/base.yaml}"
