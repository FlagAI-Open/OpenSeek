#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 /path/to/LongContext-ICL-Annotation/data" >&2
  exit 1
fi

SOURCE_DIR="$1"
TARGET_DIR="$(cd "$(dirname "$0")/.." && pwd)/data/raw/openseek"

if [[ ! -d "$SOURCE_DIR" ]]; then
  echo "Source directory not found: $SOURCE_DIR" >&2
  exit 1
fi

mkdir -p "$TARGET_DIR"
find "$SOURCE_DIR" -maxdepth 1 -type f -name 'openseek-*.json' -exec cp {} "$TARGET_DIR"/ \;

COUNT="$(find "$TARGET_DIR" -maxdepth 1 -type f -name 'openseek-*.json' | wc -l | tr -d ' ')"
echo "Imported $COUNT official dataset files into $TARGET_DIR"
