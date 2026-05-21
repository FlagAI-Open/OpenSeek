#!/usr/bin/env bash
# ==============================================================================
# Verify that outputs/openseek-{1..8}-v*.jsonl all exist (latest version of each
# task is picked) and are valid JSONL with 'prediction' field on every row.
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "${SCRIPT_DIR}/.." && pwd )"
OUT="${REPO_ROOT}/outputs"

EXIT=0
for i in 1 2 3 4 5 6 7 8; do
    F=$(ls "${OUT}"/openseek-${i}-v*.jsonl 2>/dev/null \
        | sed -E 's/.*-v([0-9]+)\.jsonl$/\1 &/' | sort -n | tail -1 | cut -d' ' -f2-)
    if [ -z "${F}" ] || [ ! -f "${F}" ]; then
        echo "[MISS] task${i}: no openseek-${i}-v*.jsonl found under ${OUT}/"
        EXIT=1
        continue
    fi
    LINES=$(wc -l < "${F}")
    if ! python -c "
import json, sys
ok = 0
bad = 0
with open('${F}', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if 'prediction' not in obj:
                bad += 1
            else:
                ok += 1
        except Exception:
            bad += 1
print(f'  rows ok={ok}, bad={bad}')
sys.exit(0 if bad == 0 else 1)
"; then
        echo "[BAD ] ${F} (${LINES} lines, contains invalid rows or missing 'prediction')"
        EXIT=1
    else
        echo "[OK  ] ${F} (${LINES} lines)"
    fi
done

exit ${EXIT}
