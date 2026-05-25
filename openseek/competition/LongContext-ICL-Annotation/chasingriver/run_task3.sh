#!/usr/bin/env bash
set -euo pipefail

TASK_ID=3 exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_task_common.sh"
