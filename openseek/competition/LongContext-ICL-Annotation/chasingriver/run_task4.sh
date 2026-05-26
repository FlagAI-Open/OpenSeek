#!/usr/bin/env bash
set -euo pipefail

TASK_ID=4 exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_task_common.sh"
