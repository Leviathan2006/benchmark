#!/usr/bin/env bash
#
# Phase 1 sweep driver.
#
# NOT RUN as part of scaffolding. This is the entry point once
# rollout_error.runner.run_cell is wired to APEBench.
#
# Usage:
#   scripts/run_phase1.sh --dry-run          # print grid size only
#   scripts/run_phase1.sh                     # run the full grid (long)
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DRY_RUN=""
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN="--dry-run"
fi

echo "== rollout-error phase 1 =="
echo "grid:"
python -m rollout_error.sweep --dry-run

if [[ -n "$DRY_RUN" ]]; then
  exit 0
fi

# One JSON job per line -> run_cell per job. run_cell appends to
# results/phase1.parquet and is idempotent per (scenario, arch, train_mode,
# weight_mode, gamma, seed).
python -m rollout_error.sweep --json | while read -r job; do
  echo "-- $job"
  python - "$job" <<'PY'
import json, sys
from rollout_error.sweep import Job
from rollout_error.runner import run_cell

job = Job(**json.loads(sys.argv[1]))
run_cell(job)
PY
done

echo "done. results in results/phase1.parquet"
