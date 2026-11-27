#!/usr/bin/env bash
set -euo pipefail

TIMEOUT="${TIMEOUT:-3600}"
XLSX="${XLSX:-sat/results_benchmark_v1.xlsx}"
DATA="${DATA:-prob027/data/benchmark_v1}"
PY="${PY:-.venv/bin/python}"

[ -x "$PY" ] || { echo "No interpreter at $PY. Create one:"; \
  echo "  python3.12 -m venv .venv && .venv/bin/pip install -r sat/requirements.txt"; exit 1; }

echo "cutoff ${TIMEOUT}s -> $XLSX"
"$PY" -c "import sys; sys.path.insert(0,'sat'); import common; print('machine:', common.machine_label())"

for MODE in binary-incr binary-new topdown-incr topdown-new \
            linear-incr linear-new maxsat feasibility; do
  echo "===== $MODE ====="
  "$PY" -u sat/solver.py --input-dir "$DATA" --xlsx "$XLSX" --mode "$MODE" \
        --resume --timeout "$TIMEOUT" --quiet
  "$PY" -u sat/solver.py --input-dir "$DATA" --xlsx "$XLSX" --mode "$MODE" \
        --upgrade-timeouts --timeout "$TIMEOUT" --quiet
done

echo "===== done ====="
