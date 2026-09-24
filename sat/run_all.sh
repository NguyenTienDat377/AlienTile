#!/usr/bin/env bash
# Full Alien Tiles SAT benchmark. Run from the repository root:
#     bash sat/run_all.sh                 # 3600s cutoff, benchmark_v1
#     TIMEOUT=600 bash sat/run_all.sh     # shorter cutoff
#
# Safe to re-run: finished rows are skipped, so an interrupted sweep resumes.
# Keep the machine otherwise idle — the Runtime column is wall-clock.
set -euo pipefail

TIMEOUT="${TIMEOUT:-3600}"
XLSX="${XLSX:-sat/results_benchmark_v1.xlsx}"
DATA="${DATA:-prob027/data/benchmark_v1}"
PY="${PY:-.venv/bin/python}"

CONFIGS=4x4_c2,4x4_c3,4x4_c4,5x5_c2,5x5_c3,5x5_c4,6x6_c2,6x6_c3,6x6_c4,\
8x8_c2,8x8_c3,8x8_c4,10x10_c2,10x10_c3,10x10_c4,12x12_c2,12x12_c3,12x12_c4

[ -x "$PY" ] || { echo "No interpreter at $PY. Create one:"; \
  echo "  python3.12 -m venv .venv && .venv/bin/pip install -r sat/requirements.txt"; exit 1; }

echo "cutoff ${TIMEOUT}s -> $XLSX"
"$PY" -c "import sys; sys.path.insert(0,'sat'); import common; print('machine:', common.machine_label())"

for MODE in feasibility optimum-binary optimum; do
  echo "===== $MODE ====="
  # --resume adds rows not yet present; --upgrade-timeouts redoes TIMEOUT rows
  # recorded under a smaller cutoff. Together they make this idempotent.
  "$PY" -u sat/solver.py --input-dir "$DATA" --xlsx "$XLSX" --mode "$MODE" \
        --resume --timeout "$TIMEOUT" --quiet
  "$PY" -u sat/solver.py --input-dir "$DATA" --xlsx "$XLSX" --mode "$MODE" \
        --upgrade-timeouts --timeout "$TIMEOUT" --quiet
done

echo "===== maxmin ====="
"$PY" -u sat/maxmin.py --sweep "$CONFIGS" --xlsx "$XLSX" --timeout "$TIMEOUT"

echo "===== done ====="
