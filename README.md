# Alien Tiles: SAT and MaxSAT solving (CSPLib prob027)

Undergraduate thesis, VNU University of Engineering and Technology (UET).
Author: Nguyễn Tiến Đạt.

## Problem

Alien Tiles ([CSPLib prob027](https://www.csplib.org/Problems/prob027/)) is played on
an N×N board where every cell has one of `c` colours, numbered `0..c-1`. All cells
start at 0. Clicking a cell advances every cell in its row and column by one,
modulo `c`; the clicked cell advances only once.

Given a target board `T`, we look for a click matrix `X`, with `x_ij` in `0..c-1`,
such that for every cell `(r, k)`:

```text
(sum of row r of X + sum of column k of X − x_rk) mod c = T[r][k]
```

This repository solves two versions of the problem:

- **Feasibility:** find any `X` that reaches `T`, or prove that none exists.
- **Optimisation:** find an `X` that reaches `T` with the fewest total clicks.

## Method

### CNF encoding ([sat/encoder.py](sat/encoder.py))

| Variables | Meaning |
|---|---|
| `d(i,j,v)` | cell `(i,j)` is clicked `v` times (exactly one `v` holds) |
| `s(r,k,l,a)` | the first `l` clicks affecting `(r,k)` sum to `a` modulo `c` |
| `u(i,j,h)` | cell `(i,j)` is clicked at least `h` times, for `h = 1..c-1` |

The last value of each `s` chain is fixed to `T[r][k]`. The number of true `u`
literals equals the total number of clicks, so a cardinality constraint over them
limits the cost.

### Optimisation ([sat/solver.py](sat/solver.py))

The optimum is the smallest `K` for which the formula plus "at most `K` clicks" is
satisfiable. A lower bound is computed first from the largest target value, the
target sum and a modular condition (each click adds `2N−1` to the board sum,
modulo `c`). This condition alone proves some targets infeasible.

| `--mode` | Choice of `K` | Solver use |
|---|---|---|
| `linear-new` | upward from the lower bound | new solver for each `K` |
| `binary-new` | bisection | new solver for each `K` |
| `topdown-new` | downward from a known solution | new solver for each `K` |
| `linear-incr` | upward from the lower bound | incremental |
| `binary-incr` (default) | bisection | incremental |
| `topdown-incr` | downward from a known solution | incremental |
| `maxsat` | decided by RC2 | MaxSAT |
| `feasibility` | no click bound | single SAT call |

The `*-new` modes use a sequential-counter encoding. The incremental modes use an
incremental totalizer and pass the bound as an assumption. All modes use Glucose 4
through [PySAT](https://pysathq.github.io/).

## Repository layout

```text
sat/
  encoder.py                     CNF encoding
  solver.py                      solving strategies and command line
  common.py                      instance loading, verification, timeouts, Excel export
  run_all.sh                     full benchmark
  requirements.txt               pinned dependencies
prob027/
  data/                          hand-made instances
  data/benchmark_v1/             generated benchmark, indexed by manifest.csv
  experiments/                   instance generator and dataset validator
  tests/                         generator tests
```

## Setup

The code was checked with Python 3.12. Run all commands from the repository root
unless stated otherwise.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r sat/requirements.txt
```

## Usage

Solve one instance (add `--no-excel` to skip writing a spreadsheet):

```bash
python sat/solver.py --input prob027/data/4x4_c3_easy.json
```

Give a target on the command line, with rows separated by `;`:

```bash
python sat/solver.py --N 3 --c 2 --target "1,0,1;0,1,0;1,0,1" --mode topdown-incr
```

Solve every instance in a directory, including its subdirectories:

```bash
python sat/solver.py --input-dir prob027/data/benchmark_v1/easy \
    --mode maxsat --timeout 600 --quiet --xlsx sat/results.xlsx
```

| Option | Effect |
|---|---|
| `--timeout S` | wall-clock limit per instance in seconds (default 600, `0` for none) |
| `--xlsx PATH` | output spreadsheet (default `sat.xlsx`), one sheet per mode |
| `--resume` | skip instances already in the sheet |
| `--upgrade-timeouts` | re-run timeouts recorded with a shorter limit |
| `--only-status LIST` | re-run rows with these statuses, e.g. `TIMEOUT,UNSAT` |
| `--only-missing` | re-run rows with missing solver fields |

The last four options filter instances against an existing spreadsheet.

Every returned solution is checked by applying its clicks directly to the board,
without using the CNF. On a timeout, `Total Clicks` is the best solution found so
far (an upper bound), not a proven optimum.

### Instance format

```json
{
  "name": "example",
  "N": 3,
  "c": 2,
  "target": [[1, 0, 1], [0, 1, 0], [1, 0, 1]]
}
```

An optional `witness_clicks` matrix gives a known solution. It is verified and then
used as the starting upper bound by the optimisation modes.

## Benchmark

`benchmark_v1` has 66 instances with N ∈ {4, 5, 6, 8, 10, 12} and c ∈ {2, 3, 4}:

- 22 each in the easy, medium and hard tiers. Tiers are ordered by `N²(c−1)`, a
  measure of model size rather than of measured difficulty.
- 60 `witness` instances, built from sampled clicks and therefore feasible. The
  witness gives an upper bound, not the optimum.
- 6 `uniform` instances with random targets, which may be infeasible.

It was generated with master seed `270027`; see the
[generation notes](prob027/experiments/README.md). To validate the dataset and test
the generator:

```bash
cd prob027
python experiments/validate_dataset.py data/benchmark_v1
python -m unittest discover -s tests -v
```

## Reproducing the experiments

```bash
bash sat/run_all.sh                 # 3600 s limit per instance
TIMEOUT=600 bash sat/run_all.sh     # shorter limit
```

The script runs every mode and writes to `sat/results_benchmark_v1.xlsx`. It can be
restarted after an interruption: recorded instances are skipped, and timeouts are
retried only with a longer limit. Set `XLSX`, `DATA` or `PY` to change the output
file, dataset or Python interpreter. Runtimes are wall-clock, so keep the machine
otherwise idle.
