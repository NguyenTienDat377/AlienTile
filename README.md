# Alien Tiles: SAT and MaxSAT solving (CSPLib prob027)

Undergraduate thesis, VNU University of Engineering and Technology (UET).
Author: Nguyễn Tiến Đạt.

This repository finds the minimum number of clicks needed to reach a target board in
the Alien Tiles puzzle, using SAT and MaxSAT solvers through
[PySAT](https://pysathq.github.io/).

- [Quick start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [Problem](#problem)
- [Method](#method)
- [Benchmark](#benchmark)
- [Reproducing the experiments](#reproducing-the-experiments)
- [Troubleshooting](#troubleshooting)

## Quick start

On macOS or Linux, with Python 3.12 installed:

```bash
git clone https://github.com/NguyenTienDat377/AlienTile.git
cd AlienTile
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r sat/requirements.txt
python sat/solver.py --input prob027/data/4x4_c3_easy.json --no-excel
```

The last command should end with `Total clicks: 3` and `Verification: PASSED`.

## Installation

### 1. Requirements

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.12 | Other 3.x versions may work but have not been tested |
| Git | any | To clone the repository |
| OS | macOS or Linux | On Windows, use WSL (see below) |

The Python dependencies are pinned in [sat/requirements.txt](sat/requirements.txt):

| Package | Version | Used for |
|---|---|---|
| `python-sat` | 1.9.dev15 | SAT solver (Glucose 4), cardinality encodings, RC2 MaxSAT |
| `openpyxl` | 3.1.5 | Writing results to Excel |

No commercial solver is needed. CPLEX, docplex and Gurobi appear in
`requirements.txt` only as comments: they belong to earlier ILP/CP experiments
and are not used by the SAT code.

### 2. Get the code

```bash
git clone https://github.com/NguyenTienDat377/AlienTile.git
cd AlienTile
```

All commands below are run from the repository root unless stated otherwise.

### 3. Create a virtual environment

A virtual environment keeps the pinned versions separate from your system
Python.

```bash
python3.12 -m venv .venv
source .venv/bin/activate          # fish: source .venv/bin/activate.fish
```

Your prompt should now start with `(.venv)`. Run `source .venv/bin/activate` again
in every new terminal.

### 4. Install the dependencies

```bash
pip install --upgrade pip
pip install -r sat/requirements.txt
```

### 5. Check the installation

```bash
python -c "import pysat, openpyxl; print('pysat', pysat.__version__, '| openpyxl', openpyxl.__version__)"
```

Expected output:

```text
pysat 1.9.dev15 | openpyxl 3.1.5
```

Then solve the small example from the thesis:

```bash
python sat/solver.py --input prob027/data/4x4_c3_easy.json --no-excel
```

Expected output (the runtime will differ):

```text
Instance: 4x4_c3_easy   (N=4, c=3)
  Target T:
  1 1 1 0
  1 0 0 2
  0 2 2 2
  1 0 0 2

  Solution X (click matrix):
  1 0 0 0
  0 0 0 0
  0 0 0 2
  0 0 0 0

  Encoding: 464 vars, 1696 clauses
  Total clicks: 3
  Extra: {'SAT Calls': 2, 'LB': 3}
  Runtime: 0.0141s
  Verification: PASSED
```

Finally, check the dataset and run the generator tests:

```bash
cd prob027
python experiments/validate_dataset.py data/benchmark_v1
python -m unittest discover -s tests -v
cd ..
```

The validator should report `"unique_targets": 66`, and all 5 tests should pass.

### Windows

The scripts assume a Unix shell (`bash`, `.venv/bin/python`). On Windows, the
simplest option is [WSL](https://learn.microsoft.com/windows/wsl/install): open an
Ubuntu terminal and follow the macOS/Linux steps above. If Python 3.12 is missing
there, install it with `sudo apt install python3.12 python3.12-venv`.

## Usage

Solve one instance (add `--no-excel` to skip writing a spreadsheet):

```bash
python sat/solver.py --input prob027/data/4x4_c3_easy.json
```

Give a target on the command line, with rows separated by `;`:

```bash
python sat/solver.py --N 3 --c 2 --target "1,1,1;1,1,1;1,1,1" --mode topdown-incr --no-excel
```

This should report `Total clicks: 3`.

Solve every instance in a directory, including its subdirectories:

```bash
python sat/solver.py --input-dir prob027/data/benchmark_v1/easy \
    --mode maxsat --timeout 600 --quiet --xlsx sat/results.xlsx
```

| Option | Effect |
|---|---|
| `--mode MODE` | solving strategy (see [Method](#method)); default `binary-incr` |
| `--timeout S` | wall-clock limit per instance in seconds (default 600, `0` for none) |
| `--xlsx PATH` | output spreadsheet (default `sat.xlsx`), one sheet per mode |
| `--no-excel` | do not write a spreadsheet |
| `--quiet` | one line per instance instead of full output |
| `--resume` | skip instances already in the sheet |
| `--upgrade-timeouts` | re-run timeouts recorded with a shorter limit |
| `--only-status LIST` | re-run rows with these statuses, e.g. `TIMEOUT,UNSAT` |
| `--only-missing` | re-run rows with missing solver fields |

The last four options filter instances against an existing spreadsheet.

Every returned solution is checked by applying its clicks directly to the board,
without using the CNF. If no click matrix reaches the target, the output says
`UNSATISFIABLE`. On a timeout, `Total Clicks` is the best solution found so far (an
upper bound), not a proven optimum.

### Instance format

```json
{
  "name": "example",
  "N": 3,
  "c": 2,
  "target": [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
}
```

An optional `witness_clicks` matrix gives a known solution. It is verified and then
used as the starting upper bound by the optimisation modes.

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
through PySAT.

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

### Hand-made instances (`prob027/data/`)

| File | N | c | Expected result |
|---|---|---|---|
| `3x3_c2_allones.json` | 3 | 2 | minimum 3 |
| `4x4_c3_allones.json` | 4 | 3 | minimum 4 |
| `4x4_c3_easy.json` | 4 | 3 | minimum 3 (running example) |
| `4x4_c3_infeasible.json` | 4 | 3 | UNSATISFIABLE |
| `4x4_c3_hardest_gent2000.json` | 4 | 3 | minimum 10 (Gent, Linton and Smith, 2000) |
| `7x7_c4_solvable.json` | 7 | 4 | minimum 6 |

## Benchmark

`benchmark_v1` has 66 instances with N ∈ {4, 5, 6, 8, 10, 12} and c ∈ {2, 3, 4}:

- 22 each in the easy, medium and hard tiers. Tiers are ordered by `N²(c−1)`, a
  measure of model size rather than of measured difficulty.
- 60 `witness` instances, built from sampled clicks and therefore feasible. The
  witness gives an upper bound, not the optimum.
- 6 `uniform` instances with random targets, which may be infeasible.

It was generated with master seed `270027`; see the
[generation notes](prob027/experiments/README.md).

## Reproducing the experiments

```bash
bash sat/run_all.sh                 # 3600 s limit per instance
TIMEOUT=600 bash sat/run_all.sh     # shorter limit
```

The script expects the virtual environment at `.venv` (see
[Installation](#installation)). It runs every mode and writes to
`sat/results_benchmark_v1.xlsx`. It can be restarted after an interruption:
recorded instances are skipped, and timeouts are retried only with a longer limit.
Set `XLSX`, `DATA` or `PY` to change the output file, dataset or Python
interpreter. Runtimes are wall-clock, so keep the machine otherwise idle.

With a 3600 s limit, the full run can take many hours, because some hard instances
use the whole limit in several modes. For a quick check, run only the easy tier:

```bash
DATA=prob027/data/benchmark_v1/easy TIMEOUT=60 XLSX=sat/easy_check.xlsx bash sat/run_all.sh
```

## Troubleshooting

| Problem | Fix |
|---|---|
| `python3.12: command not found` | Install Python 3.12 (macOS: `brew install python@3.12`; Ubuntu/WSL: `sudo apt install python3.12 python3.12-venv`) |
| `ModuleNotFoundError: No module named 'pysat'` | The virtual environment is not active. Run `source .venv/bin/activate`, or call `.venv/bin/python` directly |
| `pip install` fails while building `python-sat` | No pre-built package matched your system, so pip tried to compile it. Use Python 3.12 on macOS, Linux or WSL, and install a C/C++ compiler (macOS: `xcode-select --install`; Ubuntu: `sudo apt install build-essential`) |
| `No interpreter at .venv/bin/python` from `run_all.sh` | Create the virtual environment in the repository root, or set `PY=/path/to/python` |
| The Excel file is not updated | Close it in Excel first: an open file cannot be written on some systems |

## References

- CSPLib, *Problem 027: Alien Tiles*. https://www.csplib.org/Problems/prob027/
- M. J. Chlond, "An Alien IP", *INFORMS Transactions on Education* 7(1), 2006.
- I. P. Gent, S. A. Linton and B. M. Smith, *Symmetry breaking in the alien tiles
  puzzle*, Technical Report APES-22-2000, 2000.
- C. Sinz, "Towards an optimal CNF encoding of Boolean cardinality constraints",
  CP 2005.
- O. Bailleux and Y. Boufkhad, "Efficient CNF encoding of Boolean cardinality
  constraints", CP 2003.