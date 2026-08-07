#!/usr/bin/env python3
from __future__ import annotations
"""
SAT encoding for the Alien Tiles Problem (CSPLib prob027)
=========================================================

Variant 1: FEASIBILITY
    Given a target state T (N×N matrix over Z_c), find a click
    matrix X such that for each cell (r,k):
        (sum_j x[r][j] + sum_i x[i][k] - x[r][k]) mod c = T[r][k]

Encoding
--------
- Direct (one-hot) encoding: for each x[i][j], c Boolean variables
  d(i,j,v) meaning "x[i][j] = v", with exactly-one constraints.
- Running-sum-modulo-c:  for each constraint (r,k), chain auxiliary
  variables  s(r,k,step,v)  meaning "partial sum at step ≡ v (mod c)".
  Transition: ¬s(ℓ-1,a) ∨ ¬d(term,b) ∨ s(ℓ,(a+b)%c).
  Final: assert s(last_step, T[r][k]).

Usage
-----
    python3 sat_variant1.py --input data/4x4_c3_easy.json
    python3 sat_variant1.py --input-dir data/
    python3 sat_variant1.py --N 4 --c 3 --target "1,0,2,1;0,1,0,2;2,0,1,0;1,2,0,1"
    python3 sat_variant1.py --examples
"""

import argparse
import glob
import json
import os
import sys
import time
from pysat.solvers import Glucose4
from pysat.formula import CNF, IDPool


# =====================================================================
#  SAT Encoder
# =====================================================================

class AlienTilesSAT:
    """SAT encoder/solver for the Alien Tiles feasibility problem."""

    def __init__(self, N: int, c: int, target: list[list[int]]):
        """
        Parameters
        ----------
        N : int       Grid size (N×N).
        c : int       Number of colours / modulus.
        target : list Target matrix, N×N with entries in {0,..,c-1}.
        """
        self.N = N
        self.c = c
        self.target = target
        self.pool = IDPool()
        self.cnf = CNF()
        self.stats = {"vars": 0, "clauses": 0, "time_encode": 0.0, "time_solve": 0.0}

    # -- Variable helpers ---------------------------------------------

    def var_click(self, i: int, j: int, v: int) -> int:
        """Boolean variable: x[i][j] == v."""
        return self.pool.id(("x", i, j, v))

    def var_sum(self, r: int, k: int, step: int, v: int) -> int:
        """Boolean variable: running sum at `step` for constraint (r,k) ≡ v (mod c)."""
        return self.pool.id(("s", r, k, step, v))

    # -- Exactly-one encoding -----------------------------------------

    def _add_exactly_one(self, lits: list[int]):
        """Add ALO + pairwise AMO clauses."""
        # At-least-one
        self.cnf.append(list(lits))
        # At-most-one (pairwise)
        for a in range(len(lits)):
            for b in range(a + 1, len(lits)):
                self.cnf.append([-lits[a], -lits[b]])

    # -- Main encoding ------------------------------------------------

    def encode(self):
        """Build the CNF formula."""
        t0 = time.perf_counter()
        N, c = self.N, self.c

        # 1. Click variables with exactly-one constraints
        for i in range(N):
            for j in range(N):
                lits = [self.var_click(i, j, v) for v in range(c)]
                self._add_exactly_one(lits)

        # 2. For each cell (r,k): modular sum constraint
        for r in range(N):
            for k in range(N):
                self._encode_cell_constraint(r, k)

        self.stats["vars"] = self.pool.top
        self.stats["clauses"] = len(self.cnf.clauses)
        self.stats["time_encode"] = time.perf_counter() - t0

    def _encode_cell_constraint(self, r: int, k: int):
        """
        Encode:  (sum_j x[r][j] + sum_{i≠r} x[i][k]) mod c == target[r][k]

        We list 2N-1 terms (row r, then column k excluding row r)
        and chain a running-sum modulo c using auxiliary variables.
        """
        N, c = self.N, self.c

        # Collect the 2N-1 terms
        terms = [(r, j) for j in range(N)]              # row r (N terms)
        terms += [(i, k) for i in range(N) if i != r]    # col k, skip (r,k)

        num_terms = len(terms)  # == 2N - 1

        # --- Step 0: s(r,k,0,v) ↔ d(term_0, v) ----------------------
        ti, tj = terms[0]
        s0_lits = [self.var_sum(r, k, 0, v) for v in range(c)]
        self._add_exactly_one(s0_lits)
        for v in range(c):
            sv = self.var_sum(r, k, 0, v)
            dv = self.var_click(ti, tj, v)
            # sv → dv  and  dv → sv  (channelling)
            self.cnf.append([-sv, dv])
            self.cnf.append([sv, -dv])

        # --- Steps 1 .. num_terms-1: transition ----------------------
        for step in range(1, num_terms):
            ti, tj = terms[step]

            # Exactly-one for s(r,k,step,*)
            s_lits = [self.var_sum(r, k, step, v) for v in range(c)]
            self._add_exactly_one(s_lits)

            # Transition: if s(step-1)=a AND x[ti][tj]=b → s(step)=(a+b)%c
            for a in range(c):
                s_prev = self.var_sum(r, k, step - 1, a)
                for b in range(c):
                    result = (a + b) % c
                    x_cur = self.var_click(ti, tj, b)
                    s_next = self.var_sum(r, k, step, result)
                    self.cnf.append([-s_prev, -x_cur, s_next])

        # --- Final: s(r,k, last_step) must equal target[r][k] --------
        target_val = self.target[r][k]
        last_step = num_terms - 1
        self.cnf.append([self.var_sum(r, k, last_step, target_val)])

    # -- Solve --------------------------------------------------------

    def solve(self) -> list[list[int]] | None:
        """
        Encode, solve, and return the click matrix (or None if UNSAT).

        Returns
        -------
        list[list[int]] or None
            N×N click matrix with entries in {0,..,c-1}, or None.
        """
        self.encode()

        t0 = time.perf_counter()
        with Glucose4(bootstrap_with=self.cnf) as solver:
            sat = solver.solve()
            self.stats["time_solve"] = time.perf_counter() - t0

            if not sat:
                return None

            model_set = set(solver.get_model())

        # Extract click matrix from model
        solution = []
        for i in range(self.N):
            row = []
            for j in range(self.N):
                for v in range(self.c):
                    if self.var_click(i, j, v) in model_set:
                        row.append(v)
                        break
                else:
                    row.append(-1)  # should never happen
            solution.append(row)
        return solution


# =====================================================================
#  Verification
# =====================================================================

def verify_solution(N: int, c: int, target: list[list[int]],
                    X: list[list[int]]) -> bool:
    """Check that click matrix X produces target T under modular arithmetic."""
    for r in range(N):
        for k in range(N):
            sigma = sum(X[r][j] for j in range(N)) \
                  + sum(X[i][k] for i in range(N)) \
                  - X[r][k]
            if sigma % c != target[r][k]:
                print(f"  ✗ Mismatch at ({r},{k}): "
                      f"σ={sigma}, σ mod {c}={sigma % c}, "
                      f"target={target[r][k]}")
                return False
    return True


# =====================================================================
#  Pretty printing
# =====================================================================

def print_matrix(label: str, M: list[list[int]]):
    """Print a matrix with a label."""
    print(f"\n{label}:")
    for row in M:
        print("  " + "  ".join(f"{v}" for v in row))


def generate_target_from_clicks(N: int, c: int,
                                X: list[list[int]]) -> list[list[int]]:
    """Compute the target state produced by a given click matrix."""
    T = []
    for r in range(N):
        row = []
        for k in range(N):
            sigma = sum(X[r][j] for j in range(N)) \
                  + sum(X[i][k] for i in range(N)) \
                  - X[r][k]
            row.append(sigma % c)
        T.append(row)
    return T


# =====================================================================
#  Built-in example instances
# =====================================================================

def example_instances() -> list[dict]:
    """Return a list of built-in test instances."""
    return [
        {
            "name": "3×3, c=2 (Lights Out variant)",
            "N": 3, "c": 2,
            "target": [[1, 1, 1],
                       [1, 1, 1],
                       [1, 1, 1]],
        },
        {
            "name": "4×4, c=3 (classic Alien Tiles)",
            "N": 4, "c": 3,
            # Target generated by clicking (0,0) once and (2,3) twice
            "target": generate_target_from_clicks(
                4, 3,
                [[1, 0, 0, 0],
                 [0, 0, 0, 0],
                 [0, 0, 0, 2],
                 [0, 0, 0, 0]]),
        },
        {
            "name": "4×4, c=3 (all ones — feasibility unknown)",
            "N": 4, "c": 3,
            "target": [[1, 1, 1, 1],
                       [1, 1, 1, 1],
                       [1, 1, 1, 1],
                       [1, 1, 1, 1]],
        },
    ]


# =====================================================================
#  Instance I/O
# =====================================================================

def load_instance(filepath: str) -> dict:
    """Load an instance from a JSON file."""
    with open(filepath) as f:
        data = json.load(f)
    name = data.get("name", os.path.basename(filepath))
    return {"name": name, "N": data["N"], "c": data["c"],
            "target": data["target"]}


def load_instances_from_dir(dirpath: str) -> list[dict]:
    """Load instance JSON files recursively, ignoring dataset metadata."""
    files = sorted(glob.glob(os.path.join(dirpath, "**", "*.json"), recursive=True))
    instance_files = []
    for filepath in files:
        with open(filepath) as f:
            data = json.load(f)
        if all(key in data for key in ("N", "c", "target")):
            instance_files.append(filepath)
    if not instance_files:
        print(f"No .json files found in {dirpath}")
        sys.exit(1)
    return [load_instance(f) for f in instance_files]


def parse_target(s: str, N: int, c: int) -> list[list[int]]:
    """Parse target string "v,v,...;v,v,...;..." into N×N int matrix."""
    rows = s.strip().split(";")
    if len(rows) != N:
        raise ValueError(f"Expected {N} rows, got {len(rows)}")
    matrix = []
    for ri, row_str in enumerate(rows):
        vals = [int(x.strip()) for x in row_str.split(",")]
        if len(vals) != N:
            raise ValueError(f"Row {ri}: expected {N} values, got {len(vals)}")
        for v in vals:
            if not 0 <= v < c:
                raise ValueError(f"Value {v} out of range [0, {c-1}]")
        matrix.append(vals)
    return matrix


# =====================================================================
#  CLI
# =====================================================================

def solve_instance(inst: dict):
    """Solve a single instance and print results."""
    N, c, target = inst["N"], inst["c"], inst["target"]
    print("=" * 60)
    print(f"Instance: {inst['name']}")
    print(f"  N={N}, c={c}")
    print_matrix("  Target T", target)

    sat = AlienTilesSAT(N, c, target)
    solution = sat.solve()

    print(f"\n  Encoding: {sat.stats['vars']} variables, "
          f"{sat.stats['clauses']} clauses")
    print(f"  Encode time: {sat.stats['time_encode']:.4f}s")
    print(f"  Solve time:  {sat.stats['time_solve']:.4f}s")

    if solution is None:
        print("\n  Result: UNSATISFIABLE — no solution exists.")
    else:
        print_matrix("  Solution X (click matrix)", solution)
        total = sum(sum(row) for row in solution)
        print(f"\n  Total clicks: {total}")

        ok = verify_solution(N, c, target, solution)
        print(f"  Verification: {'✓ PASSED' if ok else '✗ FAILED'}")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="SAT solver for Alien Tiles (Variant 1: Feasibility)")
    parser.add_argument("--input", type=str, default=None,
                        help="Path to a JSON instance file")
    parser.add_argument("--input-dir", type=str, default=None,
                        help="Path to a directory of JSON instance files")
    parser.add_argument("--N", type=int, help="Grid size (N×N)")
    parser.add_argument("--c", type=int, help="Number of colours")
    parser.add_argument("--target", type=str, default=None,
                        help='Target matrix, e.g. "1,0,2;0,1,0;2,0,1"')
    parser.add_argument("--examples", action="store_true",
                        help="Run built-in example instances")
    args = parser.parse_args()

    # Determine instances
    if args.input:
        instances = [load_instance(args.input)]
    elif args.input_dir:
        instances = load_instances_from_dir(args.input_dir)
    elif args.examples:
        instances = example_instances()
    elif args.N is not None and args.c is not None and args.target is not None:
        target = parse_target(args.target, args.N, args.c)
        instances = [{"name": f"User instance ({args.N}×{args.N}, c={args.c})",
                      "N": args.N, "c": args.c, "target": target}]
    else:
        parser.print_help()
        sys.exit(1)

    # Solve each instance
    for inst in instances:
        solve_instance(inst)


if __name__ == "__main__":
    main()
