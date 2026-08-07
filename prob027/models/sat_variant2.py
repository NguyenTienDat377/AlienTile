#!/usr/bin/env python3
from __future__ import annotations
"""
SAT encoding for the Alien Tiles Problem (CSPLib prob027)
=========================================================

Variant 2: MINIMISATION
    Given a target state T, find a click matrix X that achieves T
    with the MINIMUM total number of clicks  sum_{i,j} x[i][j].

Approach
--------
Extends Variant 1 (feasibility) with:
1.  Unary expansion of click values:  for each x[i][j] and each
    v in {1,..,c-1}, create u(i,j,v) = true iff x[i][j] >= v.
    Then  total_clicks = sum of all u literals.
2.  Cardinality constraint:  "at most K of the u-literals are true"
    encoded via a sequential counter (CardEnc.atmost).
3.  Binary search on K to find the minimum feasible total.

Usage
-----
    python3 sat_variant2.py --input data/4x4_c3_easy.json
    python3 sat_variant2.py --input-dir data/
    python3 sat_variant2.py --N 4 --c 3 --target "1,1,1,1;1,1,1,1;1,1,1,1;1,1,1,1"
    python3 sat_variant2.py --examples
"""

import argparse
import glob
import json
import os
import sys
import time
from pysat.solvers import Glucose4
from pysat.formula import CNF, IDPool
from pysat.card import CardEnc, EncType


# =====================================================================
#  Import shared utilities from Variant 1
# =====================================================================

# Allow importing from same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sat_variant1 import (
    AlienTilesSAT,
    verify_solution,
    print_matrix,
    generate_target_from_clicks,
    example_instances,
    load_instance,
    load_instances_from_dir,
    parse_target,
)


# =====================================================================
#  Minimisation Solver
# =====================================================================

class AlienTilesMinSAT:
    """
    SAT-based minimisation solver for Alien Tiles.

    Uses the feasibility encoding from AlienTilesSAT plus a
    cardinality constraint on "unit contribution" literals,
    with binary search to find the minimum total clicks.
    """

    def __init__(self, N: int, c: int, target: list[list[int]]):
        self.N = N
        self.c = c
        self.target = target
        self.stats = {
            "vars": 0, "clauses": 0,
            "time_encode": 0.0, "time_solve": 0.0,
            "sat_calls": 0, "optimum": None,
        }

    def _build_cnf_with_bound(self, K: int):
        """
        Build a complete CNF with:
          (a) feasibility constraints  (Variant 1), and
          (b) total clicks <= K  (cardinality on unit-expansion literals).

        Returns (cnf, encoder) where encoder is the AlienTilesSAT instance
        (needed to extract the solution from the model).
        """
        N, c = self.N, self.c

        # --- (a) Feasibility encoding --------------------------------
        encoder = AlienTilesSAT(N, c, self.target)
        encoder.encode()
        cnf = encoder.cnf
        pool = encoder.pool

        # --- (b) Unit-expansion literals -----------------------------
        # For each cell (i,j) and each v in {1,..,c-1}:
        #   u(i,j,v) is true  iff  x[i][j] >= v
        # Then  total_clicks = #{u-literals that are true}.
        unit_lits = []
        for i in range(N):
            for j in range(N):
                for v in range(1, c):
                    u = pool.id(("u", i, j, v))

                    # Channelling:  u <-> OR( d[i][j][v], d[i][j][v+1], ..., d[i][j][c-1] )
                    d_lits = [encoder.var_click(i, j, w) for w in range(v, c)]

                    # Forward:  d_w -> u   for each w >= v
                    for d_w in d_lits:
                        cnf.append([-d_w, u])

                    # Backward:  u -> OR(d_lits)
                    cnf.append([-u] + d_lits)

                    unit_lits.append(u)

        # --- (c) Cardinality constraint:  at most K of unit_lits -----
        if K < len(unit_lits):
            card_cnf = CardEnc.atmost(
                lits=unit_lits,
                bound=K,
                top_id=pool.top,
                encoding=EncType.seqcounter,
            )
            for cl in card_cnf.clauses:
                cnf.append(cl)

        return cnf, encoder

    def _extract_solution(self, model_set: set, encoder: AlienTilesSAT):
        """Extract click matrix from a SAT model."""
        solution = []
        for i in range(self.N):
            row = []
            for j in range(self.N):
                for v in range(self.c):
                    if encoder.var_click(i, j, v) in model_set:
                        row.append(v)
                        break
                else:
                    row.append(-1)
            solution.append(row)
        return solution

    def solve(self):
        """
        Find the minimum-click solution via binary search.

        Returns
        -------
        (solution, min_total) or (None, None)
            solution is the N×N click matrix, min_total is its total clicks.
        """
        N, c = self.N, self.c
        max_total = N * N * (c - 1)

        t0_total = time.perf_counter()

        # --- First check feasibility (no bound) ----------------------
        t0 = time.perf_counter()
        cnf, encoder = self._build_cnf_with_bound(max_total)
        self.stats["time_encode"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        with Glucose4(bootstrap_with=cnf) as solver:
            if not solver.solve():
                self.stats["time_solve"] = time.perf_counter() - t0
                self.stats["sat_calls"] = 1
                return None, None
            model_set = set(solver.get_model())
        self.stats["sat_calls"] = 1

        best_solution = self._extract_solution(model_set, encoder)
        best_total = sum(sum(row) for row in best_solution)

        # --- Binary search for minimum K ----------------------------
        lo, hi = 0, best_total - 1

        while lo <= hi:
            mid = (lo + hi) // 2

            cnf, encoder = self._build_cnf_with_bound(mid)
            self.stats["sat_calls"] += 1

            with Glucose4(bootstrap_with=cnf) as solver:
                if solver.solve():
                    model_set = set(solver.get_model())
                    best_solution = self._extract_solution(model_set, encoder)
                    best_total = sum(sum(row) for row in best_solution)
                    hi = mid - 1
                else:
                    lo = mid + 1

        self.stats["time_solve"] = time.perf_counter() - t0_total
        self.stats["optimum"] = best_total
        self.stats["vars"] = cnf.nv
        self.stats["clauses"] = len(cnf.clauses)

        return best_solution, best_total


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

    solver = AlienTilesMinSAT(N, c, target)
    solution, min_total = solver.solve()

    print(f"\n  SAT calls:   {solver.stats['sat_calls']}")
    print(f"  Total time:  {solver.stats['time_solve']:.4f}s")

    if solution is None:
        print("\n  Result: UNSATISFIABLE — no solution exists.")
    else:
        print_matrix("  Optimal X (click matrix)", solution)
        print(f"\n  Minimum total clicks: {min_total}")

        ok = verify_solution(N, c, target, solution)
        print(f"  Verification: {'✓ PASSED' if ok else '✗ FAILED'}")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="SAT solver for Alien Tiles (Variant 2: Minimisation)")
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

    for inst in instances:
        solve_instance(inst)


if __name__ == "__main__":
    main()
