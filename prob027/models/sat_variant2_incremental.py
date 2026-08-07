#!/usr/bin/env python3
from __future__ import annotations
"""
SAT encoding for the Alien Tiles Problem (CSPLib prob027)
=========================================================

Variant 2 — INCREMENTAL SAT MINIMISATION
    Given a target state T, find a click matrix X that achieves T
    with the MINIMUM total number of clicks  sum_{i,j} x[i][j].

Approach
--------
Uses a SINGLE solver instance kept alive across all iterations:
1.  Build feasibility encoding (Variant 1) + unit-expansion literals.
2.  Build an Incremental Totalizer (ITotalizer) on the unit literals.
3.  Binary search on bound K using ASSUMPTIONS on the totalizer output
    (no re-encoding, full learned-clause reuse).

This is typically 2–10× faster than non-incremental binary search
(sat_variant2.py) because:
  - The totalizer is built ONCE.
  - Learned clauses from earlier iterations accelerate later calls.
  - Bound changes only modify assumptions, not the clause database.

Usage
-----
    python3 sat_variant2_incremental.py --input data/4x4_c3_easy.json
    python3 sat_variant2_incremental.py --input-dir data/
    python3 sat_variant2_incremental.py --examples
"""

import argparse
import os
import sys
import time
from pysat.solvers import Glucose4
from pysat.formula import CNF, IDPool
from pysat.card import ITotalizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sat_variant1 import (
    AlienTilesSAT,
    verify_solution,
    print_matrix,
    example_instances,
    load_instance,
    load_instances_from_dir,
    parse_target,
)


# =====================================================================
#  Incremental Minimisation Solver
# =====================================================================

class AlienTilesMinIncrSAT:
    """
    Incremental SAT-based minimisation solver for Alien Tiles.

    Keeps a single solver alive and uses ITotalizer with
    assumption-based bound control for binary search.
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

    def solve(self):
        """
        Find the minimum-click solution via incremental binary search.

        Returns (solution, min_total) or (None, None).
        """
        N, c = self.N, self.c
        max_total = N * N * (c - 1)

        # =============================================================
        # Phase 1: Build feasibility CNF + unit-expansion literals
        # =============================================================
        t0 = time.perf_counter()

        encoder = AlienTilesSAT(N, c, self.target)
        encoder.encode()
        cnf = encoder.cnf
        pool = encoder.pool

        # Unit-expansion: u(i,j,v) = true iff x[i][j] >= v
        # total_clicks = #{true u-literals}
        unit_lits = []
        for i in range(N):
            for j in range(N):
                for v in range(1, c):
                    u = pool.id(("u", i, j, v))
                    d_lits = [encoder.var_click(i, j, w) for w in range(v, c)]
                    # Forward: d_w -> u
                    for d_w in d_lits:
                        cnf.append([-d_w, u])
                    # Backward: u -> OR(d_lits)
                    cnf.append([-u] + d_lits)
                    unit_lits.append(u)

        self.stats["time_encode"] = time.perf_counter() - t0

        # =============================================================
        # Phase 2: Build Incremental Totalizer (ONCE)
        # =============================================================
        t0_solve = time.perf_counter()

        with ITotalizer(unit_lits, ubound=max_total) as itot:
            # Create solver with feasibility + totalizer clauses
            solver = Glucose4(bootstrap_with=cnf)
            solver.append_formula(itot.cnf)

            self.stats["vars"] = solver.nof_vars()
            self.stats["clauses"] = solver.nof_clauses()

            # ==========================================================
            # Phase 3: Check feasibility (no bound restriction)
            # ==========================================================
            self.stats["sat_calls"] += 1
            if not solver.solve():
                self.stats["time_solve"] = time.perf_counter() - t0_solve
                solver.delete()
                return None, None

            # Extract first solution and its total
            model_set = set(solver.get_model())
            best_solution = self._extract(model_set, encoder)
            best_total = sum(sum(row) for row in best_solution)

            # ==========================================================
            # Phase 4: Binary search with assumptions
            # ==========================================================
            lo, hi = 0, best_total - 1

            while lo <= hi:
                mid = (lo + hi) // 2
                self.stats["sat_calls"] += 1

                # ITotalizer.rhs[k] (0-indexed) = "at least k+1 are true"
                # "at most mid" = NOT "at least mid+1" = assume -rhs[mid]
                if mid < len(itot.rhs):
                    assumps = [-itot.rhs[mid]]
                else:
                    assumps = []  # no restriction needed

                if solver.solve(assumptions=assumps):
                    model_set = set(solver.get_model())
                    best_solution = self._extract(model_set, encoder)
                    best_total = sum(sum(row) for row in best_solution)
                    hi = mid - 1
                else:
                    lo = mid + 1

            solver.delete()

        self.stats["time_solve"] = time.perf_counter() - t0_solve
        self.stats["optimum"] = best_total
        return best_solution, best_total

    def _extract(self, model_set: set, encoder: AlienTilesSAT):
        """Extract click matrix from SAT model."""
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

    solver = AlienTilesMinIncrSAT(N, c, target)
    solution, min_total = solver.solve()

    print(f"\n  Encoding:    {solver.stats['vars']} vars, "
          f"{solver.stats['clauses']} clauses")
    print(f"  Encode time: {solver.stats['time_encode']:.4f}s")
    print(f"  SAT calls:   {solver.stats['sat_calls']}  (single solver)")
    print(f"  Solve time:  {solver.stats['time_solve']:.4f}s")

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
        description="Incremental SAT solver for Alien Tiles "
                    "(Variant 2: Minimisation)")
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
