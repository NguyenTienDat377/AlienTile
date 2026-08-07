#!/usr/bin/env python3
from __future__ import annotations
"""
SAT encoding for the Alien Tiles Problem (CSPLib prob027)
=========================================================

Variant 3: MAX-MIN (Hardest Puzzle)
    Find the target T whose minimum-click solution has the MAXIMUM
    total clicks.  i.e.  max_T  min_X { sum x[i][j] : X solves T }.

Approach  (CEGAR – Counter-Example Guided)
------------------------------------------
1.  Build a SAT formula with click variables X (no fixed target),
    running-sum encoding (to track the implicit target), unit-expansion
    literals, and a cardinality constraint  total(X) >= K.
2.  Find a satisfying X.  Compute T = effect(X).
3.  Call Variant 2 on T to get  min_total(T).
4.  If min_total > best_total:  update best, raise K, restart.
5.  Otherwise: block T via a clause on the running-sum final values
    ("at least one cell's effect must differ from T"), continue.
6.  When SAT returns UNSAT at the current K: done.

Usage
-----
    python3 sat_variant3.py --N 4 --c 3
    python3 sat_variant3.py --N 3 --c 2
"""

import argparse
import sys
import os
import time
from pysat.solvers import Glucose4
from pysat.formula import CNF, IDPool
from pysat.card import CardEnc, EncType

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sat_variant1 import (
    verify_solution,
    print_matrix,
    generate_target_from_clicks,
)
from sat_variant2 import AlienTilesMinSAT


# =====================================================================
#  Max-Min Solver
# =====================================================================

class AlienTilesMaxMinSAT:
    """CEGAR-based max-min solver for the Alien Tiles problem."""

    def __init__(self, N: int, c: int):
        self.N = N
        self.c = c
        self.pool = IDPool()
        self.unit_lits = []
        self.base_clauses = []
        self.stats = {
            "var2_calls": 0, "targets_checked": 0,
            "iterations": 0, "time_total": 0.0,
        }
        self._build_base()

    # -- Variable helpers ---------------------------------------------

    def var_click(self, i, j, v):
        return self.pool.id(("x", i, j, v))

    def var_sum(self, r, k, step, v):
        return self.pool.id(("s", r, k, step, v))

    # -- Exactly-one --------------------------------------------------

    def _eo(self, lits, clauses):
        clauses.append(list(lits))
        for a in range(len(lits)):
            for b in range(a + 1, len(lits)):
                clauses.append([-lits[a], -lits[b]])

    # -- Build base CNF -----------------------------------------------

    def _build_base(self):
        """Click vars + running-sum (no target fixation) + unit lits."""
        N, c = self.N, self.c
        cls = []

        # Click variables with exactly-one
        for i in range(N):
            for j in range(N):
                lits = [self.var_click(i, j, v) for v in range(c)]
                self._eo(lits, cls)

        # Running-sum for each cell (NO final assertion)
        for r in range(N):
            for k in range(N):
                terms = [(r, j) for j in range(N)]
                terms += [(i, k) for i in range(N) if i != r]

                # Step 0: channel to first term
                ti, tj = terms[0]
                s0 = [self.var_sum(r, k, 0, v) for v in range(c)]
                self._eo(s0, cls)
                for v in range(c):
                    cls.append([-s0[v], self.var_click(ti, tj, v)])
                    cls.append([s0[v], -self.var_click(ti, tj, v)])

                # Steps 1..2N-2: transition
                for step in range(1, len(terms)):
                    ti, tj = terms[step]
                    sl = [self.var_sum(r, k, step, v) for v in range(c)]
                    self._eo(sl, cls)
                    for a in range(c):
                        sp = self.var_sum(r, k, step - 1, a)
                        for b in range(c):
                            cls.append([-sp,
                                        -self.var_click(ti, tj, b),
                                        self.var_sum(r, k, step, (a + b) % c)])

        # Unit-expansion literals
        self.unit_lits = []
        for i in range(N):
            for j in range(N):
                for v in range(1, c):
                    u = self.pool.id(("u", i, j, v))
                    d_lits = [self.var_click(i, j, w) for w in range(v, c)]
                    for dw in d_lits:
                        cls.append([-dw, u])
                    cls.append([-u] + d_lits)
                    self.unit_lits.append(u)

        self.base_clauses = cls
        self.base_top = self.pool.top

    # -- Helpers ------------------------------------------------------

    def _make_cnf(self, K, blocked_targets):
        """Create CNF = base + atleast(K) + blocking clauses."""
        cnf = CNF(from_clauses=self.base_clauses)

        # Cardinality: at least K unit lits
        if K > 0 and K <= len(self.unit_lits):
            card = CardEnc.atleast(
                lits=self.unit_lits, bound=K,
                top_id=self.base_top, encoding=EncType.seqcounter,
            )
            for cl in card.clauses:
                cnf.append(cl)

        # Block known targets
        for T_key in blocked_targets:
            cnf.append(self._block_clause_from_key(T_key))

        return cnf

    def _block_clause(self, T):
        """Clause: at least one cell's effect differs from T."""
        N = self.N
        last = 2 * N - 2
        return [-self.var_sum(r, k, last, T[r][k])
                for r in range(N) for k in range(N)]

    def _block_clause_from_key(self, T_key):
        N = self.N
        last = 2 * N - 2
        idx = 0
        clause = []
        for r in range(N):
            for k in range(N):
                clause.append(-self.var_sum(r, k, last, T_key[idx]))
                idx += 1
        return clause

    def _extract_X(self, model_set):
        X = []
        for i in range(self.N):
            row = []
            for j in range(self.N):
                for v in range(self.c):
                    if self.var_click(i, j, v) in model_set:
                        row.append(v)
                        break
            X.append(row)
        return X

    def _compute_target(self, X):
        return generate_target_from_clicks(self.N, self.c, X)

    # -- Main solve ---------------------------------------------------

    def solve(self):
        """
        Find the hardest target T and its optimal solution.

        Returns (target, solution, max_min_total) or (None, None, 0).
        """
        N, c = self.N, self.c
        t0 = time.perf_counter()

        best_total = 0
        best_target = None
        best_solution = None
        checked = {}  # T_key -> min_total

        while True:
            K = best_total + 1
            print(f"  [CEGAR] Searching K >= {K}  "
                  f"(best={best_total}, checked={len(checked)} targets)")

            cnf = self._make_cnf(K, checked)
            improved = False

            with Glucose4(bootstrap_with=cnf) as solver:
                while solver.solve():
                    self.stats["iterations"] += 1
                    model_set = set(solver.get_model())
                    X = self._extract_X(model_set)
                    T = self._compute_target(X)
                    T_key = tuple(v for row in T for v in row)

                    if T_key in checked:
                        solver.add_clause(self._block_clause(T))
                        continue

                    # Variant 2: find min_total for this target
                    self.stats["var2_calls"] += 1
                    self.stats["targets_checked"] += 1
                    min_solver = AlienTilesMinSAT(N, c, T)
                    min_X, min_total = min_solver.solve()
                    checked[T_key] = min_total

                    if min_total is not None and min_total > best_total:
                        best_total = min_total
                        best_target = T
                        best_solution = min_X
                        improved = True
                        break  # restart with higher K

                    solver.add_clause(self._block_clause(T))

            if not improved:
                break

        self.stats["time_total"] = time.perf_counter() - t0
        return best_target, best_solution, best_total


# =====================================================================
#  CLI
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="SAT solver for Alien Tiles (Variant 3: Max-Min)")
    parser.add_argument("--N", type=int, required=True, help="Grid size")
    parser.add_argument("--c", type=int, required=True, help="Num colours")
    args = parser.parse_args()

    N, c = args.N, args.c
    print("=" * 60)
    print(f"Alien Tiles Max-Min: N={N}, c={c}")
    print(f"  Max possible total: {N*N*(c-1)}")
    print()

    solver = AlienTilesMaxMinSAT(N, c)
    target, solution, max_min = solver.solve()

    print()
    print(f"  Variant-2 calls: {solver.stats['var2_calls']}")
    print(f"  Targets checked: {solver.stats['targets_checked']}")
    print(f"  Total iterations: {solver.stats['iterations']}")
    print(f"  Total time: {solver.stats['time_total']:.2f}s")

    if target is None:
        print("\n  No non-trivial solvable target found.")
    else:
        print_matrix("  Hardest target T", target)
        print_matrix("  Optimal solution X", solution)
        print(f"\n  MAX-MIN total clicks: {max_min}")

        ok = verify_solution(N, c, target, solution)
        print(f"  Verification: {'✓ PASSED' if ok else '✗ FAILED'}")

    print()


if __name__ == "__main__":
    main()
