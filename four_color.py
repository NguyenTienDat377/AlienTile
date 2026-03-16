"""
SAT-Based Optimisation for the Alien Tiles Problem (general c >= 2).

Implements the three approaches from Section 7:
  7.1  Non-incremental SAT  (binary search + linear search top-down)
  7.2  Incremental SAT      (assumption-based bound control with totalizer)
  7.3  MaxSAT / PBO         (partial weighted MaxSAT)

Each click variable x_{i,j} in {0,...,c-1} is binary-encoded (Section 6.1).
Feasibility uses a running modular-sum encoding (Sections 6.2-6.3).
The objective uses unit-contribution literals u_{i,j,v} (Section 7).
"""

import json
import math
import sys
import time

from pysat.card import CardEnc, EncType
from pysat.examples.rc2 import RC2
from pysat.formula import WCNF
from pysat.solvers import Solver as PySATSolver


# ── helpers ──────────────────────────────────────────────────────────────

def load_instance(path: str):
    with open(path) as f:
        data = json.load(f)
    return data["N"], data["c"], data["target"]


def print_board(board, title="Board"):
    print(f"{title}:")
    for row in board:
        print(" ".join(str(cell) for cell in row))
    print()


# ── CNF encoder ──────────────────────────────────────────────────────────

class AlienTilesEncoder:
    """Encodes an Alien Tiles instance (N, c, target) as a CNF formula."""

    def __init__(self, N, c, target):
        self.N = N
        self.c = c
        self.target = target
        self.b = math.ceil(math.log2(c)) if c > 1 else 1
        self.top = 0
        self.clauses = []
        self.p = None       # p[i][j][l]: SAT var for bit l of x_{i,j}
        self.d = None       # d[i][j][w]: SAT var for "x_{i,j} = w"
        self.u_lits = None  # all unit-contribution literals

    def _new(self):
        self.top += 1
        return self.top

    def build(self):
        """Build feasibility CNF + unit-contribution literals."""
        self._encode_click_vars()
        self._encode_value_indicators()
        self._encode_feasibility()
        self._encode_unit_lits()
        return self.clauses, self.u_lits, self.top

    # ── Section 6.1: binary encoding of click variables ──────────────

    def _encode_click_vars(self):
        N, c, b = self.N, self.c, self.b
        self.p = [[[self._new() for _ in range(b)]
                    for _ in range(N)]
                   for _ in range(N)]

        # Exclude invalid bit-patterns when c is not a power of 2
        if c < (1 << b):
            for i in range(N):
                for j in range(N):
                    for v in range(c, 1 << b):
                        clause = []
                        for l in range(b):
                            if (v >> l) & 1:
                                clause.append(-self.p[i][j][l])
                            else:
                                clause.append(self.p[i][j][l])
                        self.clauses.append(clause)

    # ── value-indicator variables d[i][j][w] ⟺ "x_{i,j} = w" ────────

    def _encode_value_indicators(self):
        N, c, b = self.N, self.c, self.b
        self.d = [[[None] * c for _ in range(N)] for _ in range(N)]

        for i in range(N):
            for j in range(N):
                for w in range(c):
                    dv = self._new()
                    self.d[i][j][w] = dv

                    # Forward: dv → each bit matches w's pattern
                    backward = [dv]
                    for l in range(b):
                        if (w >> l) & 1:
                            self.clauses.append([-dv, self.p[i][j][l]])
                            backward.append(-self.p[i][j][l])
                        else:
                            self.clauses.append([-dv, -self.p[i][j][l]])
                            backward.append(self.p[i][j][l])
                    # Backward: all bits match w → dv
                    self.clauses.append(backward)

    # ── Sections 6.2–6.3: feasibility via running modular sum ────────

    def _encode_feasibility(self):
        """
        For each cell (r,k):
          σ_{r,k} = (Σ_j x_{r,j} + Σ_i x_{i,k} − x_{r,k}) mod c = target[r][k]

        Encoded as a running partial-sum mod c using one-hot state variables.
        """
        N, c = self.N, self.c

        for r in range(N):
            for k in range(N):
                # 2N−1 terms: full row r, then column k excluding (r,k)
                terms = [(r, j) for j in range(N)]
                terms += [(i, k) for i in range(N) if i != r]

                prev = None  # prev[v] = var for "partial sum ≡ v (mod c)"

                for ti, tj in terms:
                    curr = [self._new() for _ in range(c)]

                    # Exactly-one constraint on curr
                    self.clauses.append(list(curr))           # at-least-one
                    for a in range(c):
                        for a2 in range(a + 1, c):
                            self.clauses.append([-curr[a], -curr[a2]])  # at-most-one

                    if prev is None:
                        # First term: curr[w] ⟺ d[ti][tj][w]
                        for w in range(c):
                            self.clauses.append([-curr[w], self.d[ti][tj][w]])
                            self.clauses.append([curr[w], -self.d[ti][tj][w]])
                    else:
                        # Forward: prev[a] ∧ d[w] → curr[(a+w)%c]
                        for a in range(c):
                            for w in range(c):
                                self.clauses.append(
                                    [-prev[a], -self.d[ti][tj][w], curr[(a + w) % c]])

                        # Backward: curr[v] ∧ prev[a] → d[(v−a)%c]
                        for v in range(c):
                            for a in range(c):
                                w = (v - a) % c
                                self.clauses.append(
                                    [-curr[v], -prev[a], self.d[ti][tj][w]])

                    prev = curr

                # Assert final sum ≡ target[r][k]
                self.clauses.append([prev[self.target[r][k]]])

    # ── Section 7: unit-contribution literals ────────────────────────

    def _encode_unit_lits(self):
        """
        u_{i,j,v} = 1 ⟺ x_{i,j} ≥ v   for v ∈ {1,…,c−1}.
        total(X) = Σ_{i,j} Σ_{v=1}^{c-1} u_{i,j,v} = Σ_{i,j} x_{i,j}.
        """
        N, c = self.N, self.c
        self.u_lits = []

        for i in range(N):
            for j in range(N):
                for v in range(1, c):
                    u = self._new()
                    self.u_lits.append(u)

                    # u ⟺ ∨_{w=v}^{c-1} d[i][j][w]
                    backward = [-u]
                    for w in range(v, c):
                        self.clauses.append([-self.d[i][j][w], u])  # d[w] → u
                        backward.append(self.d[i][j][w])
                    self.clauses.append(backward)  # u → ∨ d[w]

    # ── decode / print ───────────────────────────────────────────────

    def decode(self, model_set):
        """Return (click_matrix, total_clicks) from a model."""
        N, b = self.N, self.b
        matrix = []
        total = 0
        for i in range(N):
            row = []
            for j in range(N):
                val = sum((1 << l) for l in range(b)
                          if self.p[i][j][l] in model_set)
                row.append(val)
                total += val
            matrix.append(row)
        return matrix, total

    def print_solution(self, model_set):
        matrix, total = self.decode(model_set)
        print("Click matrix:")
        for row in matrix:
            print(" ".join(str(x) for x in row))
        print(f"Total clicks: {total}\n")
        return total


def _build(N, c, target):
    """Create a fresh encoder and build its CNF."""
    enc = AlienTilesEncoder(N, c, target)
    clauses, u_lits, top = enc.build()
    return enc, clauses, u_lits, top


# ══════════════════════════════════════════════════════════════════════════
#  7.1  Non-incremental SAT (multiple independent calls)
# ══════════════════════════════════════════════════════════════════════════

def approach_71_binary(N, c, target):
    """
    Section 7.1.1 — Binary search.

    Fresh solver per iteration.  O(log(N²(c−1))) SAT calls.
    """
    print("=" * 60)
    print("Approach 7.1.1: Non-incremental SAT — Binary Search")
    print("=" * 60)

    enc, clauses, u_lits, top = _build(N, c, target)
    solver = PySATSolver(name='cadical153', bootstrap_with=clauses)

    if not solver.solve():
        print("UNSATISFIABLE\n")
        solver.delete()
        return None

    model_set = set(solver.get_model())
    solver.delete()

    hi = sum(1 for u in u_lits if u in model_set)
    best_set, best_enc = model_set, enc
    lo, calls = 0, 1
    print(f"Initial feasible solution: {hi} clicks")

    while lo <= hi:
        mid = (lo + hi) // 2
        calls += 1

        enc2, cl2, ul2, top2 = _build(N, c, target)
        am = CardEnc.atmost(ul2, bound=mid, top_id=top2,
                            encoding=EncType.seqcounter)
        s = PySATSolver(name='cadical153', bootstrap_with=cl2 + am.clauses)

        if s.solve():
            ms = set(s.get_model())
            cost = sum(1 for u in ul2 if u in ms)
            best_set, best_enc = ms, enc2
            hi = mid - 1
            print(f"  SAT call #{calls}: AtMost({mid}) → SAT  (actual = {cost})")
        else:
            lo = mid + 1
            print(f"  SAT call #{calls}: AtMost({mid}) → UNSAT")
        s.delete()

    opt = best_enc.print_solution(best_set)
    print(f"Optimal: {opt} clicks  ({calls} SAT calls)")
    return opt


def approach_71_linear(N, c, target):
    """
    Section 7.1.2 — Linear search (top-down).

    Find any feasible solution with cost K, then tighten to K−1 until UNSAT.
    """
    print("=" * 60)
    print("Approach 7.1.2: Non-incremental SAT — Linear Search (top-down)")
    print("=" * 60)

    enc, clauses, u_lits, top = _build(N, c, target)
    solver = PySATSolver(name='cadical153', bootstrap_with=clauses)

    if not solver.solve():
        print("UNSATISFIABLE\n")
        solver.delete()
        return None

    model_set = set(solver.get_model())
    solver.delete()

    UB = sum(1 for u in u_lits if u in model_set)
    best_set, best_enc = model_set, enc
    calls = 1
    print(f"Initial feasible solution: {UB} clicks")

    while True:
        calls += 1
        enc2, cl2, ul2, top2 = _build(N, c, target)
        am = CardEnc.atmost(ul2, bound=UB - 1, top_id=top2,
                            encoding=EncType.seqcounter)
        s = PySATSolver(name='cadical153', bootstrap_with=cl2 + am.clauses)

        if s.solve():
            ms = set(s.get_model())
            UB = sum(1 for u in ul2 if u in ms)
            best_set, best_enc = ms, enc2
            print(f"  SAT call #{calls}: AtMost({UB}) → SAT  (actual = {UB})")
        else:
            print(f"  SAT call #{calls}: AtMost({UB - 1}) → UNSAT")
            s.delete()
            break
        s.delete()

    opt = best_enc.print_solution(best_set)
    print(f"Optimal: {opt} clicks  ({calls} SAT calls)")
    return opt


# ══════════════════════════════════════════════════════════════════════════
#  7.2  Incremental SAT — assumption-based bound control
# ══════════════════════════════════════════════════════════════════════════

def _build_totalizer(solver, lits, top):
    """
    Incremental totalizer (Bailleux & Boufkhad, CP 2003).

    Returns output literals o_1,…,o_n where o_k is true iff
    at least k of the inputs are true.
    """
    n = len(lits)
    if n == 0:
        return [], top
    if n == 1:
        return list(lits), top

    mid = n // 2
    left, top = _build_totalizer(solver, lits[:mid], top)
    right, top = _build_totalizer(solver, lits[mid:], top)

    a, b_len = len(left), len(right)
    total = a + b_len
    outputs = []
    for _ in range(total):
        top += 1
        outputs.append(top)

    for i in range(a):
        solver.add_clause([-left[i], outputs[i]])
    for j in range(b_len):
        solver.add_clause([-right[j], outputs[j]])
    for i in range(a):
        for j in range(b_len):
            k = i + j + 1
            if k < total:
                solver.add_clause([-left[i], -right[j], outputs[k]])

    return outputs, top


def approach_72_incremental(N, c, target):
    """
    Section 7.2.1 — Incremental SAT with assumption-based bound control.

    Single solver; totalizer built once; binary search via assumptions.
    """
    print("=" * 60)
    print("Approach 7.2: Incremental SAT — Assumption-based Bound Control")
    print("=" * 60)

    enc, clauses, u_lits, top = _build(N, c, target)
    solver = PySATSolver(name='cadical153', bootstrap_with=clauses)

    outputs, top = _build_totalizer(solver, u_lits, top)

    if not solver.solve():
        print("UNSATISFIABLE\n")
        solver.delete()
        return None

    model_set = set(solver.get_model())
    hi = sum(1 for u in u_lits if u in model_set)
    best_set = model_set
    lo, calls = 0, 1
    print(f"Initial feasible solution: {hi} clicks")

    while lo <= hi:
        mid = (lo + hi) // 2
        calls += 1

        # ¬o_{mid+1} asserts "at most mid"
        assumptions = [-outputs[mid]] if mid < len(outputs) else []

        if solver.solve(assumptions=assumptions):
            ms = set(solver.get_model())
            cost = sum(1 for u in u_lits if u in ms)
            best_set = ms
            hi = mid - 1
            print(f"  SAT call #{calls}: AtMost({mid}) → SAT  (actual = {cost})")
        else:
            lo = mid + 1
            print(f"  SAT call #{calls}: AtMost({mid}) → UNSAT")

    opt = enc.print_solution(best_set)
    print(f"Optimal: {opt} clicks  ({calls} SAT calls)")
    solver.delete()
    return opt


# ══════════════════════════════════════════════════════════════════════════
#  7.3  MaxSAT — partial weighted MaxSAT
# ══════════════════════════════════════════════════════════════════════════

def approach_73_maxsat(N, c, target):
    """
    Section 7.3.1 — Partial weighted MaxSAT via RC2.

    Hard clauses : feasibility F.
    Soft clauses : (¬u_{i,j,v}) with weight 1 for each unit literal.
    Maximising satisfied soft clauses ⟺ minimising total clicks.
    """
    print("=" * 60)
    print("Approach 7.3: MaxSAT — Partial Weighted MaxSAT")
    print("=" * 60)

    enc, clauses, u_lits, top = _build(N, c, target)

    wcnf = WCNF()
    wcnf.nv = top
    for cl in clauses:
        wcnf.append(cl)
    for u in u_lits:
        wcnf.append([-u], weight=1)

    rc2 = RC2(wcnf)
    model = rc2.compute()

    if model is None:
        print("UNSATISFIABLE\n")
        rc2.delete()
        return None

    model_set = set(model)
    opt = enc.print_solution(model_set)
    print(f"Optimal: {opt} clicks")
    rc2.delete()
    return opt


# ══════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <instance.json> [approach]")
        print("  approach: 'binary' | 'linear' | 'incremental' | 'maxsat' | 'all' (default)")
        sys.exit(1)

    path = sys.argv[1]
    approach = sys.argv[2] if len(sys.argv) > 2 else "all"

    N, c, target = load_instance(path)
    print(f"Instance: N={N}, c={c}")
    print_board(target, "Target")

    results = {}

    if approach in ("binary", "all"):
        t0 = time.perf_counter()
        results["7.1.1 Binary search"] = approach_71_binary(N, c, target)
        print(f"  Time: {time.perf_counter() - t0:.4f}s\n")

    if approach in ("linear", "all"):
        t0 = time.perf_counter()
        results["7.1.2 Linear search"] = approach_71_linear(N, c, target)
        print(f"  Time: {time.perf_counter() - t0:.4f}s\n")

    if approach in ("incremental", "all"):
        t0 = time.perf_counter()
        results["7.2 Incremental SAT"] = approach_72_incremental(N, c, target)
        print(f"  Time: {time.perf_counter() - t0:.4f}s\n")

    if approach in ("maxsat", "all"):
        t0 = time.perf_counter()
        results["7.3 MaxSAT"] = approach_73_maxsat(N, c, target)
        print(f"  Time: {time.perf_counter() - t0:.4f}s\n")

    if approach == "all" and results:
        print("=" * 60)
        print("Summary")
        print("=" * 60)
        for name, opt in results.items():
            print(f"  {name}: {opt} clicks")