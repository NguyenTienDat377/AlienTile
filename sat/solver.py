#!/usr/bin/env python3
"""
Solving strategies for Alien Tiles, and the benchmark CLI.

Φ_K = F ∧ AtMost({u_ijh}, K); the optimum is the smallest satisfiable K.
Modes follow Table 2 of the report: how K is chosen × how long a solver lives.

    python3 sat/solver.py --input-dir prob027/data/benchmark_v1 --xlsx sat/results.xlsx
    python3 sat/solver.py --input data/4x4_c3_easy.json --mode topdown-new
    python3 sat/solver.py --input-dir data/ --mode feasibility
"""

from __future__ import annotations

import math
import os
import sys
from functools import partial

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pysat.card import CardEnc, EncType, ITotalizer
from pysat.examples.rc2 import RC2
from pysat.formula import CNF, WCNF
from pysat.solvers import Glucose4

import common
from encoder import AlienTilesEncoder


def _solver_for_bound(enc: AlienTilesEncoder, K: int) -> Glucose4:
    """A solver loaded with Φ_K."""
    solver = Glucose4(bootstrap_with=enc.clauses.clauses)

    if K < len(enc.unit_lits):        # a bound >= the literal count constrains nothing
        card: CNF = CardEnc.atmost(
            lits=enc.unit_lits,
            bound=K,
            top_id=enc.pool.top,
            encoding=EncType.seqcounter,
        )
        solver.append_formula(card.clauses)

    return solver


def _total_clicks(X: list[list[int]]) -> int:
    return sum(sum(row) for row in X)


# Module-level and picklable: run_with_timeout runs these in a spawned child.

def solve_feasibility(N: int, c: int, target: list[list[int]],
                      witness=None) -> common.SolveResult:
    """Variant 1: is any click matrix reaching `target` possible at all?"""
    enc = AlienTilesEncoder(N, c, target)
    enc.build()
    common.report({"Variables": enc.stats["vars"], "Clauses": enc.stats["clauses"],
                   "SAT Calls": 1})

    with Glucose4(bootstrap_with=enc.clauses.clauses) as solver:
        if not solver.solve():
            return common.SolveResult(X=None, n_vars=enc.stats["vars"],
                                      n_clauses=enc.stats["clauses"],
                                      extra={"SAT Calls": 1})
        X = enc.decode(set(solver.get_model()))

    return common.SolveResult(X=X, optimum=None, n_vars=enc.stats["vars"],
                              n_clauses=enc.stats["clauses"],
                              extra={"SAT Calls": 1})


def _built_encoder(N: int, c: int, target: list[list[int]]) -> AlienTilesEncoder:
    enc = AlienTilesEncoder(N, c, target)
    enc.build()
    common.report({"Variables": enc.stats["vars"], "Clauses": enc.stats["clauses"]})
    return enc


def _result(enc: AlienTilesEncoder, X, calls: int, LB: int | None = None) -> common.SolveResult:
    extra = {"SAT Calls": calls} if LB is None else {"SAT Calls": calls, "LB": LB}
    return common.SolveResult(X=X, optimum=_total_clicks(X) if X else None,
                              n_vars=enc.stats["vars"],
                              n_clauses=enc.stats["clauses"],
                              extra=extra)


def _calculate_lower_bound(N: int, c: int, target: list[list[int]]):
    max_t = max(max(row) for row in target) 

    total = sum(sum(row) for row in target)

    L = 2 * N - 1

    LB0 = max(max_t, (total + L - 1) // L)

    g = math.gcd(L, c)

    if (total % g != 0): 
        return None

    f = LB0
    while (L * f) % c != total % c:
        f += 1
    return f


def _bound_literals(enc: AlienTilesEncoder, UB: int) -> tuple[list[int], list[list[int]]]:
    """U[k] means F0 ≤ k (k = 0..UB), and the clauses tying U to the unit literals."""
    with ITotalizer(enc.unit_lits, ubound=UB, top_id=enc.pool.top) as itot:
        U = [-r for r in itot.rhs]                  # rhs[k] is forced once k+1 unit literals are true
        clauses = itot.cnf.clauses + [[-U[k], U[k + 1]] for k in range(len(U) - 1)]  # U_k -> U_k+1
    return U, clauses


def _next_bound(strategy: str, LB: int, UB: int) -> int:
    """The K to ask next, inside [LB, UB - 1] (5.6)."""
    if strategy == "linear":
        return LB                                   # bottom-up
    if strategy == "binary":
        return (LB + UB - 1) // 2                   # midpoint
    return UB - 1                                   # topdown


def solve_optimum(N: int, c: int, target: list[list[int]], witness=None,
                  strategy: str = "binary", incremental: bool = True) -> common.SolveResult:
    """Minimise clicks. strategy picks K; incremental picks the solver lifecycle (Table 2)."""
    enc = _built_encoder(N, c, target)
    calls = 0
    LB = _calculate_lower_bound(N, c, target)
    if LB is None:                                  # (27) has no solution: target infeasible
        return _result(enc, None, calls)

    solver = Glucose4(bootstrap_with=enc.clauses.clauses) if incremental else None
    U = []                                          # U[K] means F0 ≤ K; filled once UB is known

    def solve_at_most(K):
        nonlocal calls
        calls += 1
        common.report({"SAT Calls": calls})
        if incremental:
            ok = solver.solve(assumptions=[U[K]] if K < len(U) else [])
            return enc.decode(set(solver.get_model())) if ok else None
        with _solver_for_bound(enc, K) as fresh:    # a new solver for every K
            return enc.decode(set(fresh.get_model())) if fresh.solve() else None

    try:
        best_X = witness                            # 5.5 #1, already verified by run_instance
        if best_X is None:
            best_X = solve_at_most(len(enc.unit_lits))   # 5.5 #2: F alone, no bound
            if best_X is None:
                return _result(enc, None, calls)
        UB = _total_clicks(best_X)
        assert LB <= UB, f"lower bound {LB} is above a known solution {UB}"
        common.report({"LB": LB, "Total Clicks": UB, "_best_X": best_X})

        if incremental and LB < UB:
            bound_lits, clauses = _bound_literals(enc, UB - 1)   # totalizer cut at UB - 1 (5.7)
            U.extend(bound_lits)
            solver.append_formula(clauses)

        while LB < UB:
            K = _next_bound(strategy, LB, UB)
            X = solve_at_most(K)
            if X is None:
                LB = K + 1                          # nothing costs ≤ K
                common.report({"LB": LB})
            else:
                best_X, UB = X, _total_clicks(X)    # the real cost, which may be below K
                common.report({"Total Clicks": UB, "_best_X": best_X})
        return _result(enc, best_X, calls, LB)
    finally:
        if solver is not None:
            solver.delete()


def solve_maxsat(N: int, c: int, target: list[list[int]],
                 witness=None) -> common.SolveResult:
    """MaxSAT: hard = F + U clauses, soft = [U_LB … U_UB], solved by RC2."""
    enc = _built_encoder(N, c, target)
    calls = 0
    LB = _calculate_lower_bound(N, c, target)
    if LB is None:
        return _result(enc, None, calls)

    best_X = witness
    if best_X is None:
        with Glucose4(bootstrap_with=enc.clauses.clauses) as solver:
            calls = 1
            common.report({"SAT Calls": calls})
            if not solver.solve():
                return _result(enc, None, calls)
            best_X = enc.decode(set(solver.get_model()))
    UB = _total_clicks(best_X)
    assert LB <= UB, f"lower bound {LB} is above a known solution {UB}"
    common.report({"SAT Calls": calls, "LB": LB, "Total Clicks": UB, "_best_X": best_X})

    if LB < UB:
        U, clauses = _bound_literals(enc, UB)
        wcnf = WCNF()
        wcnf.extend(enc.clauses.clauses + clauses)
        for u_k in U[LB:UB + 1]:
            wcnf.append([u_k], weight=1)
        with RC2(wcnf, solver="g4") as rc2:
            best_X = enc.decode(set(rc2.compute()))

    return _result(enc, best_X, calls, _total_clicks(best_X))


MODES = {
    "feasibility": solve_feasibility,
    "linear-new": partial(solve_optimum, strategy="linear", incremental=False),
    "binary-new": partial(solve_optimum, strategy="binary", incremental=False),
    "topdown-new": partial(solve_optimum, strategy="topdown", incremental=False),
    "linear-incr": partial(solve_optimum, strategy="linear", incremental=True),
    "binary-incr": partial(solve_optimum, strategy="binary", incremental=True),
    "topdown-incr": partial(solve_optimum, strategy="topdown", incremental=True),
    "maxsat": solve_maxsat,
}


def main() -> None:
    parser = common.build_parser("Alien Tiles SAT solver (prob027)")
    parser.add_argument("--mode", choices=sorted(MODES), default="binary-incr",
                        help="binary-incr: the report's main method (default); "
                             "feasibility: any solution will do")
    args = parser.parse_args()

    instances = common.collect_instances(args)
    if not instances:
        parser.print_help()
        sys.exit(1)

    solve_fn = MODES[args.mode]
    timeout = args.timeout if args.timeout and args.timeout > 0 else None

    if args.upgrade_timeouts:
        names = common.names_to_upgrade(args.xlsx, args.mode, timeout or 0)
        instances = [i for i in instances if i["name"] in names]
        print(f"Re-running {len(instances)} TIMEOUT instances at "
              f"{timeout}s.", flush=True)

    if args.only_missing:
        names = common.names_missing_fields(
            args.xlsx, args.mode, ["Variables", "Clauses", "SAT Calls"])
        instances = [i for i in instances if i["name"] in names]
        print(f"Re-running {len(instances)} instances with missing fields.",
              flush=True)

    if args.only_status:
        wanted = {s.strip().upper() for s in args.only_status.split(",")}
        names = common.names_with_status(args.xlsx, args.mode, wanted)
        instances = [i for i in instances if i["name"] in names]
        print(f"Re-running {len(instances)} instances with status "
              f"{', '.join(sorted(wanted))}.", flush=True)

    if args.resume:
        done = common.already_done(args.xlsx, args.mode)
        before = len(instances)
        instances = [i for i in instances if i["name"] not in done]
        print(f"Resuming: {before - len(instances)} already done, "
              f"{len(instances)} to go.", flush=True)

    # Exported per instance so an interrupted sweep keeps what it finished.
    rows = []
    for idx, inst in enumerate(instances, start=1):
        row = common.run_instance(solve_fn, inst, timeout, verbose=not args.quiet)
        rows.append(row)

        if not args.no_excel:
            common.export_to_excel([row], sheet_name=args.mode,
                                   xlsx_path=args.xlsx, quiet=True)

        if args.quiet:
            clicks = row.get("Optimal Clicks")
            label = f"opt={clicks}" if clicks is not None else \
                    f"clicks={row.get('Total Clicks', '-')}"
            print(f"[{idx}/{len(instances)}] {inst['name']}: "
                  f"{row.get('Status')} ({label}, "
                  f"{row.get('Runtime (s)')}s)", flush=True)

    solved = sum(1 for r in rows if r.get("Status") == "OK")
    unsat = sum(1 for r in rows if r.get("Status") == "UNSAT")
    timeouts = sum(1 for r in rows if r.get("Status") == "TIMEOUT")
    print(f"Done: {solved} solved+verified, {unsat} UNSAT, "
          f"{timeouts} timeout, {len(rows)} total.")


if __name__ == "__main__":
    # Required: multiprocessing spawns a child that re-imports this module.
    main()
