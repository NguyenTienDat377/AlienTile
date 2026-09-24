#!/usr/bin/env python3
"""
Solving strategies for Alien Tiles, and the benchmark CLI.

Φ_K = Φ ∧ AtMost({u_ijh}, K); the optimum is the smallest satisfiable K.

    python3 sat/solver.py --input-dir data/ --xlsx sat/results.xlsx
    python3 sat/solver.py --input data/4x4_c3_easy.json
    python3 sat/solver.py --input-dir data/ --mode feasibility
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pysat.card import CardEnc, EncType
from pysat.formula import CNF
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

def solve_feasibility(N: int, c: int, target: list[list[int]]) -> common.SolveResult:
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


def _solve_optimum(N: int, c: int, target: list[list[int]],
                   binary: bool) -> common.SolveResult:
    enc = AlienTilesEncoder(N, c, target)
    enc.build()
    stats = {"vars": enc.stats["vars"], "clauses": enc.stats["clauses"]}
    calls = 0
    common.report({"Variables": stats["vars"], "Clauses": stats["clauses"],
                   "SAT Calls": calls})

    # Reported before each solve, so a TIMEOUT row counts the call it died in.
    def started_call():
        nonlocal calls
        calls += 1
        common.report({"SAT Calls": calls})

    def new_incumbent(X):
        common.report({"Total Clicks": _total_clicks(X), "_best_X": X})

    # Unbounded call first: settles UNSAT in one call, and its click total is a
    # real upper bound for the search below.
    with Glucose4(bootstrap_with=enc.clauses.clauses) as solver:
        started_call()
        if not solver.solve():
            return common.SolveResult(X=None, n_vars=stats["vars"],
                                      n_clauses=stats["clauses"],
                                      extra={"SAT Calls": calls})
        best_X = enc.decode(set(solver.get_model()))

    best_K = _total_clicks(best_X)
    new_incumbent(best_X)

    if binary:
        lo, hi = 0, best_K - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            with _solver_for_bound(enc, mid) as solver:
                started_call()
                if solver.solve():
                    best_X = enc.decode(set(solver.get_model()))
                    best_K = _total_clicks(best_X)
                    new_incumbent(best_X)
                    hi = best_K - 1
                else:
                    lo = mid + 1
    else:
        for K in range(best_K + 1):
            with _solver_for_bound(enc, K) as solver:
                started_call()
                if solver.solve():
                    best_X = enc.decode(set(solver.get_model()))
                    best_K = _total_clicks(best_X)
                    new_incumbent(best_X)
                    break

    return common.SolveResult(X=best_X, optimum=best_K, n_vars=stats["vars"],
                              n_clauses=stats["clauses"],
                              extra={"SAT Calls": calls})


def solve_optimum_linear(N: int, c: int, target: list[list[int]]) -> common.SolveResult:
    """Variant 2, K = 0,1,2,… — the paper's presentation."""
    return _solve_optimum(N, c, target, binary=False)


def solve_optimum_binary(N: int, c: int, target: list[list[int]]) -> common.SolveResult:
    """Variant 2, binary search — same answer, O(log K) solver calls."""
    return _solve_optimum(N, c, target, binary=True)


MODES = {
    "optimum": solve_optimum_linear,
    "optimum-binary": solve_optimum_binary,
    "feasibility": solve_feasibility,
}


def main() -> None:
    parser = common.build_parser("Alien Tiles SAT solver (prob027)")
    parser.add_argument("--mode", choices=sorted(MODES), default="optimum",
                        help="optimum: minimise clicks (default); "
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
