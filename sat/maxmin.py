#!/usr/bin/env python3
"""
Variant 3: max-min (hardest target) for Alien Tiles.

    max_T  min_X { Σ x_ij : X solves T }

    python3 sat/maxmin.py --N 3 --c 2
    python3 sat/maxmin.py --sweep 3x3_c2,3x3_c3,4x4_c2
"""

from __future__ import annotations

import argparse
import os
import sys 
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pysat.card import CardEnc, EncType
from pysat.formula import CNF
from pysat.solvers import Glucose4

import common
from encoder import AlienTilesEncoder
from solver import solve_optimum_binary

HEADERS = ["Config", "N", "c", "Variables", "Clauses", "Max-Min Clicks",
           "Targets Checked", "CEGAR Iterations", "Variant-2 Calls",
           "Runtime (s)", "Timeout (s)", "Status", "Hardest Target", "Machine"]


def _target_str(T):
    return ";".join(",".join(str(v) for v in r) for r in T) if T else ""


def _cnf_with_lower_bound(enc: AlienTilesEncoder, K: int,
                          blocked: list[list[int]]) -> CNF:
    """Base Φ (target not pinned) + AtLeast(unit_lits, K) + blocking clauses."""
    cnf = CNF(from_clauses=list(enc.clauses.clauses))

    if 0 < K <= len(enc.unit_lits):
        card = CardEnc.atleast(lits=enc.unit_lits, bound=K,
                               top_id=enc.pool.top, encoding=EncType.seqcounter)
        cnf.extend(card.clauses)

    cnf.extend(blocked)
    return cnf


def solve_max_min(N: int, c: int, _unused=None) -> dict:
    """Find the target whose optimum is largest. Returns a result row dict."""
    t0 = time.perf_counter()

    # No target: the chain is built but its last level is left free, because the
    # target is what we are searching over.
    enc = AlienTilesEncoder(N, c)
    enc.build()
    common.report({"Variables": enc.stats["vars"], "Clauses": enc.stats["clauses"]})

    best_total, best_T, best_X = 0, None, None
    checked: dict[tuple, int] = {}
    blocked: list[list[int]] = []
    iterations = 0
    var2_calls = 0

    while True:
        # Only targets strictly harder than the incumbent are of interest.
        K = best_total + 1
        if K > len(enc.unit_lits):
            break                  # no X has more than N²(c-1) clicks: proven
        cnf = _cnf_with_lower_bound(enc, K, blocked)
        improved = False

        with Glucose4(bootstrap_with=cnf.clauses) as solver:
            while solver.solve():
                iterations += 1
                X = enc.decode(set(solver.get_model()))
                T = common.apply_clicks(N, c, X)
                key = tuple(v for row in T for v in row)

                if key in checked:
                    solver.add_clause(enc.differs_from(T))
                    continue

                # X proves T needs at most sum(X) clicks; variant 2 says how few
                # it actually needs, which is the value being maximised.
                var2_calls += 1
                with common.quiet_reports():
                    res = solve_optimum_binary(N, c, T)
                checked[key] = res.optimum
                blocked.append(enc.differs_from(T))

                if res.optimum is not None and res.optimum > best_total:
                    best_total, best_T, best_X = res.optimum, T, res.X
                    improved = True

                # At a TIMEOUT, Max-Min Clicks is the best found so far: a lower bound.
                common.report({"Targets Checked": len(checked),
                               "CEGAR Iterations": iterations,
                               "Variant-2 Calls": var2_calls,
                               "Max-Min Clicks": best_total,
                               "Hardest Target": _target_str(best_T)})
                if improved:
                    break                      # restart with the raised bound

                solver.add_clause(enc.differs_from(T))

        if not improved:
            break

    elapsed = time.perf_counter() - t0
    row = {
        "Config": f"{N}x{N}_c{c}", "N": N, "c": c,
        "Variables": enc.stats["vars"], "Clauses": enc.stats["clauses"],
        "Max-Min Clicks": best_total,
        "Targets Checked": len(checked),
        "CEGAR Iterations": iterations,
        "Variant-2 Calls": var2_calls,
        "Runtime (s)": round(elapsed, 4),
        "Status": "OK" if best_T is not None else "NONE",
        "Hardest Target": _target_str(best_T),
    }
    row["_target"], row["_solution"] = best_T, best_X
    return row


def _parse_configs(spec: str) -> list[tuple[int, int]]:
    """"3x3_c2,4x4_c3" -> [(3,2), (4,3)]"""
    out = []
    for part in spec.split(","):
        grid, _, cpart = part.strip().partition("_c")
        n = int(grid.split("x")[0])
        out.append((n, int(cpart)))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Alien Tiles variant 3: max-min")
    p.add_argument("--N", type=int)
    p.add_argument("--c", type=int)
    p.add_argument("--sweep", type=str,
                   help='Configs to run, e.g. "3x3_c2,3x3_c3,4x4_c2"')
    p.add_argument("--xlsx", type=str, default="sat/results_benchmark_v1.xlsx")
    p.add_argument("--no-excel", action="store_true")
    p.add_argument("--timeout", type=float, default=600.0)
    p.add_argument("--upgrade-timeouts", action="store_true",
                   help="Only configs whose TIMEOUT row used a smaller cutoff")
    args = p.parse_args()

    if args.sweep:
        configs = _parse_configs(args.sweep)
    elif args.N and args.c:
        configs = [(args.N, args.c)]
    else:
        p.print_help()
        sys.exit(1)

    timeout = args.timeout if args.timeout and args.timeout > 0 else None

    if args.upgrade_timeouts:
        names = common.names_to_upgrade(args.xlsx, "maxmin", timeout or 0)
        configs = [(N, c) for N, c in configs if f"{N}x{N}_c{c}" in names]
        print(f"Re-running {len(configs)} TIMEOUT configs at {timeout}s.",
              flush=True)

    for N, c in configs:
        print(f"=== {N}x{N}, c={c} ===", flush=True)
        status, payload, elapsed, progress = common.run_with_timeout(
            solve_max_min, N, c, None, timeout)

        if status != "ok":
            row = {"Config": f"{N}x{N}_c{c}", "N": N, "c": c, **progress,
                   "Runtime (s)": round(elapsed, 4),
                   "Timeout (s)": timeout if timeout else "none",
                   "Machine": common.machine_label(),
                   "Status": "TIMEOUT" if status == "timeout" else f"ERROR: {payload}"}
            print(f"  {row['Status']} after {elapsed:.1f}s — "
                  f"{row.get('Targets Checked', 0)} targets checked, "
                  f"best so far {row.get('Max-Min Clicks', '-')}\n", flush=True)
        else:
            row = payload
            row["Timeout (s)"] = timeout if timeout else "none"
            row["Machine"] = common.machine_label()
            T, X = row.pop("_target"), row.pop("_solution")
            print(f"  max-min clicks: {row['Max-Min Clicks']}")
            print(f"  targets checked: {row['Targets Checked']}, "
                  f"CEGAR iterations: {row['CEGAR Iterations']}")
            if T:
                common.print_board(T, "  Hardest target T")
                common.print_board(X, "  Its optimal X")
                ok = common.verify_solution(N, c, T, X)
                print(f"  Verification: {'PASSED' if ok else 'FAILED'}")
                if not ok:
                    row["Status"] = "VERIFY FAILED"
            print(f"  Runtime: {elapsed:.2f}s\n", flush=True)

        if not args.no_excel:
            common.export_to_excel([row], sheet_name="maxmin",
                                   xlsx_path=args.xlsx, quiet=True,
                                   headers=HEADERS)


if __name__ == "__main__":
    main()
