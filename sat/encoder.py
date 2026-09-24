#!/usr/bin/env python3

from __future__ import annotations

from pysat.formula import CNF, IDPool


class AlienTilesEncoder:

    def __init__(self, N: int, c: int,
                 target: list[list[int]] | None = None) -> None:
        self.N = N
        self.c = c
        self.target = target          
        self.L = 2 * N - 1            
        self.clauses = CNF()
        self.pool = IDPool()
        self.unit_lits: list[int] = []
        self.stats = {"vars": 0, "clauses": 0, "time_encode": 0.0, "time_solve": 0.0}


    def _click_var(self, i: int, j: int, v: int) -> int:
        return self.pool.id(("d", i, j, v))

    def _var_sum(self, r: int, k: int, l: int, a: int) -> int:
        return self.pool.id(("s", r, k, l, a))

    def _unit_var(self, i: int, j: int, h: int) -> int:
        return self.pool.id(("u", i, j, h))



    def _add_exactly_one(self, lits: list[int]) -> None:
        self.clauses.append(list(lits))                     
        for a in range(len(lits)):
            for b in range(a + 1, len(lits)):
                self.clauses.append([-lits[a], -lits[b]])    


    def _encode_click_var(self) -> None:
        """One-hot encode x_ij as d_ij0 … d_ij,c-1."""
        for i in range(self.N):
            for j in range(self.N):
                self._add_exactly_one(
                    [self._click_var(i, j, v) for v in range(self.c)]
                )


    def _cell_affecting(self, r: int, k: int) -> list[tuple[int, int]]:
        cells = [(r, j) for j in range(self.N)]
        cells += [(i, k) for i in range(self.N) if i != r]
        return cells

    def _encode_modulo_sums(self) -> None:
        for r in range(self.N):
            for k in range(self.N):
                P_rk = self._cell_affecting(r, k)
                L = len(P_rk)

                for l in range(L + 1):
                    self._add_exactly_one(
                        [self._var_sum(r, k, l, a) for a in range(self.c)]
                    )

                self.clauses.append([self._var_sum(r, k, 0, 0)])   

                for l, (u, v) in enumerate(P_rk, start=1):
                    for a in range(self.c):
                        s_prev = self._var_sum(r, k, l - 1, a)
                        for b in range(self.c):
                            self.clauses.append([                 
                                -s_prev,
                                -self._click_var(u, v, b),
                                self._var_sum(r, k, l, (a + b) % self.c),
                            ])



    def _encode_unit_literals(self) -> None:
        for i in range(self.N):
            for j in range(self.N):
                for h in range(1, self.c):
                    u = self._unit_var(i, j, h)
                    d_lits = [self._click_var(i, j, w) for w in range(h, self.c)]

                    self.clauses.append([-u] + d_lits)            # eq:unit-channel
                    for d_w in d_lits:
                        self.clauses.append([-d_w, u])

                    self.unit_lits.append(u)

    # -- Assembly ---------------------------------------------------------

    def assert_target(self, target: list[list[int]]) -> None:
        for r in range(self.N):
            for k in range(self.N):
                self.clauses.append([self._var_sum(r, k, self.L, target[r][k])])

    def differs_from(self, target: list[list[int]]) -> list[int]:
        return [-self._var_sum(r, k, self.L, target[r][k])
                for r in range(self.N) for k in range(self.N)]

    def build(self) -> CNF:
        self._encode_click_var()
        self._encode_modulo_sums()
        self._encode_unit_literals()
        if self.target is not None:
            self.assert_target(self.target)

        self.stats["vars"] = self.pool.top
        self.stats["clauses"] = len(self.clauses.clauses)
        return self.clauses

    def decode(self, model_set: set[int]) -> list[list[int]]:
        X = []
        for i in range(self.N):
            row = []
            for j in range(self.N):
                for v in range(self.c):
                    if self._click_var(i, j, v) in model_set:
                        row.append(v)
                        break
                else:
                    row.append(-1)
            X.append(row)
        return X
