import numpy
from pycryptosat import Solver
import re
import numpy as np
from pathlib import Path
import random

def apply_move(board, i, j):
    m = len(board)
    n = len(board[0])

    for di, dj in [(0,0),(1,0),(-1,0),(0,1),(0,-1)]:
        ni, nj = i + di, j + dj
        if 0 <= ni < m and 0 <= nj < n:
            board[ni][nj] ^= 1


def generate_puzzle(m, n, num_moves=None):
    board = [[0]*n for _ in range(m)]

    if num_moves is None:
        num_moves = random.randint(1, m*n)

    solution = [[0]*n for _ in range(m)]

    for _ in range(num_moves):
        i = random.randrange(m)
        j = random.randrange(n)

        solution[i][j] ^= 1
        apply_move(board, i, j)

    return board, solution



def build_XOR_model(initial, target):
    m = len(initial)
    n = len(initial[0])
    solver = Solver()

    def var(i, j):
        return i * n + j + 1

    for i in range(m):
        for j in range(n):
            lists = [var(i, j)]

            for di, dj in [(1,0),(-1,0),(0,1),(0,-1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < m and 0 <= nj < n:
                    lists.append(var(ni, nj))

            rhs = bool(initial[i][j] ^ target[i][j])

            solver.add_xor_clause(lists, rhs)

    click_vars = [var(i, j) for i in range(m) for j in range(n)]
    return solver, click_vars

def add_atmost_k(solver, vars, K):
    n = len(vars)

    if K >= n:
        return

    # helper variables
    s = {}

    next_var = max(vars) + 1

    def new_var():
        nonlocal next_var
        v = next_var
        next_var += 1
        return v

    for i in range(n):
        for j in range(K+1):
            s[(i,j)] = new_var()

    # constraints
    for i in range(n):

        # xi → s(i,0)
        solver.add_clause([-vars[i], s[(i,0)]])

        if i > 0:
            solver.add_clause([-s[(i-1,0)], s[(i,0)]])

        for j in range(1, K+1):

            if i > 0:
                solver.add_clause([-s[(i-1,j)], s[(i,j)]])

                solver.add_clause([-vars[i], -s[(i-1,j-1)], s[(i,j)]])

    # forbid K+1 trues
    solver.add_clause([-s[(n-1,K)]])

def print_board(board, title="Board"):
    print(f"{title}:")
    for row in board:
        print(" ".join(str(cell) for cell in row))
    print()

if __name__ == "__main__":

    m = 8
    n = 8
    initial, _ = generate_puzzle(m, n)
    target = [[0]*n for _ in range(m)]

    print_board(initial, "Initial Board")
    print_board(target, "Target Board")

    solver, click_vars = build_XOR_model(initial, target)

    sat, model = solver.solve()

    if not sat:
        print("No solution found")
        exit()

    UB = sum(model[v] for v in click_vars)

    best = model

    while True:

        solver, click_vars = build_XOR_model(initial, target)

        add_atmost_k(solver, click_vars, UB - 1)

        sat, model = solver.solve()

        if not sat:
            break

        best = model
        UB = sum(model[v] for v in click_vars)

        print("Found better solution with", UB, "clicks")

    print("Optimal clicks:", UB)
