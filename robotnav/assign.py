"""Who goes where: the assignment problem.

Several vehicles, several places to be; `cost[i][j]` is what it costs to send column j's vehicle to row i's place.
`hungarian` returns, for every row, the column that makes the total least (each column used at most once). It is the
Hungarian method in its form with potentials, O(rows^2 x columns); the console in the viewer runs the same steps in
JavaScript to decide which drone flies to which survey point. `nearest_first` is the obvious alternative it is
compared with: each row in turn takes the cheapest column still free.
"""
from __future__ import annotations

import math


def hungarian(cost) -> list[int]:
    """The column for each row that minimises the total cost. Needs rows <= columns."""
    n, m = len(cost), len(cost[0])
    if n > m:
        raise ValueError("more rows than columns: give every row a column to take")
    u, v, p, way = [0.0] * (n + 1), [0.0] * (m + 1), [0] * (m + 1), [0] * (m + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv, used = [math.inf] * (m + 1), [False] * (m + 1)
        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], math.inf, 0
            for j in range(1, m + 1):
                if not used[j]:
                    cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j], way[j] = cur, j0
                    if minv[j] < delta:
                        delta, j1 = minv[j], j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    out = [-1] * n
    for j in range(1, m + 1):
        if p[j]:
            out[p[j] - 1] = j - 1
    return out


def nearest_first(cost) -> list[int]:
    """Each row in turn takes the cheapest column still free."""
    taken, out = set(), []
    for row in cost:
        j = min((k for k in range(len(row)) if k not in taken), key=row.__getitem__)
        taken.add(j)
        out.append(j)
    return out


def total(cost, pick) -> float:
    return sum(cost[i][j] for i, j in enumerate(pick))
