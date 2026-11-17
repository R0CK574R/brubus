"""A* with landmarks (ALT: A*, Landmarks, Triangle inequality; Goldberg and Harrelson, 2005).

Pick a few cells spread over the map (the landmarks) and work out, once per map, the quickest time from each
landmark to every cell and from every cell to each landmark (two Dijkstra searches per landmark). Then for any
cell c and goal g, the triangle inequality gives lower bounds on the time from c to g:

    time(c -> g) >= time(c -> L) - time(g -> L)        (c could go to L through g)
    time(c -> g) >= time(L -> g) - time(L -> c)        (L could go to g through c)

The largest of these over the landmarks (and the climb-aware estimate: the straight-line distance plus the least
time the climb takes) never overestimates and is consistent, so A* guided by it still finds the quickest route; it
is simply a much better guess. On the data set's test maps it is about 95% of the true time, against 67% for the
straight-line distance, and A* searches several times fewer cells.

The work per map (2 x 8 Dijkstra searches) is done once and serves every trip on that map. Landmarks are chosen
by farthest-point: first the drivable cell farthest from the map's centre, then repeatedly the cell farthest (in
drive time) from all the landmarks so far, all inside the largest drivable region.

This is the Python reference. The C++ core (cpp/src/landmarks.cpp) does the same work in the same order; the tests
hold the two equal with ==.
"""
from __future__ import annotations

import heapq
from collections import deque
from dataclasses import dataclass
from itertools import count

import numpy as np

from .algorithms import SearchResult
from .terrain import Terrain

K = 8


def cost_from(t: Terrain, source) -> np.ndarray:
    """Fewest seconds from `source` to every cell (Dijkstra forwards); inf where it can't be reached."""
    R, C = t.rows, t.cols
    s = (int(source[0]), int(source[1]))
    if t.blocked[s]:
        raise ValueError("the source must be an open cell")
    dist = np.full((R, C), np.inf)
    done = np.zeros((R, C), dtype=bool)
    dist[s] = 0.0
    tie = count()
    frontier = [(0.0, next(tie), s)]
    while frontier:
        _, _, u = heapq.heappop(frontier)
        if done[u]:
            continue
        done[u] = True
        du = float(dist[u])
        for v, step in t.neighbors(u):
            if done[v]:
                continue
            d = du + step
            if d < dist[v]:
                dist[v] = d
                heapq.heappush(frontier, (d, next(tie), v))
    return dist


def largest_region(t: Terrain) -> np.ndarray:
    """The largest set of open cells the rover can drive between (moves are legal both ways); on a tie, the one
    holding the lowest cell (row by row)."""
    R, C = t.rows, t.cols
    label = np.full((R, C), -1)
    best, best_size, n = -1, 0, 0
    for r in range(R):
        for c in range(C):
            if t.blocked[r, c] or label[r, c] >= 0:
                continue
            label[r, c] = n
            size, queue = 1, deque([(r, c)])
            while queue:
                u = queue.popleft()
                for v, _ in t.neighbors(u):
                    if label[v] < 0:
                        label[v] = n
                        size += 1
                        queue.append(v)
            if size > best_size:
                best, best_size = n, size
            n += 1
    return label == best


@dataclass
class Landmarks:
    cells: list                 # flat indices (row * cols + col)
    from_: np.ndarray           # (k, rows, cols): seconds from each landmark to every cell
    to: np.ndarray              # (k, rows, cols): seconds from every cell to each landmark


def pick(t: Terrain, k: int = K) -> Landmarks:
    """k landmarks by farthest-point in drive time, and their fields."""
    from .learn.labels import cost_to_go
    C = t.cols
    region = largest_region(t)
    cells = np.flatnonzero(region.ravel())
    if len(cells) == 0:
        raise ValueError("the map has no open cells")
    score = np.abs(2 * (cells // C) - t.rows) + np.abs(2 * (cells % C) - C)     # twice the distance from the centre
    chosen = [int(cells[np.argmax(score)])]
    fields = []
    nearest = np.full(t.rows * C, np.inf)
    while True:
        d = cost_from(t, divmod(chosen[-1], C))
        fields.append(d)
        if len(chosen) == k:
            break
        nearest = np.minimum(nearest, d.ravel())
        far = np.where(region.ravel(), nearest, -1.0)
        nxt = int(np.argmax(far))
        if far[nxt] <= 0.0:                          # the region has fewer cells than landmarks asked for
            break
        chosen.append(nxt)
    to = [cost_to_go(t, divmod(c, C)) for c in chosen]
    return Landmarks(chosen, np.stack(fields), np.stack(to))


def table(t: Terrain, lm: Landmarks, goal) -> np.ndarray:
    """The landmark bound for every cell, never below the climb-aware estimate."""
    from .learn.labels import climb_table
    g = (int(goal[0]), int(goal[1]))
    best = climb_table(t, g)
    for k in range(len(lm.cells)):
        tc, tg = lm.to[k], lm.to[k][g]
        a = np.where(np.isfinite(tc) & np.isfinite(tg), tc - tg, -np.inf)
        fc, fg = lm.from_[k], lm.from_[k][g]
        b = np.where(np.isfinite(fc) & np.isfinite(fg), fg - fc, -np.inf)
        best = np.maximum(np.maximum(best, a), b)
    return best


def astar(t: Terrain, lm: Landmarks, start, goal) -> SearchResult:
    """A* from start to goal guided by the landmark bound: the quickest route, searching fewer cells."""
    from .learn.labels import astar_table
    return astar_table(t, tuple(start), tuple(goal), table(t, lm, goal), name="A* + landmarks")
