"""The labels a learned estimate is trained on, and the search it is used in.

`cost_to_go(t, goal)` is the fewest seconds of drive time from every cell to `goal`: Dijkstra run backwards from
the goal over the moves into each cell (a move's time depends on its direction, uphill is slower, so the search
follows the moves backwards rather than simply starting at the goal). It is exact: the true value of the quantity
A*'s estimate tries to guess. Where the goal can't be reached the field is infinite.

`astar_table(t, start, goal, h)` is A* with the estimate read from a table, one value per cell. With the octile
distance in the table it is the ordinary A* (robotnav.algorithms.astar, cell for cell); with the true field it is
A* with a perfect estimate, which searches little more than the route itself. A learned estimate goes in between.

This is the Python reference. The C++ core (cpp/src/fields.cpp) does the same work in the same order, and the tests
hold the two equal with ==.
"""
from __future__ import annotations

import heapq
import time
from itertools import count

import numpy as np

from ..algorithms import SearchResult, _best_first
from ..grid import EIGHT, Cell
from ..terrain import Terrain


def cost_to_go(t: Terrain, goal: Cell) -> np.ndarray:
    """Fewest seconds from every cell to `goal`, shape (rows, cols); inf where the goal can't be reached."""
    R, C = t.rows, t.cols
    g = (int(goal[0]), int(goal[1]))
    if not (0 <= g[0] < R and 0 <= g[1] < C):
        raise ValueError("the goal is off the map")
    blocked = t.blocked
    if blocked[g]:
        raise ValueError("the goal must be an open cell")
    dist = np.full((R, C), np.inf)
    done = np.zeros((R, C), dtype=bool)
    dist[g] = 0.0
    tie = count()
    frontier = [(0.0, next(tie), g)]
    while frontier:
        _, _, u = heapq.heappop(frontier)
        if done[u]:
            continue                                   # stale duplicate entry
        done[u] = True
        ur, uc = u
        du = float(dist[u])
        for dr, dc in EIGHT:                           # p -> u is the move (dr, dc)
            pr, pc = ur - dr, uc - dc
            if not (0 <= pr < R and 0 <= pc < C):
                continue
            p = (pr, pc)
            if done[p] or blocked[p]:
                continue
            if dr and dc and (blocked[ur, pc] or blocked[pr, uc]):
                continue                               # no squeezing diagonally between two obstacles
            step = t.edge_cost(p, u)
            if step is None:
                continue                               # steeper than the grade limit
            d = du + step
            if d < dist[p]:
                dist[p] = d
                heapq.heappush(frontier, (d, next(tie), p))
    return dist


def octile_table(t: Terrain, goal: Cell) -> np.ndarray:
    """A*'s own estimate as a table: the octile distance from every cell to `goal`, in seconds at 1 m/s."""
    rr, cc = np.mgrid[0:t.rows, 0:t.cols]
    dr, dc = np.abs(rr - goal[0]), np.abs(cc - goal[1])
    return ((dr + dc) + (np.sqrt(2.0) - 2.0) * np.minimum(dr, dc)) * t.cell_size


def climb_table(t: Terrain, goal: Cell) -> np.ndarray:
    """A better hand-made estimate: the octile distance plus the time the climb to the goal must take at least.

    Each move takes 3-D distance x surface factor x (1 + uphill x grade+), which is at least run + uphill x rise+
    (the 3-D distance is at least the run, every surface factor at least 1). Added over a route: at least the octile
    distance plus uphill x the net climb to the goal. So it never overestimates, and it is consistent, like the
    octile distance; it is the strongest classical estimate here, and what a learned one has to beat."""
    return octile_table(t, goal) + t.uphill_penalty * np.maximum(t.height[goal] - t.height, 0.0)


class _Table:
    """The terrain with A*'s estimate read from a table."""

    def __init__(self, t: Terrain, start: Cell, goal: Cell, h: np.ndarray):
        self.t, self.start, self.goal, self.h = t, start, goal, h

    def neighbors(self, c):
        return self.t.neighbors(c)

    def heuristic(self, a, b):
        return float(self.h[a])

    def path_cost(self, path):
        return self.t.path_cost(path)


def astar_table(t: Terrain, start: Cell, goal: Cell, h: np.ndarray, w: float = 1.0,
                name: str = "A* (table)") -> SearchResult:
    """A* from start to goal with h[cell] (times w) as the estimate of the time left."""
    h = np.asarray(h, dtype=float)
    if h.shape != (t.rows, t.cols):
        raise ValueError("the table needs one estimate per cell")
    if np.isnan(h).any() or (h < 0).any():
        raise ValueError("every estimate must be a number, zero or more")
    grid = _Table(t, tuple(start), tuple(goal), h)
    if w == 1.0:
        return _best_first(grid, name, lambda g, est: g + est)
    return _best_first(grid, name, lambda g, est: g + w * est)


def astar_reopen(t: Terrain, start: Cell, goal: Cell, h: np.ndarray, name: str = "A* (re-expanding)") -> SearchResult:
    """A* with h[cell] as the estimate that searches a cell again when a quicker way to it turns up.

    An estimate that isn't consistent (a learned one) can close a cell before its quickest way in is known; this
    search reopens it, which is what guarantees a route at most w times the quickest when the estimate is never
    more than w times the truth. With a consistent estimate it does exactly what astar_table does."""
    t0 = time.perf_counter()
    h = np.asarray(h, dtype=float)
    if h.shape != (t.rows, t.cols):
        raise ValueError("the table needs one estimate per cell")
    start, goal = tuple(start), tuple(goal)
    tie = count()
    g = {start: 0.0}
    came_from = {start: None}
    frontier = [(float(h[start]), next(tie), start, 0.0)]
    explored, peak, reached = [], 1, False
    while frontier:
        _, _, cur, gc = heapq.heappop(frontier)
        if gc != g[cur]:
            continue                                   # stale: a quicker way here was found since
        explored.append(cur)
        if cur == goal:
            reached = True
            break
        for nxt, step in t.neighbors(cur):
            ng = gc + step
            if nxt not in g or ng < g[nxt]:
                g[nxt] = ng
                came_from[nxt] = cur
                heapq.heappush(frontier, (ng + float(h[nxt]), next(tie), nxt, ng))
        peak = max(peak, len(frontier))
    path = []
    if reached:
        node = goal
        while node is not None:
            path.append(node)
            node = came_from[node]
        path.reverse()
    return SearchResult(name, path, t.path_cost(path) if path else float("inf"), len(explored), peak,
                        (time.perf_counter() - t0) * 1e3, explored)

