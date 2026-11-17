"""What the network sees: 16 numbers about a cell and the goal.

The climb-aware estimate (the straight-line distance plus the least time the climb to the goal takes, labels.
climb_table) never overestimates, but it ignores what lies between the cell and the goal. The features describe
exactly that, mostly along the straight line from the cell to the goal, walked one cell at a time (the cells at
round(start + (goal - start) * k / n), n the larger of the row and column distance):

     0  octile distance (m)                       8  longest run of blocked cells on the line / n
     1  height of the goal minus the cell's (m)   9  that run, in cells
     2  drive time along the line / climb         10 metres climbed along the line / octile
     3  drive time along the line / octile        11 metres descended along the line / octile
     4  steps of the line that can't be driven / n   12 moves out of the cell that can't be driven / 8
     5  steps too steep to drive / n              13 moves into the goal that can't be driven / 8
     6  blocked cells on the line / n             14 blocked cells around the cell (5 x 5, edge counts) / 25
     7  water cells on the line / n               15 blocked cells around the goal (5 x 5) / 25

"Drive time along the line" adds the real move times of the line's drivable steps (it is what driving straight
would take, if it could be done). The network learns how much longer than the climb-aware estimate the true
drive is, from these.

The arithmetic is +, -, x, /, sqrt, min and max on 64-bit floats, in the same order as the C++ core
(cpp/src/learned.cpp): the tests hold the two equal with ==. The goal itself and blocked cells get all zeros.
"""
from __future__ import annotations

import numpy as np

from ..terrain import SQRT2, WATER, Terrain

FEATURES = 16
NAMES = ("octile", "goal height minus cell height", "line time / climb", "line time / octile", "undrivable steps / n",
         "too-steep steps / n", "blocked on line / n", "water on line / n", "longest blocked run / n",
         "longest blocked run", "climbed along line / octile", "descended along line / octile",
         "blocked moves out / 8", "blocked moves into goal / 8", "blocked around cell / 25", "blocked around goal / 25")


def _moves_into(t: Terrain, goal) -> int:
    from ..grid import EIGHT
    ur, uc = goal
    count = 0
    for dr, dc in EIGHT:
        pr, pc = ur - dr, uc - dc
        if not (0 <= pr < t.rows and 0 <= pc < t.cols) or t.blocked[pr, pc]:
            continue
        if dr and dc and (t.blocked[ur, pc] or t.blocked[pr, uc]):
            continue
        if t.edge_cost((pr, pc), (ur, uc)) is not None:
            count += 1
    return count


def features(t: Terrain, goal, cells=None):
    """(x, climb): x is (n, 16), one row per cell (flat indices row * cols + col; default every cell), and climb
    the climb-aware estimate for each. All zeros for the goal and for blocked cells."""
    R, C = t.rows, t.cols
    gr, gc = int(goal[0]), int(goal[1])
    cells = np.arange(R * C) if cells is None else np.asarray(cells, dtype=np.int64)
    Hf = np.asarray(t.height, dtype=float).ravel()
    Bf = np.asarray(t.blocked, dtype=bool).ravel()
    Wf = (np.asarray(t.obstacle).ravel() == WATER)
    Sf = np.asarray(t.surface_costs, dtype=float)[np.asarray(t.surface).ravel()]
    g = gr * C + gc

    r0, c0 = cells // C, cells % C
    live = (cells != g) & ~Bf[cells]
    dr, dc = gr - r0, gc - c0
    adr, adc = np.abs(dr), np.abs(dc)
    n = np.where(live, np.maximum(adr, adc), 0)
    octile = ((adr + adc) + (SQRT2 - 2.0) * np.minimum(adr, adc)) * t.cell_size
    dh = Hf[g] - Hf[cells]
    climb = octile + t.uphill_penalty * np.maximum(dh, 0.0)

    line_time = np.zeros(len(cells))
    rise_up = np.zeros(len(cells))
    rise_down = np.zeros(len(cells))
    bad = np.zeros(len(cells), dtype=np.int64)
    steep = np.zeros_like(bad)
    blocked = np.zeros_like(bad)
    water = np.zeros_like(bad)
    run = np.zeros_like(bad)
    longest = np.zeros_like(bad)
    pr, pc = r0.copy(), c0.copy()
    for k in range(1, int(n.max(initial=0)) + 1):
        act = k <= n
        nn = np.where(act, n, 1)
        r = np.where(act, r0 + (2 * dr * k + nn) // (2 * nn), pr)
        c = np.where(act, c0 + (2 * dc * k + nn) // (2 * nn), pc)
        a, b = pr * C + pc, r * C + c
        both = act & ~Bf[a] & ~Bf[b]
        rise = Hf[b] - Hf[a]
        run_len = np.where((r != pr) & (c != pc), SQRT2, 1.0) * t.cell_size
        grade = rise / run_len
        too_steep = both & (np.abs(grade) > t.max_grade)
        legal = both & ~too_steep
        step = np.sqrt(run_len * run_len + rise * rise) * Sf[b] * (1.0 + t.uphill_penalty * np.maximum(grade, 0.0))
        line_time = line_time + np.where(legal, step, 0.0)
        rise_up = rise_up + np.where(both, np.maximum(rise, 0.0), 0.0)
        rise_down = rise_down + np.where(both, np.maximum(-rise, 0.0), 0.0)
        steep += too_steep
        bad += act & ~legal
        inner = act & (k < n)
        hit = inner & Bf[b]
        blocked += hit
        water += hit & Wf[b]
        run = np.where(inner, np.where(Bf[b], run + 1, 0), run)
        longest = np.maximum(longest, run)
        pr, pc = r, c

    out = np.zeros(len(cells), dtype=np.int64)                 # legal moves out of each cell
    for mr, mc in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
        nr, nc = r0 + mr, c0 + mc
        inside = (nr >= 0) & (nr < R) & (nc >= 0) & (nc < C)
        nr_, nc_ = np.clip(nr, 0, R - 1), np.clip(nc, 0, C - 1)
        b = nr_ * C + nc_
        ok = inside & ~Bf[b]
        if mr and mc:
            ok &= ~Bf[nr_ * C + c0] & ~Bf[r0 * C + nc_]
        run_len = (SQRT2 if mr and mc else 1.0) * t.cell_size
        ok &= np.abs((Hf[b] - Hf[cells]) / run_len) <= t.max_grade
        out += ok
    pad = np.pad(np.asarray(t.blocked, dtype=np.int64), 2, constant_values=1)
    win = sum(pad[i:i + R, j:j + C] for i in range(5) for j in range(5)).ravel()

    safe_n = np.where(n > 0, n, 1).astype(float)
    safe_oct = np.where(live, octile, 1.0)
    safe_climb = np.where(live, climb, 1.0)
    x = np.stack([octile, dh, line_time / safe_climb, line_time / safe_oct, bad / safe_n, steep / safe_n,
                  blocked / safe_n, water / safe_n, longest / safe_n, longest.astype(float), rise_up / safe_oct,
                  rise_down / safe_oct, (8 - out) / 8.0, np.full(len(cells), (8 - _moves_into(t, (gr, gc))) / 8.0),
                  win[cells] / 25.0, np.full(len(cells), win[g] / 25.0)], axis=1)
    x[~live] = 0.0
    return x, np.where(live, climb, 0.0)
