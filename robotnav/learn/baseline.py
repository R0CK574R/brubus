"""The baseline a learned estimate has to beat: how much A* searches with today's estimate, and with a perfect one.

A* is guided by an estimate of the time left to the goal. Today's is the octile distance (the straight-line
length at 1 m/s), which never overestimates, so A* finds the quickest route, but it ignores hills, slow ground and
obstacles, so it underestimates, and A* searches far more cells than the route it ends up with. A perfect estimate
(the true time left, cost_to_go) would lead A* almost straight down the route. A learned estimate sits somewhere in
between; this measures the two ends, on maps from the data set's test split, so Stage 2 has numbers to beat.

For each test map and each of its goals, `starts` start cells are picked at random among the cells that can reach
the goal in at least `min_time` seconds (so the trips are worth planning). On each trip:
    route     cells on the quickest route
    A*        cells A* searched with the octile estimate (and Dijkstra, with none; Weighted A*, w = 2)
    climb     cells A* searched with the climb-aware estimate (octile + the least time the climb takes), the best
              hand-made estimate here, also never too high
    perfect   cells A* searched with the true field as its estimate
    estimate  the octile estimate at the start, as a share of the true time (and the climb-aware one's)
"""
from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from ..terrain import THEMES
from .dataset import DEFAULT_DIR, load, load_manifest, terrain_of
from .labels import astar_table, climb_table, cost_to_go, octile_table

STARTS = 4
MIN_TIME = 20.0
SEED = 7


def _trips(field: np.ndarray, starts: int, min_time: float, rng) -> list:
    ok = np.argwhere(np.isfinite(field) & (field >= min_time))
    if len(ok) == 0:
        return []
    pick = rng.choice(len(ok), size=min(starts, len(ok)), replace=False)
    return [tuple(int(v) for v in ok[i]) for i in sorted(pick)]


def measure(directory: Path = DEFAULT_DIR, split: str = "test", starts: int = STARTS, min_time: float = MIN_TIME,
            seed: int = SEED, engine: str = "auto", log=print) -> list:
    """One record per trip (theme, map, goal, start, and the numbers above)."""
    from .. import cpp
    use_cpp = cpp.available() if engine == "auto" else engine == "cpp"
    m = load_manifest(directory)
    rng = np.random.default_rng(seed)
    rows, t0 = [], time.perf_counter()
    for e in [e for e in m["maps"] if e["split"] == split]:
        arrays = load(directory, e)
        t = terrain_of(arrays)
        cmap = cpp.to_cpp(t) if use_cpp else None
        for k, g in enumerate(arrays["goals"]):
            goal = tuple(int(v) for v in g)
            field = cpp.cost_to_go(t, goal, cmap=cmap) if use_cpp else cost_to_go(t, goal)   # exact, 64-bit
            octile, climb = octile_table(t, goal), climb_table(t, goal)
            for s in _trips(field, starts, min_time, rng):
                if use_cpp:
                    core, si, gi = cpp.core(), s[0] * t.cols + s[1], goal[0] * t.cols + goal[1]
                    a, d, w2 = core.astar(cmap, si, gi), core.dijkstra(cmap, si, gi), core.weighted_astar(cmap, si, gi, 2.0)
                    best = core.astar_table(cmap, si, gi, field, 1.0, "perfect")
                    cl = core.astar_table(cmap, si, gi, climb, 1.0, "climb")
                    route = len(a.path)
                else:
                    from ..algorithms import astar, dijkstra, weighted_astar
                    leg = replace(t, start=s, goal=goal)
                    a, d, w2 = astar(leg), dijkstra(leg), weighted_astar(leg)
                    best = astar_table(t, s, goal, field, name="perfect")
                    cl = astar_table(t, s, goal, climb, name="climb")
                    route = len(a.path)
                rows.append(dict(theme=e["theme"], map=e["id"], goal=goal, start=s, time=float(field[s]),
                                 route=route, astar=int(a.nodes_expanded), dijkstra=int(d.nodes_expanded),
                                 wastar=int(w2.nodes_expanded), wastar_extra=float(w2.cost / a.cost - 1.0),
                                 climb=int(cl.nodes_expanded), climb_cost=float(cl.cost),
                                 perfect=int(best.nodes_expanded), perfect_cost=float(best.cost), astar_cost=float(a.cost),
                                 estimate=float(octile[s] / field[s]), climb_estimate=float(climb[s] / field[s])))
    if log:
        log(f"{len(rows)} trips on the {split} maps measured in {time.perf_counter() - t0:.0f} s")
    return rows


def summarize(rows: list) -> list:
    """Per theme and overall: trips, route cells, cells searched (mean) and the estimate's share of the truth."""
    out = []
    for theme in [*THEMES, "all"]:
        r = [x for x in rows if theme == "all" or x["theme"] == theme]
        if not r:
            continue
        mean = lambda key: float(np.mean([x[key] for x in r]))
        out.append(dict(theme=theme, trips=len(r), time=mean("time"), route=mean("route"), dijkstra=mean("dijkstra"),
                        astar=mean("astar"), wastar=mean("wastar"), wastar_extra=mean("wastar_extra"),
                        climb=mean("climb"), perfect=mean("perfect"), estimate=mean("estimate"),
                        climb_estimate=mean("climb_estimate"),
                        astar_over_route=float(np.mean([x["astar"] / x["route"] for x in r])),
                        perfect_over_route=float(np.mean([x["perfect"] / x["route"] for x in r]))))
    return out


def table(summary: list) -> str:
    head = (f"  {'theme':<10}{'trips':>6}{'route':>7}{'Dijkstra':>10}{'A*':>8}{'climb':>8}{'W-A* (w=2)':>12}{'+time':>7}"
            f"{'perfect':>9}{'A*/route':>10}{'estimate':>10}{'climb est.':>12}")
    lines = [head]
    for s in summary:
        lines.append(f"  {s['theme']:<10}{s['trips']:>6}{s['route']:>7.0f}{s['dijkstra']:>10.0f}{s['astar']:>8.0f}"
                     f"{s['climb']:>8.0f}{s['wastar']:>12.0f}{100 * s['wastar_extra']:>6.1f}%{s['perfect']:>9.0f}"
                     f"{s['astar_over_route']:>9.1f}x{100 * s['estimate']:>9.0f}%{100 * s['climb_estimate']:>11.0f}%")
    return "\n".join(lines)
