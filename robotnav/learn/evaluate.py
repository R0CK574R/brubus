"""The stage-2 benchmark: A* with the learned estimate against the classical ones, on maps no model has seen.

For each map of a split and each of its goals, `starts` start cells are picked at random among the cells that can
reach the goal in at least `min_time` seconds (the same trips the baseline uses, for the same seed). Each trip is
planned by Dijkstra; A* with the straight-line estimate and with the climb-aware one; Weighted A* (the climb-aware
estimate inflated 1.25 and 1.5 times: fewer cells, routes allowed to be longer); A* with the network's estimate
(re-expanding, since a learned estimate isn't consistent), plain and capped at 1.1 x the climb-aware estimate (which
proves routes at most 10% longer than the quickest); A* with landmarks (8 per map, worked out once per map); and A*
with the true time (the perfect estimate). For each: cells searched, how much longer the route is than the quickest,
and the search's own time in the C++ core (the network's includes working out its estimate for every cell the search
reaches; the landmarks' one-off set-up per map is reported apart).

`accuracy` measures the estimates themselves against the true time, without any search.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np

from ..terrain import THEMES
from .baseline import MIN_TIME, SEED, STARTS, _trips
from .dataset import DEFAULT_DIR, load, load_manifest, terrain_of
from .labels import climb_table

CAP = 1.1           # one capped run: the learned estimate never above 1.1 x the climb-aware one (proven bound)


def evaluate(model_text: str, directory: Path = DEFAULT_DIR, split: str = "test", starts: int = STARTS,
             min_time: float = MIN_TIME, seed: int = SEED, themes=THEMES, max_maps: int | None = None,
             landmarks: int = 8, log=print) -> tuple:
    """(rows, setup): one record per trip and planner, and the landmarks' set-up time per map (ms)."""
    from .. import cpp
    core = cpp.core()
    model = cpp.heuristic(model_text)
    m = load_manifest(directory)
    rng = np.random.default_rng(seed)
    entries = [e for e in m["maps"] if e["split"] == split]
    rows, setup, t0 = [], [], time.perf_counter()
    for i, e in enumerate(entries[:max_maps] if max_maps else entries):
        arrays = load(directory, e)
        t = terrain_of(arrays)
        cmap = cpp.to_cpp(t)
        lm = core.pick_landmarks(cmap, landmarks) if e["theme"] in themes else None
        if lm is not None:
            setup.append(lm.setup_ms)
        for g in arrays["goals"]:
            goal = (int(g[0]), int(g[1]))
            field = core.cost_to_go(cmap, goal[0] * t.cols + goal[1])
            trips = _trips(field, starts, min_time, rng)          # drawn for every map, so the trips don't change
            if lm is None:                                        # with the themes asked for
                continue
            climb = climb_table(t, goal)
            gi = goal[0] * t.cols + goal[1]
            for s in trips:
                si = s[0] * t.cols + s[1]
                best = float(field[s])
                runs = [("Dijkstra", core.dijkstra(cmap, si, gi)),
                        ("A* straight-line", core.astar(cmap, si, gi)),
                        ("A* climb-aware", core.astar_table(cmap, si, gi, climb, 1.0, "climb")),
                        ("Weighted A* 1.25", core.astar_table(cmap, si, gi, climb, 1.25, "w")),
                        ("Weighted A* 1.5", core.astar_table(cmap, si, gi, climb, 1.5, "w")),
                        ("A* + network", core.astar_learned(cmap, si, gi, model, math.inf, "n")),
                        (f"A* + network, capped", core.astar_learned(cmap, si, gi, model, CAP, "n")),
                        (f"A* + {landmarks} landmarks", core.astar_landmarks(cmap, lm, si, gi)),
                        ("A* perfect", core.astar_table(cmap, si, gi, field, 1.0, "perfect"))]
                for name, r in runs:
                    extra = float(r.cost / best - 1.0)
                    rows.append(dict(theme=e["theme"], map=e["id"], goal=goal, start=s, planner=name,
                                     cells=int(r.nodes_expanded), route=len(r.path), cost=float(r.cost),
                                     extra=extra if abs(extra) > 1e-12 else 0.0, ms=float(r.runtime_ms)))
        if log and (i + 1) % 15 == 0:
            log(f"  {i + 1}/{len(entries)} maps ({time.perf_counter() - t0:.0f} s)")
    return rows, setup


def accuracy(model_text: str, directory: Path = DEFAULT_DIR, split: str = "test", themes=THEMES,
             landmarks: int = 8) -> list:
    """How close each estimate is to the true time, over every cell that can reach each goal (at least 10 s away):
    mean estimate / truth, mean |error|, and the share of cells where it is too high."""
    from .. import cpp
    from .labels import octile_table
    core = cpp.core()
    model = cpp.heuristic(model_text)
    m = load_manifest(directory)
    acc = {}
    for e in [e for e in m["maps"] if e["split"] == split and e["theme"] in themes]:
        arrays = load(directory, e)
        t = terrain_of(arrays)
        cmap = cpp.to_cpp(t)
        lm = core.pick_landmarks(cmap, landmarks)
        for k, g in enumerate(arrays["goals"]):
            goal = (int(g[0]), int(g[1]))
            gi = goal[0] * t.cols + goal[1]
            field = core.cost_to_go(cmap, gi)
            ok = np.isfinite(field) & (field >= 10.0)
            for name, est in (("straight-line", octile_table(t, goal)), ("climb-aware", climb_table(t, goal)),
                              ("network", core.learned_table(model, cmap, gi, math.inf)),
                              (f"{landmarks} landmarks", core.landmark_table(cmap, lm, gi))):
                r = est[ok] / field[ok]
                for key in (e["theme"], "all"):
                    acc.setdefault((name, key), []).append(r)
    out = []
    for (name, theme), parts in acc.items():
        r = np.concatenate(parts)
        out.append(dict(estimate=name, theme=theme, cells=int(len(r)), ratio=float(r.mean()),
                        error=float(np.abs(r - 1).mean()), too_high=float((r > 1 + 1e-12).mean())))
    return out


def summarize(rows: list) -> list:
    """Per planner (and theme): trips, mean cells searched, routes' extra time (mean and worst), share of trips
    that found the quickest route, mean search time."""
    out = []
    planners = list(dict.fromkeys(r["planner"] for r in rows))
    for theme in [*THEMES, "all"]:
        for p in planners:
            r = [x for x in rows if x["planner"] == p and (theme == "all" or x["theme"] == theme)]
            if not r:
                continue
            extra = np.array([x["extra"] for x in r])
            out.append(dict(theme=theme, planner=p, trips=len(r), cells=float(np.mean([x["cells"] for x in r])),
                            route=float(np.mean([x["route"] for x in r])), extra_mean=float(extra.mean()),
                            extra_max=float(extra.max()), quickest=float(np.mean(extra <= 1e-9)),
                            ms=float(np.mean([x["ms"] for x in r]))))
    return out


def table(summary: list, theme: str = "all") -> str:
    lines = [f"  {'planner':<24}{'trips':>6}{'cells':>8}{'route':>7}{'extra (mean)':>14}{'extra (worst)':>15}"
             f"{'quickest':>10}{'ms':>8}"]
    for s in summary:
        if s["theme"] != theme:
            continue
        lines.append(f"  {s['planner']:<24}{s['trips']:>6}{s['cells']:>8.0f}{s['route']:>7.0f}"
                     f"{100 * s['extra_mean']:>13.2f}%{100 * s['extra_max']:>14.1f}%{100 * s['quickest']:>9.0f}%"
                     f"{s['ms']:>8.2f}")
    return "\n".join(lines)
