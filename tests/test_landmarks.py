"""A* with landmarks (robotnav/landmarks.py, cpp/src/landmarks.cpp): a lower bound that is never too high and
consistent, so A* still finds the quickest route while searching fewer cells; the C++ core agrees with ==."""
import sys
from argparse import Namespace
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from robotnav import cpp, landmarks  # noqa: E402
from robotnav.algorithms import astar, dijkstra  # noqa: E402
from robotnav.learn import dataset as ds  # noqa: E402
from robotnav.learn.labels import cost_to_go  # noqa: E402
from robotnav.terrain import THEMES, make_map  # noqa: E402

needs_cpp = pytest.mark.skipif(not cpp.available(), reason="the C++ core is not built (cmake -S cpp -B cpp/build)")


@pytest.fixture(scope="module", params=THEMES)
def small(request):
    t = make_map(request.param, size=36, seed=1005)
    return t, landmarks.pick(t, 6)


def trips(t, n=6, seed=0):
    region = landmarks.largest_region(t)
    cells = [tuple(int(v) for v in c) for c in np.argwhere(region)]
    rng = np.random.default_rng(seed)
    return [(cells[i], cells[j]) for i, j in rng.choice(len(cells), size=(n, 2)) if i != j]


def test_cost_from_is_the_quickest_time_to_every_cell(small):
    t, lm = small
    src = divmod(lm.cells[0], t.cols)
    d = landmarks.cost_from(t, src)
    rng = np.random.default_rng(1)
    for i in rng.choice(t.rows * t.cols, size=25, replace=False):
        cell = divmod(int(i), t.cols)
        if t.blocked[cell] or cell == src:
            continue
        r = dijkstra(replace(t, start=src, goal=cell))
        assert r.found == bool(np.isfinite(d[cell]))
        if r.found:
            assert d[cell] == pytest.approx(r.cost, rel=1e-12)


def test_largest_region_matches_the_data_sets(small):
    t, _ = small
    assert np.array_equal(landmarks.largest_region(t), ds.largest_region(t))


def test_landmarks_are_spread_over_the_region(small):
    t, lm = small
    region = landmarks.largest_region(t)
    assert len(lm.cells) == 6 == len(set(lm.cells))
    assert all(region[divmod(c, t.cols)] for c in lm.cells)
    assert lm.from_.shape == lm.to.shape == (6, t.rows, t.cols)
    for k, c in enumerate(lm.cells):
        assert lm.from_[k][divmod(c, t.cols)] == 0.0 == lm.to[k][divmod(c, t.cols)]


def test_bound_is_never_too_high_and_consistent(small):
    t, lm = small
    for _, goal in trips(t, 3):
        h = landmarks.table(t, lm, goal)
        truth = cost_to_go(t, goal)
        ok = np.isfinite(truth)
        assert (h[ok] <= truth[ok] * (1 + 1e-12) + 1e-12).all()
        assert h[goal] == 0.0
        for r in range(t.rows):                            # h(a) <= time(a -> b) + h(b) on every legal move
            for c in range(t.cols):
                if not t.blocked[r, c]:
                    for n, step in t.neighbors((r, c)):
                        assert h[r, c] <= step + h[n] + 1e-9


def test_astar_with_landmarks_finds_the_quickest_route_searching_less(small):
    t, lm = small
    fewer = 0
    for start, goal in trips(t, 6, seed=2):
        a, b = astar(replace(t, start=start, goal=goal)), landmarks.astar(t, lm, start, goal)
        assert b.found == a.found
        if a.found:
            assert b.cost == pytest.approx(a.cost, rel=1e-12) and b.path[0] == start and b.path[-1] == goal
            assert b.nodes_expanded <= a.nodes_expanded
            fewer += b.nodes_expanded < a.nodes_expanded
    assert fewer >= 3


def test_more_landmarks_than_cells_stops_early():
    t = make_map("grassland", size=12, seed=1)
    lm = landmarks.pick(t, 500)
    assert len(lm.cells) == int(landmarks.largest_region(t).sum())


@needs_cpp
def test_cpp_landmarks_equal_the_python_reference(small):
    t, lm = small
    c = cpp.pick_landmarks(t, 6)
    assert list(c.cells) == lm.cells
    assert np.array_equal(c.from_.reshape(lm.from_.shape), lm.from_)
    assert np.array_equal(c.to.reshape(lm.to.shape), lm.to)
    assert np.array_equal(cpp.core().largest_region(cpp.to_cpp(t)), landmarks.largest_region(t))
    for start, goal in trips(t, 4, seed=3):
        assert np.array_equal(cpp.landmark_table(t, c, goal), landmarks.table(t, lm, goal))
        p, q = landmarks.astar(t, lm, start, goal), cpp.astar_landmarks(t, c, start, goal)
        assert (q.path, q.cost, q.nodes_expanded, q.explored_order) == (p.path, p.cost, p.nodes_expanded, p.explored_order)


def test_route_command_adds_landmarks(capsys):
    import main
    args = Namespace(map="island", start=None, goal=None, preset=None, size=48, grade=0.6, seed=3, engine="python",
                     landmarks=True)
    main.route(args)
    out = capsys.readouterr().out
    assert "A* + landmarks" in out and "A* " in out
