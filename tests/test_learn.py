"""The learning data (robotnav/learn): exact labels, A* with a table of estimates, the split by map, checksums and
the baseline. The C++ fields are held equal to the Python reference with ==."""
import json
import shutil
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from robotnav import cpp  # noqa: E402
from robotnav.algorithms import astar, dijkstra  # noqa: E402
from robotnav.learn import baseline, dataset as ds  # noqa: E402
from robotnav.learn.labels import astar_table, climb_table, cost_to_go, octile_table  # noqa: E402
from robotnav.terrain import THEMES, make_map  # noqa: E402

needs_cpp = pytest.mark.skipif(not cpp.available(), reason="the C++ core is not built (cmake -S cpp -B cpp/build)")


@pytest.fixture(scope="module", params=THEMES)
def small(request):
    return make_map(request.param, size=32, seed=1003)


def far_pair(t):
    """A goal in the middle of the largest region and the cell of that region farthest from it."""
    region = ds.largest_region(t)
    cells = np.argwhere(region)
    goal = tuple(int(v) for v in cells[len(cells) // 2])
    field = cost_to_go(t, goal)
    start = tuple(int(v) for v in np.unravel_index(np.argmax(np.where(np.isfinite(field), field, -1)), field.shape))
    return start, goal, field


def test_cost_to_go_is_the_quickest_time_from_every_cell(small):
    start, goal, field = far_pair(small)
    assert field[goal] == 0.0
    assert np.isinf(field[small.blocked]).all()                       # nothing drives from inside a rock or the sea
    rng = np.random.default_rng(0)
    cells = np.argwhere(~small.blocked)
    for i in rng.choice(len(cells), size=30, replace=False):
        s = tuple(int(v) for v in cells[i])
        d = dijkstra(replace(small, start=s, goal=goal))                 # forward, from each cell on its own
        assert d.found == bool(np.isfinite(field[s]))
        if d.found:
            assert field[s] == pytest.approx(d.cost, rel=1e-12, abs=1e-12)   # the same sums, added the other way round


def test_octile_table_is_a_stars_own_estimate(small):
    start, goal, _ = far_pair(small)
    table = octile_table(small, goal)
    assert all(table[r, c] == small.heuristic((r, c), goal) for r in range(small.rows) for c in range(small.cols))
    a, t = astar(replace(small, start=start, goal=goal)), astar_table(small, start, goal, table)
    assert (t.path, t.cost, t.nodes_expanded, t.explored_order) == (a.path, a.cost, a.nodes_expanded, a.explored_order)


def test_climb_aware_estimate_never_overestimates_and_is_consistent(small):
    start, goal, field = far_pair(small)
    climb, octile = climb_table(small, goal), octile_table(small, goal)
    ok = np.isfinite(field)
    assert (climb >= octile).all() and (climb[ok] <= field[ok] * (1 + 1e-12)).all()
    for r in range(small.rows):                                        # h(a) <= time(a -> b) + h(b) on every move
        for c in range(small.cols):
            if small.blocked[r, c]:
                continue
            for n, step in small.neighbors((r, c)):
                assert climb[r, c] <= step + climb[n] + 1e-9
    a, cl = astar(replace(small, start=start, goal=goal)), astar_table(small, start, goal, climb)
    assert cl.cost == pytest.approx(a.cost, rel=1e-12) and cl.nodes_expanded <= a.nodes_expanded


def test_the_true_time_is_a_perfect_estimate(small):
    start, goal, field = far_pair(small)
    a, best = astar(replace(small, start=start, goal=goal)), astar_table(small, start, goal, field)
    assert best.cost == pytest.approx(a.cost, rel=1e-12)
    assert best.nodes_expanded == len(best.path) < a.nodes_expanded     # straight down the route, nothing else
    zero = astar_table(small, start, goal, np.zeros_like(field))
    d = dijkstra(replace(small, start=start, goal=goal))
    assert (zero.cost, zero.nodes_expanded) == (d.cost, d.nodes_expanded)   # no estimate at all: Dijkstra


def test_bad_tables_and_goals_are_refused(small):
    start, goal, field = far_pair(small)
    with pytest.raises(ValueError):
        astar_table(small, start, goal, field[:-1])
    with pytest.raises(ValueError):
        astar_table(small, start, goal, np.full_like(field, np.nan))
    with pytest.raises(ValueError):
        cost_to_go(small, tuple(int(v) for v in np.argwhere(small.blocked)[0]))


@needs_cpp
def test_cpp_fields_and_table_search_equal_the_python_reference(small):
    start, goal, field = far_pair(small)
    assert np.array_equal(cpp.cost_to_go(small, goal), field)           # bit for bit, inf for inf
    for table in (octile_table(small, goal), field, 1.7 * octile_table(small, goal)):
        p, c = astar_table(small, start, goal, table), cpp.astar_table(small, start, goal, table)
        assert (c.path, c.cost, c.nodes_expanded, c.explored_order) == (p.path, p.cost, p.nodes_expanded, p.explored_order)
    p, c = astar_table(small, start, goal, field, w=1.5), cpp.astar_table(small, start, goal, field, w=1.5)
    assert (c.path, c.cost, c.nodes_expanded) == (p.path, p.cost, p.nodes_expanded)


def test_split_follows_the_seed_alone():
    assert [ds.split_of(s) for s in range(1000, 1010)] == ["train"] * 8 + ["val", "test"]
    assert ds.split_of(2019) == "test" and ds.split_of(77) == "train"


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    """Ten small maps per theme (every split gets some), two goals each, labelled by the Python reference."""
    out = tmp_path_factory.mktemp("terrain")
    manifest = ds.build(out, per_theme=10, goals=2, size=32, jobs=1, engine="python", log=None)
    return out, manifest


def test_data_set_is_split_by_map_with_no_map_in_two_splits(tiny):
    out, m = tiny
    assert len(m["maps"]) == 30 and m["counts"]["train"] == {th: 8 for th in THEMES}
    assert m["counts"]["val"] == m["counts"]["test"] == {th: 1 for th in THEMES}
    seen = {}
    for e in m["maps"]:
        assert e["split"] == ds.split_of(e["seed"])
        assert seen.setdefault((e["theme"], e["seed"]), e["split"]) == e["split"]
        assert m["demo_maps_excluded"][e["theme"]] != e["seed"]        # the viewer's maps stay out
    assert len(seen) == 30
    for sp in ds.SPLITS:                                                # no map is in two splits
        others = {k for k, v in seen.items() if v != sp}
        assert not others & {k for k, v in seen.items() if v == sp}


def test_every_file_holds_a_map_its_goals_and_their_labels(tiny):
    out, m = tiny
    for e in m["maps"]:
        a = ds.load(out, e)
        assert a["cost_to_go"].shape == (2, 32, 32) and a["cost_to_go"].dtype == np.float32
        assert a["height"].dtype == np.float64                           # the exact map the labels are exact for
        for k, (r, c) in enumerate(a["goals"]):
            assert a["region"][r, c] and a["cost_to_go"][k, r, c] == 0.0
            assert np.isfinite(a["cost_to_go"][k][a["region"]]).all()   # the whole region can reach its goals
        assert ds.content_hash(a) == e["sha256"]
        assert sum(int(np.isfinite(f).sum()) for f in a["cost_to_go"]) == e["labelled"]


def test_check_passes_and_catches_a_changed_label(tiny, tmp_path):
    out, m = tiny
    assert ds.check(out, audit=1, log=None) == []
    copy = tmp_path / "copy"
    shutil.copytree(out, copy)
    e = m["maps"][0]
    a = ds.load(copy, e)
    r, c = a["goals"][0]
    a["cost_to_go"][0, r, c] += 1.0                                     # the goal: 0 s becomes 1 s
    np.savez_compressed(copy / e["file"], **a)
    problems = ds.check(copy, audit=1, log=None)
    assert any("checksum" in p for p in problems) and any("Python reference" in p for p in problems)
    (copy / m["maps"][1]["file"]).unlink()
    assert any("missing" in p for p in ds.check(copy, audit=0, log=None))


def test_rebuilding_gives_the_same_data_set(tiny, tmp_path):
    out, m = tiny
    again = ds.build(tmp_path, per_theme=10, goals=2, size=32, jobs=2, engine="python", log=None)
    assert again["sha256"] == m["sha256"]                               # same seeds, same bytes (on this machine)
    assert ds.rebuilt_from_seed(out, m["maps"][0])


@needs_cpp
def test_cpp_and_python_label_the_same_data_set(tiny, tmp_path):
    out, m = tiny
    by_cpp = ds.build(tmp_path, per_theme=10, goals=2, size=32, jobs=1, engine="cpp", log=None)
    assert by_cpp["sha256"] == m["sha256"]


def test_baseline_brackets_what_a_learned_estimate_can_do(tiny):
    out, _ = tiny
    rows = baseline.measure(out, split="train", starts=2, min_time=10.0, engine="python", log=None)
    assert len(rows) > 20
    for r in rows:
        assert r["route"] == r["perfect"] <= r["climb"] <= r["astar"] <= r["dijkstra"]
        assert r["perfect_cost"] == pytest.approx(r["astar_cost"], rel=1e-12)
        assert r["climb_cost"] == pytest.approx(r["astar_cost"], rel=1e-12)
        assert 0.0 < r["estimate"] <= r["climb_estimate"] <= 1.0 + 1e-12  # neither estimate is ever too high
        assert r["wastar_extra"] >= -1e-12
    summary = baseline.summarize(rows)
    assert [s["theme"] for s in summary] == [*THEMES, "all"]
    assert "A*/route" in baseline.table(summary)
    if cpp.available():
        fast = baseline.measure(out, split="train", starts=2, min_time=10.0, engine="cpp", log=None)
        assert [(r["astar"], r["dijkstra"], r["climb"], r["perfect"], r["route"]) for r in fast] == \
               [(r["astar"], r["dijkstra"], r["climb"], r["perfect"], r["route"]) for r in rows]


def test_cli_builds_checks_and_measures(tmp_path, capsys):
    import main
    from argparse import Namespace
    base = dict(dir=str(tmp_path / "d"), per_theme=1, goals=1, size=32, jobs=1, engine="python", audit=1,
                split="train", starts=1, out=str(tmp_path / "b.json"))
    for action in ("build", "check", "baseline"):
        main.dataset(Namespace(action=action, **base))
    out = capsys.readouterr().out
    assert "3 maps with" in out and "all good" in out and "A*/route" in out
    assert json.loads((tmp_path / "b.json").read_text())["split"] == "train"


# ---------------------------------------------------------------- stage 2: the network and A* guided by it
from robotnav.learn import evaluate as ev  # noqa: E402
from robotnav.learn import model as mdl  # noqa: E402
from robotnav.learn.features import FEATURES, features  # noqa: E402
from robotnav.learn.labels import astar_reopen  # noqa: E402


@pytest.fixture(scope="module")
def shipped():
    return mdl.load()


def test_shipped_network_file(shipped):
    assert shipped.sizes == [FEATURES, 64, 64, 1]
    assert any("PyTorch" in line for line in shipped.note)
    assert (shipped.scale > 0).all()
    again = mdl.parse(shipped.text())                                  # the file round-trips exactly
    assert all(np.array_equal(a, b) and np.array_equal(c, d) for (a, c), (b, d) in zip(shipped.layers, again.layers))
    card = json.loads(mdl.ASSET.with_suffix(".card.json").read_text())
    assert card["network"]["sizes"] == shipped.sizes and card["data"]["train_maps"] == 360


def test_shipped_network_still_estimates_the_time_well(shipped):
    """On maps outside the data set, the network must stay far closer to the true time than the straight line
    (the benchmark measured 9% off against 33%); this holds the committed weights to that without retraining."""
    errs, octile_errs = [], []
    for theme in THEMES:
        t = make_map(theme, size=96, seed=5000)
        region = ds.largest_region(t)
        cells = np.argwhere(region)
        goal = tuple(int(v) for v in cells[len(cells) // 2])
        truth = cost_to_go(t, goal)
        est = shipped.table(t, goal)
        ok = np.isfinite(truth) & (truth >= 10.0)
        errs.append(np.abs(est[ok] / truth[ok] - 1))
        octile_errs.append(np.abs(octile_table(t, goal)[ok] / truth[ok] - 1))
    err, oct_err = np.concatenate(errs).mean(), np.concatenate(octile_errs).mean()
    assert err < 0.15 and err < oct_err / 2


@pytest.fixture(scope="module", params=THEMES)
def mid(request):
    return make_map(request.param, size=48, seed=1003)


@needs_cpp
def test_cpp_features_and_network_equal_the_reference(mid, shipped):
    start, goal, _ = far_pair(mid)
    x, climb = features(mid, goal)
    xc, cc = cpp.features(mid, goal)
    assert np.array_equal(x, xc) and np.array_equal(climb, cc)        # bit for bit
    h = cpp.heuristic(mdl.ASSET.read_text())
    live = climb > 0
    z = shipped.output(x[live][:300])
    assert all(h.output(list(row)) == v for row, v in zip(x[live][:300], z))
    for w in (np.inf, 1.1):
        assert np.array_equal(cpp.learned_table(mid, goal, h, w), shipped.table(mid, goal, w))


@needs_cpp
def test_cpp_search_with_the_network_equals_the_reference(mid, shipped):
    h = cpp.heuristic(mdl.ASSET.read_text())
    start, goal, _ = far_pair(mid)
    for w in (np.inf, 1.1):
        p, c = mdl.plan(mid, start, goal, shipped, w), cpp.astar_learned(mid, start, goal, h, w)
        assert (c.path, c.cost, c.nodes_expanded, c.explored_order) == (p.path, p.cost, p.nodes_expanded, p.explored_order)


def test_reexpanding_astar_is_plain_astar_with_a_consistent_estimate(mid):
    start, goal, _ = far_pair(mid)
    for table in (octile_table(mid, goal), climb_table(mid, goal)):
        a, r = astar_table(mid, start, goal, table), astar_reopen(mid, start, goal, table)
        assert (r.path, r.cost, r.nodes_expanded, r.explored_order) == (a.path, a.cost, a.nodes_expanded, a.explored_order)


def test_capped_network_keeps_its_promise(mid, shipped):
    """Capped at 1.1 x the climb-aware estimate (never too high), the estimate is at most 1.1 x the truth, and the
    re-expanding search then never returns a route more than 10% longer than the quickest."""
    rng = np.random.default_rng(4)
    region = np.argwhere(ds.largest_region(mid))
    for i, j in rng.choice(len(region), size=(6, 2)):
        start, goal = tuple(int(v) for v in region[i]), tuple(int(v) for v in region[j])
        if start == goal:
            continue
        truth = cost_to_go(mid, goal)
        h = shipped.table(mid, goal, 1.1)
        ok = np.isfinite(truth)
        assert (h[ok] <= 1.1 * truth[ok] * (1 + 1e-12) + 1e-12).all()
        r = astar_reopen(mid, start, goal, h)
        assert r.cost <= 1.1 * truth[start] * (1 + 1e-12)


def test_model_file_errors_are_explained():
    with pytest.raises(ValueError):
        mdl.parse("brubus-heuristic 1\ninputs 3\n")
    with pytest.raises(ValueError):
        mdl.parse("brubus-heuristic 1\ninputs 16\nmean " + "0 " * 16)
    with pytest.raises(FileNotFoundError):
        mdl.load(ROOT / "no-such-model.txt")


@needs_cpp
def test_benchmark_runs_on_the_tiny_data_set(tiny, shipped):
    out, _ = tiny
    rows, setup = ev.evaluate(mdl.ASSET.read_text(), out, split="train", starts=1, landmarks=4, log=None,
                              max_maps=6)
    assert setup and rows
    by = {}
    for r in rows:
        by.setdefault((r["map"], r["goal"], r["start"]), {})[r["planner"]] = r
    for trip in by.values():
        best = trip["A* perfect"]
        for name in ("Dijkstra", "A* straight-line", "A* climb-aware", "A* + 4 landmarks"):
            assert trip[name]["cost"] == pytest.approx(best["cost"], rel=1e-12)      # all of these are optimal
        assert trip["A* + 4 landmarks"]["cells"] <= trip["A* straight-line"]["cells"]
        assert trip["A* + network, capped"]["extra"] <= 0.1 + 1e-12
    names = {s["planner"] for s in ev.summarize(rows)}
    assert "A* + network" in names and "Weighted A* 1.5" in names
    acc = ev.accuracy(mdl.ASSET.read_text(), out, split="train", landmarks=4)
    lm = next(a for a in acc if a["estimate"] == "4 landmarks" and a["theme"] == "all")
    assert lm["too_high"] == 0.0 and lm["ratio"] <= 1.0
