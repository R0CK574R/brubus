"""Stage 3 (robotnav/learn/speeds.py): Bruce's moves as tables, the hidden true world, planning on a table (the
Python reference and the C++ core held equal with ==), the speeds file, and the committed network held to what the
study measured, without retraining it (PyTorch is not needed to run these)."""
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from robotnav import cpp  # noqa: E402
from robotnav.algorithms import astar  # noqa: E402
from robotnav.grid import EIGHT  # noqa: E402
from robotnav.learn import dataset as ds  # noqa: E402
from robotnav.learn import speeds as sp  # noqa: E402
from robotnav.learn.labels import cost_to_go  # noqa: E402
from robotnav.terrain import THEMES, make_map  # noqa: E402

needs_cpp = pytest.mark.skipif(not cpp.available(), reason="the C++ core is not built (cmake -S cpp -B cpp/build)")


@pytest.fixture(scope="module", params=THEMES)
def small(request):
    t = make_map(request.param, size=32, seed=1003)
    return t, sp.moves(t)


@pytest.fixture(scope="module")
def shipped():
    return sp.load()


def far_pair(t):
    """A goal in the middle of the largest region and the cell of that region farthest from it."""
    cells = np.argwhere(ds.largest_region(t))
    goal = tuple(int(v) for v in cells[len(cells) // 2])
    field = cost_to_go(t, goal)
    start = tuple(int(v) for v in np.unravel_index(np.argmax(np.where(np.isfinite(field), field, -1)), field.shape))
    return start, goal


def test_built_in_table_is_the_planners_drive_time(small):
    t, mv = small
    nominal = sp.nominal_costs(t, mv)
    for r in range(t.rows):
        for c in range(t.cols):
            i = r * t.cols + c
            moves = {} if t.blocked[r, c] else dict(t.neighbors((r, c)))
            for k, (dr, dc) in enumerate(EIGHT):
                n = (r + dr, c + dc)
                assert mv.legal[i, k] == (n in moves)
                assert nominal[i, k] == moves.get(n, np.inf)              # bit for bit, inf where it can't be made
                if n in moves:
                    assert mv.dest[i, k] == n[0] * t.cols + n[1]


def test_planning_on_the_built_in_table_is_plain_astar(small):
    t, mv = small
    start, goal = far_pair(t)
    nominal = sp.nominal_costs(t, mv)
    a, s = astar(replace(t, start=start, goal=goal)), sp.astar_costs(t, nominal, start, goal)
    assert (s.path, s.cost, s.nodes_expanded, s.explored_order) == (a.path, a.cost, a.nodes_expanded, a.explored_order)
    assert np.array_equal(sp.cost_to_go_costs(t, nominal, goal), cost_to_go(t, goal))
    assert sp.route_time(t, nominal, a.path) == t.path_cost(a.path)


def test_true_world_is_never_quicker_than_flat_ground(small):
    t, mv = small
    true = sp.true_costs(t, mv)
    assert np.array_equal(np.isfinite(true), mv.legal)
    run = np.where(mv.diag, np.sqrt(2.0), 1.0) * t.cell_size
    assert (true[mv.legal] >= run[mv.legal]).all()                    # so the straight-line estimate stays a bound
    flat = replace(t, height=np.zeros_like(t.height))                  # flat, smooth ground: only the ground's slowdown
    fm = sp.moves(flat)
    slow = np.array([sp.TRUE_SLOW[k] for k in sp.KINDS])[fm.kind[fm.dest]]
    assert np.allclose(sp.true_costs(flat, fm)[fm.legal], (run * slow)[fm.legal], rtol=1e-15, atol=0)


def test_true_world_charges_for_what_the_built_in_model_ignores():
    t = make_map("desert", size=48, seed=1003)
    mv = sp.moves(t)
    ratio = sp.true_costs(t, mv)[mv.legal] / sp.nominal_costs(t, mv)[mv.legal]
    side, kind = mv.side[mv.legal], mv.kind[mv.dest][mv.legal]
    for k in (sp.KINDS.index("Firm sand"), sp.KINDS.index("Soft sand")):
        on = kind == k
        steep, gentle = on & (side > 0.3), on & (side < 0.05)
        assert np.median(ratio[steep]) > np.median(ratio[gentle])     # across a slope is slower than he thinks


def test_features_are_what_bruce_can_sense(small):
    t, mv = small
    x = sp.features(mv)
    assert x.shape == (t.rows * t.cols, 8, sp.FEATURES) and len(sp.NAMES) == sp.FEATURES
    assert (x[..., 6:].sum(axis=2) == 1.0).all()                       # one kind of ground per move
    assert np.array_equal(x[..., 2] - x[..., 3], x[..., 1])            # uphill - downhill = grade
    assert (x[..., 2:6] >= 0).all()
    assert np.array_equal(x[..., 0], mv.diag.astype(float))


def test_learned_table(small, shipped):
    t, mv = small
    learned = shipped.costs(t, mv)
    assert np.array_equal(np.isfinite(learned), mv.legal)
    assert (learned[mv.legal] >= mv.d3[mv.legal]).all()                 # 1 s per metre at best


@needs_cpp
def test_cpp_plans_on_tables_like_the_reference(small, shipped):
    t, mv = small
    start, goal = far_pair(t)
    cmap = cpp.to_cpp(t)
    for costs in (sp.nominal_costs(t, mv), sp.true_costs(t, mv), shipped.costs(t, mv)):
        p, c = sp.astar_costs(t, costs, start, goal), cpp.astar_costs(t, costs, start, goal, cmap=cmap)
        assert (c.path, c.cost, c.nodes_expanded, c.explored_order) == (p.path, p.cost, p.nodes_expanded, p.explored_order)
        assert np.array_equal(cpp.cost_to_go_costs(t, costs, goal, cmap=cmap), sp.cost_to_go_costs(t, costs, goal))
        assert cpp.route_time(t, costs, p.path, cmap=cmap) == sp.route_time(t, costs, p.path)
    with pytest.raises(ValueError):
        cpp.astar_costs(t, np.full((t.rows * t.cols, 8), -1.0), start, goal, cmap=cmap)


def test_the_true_table_finds_the_quickest_true_route(small, shipped):
    t, mv = small
    start, goal = far_pair(t)
    true = sp.true_costs(t, mv)
    best = sp.cost_to_go_costs(t, true, goal)[start]
    oracle = sp.astar_costs(t, true, start, goal)
    assert sp.route_time(t, true, oracle.path) == pytest.approx(best, rel=1e-12)
    for costs in (sp.nominal_costs(t, mv), shipped.costs(t, mv)):       # any other plan, driven for real, is no quicker
        route = sp.astar_costs(t, costs, start, goal).path
        assert sp.route_time(t, true, route) >= best * (1 - 1e-12)


def test_shipped_speeds_file(shipped):
    assert [shipped.layers[0][0].shape[0]] + [W.shape[1] for W, _ in shipped.layers] == [sp.FEATURES, 32, 32, 1]
    assert any("PyTorch" in line for line in shipped.note)
    from robotnav.learn.model import parse
    again = parse(shipped.text(), header="brubus-speeds", inputs=sp.FEATURES)    # the file round-trips exactly
    assert all(np.array_equal(a, b) and np.array_equal(c, d) for (a, c), (b, d) in zip(shipped.layers, again.layers))
    assert np.array_equal(again.mean, shipped.mean) and np.array_equal(again.scale, shipped.scale)
    card = json.loads(sp.ASSET.with_suffix(".card.json").read_text())
    assert card["network"]["sizes"] == [sp.FEATURES, 32, 32, 1] and card["network"]["features"] == list(sp.NAMES)
    world = card["hidden_world"]
    assert world["slowdown"] == sp.TRUE_SLOW and world["uphill"] == sp.TRUE_UPHILL and world["noise"] == sp.NOISE
    assert [c["trips"] for c in card["curve"]] == [10, 30, 100, 300]
    learned = {w["ground"]: w for w in card["learned"]}
    for w in sp.what_it_learned(shipped):                              # the card describes this very file
        assert learned[w["ground"]]["learned"] == w["learned"]


def test_shipped_speeds_learned_the_hidden_world(shipped):
    """Flat ground: within 5% of the true slowdown on every kind, and nearer it than the built-in guess. Uphill:
    within 15% on every kind but beach sand, which Bruce drove 20 times in 300 trips (the README says so)."""
    for w in sp.what_it_learned(shipped):
        assert abs(w["learned"] / w["true"] - 1) < 0.05, w
        if w["built_in"] != w["true"]:
            assert abs(w["learned"] - w["true"]) < abs(w["built_in"] - w["true"]), w
        if w["ground"] != "Sand":
            assert abs(w["learned_uphill"] / w["true_uphill"] - 1) < 0.15, w


def test_shipped_speeds_plan_quicker_routes(shipped):
    """On maps outside the data set, routes planned with what Bruce learned, driven in the true world, are within
    a whisker of the quickest; routes planned with his built-in model are well off it (the study: 0.09% vs 6.9%)."""
    plan, field_of, to_cmap = sp._planner()
    rng = np.random.default_rng(1)
    regret = {"built-in": [], "learned": []}
    for theme in THEMES:
        t = make_map(theme, size=48, seed=5000)
        mv, cmap = sp.moves(t), to_cmap(t)
        true = sp.true_costs(t, mv)
        tables = {"built-in": sp.nominal_costs(t, mv), "learned": shipped.costs(t, mv)}
        cells = np.argwhere(ds.largest_region(t))
        for _ in range(4):
            goal = tuple(int(v) for v in cells[rng.integers(len(cells))])
            best = field_of(t, cmap, true, goal)
            ok = np.argwhere(np.isfinite(best) & (best >= 20.0))
            start = tuple(int(v) for v in ok[rng.integers(len(ok))])
            for name, costs in tables.items():
                regret[name].append(sp.route_time(t, true, plan(t, cmap, costs, start, goal)) / best[start] - 1)
    built_in, learned = np.mean(regret["built-in"]), np.mean(regret["learned"])
    assert min(min(v) for v in regret.values()) > -1e-12              # nothing beats the quickest route
    assert learned < 0.01 and learned < built_in / 5


def test_speeds_file_errors_are_explained():
    from robotnav.learn.model import parse
    with pytest.raises(ValueError):
        parse("brubus-heuristic 1\ninputs 14\n", header="brubus-speeds", inputs=sp.FEATURES)   # another file
    with pytest.raises(ValueError):
        parse("brubus-speeds 1\ninputs 16\n", header="brubus-speeds", inputs=sp.FEATURES)      # the wrong inputs
    with pytest.raises(FileNotFoundError, match="speeds learn"):
        sp.load(ROOT / "no-such-speeds.txt")


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    out = tmp_path_factory.mktemp("terrain")
    ds.build(out, per_theme=10, goals=2, size=32, jobs=1, engine="python", log=None)
    return out


def test_evaluation_on_a_tiny_data_set(tiny, shipped):
    rows = sp.evaluate({"built-in": "built-in", "learned": shipped, "true": "true"}, tiny, split="train", starts=1,
                       log=None)
    by = {}
    for r in rows:
        by.setdefault(r["model"], []).append(r["regret"])
    assert len(by["true"]) == len(by["learned"]) == len(by["built-in"]) > 20
    assert all(v == 0.0 for v in by["true"]) and min(by["learned"] + by["built-in"]) >= 0.0
    assert np.mean(by["learned"]) < np.mean(by["built-in"])
    summary = sp.summarize(rows)
    assert {s["theme"] for s in summary} == {"grassland", "desert", "island", "all"}
    assert "built-in" in sp.table(summary)


def cli(action, folder):
    from argparse import Namespace
    return Namespace(action=action, dir=str(folder), model=None, split="train", checkpoints=[10, 30, 100, 300])


def test_cli_scores_the_shipped_speeds(tiny, capsys):
    import main
    main.speeds(cli("eval", tiny))
    out = capsys.readouterr().out
    assert "slower than the quickest" in out and "What Bruce learned" in out and "Soft sand" in out


def test_learning_needs_pytorch_and_says_so(tiny, monkeypatch):
    import main
    monkeypatch.setitem(sys.modules, "torch", None)                     # as if PyTorch were not installed
    with pytest.raises(SystemExit, match="PyTorch"):
        main.speeds(cli("learn", tiny))


def test_fit_recovers_a_simple_rule():
    """The trainer itself (needs PyTorch): seconds per metre = ground's slowdown x (1 + 3 x uphill grade)."""
    pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    n = 4000
    x = np.zeros((n, sp.FEATURES))
    kind = rng.integers(0, 3, n)
    x[np.arange(n), 6 + kind] = 1.0
    g = rng.uniform(-0.3, 0.3, n)
    x[:, 1], x[:, 2], x[:, 3] = g, np.maximum(g, 0), np.maximum(-g, 0)
    slow = np.array([1.0, 1.5, 2.0])[kind]
    y = slow * (1 + 3 * np.maximum(g, 0))
    model = sp.fit(x, y, steps=1500)
    assert np.abs(model.per_metre(x) / y - 1).mean() < 0.03
