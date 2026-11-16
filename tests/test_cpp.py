"""The C++ core finds exactly what the Python reference finds: the same routes, cells searched and times.

Everything is compared with == (no tolerance). Skipped unless the C++ core is built:
    cmake -S cpp -B cpp/build && cmake --build cpp/build
`python main.py parity` runs the same comparison on every preset trip of the full-size maps.
"""
import re
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import main  # noqa: E402
from robotnav import cpp  # noqa: E402
from robotnav.airspace import fly, make_airspace  # noqa: E402
from robotnav.mapfile import read_map, write_map  # noqa: E402
from robotnav.parity import DRIVE_FIELDS, FLIGHT_FIELDS, PLANNER_FIELDS, compare_edges  # noqa: E402
from robotnav.replan import Roadblock, sensor_drive  # noqa: E402
from robotnav import sensing  # noqa: E402
from robotnav.terrain import THEMES, plan_pair, preset_routes, shortest_route, solvable_map  # noqa: E402

pytestmark = pytest.mark.skipif(not cpp.available(), reason="the C++ core is not built (cmake -S cpp -B cpp/build)")
TOOL = next((p for p in (ROOT / "cpp/build/robotnav", ROOT / "cpp/build/Release/robotnav.exe",
                         ROOT / "cpp/build/robotnav.exe") if p.exists()), None)


@pytest.fixture(scope="module", params=THEMES)
def themed(request):
    return solvable_map(request.param, seed=3, size=48)


def _pairs(t, n, seed):
    """n random pairs of open cells, the same every run."""
    rng = np.random.default_rng(seed)
    cells = [tuple(int(v) for v in rc) for rc in np.argwhere(~t.blocked)]
    return [tuple(cells[i] for i in rng.choice(len(cells), 2, replace=False)) for _ in range(n)]


def _diffs(a, b, fields):
    return [f for f in fields if getattr(a, f) != getattr(b, f)]


def test_every_edge_cost_matches(themed):
    tally = compare_edges(themed, cpp.to_cpp(themed))
    assert tally.total > 8 * 40 * 40 and tally.same == tally.total, tally.diffs


def test_planners_match_cell_for_cell(themed):
    for a, b in [(themed.start, themed.goal), *_pairs(themed, 8, seed=1)]:
        py = plan_pair(themed, a, b) + [shortest_route(themed, a, b)]
        cc = cpp.race(themed, a, b)
        assert [r.algorithm for r in cc] == [r.algorithm for r in py]
        for x, y in zip(py, cc):
            assert _diffs(x, y, PLANNER_FIELDS) == [], (a, b, x.algorithm)     # path, time, every cell searched


def test_path_stats_match(themed):
    m, rc = cpp.to_cpp(themed), cpp.core()
    for a, b in _pairs(themed, 4, seed=2):
        path = shortest_route(themed, a, b).path
        assert themed.path_stats(path) == rc.path_stats(m, [r * themed.cols + c for r, c in path])


@pytest.mark.parametrize("sensor, prior", [(3.0, "none"), (7.5, "elevation"), (12.0, "map")])
def test_sensor_drives_match_step_for_step(themed, sensor, prior):
    for a, b in _pairs(themed, 2, seed=3):
        best = plan_pair(themed, a, b)[2]
        blocks = [Roadblock(best.path[len(best.path) // 2], appears=2)] if best.found and len(best.path) > 4 else []
        x = sensor_drive(themed, a, b, sensor_range=sensor, prior=prior, roadblocks=blocks, check=True)
        y = cpp.sensor_drive(themed, a, b, sensor_range=sensor, prior=prior, roadblocks=blocks, check=True)
        assert _diffs(x, y, DRIVE_FIELDS) == [], (a, b)                       # every step: news, work, plans


def test_airspace_and_flights_match(themed):
    pa, ca = make_airspace(themed), cpp.make_airspace(themed)
    assert pa.towers == ca.towers and pa.alt == ca.alt and pa.eyes == ca.eyes
    assert np.array_equal(pa.free, ca.free) and np.array_equal(pa.seen, ca.seen)
    for a, b in _pairs(themed, 3, seed=4):
        for stealth in (True, False):
            assert _diffs(fly(pa, a, b, stealth=stealth), cpp.fly(ca, a, b, stealth=stealth), FLIGHT_FIELDS) == []
    a, b = _pairs(themed, 1, seed=5)[0]
    assert _diffs(fly(pa, a, b, heuristic=False), cpp.fly(ca, a, b, heuristic=False), FLIGHT_FIELDS) == []


def test_map_files_carry_every_bit(themed, tmp_path):
    presets = [dict(name="NW → SE", start=themed.start, goal=themed.goal)]
    path = write_map(themed, tmp_path / "map.map", presets=presets)
    m = cpp.core().load_map(str(path))
    assert m.height.tobytes() == np.asarray(themed.height, dtype=float).tobytes()       # the very same doubles
    assert np.array_equal(m.surface, themed.surface) and np.array_equal(m.obstacle, themed.obstacle)
    assert [(p.start, p.goal, p.name) for p in m.presets] == [
        (themed.start[0] * themed.cols + themed.start[1], themed.goal[0] * themed.cols + themed.goal[1], "NW → SE")]
    t2, p2 = read_map(path)                                                              # and Python reads it back
    assert t2.height.tobytes() == themed.height.tobytes() and np.array_equal(t2.obstacle, themed.obstacle)
    assert t2.surface_costs == themed.surface_costs and t2.sea_level == themed.sea_level and p2 == presets
    assert [(p["level"], p["cells"]) for p in t2.ponds] == [(p["level"], p["cells"]) for p in themed.ponds]


# ------------------------------------------------------------------ the command-line tool prints what Python prints

@pytest.fixture(scope="module")
def desert_file(tmp_path_factory):
    t = solvable_map("desert", seed=3, size=48)
    return write_map(t, tmp_path_factory.mktemp("maps") / "desert.map", presets=preset_routes(t))


def _without_ms(text):
    return [re.sub(r"\s+\d+\.\d+$", "", line) for line in text.splitlines()]


@pytest.mark.skipif(TOOL is None, reason="the robotnav command-line tool is not built")
@pytest.mark.parametrize("command, options", [
    ("route", dict(preset=1)),
    ("route", dict(start="40,3", goal="5,44")),
    ("explore", dict(preset=2, sensor=5.0, prior="none", block=None)),
    ("explore", dict(preset=3, sensor=7.5, prior="map", block=["20,20"])),
    ("fly", dict(preset=4)),
])
def test_the_cpp_tool_prints_what_python_prints(desert_file, capsys, command, options):
    args = Namespace(map="desert", size=48, grade=0.6, seed=3, start=None, goal=None, preset=None, engine="python")
    for k, v in options.items():
        setattr(args, k, v)
    {"route": main.route, "explore": main.explore, "fly": main.fly_cmd}[command](args)
    python_out = capsys.readouterr().out
    argv = [str(TOOL), command, str(desert_file)]
    for k, v in options.items():
        for item in (v if isinstance(v, list) else [] if v is None else [v]):
            argv += [f"--{k}", str(item)]
    cpp_out = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8", check=True).stdout
    assert _without_ms(cpp_out) == _without_ms(python_out)


def test_the_cli_runs_on_either_engine(capsys):
    args = Namespace(map="island", size=48, grade=0.6, seed=3, start=None, goal=None, preset=None)
    main.route(Namespace(**vars(args), engine="python"))
    py = capsys.readouterr().out
    main.route(Namespace(**vars(args), engine="cpp"))
    assert _without_ms(capsys.readouterr().out) == _without_ms(py)


# ---------------------------------------------------------------- sensing: field, risk-weighted routes, Monte Carlo
@pytest.fixture(scope="module")
def sensed(themed):
    """A map, its C++ twin, the airspace, and the towers' field from C++ and from the Python reference."""
    cmap = cpp.to_cpp(themed)
    air = cpp.make_airspace(themed)
    return themed, cmap, air, cpp.make_field(cmap, air), sensing.ground_field(themed, make_airspace(themed))


def test_detection_field_matches_to_the_last_bit(sensed):
    t, _, _, field, (pd, haz) = sensed
    assert pd.any() and haz.any()
    assert np.array_equal(field.ground_pd.reshape(pd.shape), pd)            # every cell, every tower: the same 32 bits
    assert np.array_equal(field.ground_hazard.reshape(haz.shape), haz)
    assert cpp.core().pd_at(7.25) == sensing.pd_at(7.25) and cpp.core().hazard_of(0.37) == sensing.hazard(0.37)
    assert cpp.core().wilson(37, 200) == pytest.approx(sensing.wilson(37, 200), abs=1e-15)


def test_risk_weighted_routes_match_cell_for_cell(sensed):
    t, cmap, _, field, (_, haz) = sensed
    C = t.cols
    for s, g in _pairs(t, 8, seed=11):
        for w in (0.0, 4.0, 25.0):
            cells, steps = sensing.risk_drive(t, haz, s, g, w)
            for plan in (cpp.core().risk_drive, cpp.core().risk_drive_reference):
                got = plan(cmap, field, s[0] * C + s[1], g[0] * C + g[1], w)
                assert got.found == bool(cells) and [divmod(i, C) for i in got.nodes] == cells and list(got.step) == steps
                if cells:
                    assert got.risk == sensing.route_risk(haz, cells, steps)


def test_trials_match_the_python_reference(sensed):
    t, cmap, _, field, (pd, haz) = sensed
    C, core, checked, blocked = t.cols, cpp.core(), 0, 0
    for s, g in _pairs(t, 12, seed=5):
        rt = core.risk_drive(cmap, field, s[0] * C + s[1], g[0] * C + g[1], 4.0)
        if not rt.found or len(rt.nodes) < 20:
            continue
        cells = [divmod(i, C) for i in rt.nodes]
        for number in range(40):
            want = sensing.trial(t, pd, haz, cells, list(rt.step), 4.0, 99, number)
            assert core.run_trial(cmap, field, rt, 4.0, 99, number) == want     # stuck, roadblock, detected, tracked, seconds
            checked += 1
            blocked += want[1]
    assert checked >= 120 and 0.25 < blocked / checked < 0.75                   # about half meet a roadblock


def test_monte_carlo_is_the_same_on_any_number_of_threads(sensed):
    t, cmap, _, field, (_, haz) = sensed
    C, core = t.cols, cpp.core()
    plans = [core.risk_drive(cmap, field, s[0] * C + s[1], g[0] * C + g[1], 4.0) for s, g in _pairs(t, 12, seed=5)]
    rt = max(plans, key=lambda r: len(r.nodes))                               # the longest drive of a dozen
    one = cpp.monte_carlo(cmap, field, rt, 4.0, 6000, seed=3, threads=1)
    assert one["n"] == 6000 and one["block"] > 2000
    for threads in (2, 5, 0):
        assert cpp.monte_carlo(cmap, field, rt, 4.0, 6000, seed=3, threads=threads) == one
    lo, hi = sensing.wilson(one["clean_detected"], one["clean"])              # the plan's risk is what the trials find
    assert lo - 0.02 <= rt.risk <= hi + 0.02
    assert sum(sensing.trial(t, sensed[4][0], haz, [divmod(i, C) for i in rt.nodes], list(rt.step), 4.0, 3, n)[2]
               for n in range(300)) == sum(core.run_trial(cmap, field, rt, 4.0, 3, n)[2] for n in range(300))
