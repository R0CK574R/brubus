"""The learning data set: many maps from the terrain generator, each labelled with the exact drive time from every
cell to a few goals, split by map, and stored with checksums.

    python main.py dataset build      # 150 maps per theme, 8 goals each, into data/terrain-v1/
    python main.py dataset check      # checksums, splits, and labels re-derived by the Python reference
    python main.py dataset baseline   # how much A* searches with today's estimate, and with a perfect one

What is in it. For each theme (grassland, desert, island), maps with seeds 1000, 1001, ... from the same generator
as the viewer's three maps (robotnav.terrain.make_map). On each map, `goals` cells picked at random from its largest
drivable region, and for each goal the field `cost_to_go`: the fewest seconds from every cell to that goal
(robotnav.learn.labels), inf where the goal can't be reached. One file per map, `<theme>-<seed>.npz`, holds the map
itself (heights, surfaces, obstacles: everything the drive-time model reads), the goals, the fields and the largest
region; `manifest.json` lists every map with its split and a checksum of its contents.

How it is split. By map, never by cell or by route: a map's seed decides its split (last digit 0-7 train, 8
validation, 9 test), so no map has cells in two splits, and a model is always tested on terrain it has never seen.
The viewer's own three maps are never in the data set (their seeds are recorded and excluded); they stay a fair
showcase. To measure generalisation to a new kind of terrain, a model can also be trained on two themes and tested
on the third.

Exactness and reproducibility. The labels are exact for the map stored next to them: `check` rebuilds the drive-time
model from the stored arrays and re-derives fields with the Python reference, and they must equal the stored ones
(stored as 32-bit floats, the training precision; the exact 64-bit fields can always be recomputed). Building again
from the same seeds on the same machine gives the same checksums. On another machine the generator's floating point
(NumPy's and SciPy's math libraries) may differ in the last bits of the heights, so its maps may not be bit-identical;
the labels are still exact for whatever maps it builds.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from ..terrain import NONE, THEMES, Terrain, _move_graphs, make_map, solvable_map
from .labels import cost_to_go

FORMAT = 1
NAME = "terrain-v1"
DEFAULT_DIR = Path(__file__).resolve().parents[2] / "data" / NAME
FIRST_SEED = 1000
PER_THEME = 150
GOALS = 8
SIZE = 96
GOAL_SALT = 20261117                     # seeds the choice of goals (with the theme and the map's seed)
SPLITS = ("train", "val", "test")
SPLIT_RULE = "by map: the seed's last digit 0-7 train, 8 val, 9 test"
LABEL = ("cost_to_go[k, r, c]: fewest seconds of drive time from cell (r, c) to goals[k] under the map's own "
         "drive-time model (exact: Dijkstra backwards from the goal), stored as float32; inf where the goal can't be "
         "reached")


def split_of(seed: int) -> str:
    """A map's split, from its seed alone."""
    d = seed % 10
    return "train" if d < 8 else ("val" if d == 8 else "test")


def demo_seeds(size: int = SIZE) -> dict:
    """The seeds of the viewer's three maps (solvable_map(theme, seed=3)): kept out of the data set."""
    return {theme: int(solvable_map(theme, seed=3, size=size).meta["seed"]) for theme in THEMES}


def largest_region(t: Terrain) -> np.ndarray:
    """The largest set of open cells the rover can drive between (moves are legal both ways, only their times
    differ, so this is one connected component of the move graph)."""
    from scipy.sparse.csgraph import connected_components
    drive, _ = _move_graphs(t)
    _, label = connected_components(drive, directed=True, connection="weak")
    label = label.reshape(t.rows, t.cols)
    open_ = ~t.blocked
    if not open_.any():
        return np.zeros_like(open_)
    counts = np.bincount(label[open_])
    return open_ & (label == int(np.argmax(counts)))


def pick_goals(t: Terrain, region: np.ndarray, k: int, theme: str, seed: int) -> list:
    """k different goals, at random from the largest region (the same ones for the same map, every time)."""
    rng = np.random.default_rng([GOAL_SALT, THEMES.index(theme), seed])
    cells = np.argwhere(region)
    pick = rng.choice(len(cells), size=min(k, len(cells)), replace=False)
    return [tuple(int(v) for v in cells[i]) for i in sorted(pick)]


def content_hash(arrays: dict) -> str:
    """SHA-256 of the arrays' names, types, shapes and bytes (file bytes would change with the zip's timestamps)."""
    h = hashlib.sha256()
    for name in sorted(arrays):
        a = np.ascontiguousarray(arrays[name])
        h.update(f"{name}|{a.dtype.str}|{a.shape}|".encode())
        h.update(a.tobytes())
    return h.hexdigest()


def terrain_of(arrays) -> Terrain:
    """The drive-time model of a stored map, rebuilt from its arrays alone (no generator involved)."""
    meta = json.loads(str(arrays["meta"]))
    goal = tuple(int(v) for v in arrays["goals"][0])
    obstacle = np.asarray(arrays["obstacle"], dtype=np.int8)
    return Terrain(blocked=obstacle != NONE, start=goal, goal=goal, diagonal=True,
                   height=np.asarray(arrays["height"], dtype=float), surface=np.asarray(arrays["surface"], dtype=np.int8),
                   obstacle=obstacle, cell_size=meta["cell_size"], max_grade=meta["max_grade"],
                   uphill_penalty=meta["uphill_penalty"], surface_costs=tuple(float(c) for c in arrays["surface_costs"]),
                   sea_level=meta["sea_level"], meta=dict(theme=meta["theme"], seed=meta["seed"]))


def _engine(engine: str) -> str:
    if engine == "auto":
        from .. import cpp
        return "cpp" if cpp.available() else "python"
    return engine


def label_map(theme: str, seed: int, size: int = SIZE, goals: int = GOALS, engine: str = "auto") -> dict:
    """One map of the data set, as the arrays that go in its file."""
    t = make_map(theme, size=size, seed=seed)
    region = largest_region(t)
    picked = pick_goals(t, region, goals, theme, seed)
    if _engine(engine) == "cpp":
        from .. import cpp
        cmap = cpp.to_cpp(t)
        fields = [cpp.cost_to_go(t, g, cmap=cmap) for g in picked]
    else:
        fields = [cost_to_go(t, g) for g in picked]
    meta = dict(format=FORMAT, theme=theme, seed=seed, size=size, cell_size=float(t.cell_size),
                max_grade=float(t.max_grade), uphill_penalty=float(t.uphill_penalty),
                sea_level=None if t.sea_level is None else float(t.sea_level),
                surface_names=list(t.surface_names))
    return dict(height=np.asarray(t.height, dtype=np.float64), surface=np.asarray(t.surface, dtype=np.int8),
                obstacle=np.asarray(t.obstacle, dtype=np.int8),
                surface_costs=np.asarray(t.surface_costs, dtype=np.float64),
                goals=np.asarray(picked, dtype=np.int16).reshape(-1, 2), region=region,
                cost_to_go=np.stack(fields).astype(np.float32), meta=np.array(json.dumps(meta, sort_keys=True)))


def _entry(theme: str, seed: int, arrays: dict, file: str) -> dict:
    f = arrays["cost_to_go"]
    return dict(id=f"{theme}-{seed}", theme=theme, seed=seed, split=split_of(seed), file=file,
                goals=arrays["goals"].tolist(), open=round(float((arrays["obstacle"] == NONE).mean()), 4),
                region=round(float(arrays["region"].mean()), 4), labelled=int(np.isfinite(f).sum()),
                sha256=content_hash(arrays))


def _build_one(task) -> dict:
    theme, seed, size, goals, engine, out = task
    arrays = label_map(theme, seed, size, goals, engine)
    file = f"{theme}-{seed}.npz"
    np.savez_compressed(Path(out) / file, **arrays)
    return _entry(theme, seed, arrays, file)


def dataset_hash(entries) -> str:
    h = hashlib.sha256()
    for e in entries:
        h.update(f"{e['id']}|{e['split']}|{e['sha256']}\n".encode())
    return h.hexdigest()


def build(out: Path = DEFAULT_DIR, per_theme: int = PER_THEME, goals: int = GOALS, size: int = SIZE,
          first_seed: int = FIRST_SEED, jobs: int = 1, engine: str = "auto", log=print, command: str = "") -> dict:
    """Build the data set into `out`: one file per map and manifest.json. Returns the manifest."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    engine = _engine(engine)
    demo = demo_seeds(size)
    seeds = list(range(first_seed, first_seed + per_theme))
    for theme in THEMES:
        if demo[theme] in seeds:
            raise ValueError(f"seed {demo[theme]} is the viewer's {theme} map: start the data set's seeds elsewhere")
    tasks = [(theme, s, size, goals, engine, str(out)) for theme in THEMES for s in seeds]
    t0 = time.perf_counter()
    entries = []
    ctx = get_context("fork") if os.name == "posix" else get_context()
    with ctx.Pool(max(1, jobs)) as pool:
        for k, e in enumerate(pool.imap(_build_one, tasks), 1):
            entries.append(e)
            if log and (k % 25 == 0 or k == len(tasks)):
                log(f"  {k}/{len(tasks)} maps  ({time.perf_counter() - t0:.0f} s)")
    counts = {sp: {th: sum(1 for e in entries if e["split"] == sp and e["theme"] == th) for th in THEMES}
              for sp in SPLITS}
    labelled = {sp: sum(e["labelled"] for e in entries if e["split"] == sp) for sp in SPLITS}
    manifest = dict(format=FORMAT, name=NAME, built_with=command or "robotnav.learn.dataset.build",
                    engine={"cpp": "the C++ core", "python": "the Python reference"}[engine],
                    size=size, goals_per_map=goals, per_theme=per_theme, first_seed=first_seed,
                    themes=list(THEMES), split_rule=SPLIT_RULE, label=LABEL, demo_maps_excluded=demo,
                    counts=counts, labelled=labelled, maps=entries, sha256=dataset_hash(entries),
                    build_seconds=round(time.perf_counter() - t0, 1))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


def load_manifest(directory: Path = DEFAULT_DIR) -> dict:
    path = Path(directory) / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"no data set in {directory}: build it first (python main.py dataset build)")
    return json.loads(path.read_text())


def load(directory: Path, entry: dict) -> dict:
    """One map's arrays."""
    with np.load(Path(directory) / entry["file"], allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def check(directory: Path = DEFAULT_DIR, audit: int = 3, log=print) -> list:
    """Everything that must hold for the data set; returns the problems found (empty when all is well).

    Every file is present and matches its checksum; every map's split follows from its seed; no map appears twice;
    the viewer's maps are not in it; and on `audit` maps of each split, every field is re-derived from the stored map
    by the Python reference and must equal the stored labels exactly."""
    directory = Path(directory)
    m = load_manifest(directory)
    problems = []
    ids = [e["id"] for e in m["maps"]]
    if len(set(ids)) != len(ids):
        problems.append("a map appears more than once")
    for e in m["maps"]:
        if e["split"] != split_of(e["seed"]):
            problems.append(f"{e['id']}: in {e['split']}, but its seed puts it in {split_of(e['seed'])}")
        if m["demo_maps_excluded"].get(e["theme"]) == e["seed"]:
            problems.append(f"{e['id']} is one of the viewer's maps")
        path = directory / e["file"]
        if not path.exists():
            problems.append(f"{e['file']} is missing")
            continue
        if content_hash(load(directory, e)) != e["sha256"]:
            problems.append(f"{e['file']} doesn't match its checksum")
    if dataset_hash(m["maps"]) != m["sha256"]:
        problems.append("the manifest's overall checksum doesn't match its maps")
    audited = 0
    for sp in SPLITS:
        for e in [e for e in m["maps"] if e["split"] == sp][:audit]:
            if not (directory / e["file"]).exists():
                continue
            arrays = load(directory, e)
            t = terrain_of(arrays)
            for k, g in enumerate(arrays["goals"]):
                exact = cost_to_go(t, tuple(int(v) for v in g))
                if not np.array_equal(exact.astype(np.float32), arrays["cost_to_go"][k]):
                    problems.append(f"{e['id']} goal {k}: the stored labels differ from the Python reference")
                audited += 1
    if log:
        log(f"{len(m['maps'])} maps checked against their checksums; {audited} fields re-derived by the Python "
            f"reference; {len(problems)} problem(s)")
    return problems


def rebuilt_from_seed(directory: Path, entry: dict) -> bool:
    """Whether this machine's generator gives, bit for bit, the map stored for this seed (see the module notes)."""
    arrays = load(directory, entry)
    t = make_map(entry["theme"], size=int(arrays["height"].shape[0]), seed=int(entry["seed"]))
    return bool(np.array_equal(t.height, arrays["height"]) and np.array_equal(t.obstacle, arrays["obstacle"])
                and np.array_equal(t.surface, arrays["surface"]))
