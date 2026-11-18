"""Stage 3: Bruce learns his own terrain speeds from his driving.

Bruce's planners use a drive-time model (robotnav.terrain): 3-D distance x a slowdown for the kind of ground x
(1 + 3 x the uphill grade). On real ground that model is only a guess. Here the world has its own, hidden, true
speeds, and they differ from Bruce's model in ways a real robot meets:

  - each kind of ground has its own true slowdown (tall grass is quicker than he thinks, soft sand and jungle slower)
  - climbing costs more on loose ground (sand, mud-like jungle) than on firm ground
  - steep descents (beyond 25%) need braking
  - a side slope (across the direction of travel) slows him down, as it would a real tracked vehicle
  - rough, bumpy ground slows him down

Bruce never sees that model. He drives trips planned with what he currently believes, and his log records, for
every move, what he could sense (the kind of ground, from his camera; the grade, side slope and roughness, from his
height map) and how long the move really took, with 8% sensor noise. A small network (PyTorch) learns the time per
metre from those features, and Bruce plans his next trips with what it learned.

The score is what matters to a robot: the route Bruce picks, driven in the true world, against the quickest route
the true world allows (its "regret"), on test maps he has never driven. The hidden model is a simulation, made up
for this study; what is real is that Bruce has to learn it from noisy logs of his own driving.

Stage 3b fixes stage 3's two weak spots. Ground Bruce never drives (the island's beach sand) he can't learn: so he
explores (Tally, Explore): cheap detours through sorts of move he is unsure of, and a scouting trip in every ten to
the ground he is least sure of. And the network under-fitted its rarest, steepest moves even on its own log: a
bigger one (64 x 64, 6,000 steps instead of 32 x 32, 3,000) fits them. `run` measures both fixes apart: four
careers on the same missions, plain or exploring, with the small network or the bigger one.

Moves are tables: costs[cell, k] is the time of move k (the planners' order: up, down, left, right, then the
diagonals) out of the cell, inf where it can't be made. Every time is at least the move's flat length, so the
straight-line estimate stays a lower bound and A* stays optimal on any of them. The C++ core plans on such tables
(cpp/src/fields.cpp: astar_costs, cost_to_go_costs), cell for cell as this module's reference does.
"""
from __future__ import annotations

import heapq
import json
import time
from dataclasses import dataclass, field
from itertools import count
from pathlib import Path

import numpy as np

from ..algorithms import SearchResult, _best_first
from ..grid import EIGHT
from ..terrain import SQRT2, Terrain

KINDS = ("Grass", "Tall grass", "Firm sand", "Soft sand", "Rocky ground", "Oasis grass", "Sand", "Jungle")
# The hidden true world: slowdown on flat ground, and the extra time per unit of uphill grade, by kind of ground.
TRUE_SLOW = {"Grass": 1.0, "Tall grass": 1.4, "Firm sand": 1.15, "Soft sand": 2.9, "Rocky ground": 1.7,
             "Oasis grass": 1.05, "Sand": 1.9, "Jungle": 2.6}
TRUE_UPHILL = {"Grass": 2.5, "Tall grass": 3.0, "Firm sand": 4.0, "Soft sand": 6.0, "Rocky ground": 3.0,
               "Oasis grass": 2.5, "Sand": 5.0, "Jungle": 3.5}
BRAKE_FROM, BRAKE = 0.25, 2.0          # descents steeper than 25%: extra time per unit of grade beyond it
SIDE = 1.5                             # x (1 + 1.5 x side slope^2)
ROUGH = 1.0                            # x (1 + roughness in metres)
NOISE = 0.08                           # a logged move time is off by about 8% (lognormal)
FEATURES = 6 + len(KINDS)
NAMES = ("diagonal", "grade", "uphill grade", "downhill grade", "side slope", "roughness", *(f"ground: {k}" for k in KINDS))
ASSET = Path(__file__).resolve().parents[1] / "assets" / "speeds.txt"


# ---------------------------------------------------------------- the moves of a map
@dataclass
class Moves:
    """Every move (cell, k) of a map: whether it can be made, where it lands, and what Bruce can sense of it."""
    legal: np.ndarray        # (n, 8) bool
    dest: np.ndarray         # (n, 8) flat index of the cell it lands on (its own cell where off the map)
    d3: np.ndarray           # (n, 8) 3-D length (m)
    grade: np.ndarray        # (n, 8) rise / run
    diag: np.ndarray         # (n, 8) bool
    side: np.ndarray         # (n, 8) side slope at the landing cell, across the move
    rough: np.ndarray        # (n,) roughness of each cell (m)
    kind: np.ndarray         # (n,) index into KINDS of each cell's ground
    surface: np.ndarray      # (n,) the map's own surface index (for Bruce's built-in model)


def moves(t: Terrain) -> Moves:
    R, C, n = t.rows, t.cols, t.rows * t.cols
    H = np.asarray(t.height, dtype=float)
    B = np.asarray(t.blocked, dtype=bool)
    rr, cc = np.mgrid[0:R, 0:C]
    # the height map's slope at every cell (central differences, one-sided at the edges), per metre
    cl, cr_ = np.maximum(cc - 1, 0), np.minimum(cc + 1, C - 1)
    ru, rd = np.maximum(rr - 1, 0), np.minimum(rr + 1, R - 1)
    gx = (H[rr, cr_] - H[rr, cl]) / ((cr_ - cl) * t.cell_size)
    gy = (H[rd, cc] - H[ru, cc]) / ((rd - ru) * t.cell_size)
    rough = np.abs(H - (H[ru, cc] + H[rd, cc] + H[rr, cl] + H[rr, cr_]) / 4.0)
    legal = np.zeros((n, 8), dtype=bool)
    dest = np.zeros((n, 8), dtype=np.int64)
    d3 = np.zeros((n, 8))
    grade = np.zeros((n, 8))
    diag = np.zeros((n, 8), dtype=bool)
    side = np.zeros((n, 8))
    for k, (dr, dc) in enumerate(EIGHT):
        r2, c2 = rr + dr, cc + dc
        inside = (r2 >= 0) & (r2 < R) & (c2 >= 0) & (c2 < C)
        r2, c2 = np.clip(r2, 0, R - 1), np.clip(c2, 0, C - 1)
        ok = inside & ~B & ~B[r2, c2]
        if dr and dc:
            ok &= ~B[np.clip(rr + dr, 0, R - 1), cc] & ~B[rr, np.clip(cc + dc, 0, C - 1)]
        run = (SQRT2 if dr and dc else 1.0) * t.cell_size
        rise = H[r2, c2] - H
        g = rise / run
        ok &= np.abs(g) <= t.max_grade
        legal[:, k] = ok.ravel()
        dest[:, k] = np.where(inside, r2 * C + c2, rr * C + cc).ravel()
        d3[:, k] = np.sqrt(run * run + rise * rise).ravel()
        grade[:, k] = g.ravel()
        diag[:, k] = bool(dr and dc)
        norm = np.sqrt(dr * dr + dc * dc)
        side[:, k] = (np.abs(gx[r2, c2] * dr - gy[r2, c2] * dc) / norm).ravel()
    kinds = np.array([KINDS.index(name) for name in t.surface_names])
    surface = np.asarray(t.surface).ravel().astype(np.int64)
    return Moves(legal, dest, d3, grade, diag, side, rough.ravel(), kinds[surface], surface)


def nominal_costs(t: Terrain, mv: Moves | None = None) -> np.ndarray:
    """Bruce's built-in model, as a table: exactly the planners' drive times (Terrain.edge_cost)."""
    mv = mv or moves(t)
    s = np.asarray(t.surface_costs, dtype=float)[mv.surface[mv.dest]]
    cost = mv.d3 * s * (1.0 + t.uphill_penalty * np.maximum(mv.grade, 0.0))
    return np.where(mv.legal, cost, np.inf)


def true_costs(t: Terrain, mv: Moves | None = None) -> np.ndarray:
    """The hidden true world's move times."""
    mv = mv or moves(t)
    k = mv.kind[mv.dest]
    slow = np.array([TRUE_SLOW[x] for x in KINDS])[k]
    up = np.array([TRUE_UPHILL[x] for x in KINDS])[k]
    g = mv.grade
    per_m = (slow * (1.0 + up * np.maximum(g, 0.0) + BRAKE * np.maximum(-g - BRAKE_FROM, 0.0))
             * (1.0 + SIDE * mv.side * mv.side) * (1.0 + ROUGH * mv.rough[mv.dest]))
    return np.where(mv.legal, mv.d3 * per_m, np.inf)


def features(mv: Moves) -> np.ndarray:
    """What Bruce can sense of every move, (n, 8, FEATURES): the network's inputs."""
    n = mv.legal.shape[0]
    x = np.zeros((n, 8, FEATURES))
    x[..., 0] = mv.diag
    x[..., 1] = mv.grade
    x[..., 2] = np.maximum(mv.grade, 0.0)
    x[..., 3] = np.maximum(-mv.grade, 0.0)
    x[..., 4] = mv.side
    x[..., 5] = mv.rough[mv.dest]
    x[np.arange(n)[:, None], np.arange(8)[None, :], 6 + mv.kind[mv.dest]] = 1.0
    return x


# ---------------------------------------------------------------- the network: seconds per metre of a move
@dataclass
class SpeedModel:
    mean: np.ndarray
    scale: np.ndarray
    layers: list                                   # [(W (in, out), b (out,)), ...]
    note: list = field(default_factory=list)

    def per_metre(self, x: np.ndarray) -> np.ndarray:
        """Seconds per metre for rows of features: 1 + max(z, 0), never quicker than the best ground."""
        a = (np.asarray(x, dtype=float) - self.mean) / self.scale
        for i, (W, b) in enumerate(self.layers):
            z = np.empty((len(a), W.shape[1]))
            z[:] = b
            for j in range(W.shape[0]):                # inputs in order, plain + (as model.py: a fixed order of sums)
                z += a[:, j:j + 1] * W[j]
            a = np.maximum(z, 0.0) if i + 1 < len(self.layers) else z
        return 1.0 + np.maximum(a[:, 0], 0.0)

    def costs(self, t: Terrain, mv: Moves | None = None) -> np.ndarray:
        """What Bruce believes every move takes, as a table."""
        mv = mv or moves(t)
        x = features(mv)
        out = np.full(mv.legal.shape, np.inf)
        out[mv.legal] = mv.d3[mv.legal] * self.per_metre(x[mv.legal])
        return out

    def text(self) -> str:
        r = lambda v: repr(float(v))
        lines = [f"# {n}" for n in self.note] + ["brubus-speeds 1", f"inputs {FEATURES}",
                                                 "mean " + " ".join(map(r, self.mean)),
                                                 "scale " + " ".join(map(r, self.scale))]
        for W, b in self.layers:
            lines.append(f"layer {W.shape[0]} {W.shape[1]}")
            lines += [" ".join(map(r, row)) for row in W]
            lines.append(" ".join(map(r, b)))
        return "\n".join(lines + ["end"]) + "\n"

    def save(self, path: Path = ASSET) -> Path:
        Path(path).write_text(self.text())
        return Path(path)


def load(path: Path = ASSET) -> SpeedModel:
    """Read a speeds file (the stage-2 network's plain-text layout, with its own header)."""
    from .model import parse
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no learned speeds at {path}: learn them (python main.py speeds learn)")
    m = parse(path.read_text(), header="brubus-speeds", inputs=FEATURES)
    return SpeedModel(m.mean, m.scale, m.layers, m.note)


# ---------------------------------------------------------------- planning on a table (the Python reference)
class _CostsGrid:
    def __init__(self, t: Terrain, costs: np.ndarray, start, goal):
        self.t, self.costs, self.start, self.goal = t, costs, tuple(start), tuple(goal)

    def neighbors(self, c):
        i = c[0] * self.t.cols + c[1]
        for k, (dr, dc) in enumerate(EIGHT):
            v = self.costs[i, k]
            if v < np.inf:
                yield (c[0] + dr, c[1] + dc), float(v)

    def heuristic(self, a, b):
        return self.t.heuristic(a, b)

    def path_cost(self, path):
        return route_time(self.t, self.costs, path)


def route_time(t: Terrain, costs: np.ndarray, path) -> float:
    """A route's time on a table of move times, added up from its start."""
    total = 0.0
    for a, b in zip(path, path[1:]):
        total += float(costs[a[0] * t.cols + a[1], EIGHT.index((b[0] - a[0], b[1] - a[1]))])
    return total


def astar_costs(t: Terrain, costs: np.ndarray, start, goal, name: str = "A* (move times)") -> SearchResult:
    return _best_first(_CostsGrid(t, costs, start, goal), name, lambda g, h: g + h)


def cost_to_go_costs(t: Terrain, costs: np.ndarray, goal) -> np.ndarray:
    """Fewest seconds from every cell to `goal` on a table of move times (Dijkstra backwards)."""
    R, C = t.rows, t.cols
    g = (int(goal[0]), int(goal[1]))
    dist = np.full((R, C), np.inf)
    done = np.zeros((R, C), dtype=bool)
    dist[g] = 0.0
    tie = count()
    frontier = [(0.0, next(tie), g)]
    while frontier:
        _, _, u = heapq.heappop(frontier)
        if done[u]:
            continue
        done[u] = True
        du = float(dist[u])
        for k, (dr, dc) in enumerate(EIGHT):
            p = (u[0] - dr, u[1] - dc)
            if not (0 <= p[0] < R and 0 <= p[1] < C) or done[p]:
                continue
            step = costs[p[0] * C + p[1], k]
            if not step < np.inf:
                continue
            d = du + float(step)
            if d < dist[p]:
                dist[p] = d
                heapq.heappush(frontier, (d, next(tie), p))
    return dist


# ---------------------------------------------------------------- what Bruce has driven, and exploring
GRADE_EDGES = (-0.4, -0.25, -0.1, 0.1, 0.25, 0.4)      # 7 bands of grade, downhill to uphill
SIDE_EDGES = (0.25, 0.5, 0.8)                           # 4 bands of side slope
ROUGH_EDGES = (0.07, 0.15)                              # 3 bands of roughness (m)
BINS = len(KINDS) * (len(GRADE_EDGES) + 1) * (len(SIDE_EDGES) + 1) * (len(ROUGH_EDGES) + 1)


class Tally:
    """How many moves Bruce has logged of each sort: kind of ground x grade x side slope x roughness (672 bins).
    A move of a sort he has rarely driven is one he is unsure of: its novelty is 1 / sqrt(1 + moves logged in its
    bin), 1 for a sort he has never driven (the count-based bonus of UCB-style exploration)."""

    def __init__(self):
        self.counts = np.zeros(BINS, dtype=np.int64)

    @staticmethod
    def bin_of(x: np.ndarray) -> np.ndarray:
        kind = np.argmax(x[..., 6:], axis=-1)
        g = np.searchsorted(GRADE_EDGES, x[..., 1], side="right")
        s = np.searchsorted(SIDE_EDGES, x[..., 4], side="right")
        r = np.searchsorted(ROUGH_EDGES, x[..., 5], side="right")
        return ((kind * (len(GRADE_EDGES) + 1) + g) * (len(SIDE_EDGES) + 1) + s) * (len(ROUGH_EDGES) + 1) + r

    def add(self, x: np.ndarray) -> None:
        np.add.at(self.counts, self.bin_of(x), 1)

    def novelty(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / np.sqrt(1.0 + self.counts[self.bin_of(x)])


@dataclass
class Explore:
    """Exploring. On a mission Bruce plans twice: the quickest route by what he believes, and a hopeful route on
    which every move's believed time above its bare length is cut by pull x its novelty (a sort of move he has
    never driven looks as quick as the best ground: optimism in the face of uncertainty). He drives the hopeful
    route only if he believes it costs at most `budget` more than the quickest; otherwise he halves the pull and
    plans again, and after `tries` plans he drives the quickest route. And every `scout_every`-th trip (0: never)
    he goes scouting instead of driving the mission: from its start to the nearest patch of the ground he is least
    sure of (scout_target), on the hopeful route."""
    budget: float = 0.10
    pull: float = 1.0
    tries: int = 3
    scout_every: int = 10
    scout_near: float = 1.0          # scout the nearest patch at least this share as unfamiliar as the most unfamiliar
                                     # (1: the most unfamiliar; 0.8 cost half as much and taught less, on validation)


def novelty(mv: Moves, x: np.ndarray, tally: Tally) -> np.ndarray:
    """Every legal move's novelty, as a table (0 where a move can't be made)."""
    u = np.zeros(mv.legal.shape)
    u[mv.legal] = tally.novelty(x[mv.legal])
    return u


def hopeful_costs(mv: Moves, believed: np.ndarray, u: np.ndarray, pull: float) -> np.ndarray:
    """The believed times, with the part above each move's bare 3-D length cut by pull x its novelty: never less
    than that length, so A* stays exact on them too."""
    return np.where(mv.legal, mv.d3 + (believed - mv.d3) * (1.0 - pull * u), np.inf)


def explore_route(t: Terrain, mv: Moves, x: np.ndarray, believed: np.ndarray, quickest: list, tally: Tally,
                  ex: Explore, start, goal, plan=None, cmap=None) -> list:
    plan, cmap = _ready(t, plan, cmap)
    limit = (1.0 + ex.budget) * route_time(t, believed, quickest)
    u = novelty(mv, x, tally)
    pull = ex.pull
    for _ in range(ex.tries):
        route = plan(t, cmap, hopeful_costs(mv, believed, u, pull), start, goal)
        if route_time(t, believed, route) <= limit:
            return route
        pull /= 2.0
    return quickest


SCOUT_WINDOW = 3                                        # a patch: the cells within 3 of a cell (7 x 7)
SCOUT_FROM = 10.0                                       # metres from the start, at least


def scout_target(t: Terrain, mv: Moves, x: np.ndarray, tally: Tally, region: np.ndarray, start,
                 near: float = 1.0) -> tuple:
    """Where to scout: of the region's cells at least SCOUT_FROM metres from the start, those whose patch holds at
    least `near` x the most novelty there is (each cell counts its most novel move), and of those the nearest
    (then the first)."""
    u = novelty(mv, x, tally).max(axis=1).reshape(t.rows, t.cols)
    w, pad = SCOUT_WINDOW, np.pad(u, SCOUT_WINDOW)
    patch = sum(pad[w + dr:w + dr + t.rows, w + dc:w + dc + t.cols] for dr in range(-w, w + 1) for dc in range(-w, w + 1))
    rr, cc = np.mgrid[0:t.rows, 0:t.cols]
    dist = np.hypot(rr - start[0], cc - start[1]) * t.cell_size
    ok = np.asarray(region, dtype=bool) & (dist >= SCOUT_FROM)
    if not ok.any():
        ok = np.asarray(region, dtype=bool)
    score = np.where(ok, patch, -np.inf)
    top = np.flatnonzero(score.ravel() >= near * score.max() - 1e-9)
    best = top[np.argmin(dist.ravel()[top])]
    return int(best // t.cols), int(best % t.cols)


# ---------------------------------------------------------------- driving, learning and scoring
def _planner():
    """C++ for the many searches when it's built, else this module's reference (same routes, slower)."""
    from .. import cpp
    if cpp.available():
        core = cpp.core()

        def plan(t, cmap, costs, start, goal):
            r = core.astar_costs(cmap, costs, start[0] * t.cols + start[1], goal[0] * t.cols + goal[1], "a")
            return [divmod(v, t.cols) for v in r.path]

        def field(t, cmap, costs, goal):
            return core.cost_to_go_costs(cmap, costs, goal[0] * t.cols + goal[1])
        return plan, field, cpp.to_cpp
    return (lambda t, cmap, costs, s, g: astar_costs(t, costs, s, g).path,
            lambda t, cmap, costs, g: cost_to_go_costs(t, costs, g), lambda t: None)


def _ready(t: Terrain, plan, cmap):
    """A planner and its map: the ones given, or the fastest there is (with the C++ core, the map in its form)."""
    if plan is None:
        plan, _, to_cmap = _planner()
        cmap = to_cmap(t)
    return plan, cmap


def drive(t: Terrain, mv: Moves, believed: np.ndarray, true: np.ndarray, start, goal, rng, plan=None, cmap=None):
    """Plan with what Bruce believes, drive it in the true world: (features of every move driven, seconds per metre
    each took, with sensor noise), and the route."""
    plan, cmap = _ready(t, plan, cmap)
    route = plan(t, cmap, believed, start, goal)
    return (*log_route(t, mv, features(mv), true, route, rng), route)


def log_route(t: Terrain, mv: Moves, x: np.ndarray, true: np.ndarray, route: list, rng):
    """Drive a route in the true world: the features of every move and the seconds per metre each took (noisy)."""
    rows, ys = [], []
    for a, b in zip(route, route[1:]):
        i, k = a[0] * t.cols + a[1], EIGHT.index((b[0] - a[0], b[1] - a[1]))
        rows.append(x[i, k])
        ys.append(true[i, k] / mv.d3[i, k] * float(np.exp(NOISE * rng.normal())))
    return np.array(rows).reshape(-1, FEATURES), np.array(ys)


def _trip(t, region, goals, rng, min_m: float = 20.0):
    """A random goal of the map and a random start at least min_m metres (straight line) from it."""
    goal = tuple(int(v) for v in goals[rng.integers(len(goals))])
    cells = np.argwhere(region)
    d = np.hypot(cells[:, 0] - goal[0], cells[:, 1] - goal[1])
    far = cells[d >= min(min_m, d.max())]                       # (on a small map: the farthest cells there are)
    return tuple(int(v) for v in far[rng.integers(len(far))]), goal


def fit(x, y, hidden=(64, 64), steps: int = 6000, batch: int = 512, lr: float = 3e-3, seed: int = 0) -> SpeedModel:
    """Learn seconds per metre from logged moves (PyTorch): relative squared error, predict 1 + z (clamped at 1)."""
    import torch
    torch.manual_seed(seed)
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale == 0] = 1.0
    X = torch.tensor((x - mean) / scale, dtype=torch.float32)
    Y = torch.tensor(y, dtype=torch.float32)
    sizes = [FEATURES, *hidden, 1]
    layers = []
    for a, b in zip(sizes, sizes[1:]):
        layers += [torch.nn.Linear(a, b), torch.nn.ReLU()]
    net = torch.nn.Sequential(*layers[:-1])
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    gen = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        idx = torch.randint(0, len(X), (min(batch, len(X)),), generator=gen)
        pred = 1.0 + net(X[idx]).squeeze(1)
        loss = (((pred - Y[idx]) / Y[idx]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    linears = [m for m in net if isinstance(m, torch.nn.Linear)]
    return SpeedModel(mean.astype(float), scale.astype(float),
                      [(L.weight.detach().double().numpy().T.copy(), L.bias.detach().double().numpy().copy())
                       for L in linears])


@dataclass
class Career:
    curve: list              # [(trips, moves logged, model)] at each checkpoint
    x: np.ndarray            # the log: features of every move driven (m, FEATURES)
    y: np.ndarray            # and the seconds per metre each took (m,)
    missions: list           # [(map id, start, goal)] in the order driven
    tally: Tally
    driven: float = 0.0      # true seconds of the missions he drove
    quickest: float = 0.0    # true seconds of the routes he believed quickest (the same unless exploring)
    explored: int = 0        # missions on which he drove a hopeful route
    scouted: int = 0         # scouting trips
    scouting: float = 0.0    # and their true seconds


def learn(directory=None, checkpoints=(10, 30, 100, 300), seed: int = 0, log=print, explore: Explore | None = None,
          missions: list | None = None, hidden=(64, 64), steps: int = 6000) -> Career:
    """Bruce's driving career on the train maps. Trips up to the first checkpoint are planned with his built-in
    model; after each checkpoint a network is trained on every move logged so far, and the next trips are planned
    with it. With `explore`, he takes an exploring route when it is cheap (Explore); with `missions` (another
    career's), he drives those same trips (the same maps, starts and goals) in the same order."""
    from .dataset import DEFAULT_DIR, load, load_manifest, terrain_of
    directory = directory or DEFAULT_DIR
    m = load_manifest(directory)
    maps = [e for e in m["maps"] if e["split"] == "train"]
    by_id = {e["id"]: e for e in maps}
    rng = np.random.default_rng(seed)              # the trips (unless given) and the sensor noise
    plan, _, to_cmap = _planner()
    c = Career([], np.zeros((0, FEATURES)), np.zeros(0), [], Tally())
    xs, ys, model, done, t0 = [], [], None, 0, time.perf_counter()
    for n in checkpoints:
        while done < n:
            if missions is None:
                e = maps[rng.integers(len(maps))]
            else:
                e = by_id[missions[done][0]]
            arrays = load(directory, e)
            t = terrain_of(arrays)
            mv = moves(t)
            believed = nominal_costs(t, mv) if model is None else model.costs(t, mv)
            if missions is None:
                start, goal = _trip(t, arrays["region"], arrays["goals"], rng)
            else:
                start, goal = (tuple(int(v) for v in p) for p in missions[done][1:])
            c.missions.append((e["id"], start, goal))
            cmap, x, true = to_cmap(t), features(mv), true_costs(t, mv)
            if explore is not None and explore.scout_every and (done + 1) % explore.scout_every == 0:
                target = scout_target(t, mv, x, c.tally, arrays["region"], start, explore.scout_near)
                route = plan(t, cmap, hopeful_costs(mv, believed, novelty(mv, x, c.tally), explore.pull), start, target)
                c.scouted += 1
                c.scouting += route_time(t, true, route)
            else:
                route = quickest = plan(t, cmap, believed, start, goal)
                if explore is not None:
                    route = explore_route(t, mv, x, believed, quickest, c.tally, explore, start, goal, plan, cmap)
                    c.explored += route != quickest
                c.driven += route_time(t, true, route)
                c.quickest += route_time(t, true, quickest)
            xr, yr = log_route(t, mv, x, true, route, rng)
            c.tally.add(xr)
            xs.append(xr); ys.append(yr)
            done += 1
        X, Y = np.concatenate(xs), np.concatenate(ys)
        model = fit(X, Y, hidden=hidden, steps=steps, seed=seed)
        how = "exploring" if explore is not None else "planning with what he believed"
        model.note = [f"Brubus stage 3: Bruce's learned seconds per metre after {n} trips ({len(Y):,} logged moves, "
                      f"{how}) on the data set's train maps; network {[FEATURES, *hidden, 1]}, ReLU, {steps:,} "
                      f"steps; PyTorch."]
        c.curve.append((n, len(Y), model))
        if log:
            log(f"  {n:>4} trips, {len(Y):>6,} moves logged: trained ({time.perf_counter() - t0:.0f} s)")
    c.x, c.y = np.concatenate(xs), np.concatenate(ys)
    return c


def evaluate(models: dict, directory=None, split: str = "test", starts: int = 2, seed: int = 7, log=print) -> list:
    """Plan the same trips with each model (name -> SpeedModel, "built-in", "true", or any function (terrain, moves)
    -> table of move times) on the split's maps, drive
    them in the true world: one record per trip and model with the route's true time and the quickest true time."""
    from .baseline import _trips
    from .dataset import DEFAULT_DIR, load, load_manifest, terrain_of
    directory = directory or DEFAULT_DIR
    m = load_manifest(directory)
    rng = np.random.default_rng(seed)
    plan, field_of, to_cmap = _planner()
    rows, t0 = [], time.perf_counter()
    for e in [e for e in m["maps"] if e["split"] == split]:
        arrays = load(directory, e)
        t = terrain_of(arrays)
        mv = moves(t)
        cmap = to_cmap(t)
        true = true_costs(t, mv)
        tables = {name: (nominal_costs(t, mv) if mdl == "built-in" else true if mdl == "true"
                         else mdl(t, mv) if callable(mdl) else mdl.costs(t, mv)) for name, mdl in models.items()}
        for g in arrays["goals"]:
            goal = (int(g[0]), int(g[1]))
            best = field_of(t, cmap, true, goal)
            for s in _trips(best, starts, 20.0, rng):
                for name, costs in tables.items():
                    route = plan(t, cmap, costs, s, goal)
                    actual = route_time(t, true, route)
                    regret = actual / float(best[s]) - 1.0              # the same sum added the other way: last bits
                    rows.append(dict(theme=e["theme"], map=e["id"], goal=goal, start=s, model=name, time=actual,
                                     best=float(best[s]), regret=regret if abs(regret) > 1e-12 else 0.0))
    if log:
        log(f"  {len(rows) // max(1, len(models)):,} trips on the {split} maps ({time.perf_counter() - t0:.0f} s)")
    return rows


def summarize(rows: list) -> list:
    out = []
    for theme in ("grassland", "desert", "island", "all"):
        for name in dict.fromkeys(r["model"] for r in rows):
            r = np.array([x["regret"] for x in rows if x["model"] == name and theme in (x["theme"], "all")])
            if len(r):
                out.append(dict(theme=theme, model=name, trips=len(r), regret=float(r.mean()),
                                median=float(np.median(r)), worst=float(r.max()), within_1=float(np.mean(r < 0.01))))
    return out


def what_it_learned(model: SpeedModel, logged: np.ndarray | None = None) -> list:
    """Per kind of ground: seconds per metre on flat, smooth ground, and the extra per unit of a 20% climb,
    learned against the hidden truth and Bruce's built-in model; with Bruce's log (its features), how many moves
    he drove on that ground and how many of them climbed more than 10%."""
    nominal = {"Grass": 1.0, "Tall grass": 1.8, "Firm sand": 1.0, "Soft sand": 2.4, "Rocky ground": 1.3,
               "Oasis grass": 1.0, "Sand": 1.6, "Jungle": 2.0}
    out = []
    for k, name in enumerate(KINDS):
        flat = np.zeros(FEATURES); flat[6 + k] = 1.0
        up = flat.copy(); up[1] = up[2] = 0.2
        f, u = model.per_metre(np.stack([flat, up]))
        out.append(dict(ground=name, built_in=nominal[name], true=TRUE_SLOW[name], learned=float(f),
                        built_in_uphill=3.0, true_uphill=TRUE_UPHILL[name], learned_uphill=float((u / f - 1) / 0.2)))
        if logged is not None:
            on = logged[:, 6 + k] == 1.0
            out[-1].update(logged=int(on.sum()), logged_climbs=int((on & (logged[:, 2] > 0.1)).sum()))
    return out


STAGE3_ASSET = ASSET.with_name("speeds-stage3.txt")
SMALL, BIGGER = ((32, 32), 3000), ((64, 64), 6000)            # (hidden layers, training steps)
CAREERS = (("plain, small network", False, SMALL), ("plain, bigger network", False, BIGGER),
           ("exploring, small network", True, SMALL), ("exploring, bigger network", True, BIGGER))


def run(directory=None, checkpoints=(10, 30, 100, 300), seed: int = 0, out: Path = ASSET, log=print) -> dict:
    """The whole study: four driving careers on the same missions (CAREERS: planning with what he believes or
    exploring, with stage 3's small network or the bigger one), every checkpoint's network scored on the test maps
    against his built-in model and the true speeds. Saves the last network of the exploring, bigger-network career
    (`out`) and of stage 3's career (speeds-stage3.txt beside it), and a card holding the numbers."""
    import torch
    t0 = time.perf_counter()
    careers, missions = {}, None
    for name, exploring, (hidden, steps) in CAREERS:
        if log:
            log(f"{name}:")
        c = learn(directory, checkpoints, seed, log, Explore() if exploring else None, missions, hidden, steps)
        missions = missions or c.missions
        careers[name] = c
    models = {"built-in": "built-in"}
    for name, c in careers.items():
        models.update({f"{name}, {n} trips": m for n, _, m in c.curve})
    models["true"] = "true"
    rows = evaluate(models, directory, log=log)
    final, stage3 = careers[CAREERS[-1][0]].curve[-1][2], careers[CAREERS[0][0]].curve[-1][2]
    final.save(out)
    stage3.save(Path(out).with_name(STAGE3_ASSET.name))

    def describe(name, exploring, hidden, steps, c):
        return dict(name=name, exploring=exploring, network=[FEATURES, *hidden, 1], steps=steps,
                    curve=[dict(trips=n, moves=k) for n, k, _ in c.curve],
                    sorts_driven=int(np.count_nonzero(c.tally.counts)), hopeful_routes=c.explored,
                    scouting_trips=c.scouted, detours=c.driven / c.quickest - 1.0,
                    scouting_share=c.scouting / (c.driven + c.scouting), learned=what_it_learned(c.curve[-1][2], c.x))
    ex = Explore()
    card = dict(model=Path(out).name, stage3_model=STAGE3_ASSET.name, framework=f"PyTorch {torch.__version__}",
                seed=seed, network=dict(sizes=[FEATURES, *BIGGER[0], 1], steps=BIGGER[1], activation="ReLU",
                                        output="seconds per metre = 1 + max(z, 0)", features=list(NAMES)),
                hidden_world=dict(slowdown=TRUE_SLOW, uphill=TRUE_UPHILL, brake_from=BRAKE_FROM, brake=BRAKE,
                                  side=SIDE, rough=ROUGH, noise=NOISE),
                explore=dict(budget=ex.budget, pull=ex.pull, tries=ex.tries, scout_every=ex.scout_every,
                             scout_near=ex.scout_near, sorts=BINS, grade_edges=list(GRADE_EDGES),
                             side_edges=list(SIDE_EDGES), rough_edges=list(ROUGH_EDGES)),
                careers=[describe(name, e, h, st, careers[name]) for name, e, (h, st) in CAREERS],
                summary=summarize(rows), seconds=round(time.perf_counter() - t0))
    Path(out).with_suffix(".card.json").write_text(json.dumps(card, indent=1) + "\n")
    return card


def table(summary: list, theme: str = "all", models=None) -> str:
    """The summary as text, for one theme (or all), and only the named models if given."""
    rows = [s for s in summary if s["theme"] == theme and (models is None or s["model"] in models)]
    wide = max([len(s["model"]) for s in rows] + [12])
    lines = [f"  {'planned with':<{wide}}{'trips':>7}{'slower than the quickest: mean':>32}{'median':>9}{'worst':>9}"
             f"{'within 1%':>11}"]
    for s in rows:
        lines.append(f"  {s['model']:<{wide}}{s['trips']:>7}{100 * s['regret']:>31.2f}%{100 * s['median']:>8.2f}%"
                     f"{100 * s['worst']:>8.1f}%{100 * s['within_1']:>10.0f}%")
    return "\n".join(lines)

