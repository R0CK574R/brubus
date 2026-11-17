"""The learned estimate: a small network, its file, and A* guided by it (the Python reference).

The network (dense layers, ReLU between them, one linear output z) reads the 16 features of a cell and the goal
(features.py) and says how much longer than the climb-aware estimate the drive really is:

    estimate = climb x (1 + max(z, 0))

so it is never below the climb-aware estimate, which never overestimates. It can overestimate, though, and then A*
may miss the quickest route. `table(..., w)` caps it at w x the climb-aware estimate, which makes it never more
than w times the true time; A* that searches a cell again whenever a quicker way to it turns up (labels.
astar_reopen) then always finds a route at most w times as long as the quickest. w = inf: no cap, no guarantee.

The file (robotnav/assets/heuristic.txt) is plain text: comment lines, then

    brubus-heuristic 1
    inputs 16
    mean <16 numbers>
    scale <16 numbers>
    layer <in> <out>   then in x out weights (row by row) and out biases
    ...
    end

Numbers are written as Python's repr of 64-bit floats, so the C++ core (strtod) and the browser (parseFloat) read
back exactly the same doubles. Inference here adds up each neuron's inputs in order with plain +, as the C++ core
does, so the two agree to the last bit; NumPy's matrix product (BLAS) would add in another order.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..terrain import Terrain
from .features import FEATURES, features
from .labels import astar_reopen

ASSET = Path(__file__).resolve().parents[1] / "assets" / "heuristic.txt"


@dataclass
class Model:
    mean: np.ndarray
    scale: np.ndarray
    layers: list                                   # [(W (in, out), b (out,)), ...] as 64-bit floats
    note: list = field(default_factory=list)       # comment lines: where it came from

    @property
    def sizes(self) -> list:
        return [FEATURES] + [int(W.shape[1]) for W, _ in self.layers]

    def output(self, x: np.ndarray) -> np.ndarray:
        """The network's raw output z for rows of features."""
        a = (np.asarray(x, dtype=float) - self.mean) / self.scale
        for k, (W, b) in enumerate(self.layers):
            z = np.repeat(b[None, :], len(a), axis=0)
            for i in range(W.shape[0]):                # inputs in order: the same sums as the C++ core
                z = z + a[:, i:i + 1] * W[i][None, :]
            a = np.maximum(z, 0.0) if k + 1 < len(self.layers) else z
        return a[:, 0]

    def estimate(self, x: np.ndarray, climb: np.ndarray, w: float = np.inf) -> np.ndarray:
        """climb x (1 + max(z, 0)), capped at w x climb; 0 where climb is 0 (the goal, blocked cells)."""
        live = climb > 0
        est = np.zeros(len(climb))
        if live.any():
            c = climb[live]
            e = c * (1.0 + np.maximum(self.output(x[live]), 0.0))
            est[live] = np.minimum(e, w * c)
        return est

    def table(self, t: Terrain, goal, w: float = np.inf) -> np.ndarray:
        """The estimate for every cell, shape (rows, cols)."""
        x, climb = features(t, goal)
        return self.estimate(x, climb, w).reshape(t.rows, t.cols)

    # ------------------------------------------------------------------ the file
    def text(self) -> str:
        r = lambda v: repr(float(v))
        lines = [f"# {n}" for n in self.note] + ["brubus-heuristic 1", f"inputs {FEATURES}",
                                                 "mean " + " ".join(map(r, self.mean)),
                                                 "scale " + " ".join(map(r, self.scale))]
        for W, b in self.layers:
            lines.append(f"layer {W.shape[0]} {W.shape[1]}")
            lines += [" ".join(map(r, row)) for row in W]
            lines.append(" ".join(map(r, b)))
        lines.append("end")
        return "\n".join(lines) + "\n"

    def save(self, path: Path = ASSET) -> Path:
        Path(path).write_text(self.text())
        return Path(path)


def parse(text: str) -> Model:
    note, words = [], []
    for line in text.splitlines():
        if line.startswith("#"):
            note.append(line.lstrip("# ").rstrip())
        else:
            words += line.split()
    it = iter(words)

    def take(expected=None):
        try:
            w = next(it)
        except StopIteration:
            raise ValueError("heuristic file: it ends too soon") from None
        if expected is not None and w != expected:
            raise ValueError(f"heuristic file: expected {expected!r}, got {w!r}")
        return w

    take("brubus-heuristic"); take("1"); take("inputs")
    if int(take()) != FEATURES:
        raise ValueError(f"heuristic file: the network must take {FEATURES} inputs")
    take("mean")
    mean = np.array([float(take()) for _ in range(FEATURES)])
    take("scale")
    scale = np.array([float(take()) for _ in range(FEATURES)])
    layers, width = [], FEATURES
    while (w := take()) != "end":
        if w != "layer":
            raise ValueError(f"heuristic file: expected 'layer' or 'end', got {w!r}")
        n_in, n_out = int(take()), int(take())
        if n_in != width or n_out < 1:
            raise ValueError("heuristic file: the layers don't fit together")
        W = np.array([float(take()) for _ in range(n_in * n_out)]).reshape(n_in, n_out)
        b = np.array([float(take()) for _ in range(n_out)])
        layers.append((W, b))
        width = n_out
    if not layers or width != 1:
        raise ValueError("heuristic file: the network must end in one output")
    return Model(mean, scale, layers, note)


def load(path: Path = ASSET) -> Model:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no trained estimate at {path}: train one (python main.py learn train)")
    return parse(path.read_text())


def plan(t: Terrain, start, goal, model: Model, w: float = np.inf, name: str = "A* (learned)"):
    """A* from start to goal guided by the learned estimate, capped at w x the climb-aware one (re-expanding)."""
    return astar_reopen(t, tuple(start), tuple(goal), model.table(t, goal, w), name=name)
