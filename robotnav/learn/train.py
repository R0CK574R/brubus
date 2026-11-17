"""Training the learned estimate (needs PyTorch: pip install torch; nothing else in Brubus does).

    python main.py learn train                       # the shipped model: train maps of all three themes
    python main.py learn train --themes grassland desert --out ...   # the generalisation test: no island at all

What it learns. For a cell and a goal, the network predicts t = true time / climb-aware estimate - 1 (how much
longer than the climb-aware estimate the drive really is) from the 16 features (features.py). The labels are the
data set's exact times (dataset.py), from the train maps only.

The examples. From every (train map, goal) field, `per_field` cells drawn at random among those that can reach the
goal: about 2.9 million examples for the full data set. The validation maps give the early-stopping loss; the
test maps are not touched here (evaluate.py reports on them once a model is chosen).

The loss is measured in seconds of estimate, (z - t) x climb, through a Huber loss (squared below 1 s, linear above,
so the long detours around the island's water don't dominate), and an overestimate costs `alpha` times as much as
an underestimate of the same size: too high an estimate is what can cost A* the quickest route, too low only
costs it extra search.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from ..terrain import THEMES
from .dataset import DEFAULT_DIR, load, load_manifest, terrain_of
from .features import FEATURES
from .model import ASSET, Model

PER_FIELD = 1024
HIDDEN = (64, 64)
ALPHA = 2.0
EPOCHS = 40
BATCH = 4096
LR = 2e-3
SEED = 0


def collect(directory: Path, entries: list, per_field: int, seed: int, log=print):
    """(x, t, climb) for `per_field` random reachable cells of every (map, goal) in `entries`."""
    from .. import cpp
    rng = np.random.default_rng(seed)
    xs, ts, cs = [], [], []
    t0 = time.perf_counter()
    for i, e in enumerate(entries):
        arrays = load(directory, e)
        t = terrain_of(arrays)
        cmap = cpp.to_cpp(t) if cpp.available() else None
        for k, g in enumerate(arrays["goals"]):
            goal = (int(g[0]), int(g[1]))
            field = arrays["cost_to_go"][k].ravel().astype(float)
            ok = np.flatnonzero(np.isfinite(field) & (field > 0))
            cells = rng.choice(ok, size=min(per_field, len(ok)), replace=False)
            if cmap is not None:
                x, climb = cpp.features(t, goal, cells, cmap=cmap)
            else:
                from .features import features
                x, climb = features(t, goal, cells)
            xs.append(x); cs.append(climb); ts.append(field[cells] / climb - 1.0)
        if log and (i + 1) % 60 == 0:
            log(f"  features for {i + 1}/{len(entries)} maps ({time.perf_counter() - t0:.0f} s)")
    return np.concatenate(xs), np.concatenate(ts), np.concatenate(cs)


def fit(x, t, climb, xv, tv, climbv, hidden=HIDDEN, alpha=ALPHA, epochs=EPOCHS, batch=BATCH, lr=LR, seed=SEED,
        log=print):
    """Train the network; returns (Model, history). Keeps the weights of the epoch with the lowest validation loss."""
    import torch
    torch.manual_seed(seed)
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale == 0] = 1.0
    norm = lambda a: torch.tensor((a - mean) / scale, dtype=torch.float32)
    X, T, Cl = norm(x), torch.tensor(t, dtype=torch.float32), torch.tensor(climb, dtype=torch.float32)
    Xv, Tv, Clv = norm(xv), torch.tensor(tv, dtype=torch.float32), torch.tensor(climbv, dtype=torch.float32)
    sizes = [FEATURES, *hidden, 1]
    layers = []
    for a, b in zip(sizes, sizes[1:]):
        layers += [torch.nn.Linear(a, b), torch.nn.ReLU()]
    net = torch.nn.Sequential(*layers[:-1])
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    steps = epochs * ((len(X) + batch - 1) // batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)

    def loss_of(z, target, cl):
        err = (z - target) * cl                                   # seconds of estimate: + too high, - too low
        huber = torch.nn.functional.huber_loss(err, torch.zeros_like(err), reduction="none", delta=1.0)
        return (torch.where(err > 0, alpha * huber, huber)).mean()

    history, best, best_state = [], float("inf"), None
    gen = torch.Generator().manual_seed(seed)
    for epoch in range(epochs):
        t0 = time.perf_counter()
        net.train()
        order = torch.randperm(len(X), generator=gen)
        total = 0.0
        for k in range(0, len(X), batch):
            idx = order[k:k + batch]
            z = net(X[idx]).squeeze(1)
            loss = loss_of(z, T[idx], Cl[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(idx)
        net.eval()
        with torch.no_grad():
            zv = net(Xv).squeeze(1)
            val = float(loss_of(zv, Tv, Clv))
            est = Clv * (1 + torch.clamp(zv, min=0))
            truth = Clv * (1 + Tv)
            rel = ((est - truth) / truth)
            over = float((rel > 0).float().mean())
            mae = float(rel.abs().mean())
        history.append(dict(epoch=epoch + 1, train=total / len(X), val=val, val_rel_error=mae, val_over=over,
                            seconds=round(time.perf_counter() - t0, 1)))
        if val < best:
            best, best_state = val, {k: v.clone() for k, v in net.state_dict().items()}
        if log:
            log(f"  epoch {epoch + 1:>2}: train {total / len(X):.4f}  val {val:.4f}  |error| {100 * mae:.1f}%  "
                f"too high on {100 * over:.0f}%  ({time.perf_counter() - t0:.1f} s)")
    net.load_state_dict(best_state)
    linears = [m for m in net if isinstance(m, torch.nn.Linear)]
    weights = [(L.weight.detach().double().numpy().T.copy(), L.bias.detach().double().numpy().copy()) for L in linears]
    return Model(mean.astype(float), scale.astype(float), weights), history


def train(directory: Path = DEFAULT_DIR, out: Path = ASSET, themes=THEMES, hidden=HIDDEN, alpha=ALPHA,
          epochs=EPOCHS, per_field=PER_FIELD, seed=SEED, log=print) -> tuple:
    """Collect the examples, train, save the model (with where it came from in its comment lines) and a card."""
    import torch
    m = load_manifest(directory)
    train_maps = [e for e in m["maps"] if e["split"] == "train" and e["theme"] in themes]
    val_maps = [e for e in m["maps"] if e["split"] == "val" and e["theme"] in themes]
    if log:
        log(f"{len(train_maps)} train maps and {len(val_maps)} validation maps ({', '.join(themes)})")
    t0 = time.perf_counter()
    x, t, climb = collect(directory, train_maps, per_field, seed, log)
    xv, tv, climbv = collect(directory, val_maps, per_field // 4, seed + 1, None)
    if log:
        log(f"{len(x):,} training examples, {len(xv):,} validation examples ({time.perf_counter() - t0:.0f} s)")
    model, history = fit(x, t, climb, xv, tv, climbv, hidden, alpha, epochs, seed=seed, log=log)
    best = min(history, key=lambda h: h["val"])
    model.note = [
        "Brubus: learned estimate of the drive time left to the goal (stage 2 of 'Learning from the planner').",
        "estimate = climb x (1 + max(z, 0)); z = this network on the 16 features of robotnav/learn/features.py.",
        f"Trained with PyTorch {torch.__version__} on data set {m['name']} (sha256 {m['sha256'][:16]}), "
        f"{len(train_maps)} train maps ({', '.join(themes)}), {len(x):,} examples.",
        f"Layers {[FEATURES, *hidden, 1]}, ReLU; Huber loss in seconds, overestimates x{alpha:g}; "
        f"{epochs} epochs, kept epoch {best['epoch']} (validation loss {best['val']:.4f}, "
        f"mean |error| {100 * best['val_rel_error']:.1f}%).",
    ]
    out = Path(out)
    model.save(out)
    card = dict(model=out.name, data=dict(name=m["name"], sha256=m["sha256"], themes=list(themes),
                                          train_maps=len(train_maps), val_maps=len(val_maps), examples=len(x),
                                          per_field=per_field),
                network=dict(sizes=[FEATURES, *hidden, 1], activation="ReLU", output="climb x (1 + max(z, 0))"),
                training=dict(framework=f"PyTorch {torch.__version__}", loss="Huber (1 s) on seconds of estimate",
                              overestimate_weight=alpha, epochs=epochs, batch=BATCH, lr=LR, seed=seed,
                              kept_epoch=best["epoch"], seconds=round(time.perf_counter() - t0)),
                history=history)
    out.with_suffix(".card.json").write_text(json.dumps(card, indent=1) + "\n")
    if log:
        log(f"saved {out} and {out.with_suffix('.card.json').name}")
    return model, card
