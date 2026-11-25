"""Stage 4: Bruce learns to see the ground.

Until now Bruce has known his whole map. Here he knows its heights (as he would from satellite elevation data) but
not its kinds of ground, and has to see them with his camera (robotnav.learn.camera) while he drives. His
short-range sensor still finds the obstacles, as before (robotnav.replan): the camera's job is the ground, which
decides how fast he can go.

    python main.py vision build    # camera frames from stage 1's maps, with every pixel labelled: data/vision-v1/
    python main.py vision check    # checksums, splits, and frames rendered again must match bit for bit
    python main.py vision train    # the baselines and the network (needs PyTorch)
    python main.py vision eval     # how well each sees, by distance; then drives that plan with what they see

The data. Frames from the maps of stage 1's data set (robotnav.learn.dataset, terrain-v1), split the same way, by
map: the network is always scored on maps it has never seen, and the viewer's three maps are never in it. On each
map, frames from random places on open ground, looking any way, under a random light (camera.random_look). A frame
keeps its picture, every pixel's true class (camera.CLASSES) and its distance along the bare ground (what Bruce can
work out from the elevation data before he sees anything, kept as a byte on a log scale). The labels are exact: the
camera knows what each ray meets.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import replace
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from ..terrain import THEMES
from . import camera as cam
from . import dataset as ds

FORMAT = 1
NAME = "vision-v1"
DEFAULT_DIR = Path(__file__).resolve().parents[2] / "data" / NAME
FRAMES = {"train": 32, "val": 16, "test": 16}       # frames per map, by split
SALT = 20261118                                      # seeds each map's poses and looks (with its theme and seed)
NEAR, FAR = 0.25, 100.0                              # the distance byte's range: 1 is NEAR, 255 is FAR, 0 is none


# ---------------------------------------------------------------- distances as bytes
def encode_dist(d: np.ndarray) -> np.ndarray:
    """Metres as a byte on a log scale (2.4% steps from 0.25 m to 100 m); 0 where there is no ground."""
    d = np.asarray(d, dtype=float)
    ok = np.isfinite(d)
    x = np.log(np.clip(np.where(ok, d, NEAR), NEAR, FAR) / NEAR) / math.log(FAR / NEAR)
    return np.where(ok, 1 + np.rint(x * 254), 0).astype(np.uint8)


def decode_dist(code: np.ndarray) -> np.ndarray:
    code = np.asarray(code)
    return np.where(code > 0, NEAR * (FAR / NEAR) ** ((code.astype(float) - 1) / 254), np.inf)


# ---------------------------------------------------------------- the frames of one map
def poses_and_looks(t, theme: str, seed: int, n: int) -> list:
    """`n` places on the map's open ground (anywhere in the cell), each looking a random way, each with a random
    light: the same ones for the same map, every time."""
    rng = np.random.default_rng([SALT, THEMES.index(theme), seed])
    cells = np.argwhere(~np.asarray(t.blocked))
    out = []
    for _ in range(n):
        r, c = cells[int(rng.integers(len(cells)))]
        jr, jc = rng.uniform(-0.4, 0.4, 2)
        pose = (float(r + jr), float(c + jc), float(rng.uniform(0, 2 * math.pi)))
        out.append((pose, cam.random_look(rng)))
    return out


def look_row(lk: cam.Look) -> list:
    return [lk.sun_az, lk.sun_el, lk.ambient, lk.fog, lk.exposure, *lk.balance, lk.noise, float(lk.seed)]


def render_map(arrays: dict, theme: str, seed: int, n: int) -> dict:
    """The frames of one stored map, as the arrays that go in its file."""
    t = ds.terrain_of(arrays)
    sc = cam.Scene(t)
    shots = poses_and_looks(t, theme, seed, n)
    H, W = cam.CAMERA.height, cam.CAMERA.width
    images = np.zeros((n, H, W, 3), np.uint8)
    labels = np.zeros((n, H, W), np.uint8)
    dist = np.zeros((n, H, W), np.uint8)
    for i, (pose, look) in enumerate(shots):
        f = cam.render(sc, pose, look)
        images[i], labels[i], dist[i] = f.image, f.label, encode_dist(f.dem_dist)
    return dict(images=images, labels=labels, dist=dist, poses=np.array([p for p, _ in shots]),
                looks=np.array([look_row(lk) for _, lk in shots]))


def _entry(src: dict, arrays: dict, file: str) -> dict:
    counts = np.bincount(arrays["labels"].ravel(), minlength=len(cam.CLASSES))
    return dict(id=src["id"], theme=src["theme"], seed=src["seed"], split=src["split"], file=file,
                frames=int(len(arrays["images"])), map_sha256=src["sha256"], pixels=counts.tolist(),
                sha256=ds.content_hash(arrays))


def _build_one(task) -> dict:
    src, terrain_dir, out = task
    arrays = render_map(ds.load(terrain_dir, src), src["theme"], src["seed"], FRAMES[src["split"]])
    file = f"{src['id']}.npz"
    np.savez_compressed(Path(out) / file, **arrays)
    return _entry(src, arrays, file)


def dataset_hash(entries) -> str:
    h = hashlib.sha256()
    for e in entries:
        h.update(f"{e['id']}|{e['split']}|{e['sha256']}\n".encode())
    return h.hexdigest()


def build(out: Path = DEFAULT_DIR, terrain_dir: Path = ds.DEFAULT_DIR, jobs: int = 1, limit: int | None = None,
          log=print, command: str = "") -> dict:
    """Render the data set into `out` from stage 1's maps in `terrain_dir`: one file per map and manifest.json.
    `limit` keeps only the first maps of each split and theme (for a quick look)."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    src = ds.load_manifest(terrain_dir)
    maps = src["maps"]
    if limit:
        maps = [e for sp in ds.SPLITS for th in THEMES
                for e in [e for e in maps if e["split"] == sp and e["theme"] == th][:limit]]
    tasks = [(e, str(terrain_dir), str(out)) for e in maps]
    t0 = time.perf_counter()
    entries = []
    ctx = get_context("fork") if os.name == "posix" else get_context()
    with ctx.Pool(max(1, jobs)) as pool:
        for k, e in enumerate(pool.imap(_build_one, tasks), 1):
            entries.append(e)
            if log and (k % 25 == 0 or k == len(tasks)):
                log(f"  {k}/{len(tasks)} maps  ({time.perf_counter() - t0:.0f} s)")
    counts = {sp: {th: sum(e["frames"] for e in entries if e["split"] == sp and e["theme"] == th) for th in THEMES}
              for sp in ds.SPLITS}
    pixels = {sp: np.sum([e["pixels"] for e in entries if e["split"] == sp], axis=0).tolist() for sp in ds.SPLITS}
    c = cam.CAMERA
    manifest = dict(format=FORMAT, name=NAME, built_with=command or "robotnav.learn.vision.build",
                    source=dict(name=src["name"], sha256=src["sha256"]), frames_per_map=FRAMES,
                    camera=dict(width=c.width, height=c.height, hfov=c.hfov, pitch=c.pitch, eye=c.eye),
                    classes=list(cam.CLASSES), split_rule=ds.SPLIT_RULE, demo_maps_excluded=src["demo_maps_excluded"],
                    frames=counts, pixels=pixels, maps=entries, sha256=dataset_hash(entries),
                    build_seconds=round(time.perf_counter() - t0, 1))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


def load_manifest(directory: Path = DEFAULT_DIR) -> dict:
    path = Path(directory) / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"no frames in {directory}: build them first (python main.py vision build)")
    return json.loads(path.read_text())


def load(directory: Path, entry: dict) -> dict:
    with np.load(Path(directory) / entry["file"], allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def load_split(directory: Path, split: str, themes=THEMES) -> dict:
    """Every frame of a split: images, labels, distance bytes, and each frame's theme and map."""
    m = load_manifest(directory)
    parts = [(e, load(directory, e)) for e in m["maps"] if e["split"] == split and e["theme"] in themes]
    if not parts:
        raise ValueError(f"no {split} frames in {directory}")
    out = {k: np.concatenate([a[k] for _, a in parts]) for k in ("images", "labels", "dist", "poses")}
    out["theme"] = np.concatenate([[THEMES.index(e["theme"])] * e["frames"] for e, _ in parts])
    out["map"] = np.concatenate([[i] * e["frames"] for i, (e, _) in enumerate(parts)])
    out["ids"] = [e["id"] for e, _ in parts]
    return out


def check(directory: Path = DEFAULT_DIR, terrain_dir: Path = ds.DEFAULT_DIR, audit: int = 2, log=print) -> list:
    """Every file present and matching its checksum, every map in its stage 1 split and unchanged since, none of the
    viewer's maps; and on `audit` maps of each split, the frames rendered again from the stored map must equal the
    stored ones bit for bit. Returns the problems found."""
    directory = Path(directory)
    m = load_manifest(directory)
    src = {e["id"]: e for e in ds.load_manifest(terrain_dir)["maps"]}
    problems = []
    for e in m["maps"]:
        s = src.get(e["id"])
        if s is None or s["sha256"] != e["map_sha256"] or s["split"] != e["split"]:
            problems.append(f"{e['id']}: not the map stage 1 has (or not in its split)")
        if m["demo_maps_excluded"].get(e["theme"]) == e["seed"]:
            problems.append(f"{e['id']} is one of the viewer's maps")
        if not (directory / e["file"]).exists():
            problems.append(f"{e['file']} is missing")
        elif ds.content_hash(load(directory, e)) != e["sha256"]:
            problems.append(f"{e['file']} doesn't match its checksum")
    if dataset_hash(m["maps"]) != m["sha256"]:
        problems.append("the manifest's overall checksum doesn't match its maps")
    audited = 0
    for sp in ds.SPLITS:
        for e in [e for e in m["maps"] if e["split"] == sp][:audit]:
            if e["id"] not in src or not (directory / e["file"]).exists():
                continue
            again = render_map(ds.load(terrain_dir, src[e["id"]]), e["theme"], e["seed"], e["frames"])
            stored = load(directory, e)
            if any(not np.array_equal(again[k], stored[k]) for k in stored):
                problems.append(f"{e['id']}: its frames rendered again differ from the stored ones")
            audited += 1
    if log:
        log(f"{len(m['maps'])} maps checked against their checksums; {audited} rendered again; "
            f"{len(problems)} problem(s)")
    return problems


# ---------------------------------------------------------------- what a map can show, and what a pixel can be
_KINDS_OF: dict = {}


def theme_kinds(theme: str) -> tuple:
    """The kinds of ground a map of this theme has (from the generator itself)."""
    if theme not in _KINDS_OF:
        from ..terrain import make_map
        _KINDS_OF[theme] = tuple(make_map(theme, size=24, seed=1).surface_names)
    return _KINDS_OF[theme]


def allowed(theme: str) -> np.ndarray:
    """Which classes a map of this theme can show: its own kinds of ground, rocks, trees, water if it has any, sky.
    Bruce knows what sort of country he is in, so a model chooses among these alone."""
    ok = np.zeros(len(cam.CLASSES), bool)
    for k in theme_kinds(theme):
        ok[cam.KINDS.index(k)] = True
    ok[[cam.ROCK_C, cam.TREE_C, cam.SKY_C]] = True
    ok[cam.WATER_C] = theme in ("desert", "island")
    return ok


def allowed_table() -> np.ndarray:
    return np.stack([allowed(t) for t in THEMES])


def geometry(dist_code: np.ndarray) -> np.ndarray:
    """Which classes each pixel can be from where the bare ground is (..., classes): where the elevation data puts
    ground along a pixel's ray, it shows ground or something on it, never sky; where it puts none, it shows sky, or
    something standing up into it (a tree, a rock, water at the map's edge)."""
    none = np.asarray(dist_code) == 0
    g = np.ones(none.shape + (len(cam.CLASSES),), bool)
    g[..., cam.SKY_C] = none
    g[..., :cam.GROUND] &= ~none[..., None]
    return g


def finish(logits: np.ndarray, theme: str, dist_code: np.ndarray) -> np.ndarray:
    """Scores (..., classes) to log-probabilities over the classes the theme and the geometry allow (-inf for the
    rest)."""
    ok = allowed(theme) & geometry(dist_code)
    z = np.where(ok, logits, -np.inf)
    top = z.max(axis=-1, keepdims=True)
    return z - top - np.log(np.exp(z - top).sum(axis=-1, keepdims=True))


def inputs(images: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """A model's input (..., 4, height, width): red, green, blue and the distance byte, each scaled to 0-1."""
    img = np.moveaxis(np.asarray(images, np.float64) / 255.0, -1, -3)
    return np.concatenate([img, (np.asarray(dist, np.float64) / 255.0)[..., None, :, :]], axis=-3)


# ---------------------------------------------------------------- baseline 1: colour alone
class ColourModel:
    """Each class's colour as a Gaussian over red, green and blue (its mean and covariance over training pixels),
    and a pixel's class from Bayes' rule with each class's share of the pixels in that theme's maps. No texture, no
    neighbours: what a pixel's colour alone says."""
    name = "colour only"

    def __init__(self, mean: np.ndarray, cov: np.ndarray, prior: dict):
        self.mean, self.cov, self.prior = np.asarray(mean, float), np.asarray(cov, float), prior
        self.inv = np.linalg.inv(self.cov)
        self.logdet = np.linalg.slogdet(self.cov)[1]

    @classmethod
    def fit(cls, frames: dict, per_class: int = 60000, seed: int = 0) -> "ColourModel":
        rng = np.random.default_rng(seed)
        rgb = frames["images"].reshape(-1, 3).astype(float) / 255.0
        lab = frames["labels"].ravel()
        K = len(cam.CLASSES)
        mean, cov = np.zeros((K, 3)), np.tile(np.eye(3) * 1e-2, (K, 1, 1))
        for k in range(K):
            idx = np.nonzero(lab == k)[0]
            if idx.size < 10:
                continue
            idx = rng.choice(idx, min(per_class, idx.size), replace=False)
            mean[k] = rgb[idx].mean(0)
            cov[k] = np.cov(rgb[idx].T) + np.eye(3) * 1e-5
        prior = {}
        for i, th in enumerate(THEMES):
            n = np.bincount(frames["labels"][frames["theme"] == i].ravel(), minlength=K).astype(float)
            prior[th] = np.log(np.maximum(n, 1.0) / n.sum())
        return cls(mean, cov, prior)

    def log_probs(self, images: np.ndarray, dist: np.ndarray, theme: str) -> np.ndarray:
        x = np.asarray(images, float) / 255.0
        d = x[..., None, :] - self.mean                                        # (..., K, 3)
        maha = np.einsum("...ki,kij,...kj->...k", d, self.inv, d)
        return finish(-0.5 * maha - 0.5 * self.logdet + self.prior[theme], theme, dist)

    def arrays(self) -> dict:
        return {"colour.mean": self.mean, "colour.cov": self.cov,
                **{f"colour.prior.{th}": self.prior[th] for th in THEMES}}

    @classmethod
    def from_arrays(cls, a: dict) -> "ColourModel":
        return cls(a["colour.mean"], a["colour.cov"], {th: a[f"colour.prior.{th}"] for th in THEMES})


# ---------------------------------------------------------------- baseline 2: classic machine learning
PIXEL_FEATURES = ("red", "green", "blue", "red 5x5 mean", "green 5x5 mean", "blue 5x5 mean", "red 5x5 spread",
                  "green 5x5 spread", "blue 5x5 spread", "edges 5x5", "distance")


def pixel_features(images: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """Hand-made features for every pixel (..., height, width, 11): its colour, the mean and spread of colour round
    it (5 x 5), how much edge there is round it (texture), and its distance byte."""
    from scipy.ndimage import sobel, uniform_filter
    x = np.asarray(images, float) / 255.0
    lead = x.ndim - 3
    size = (1,) * lead + (5, 5)
    out = [x[..., i] for i in range(3)]
    means = [uniform_filter(x[..., i], size=size, mode="reflect") for i in range(3)]
    sq = [uniform_filter(x[..., i] ** 2, size=size, mode="reflect") for i in range(3)]
    out += means + [np.sqrt(np.maximum(s - m * m, 0.0)) for s, m in zip(sq, means)]
    lum = x.mean(-1)
    edge = np.hypot(sobel(lum, axis=-1, mode="reflect"), sobel(lum, axis=-2, mode="reflect"))
    out.append(uniform_filter(edge, size=size, mode="reflect"))
    out.append(np.asarray(dist, float) / 255.0)
    return np.stack(out, -1)


class PixelModel:
    """Multinomial logistic regression on pixel_features: a weight per feature and class, nothing deep."""
    name = "classic ML"

    def __init__(self, mean, scale, W, b):
        self.mean, self.scale = np.asarray(mean, float), np.asarray(scale, float)
        self.W, self.b = np.asarray(W, float), np.asarray(b, float)

    def log_probs(self, images: np.ndarray, dist: np.ndarray, theme: str) -> np.ndarray:
        f = (pixel_features(images, dist) - self.mean) / self.scale
        z = np.broadcast_to(self.b, f.shape[:-1] + self.b.shape).copy()
        for i in range(f.shape[-1]):                                       # a fixed order of sums
            z += f[..., i, None] * self.W[i]
        return finish(z, theme, dist)

    def arrays(self) -> dict:
        return {"pixel.mean": self.mean, "pixel.scale": self.scale, "pixel.W": self.W, "pixel.b": self.b}

    @classmethod
    def from_arrays(cls, a: dict) -> "PixelModel":
        return cls(a["pixel.mean"], a["pixel.scale"], a["pixel.W"], a["pixel.b"])


def _masks_torch(theme_idx, dist_code):
    """The allowed classes per pixel (theme and geometry), as a boolean tensor (B, K, H, W)."""
    import torch
    th = torch.tensor(allowed_table())[theme_idx]                                  # (B, K)
    g = torch.from_numpy(np.ascontiguousarray(np.moveaxis(geometry(dist_code), -1, 1)))   # (B, K, H, W)
    return g & th[:, :, None, None]


def fit_pixel_model(frames: dict, pixels: int = 1_500_000, steps: int = 3000, lr: float = 0.05,
                    seed: int = 0) -> PixelModel:
    """Logistic regression by gradient descent (PyTorch) on a random sample of training pixels."""
    import torch
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n, H, W = frames["labels"].shape
    pick = rng.choice(n * H * W, pixels, replace=False)
    fi, rest = np.divmod(pick, H * W)
    yy, xx = np.divmod(rest, W)
    feats = np.zeros((pixels, len(PIXEL_FEATURES)))
    for f in np.unique(fi):                                       # features need each pixel's neighbours: by frame
        m = fi == f
        feats[m] = pixel_features(frames["images"][f], frames["dist"][f])[yy[m], xx[m]]
    lab = frames["labels"][fi, yy, xx].astype(np.int64)
    ok = allowed_table()[frames["theme"][fi]] & geometry(frames["dist"][fi, yy, xx])
    mean, scale = feats.mean(0), feats.std(0)
    scale[scale == 0] = 1.0
    X = torch.tensor((feats - mean) / scale, dtype=torch.float32)
    Y = torch.tensor(lab)
    M = torch.tensor(ok)
    lin = torch.nn.Linear(X.shape[1], len(cam.CLASSES))
    opt = torch.optim.Adam(lin.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    gen = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        idx = torch.randint(0, len(X), (8192,), generator=gen)
        z = lin(X[idx]).masked_fill(~M[idx], -1e9)
        loss = torch.nn.functional.cross_entropy(z, Y[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    return PixelModel(mean, scale, lin.weight.detach().double().numpy().T.copy(), lin.bias.detach().double().numpy())


# ---------------------------------------------------------------- the network: a small U-Net
# (in, out) channels of each 3 x 3 convolution, by stage. The picture is seen at full size by the stem, then at half
# and quarter size; the decoder comes back to half size, where the classes are scored, and each score covers 2 x 2
# pixels of the frame.
NET = dict(stem=[(4, 16)], down1=[(16, 24), (24, 24)], down2=[(24, 48), (48, 48)], up1=[(72, 24), (24, 24)],
           head=(24, len(cam.CLASSES)))
EPOCHS = 16                                    # chosen on validation frames: 16 beat 8 (ground right 94.3% vs 93.2%)


def _conv3(x: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    """A 3 x 3 convolution (zero padding) of x (C_in, H, W) by W (C_out, C_in, 3, 3), summed in one fixed order
    (bias, then input channel by input channel, row by row of the kernel), so it gives the same numbers anywhere."""
    cin, h, w = x.shape
    p = np.zeros((cin, h + 2, w + 2))
    p[:, 1:-1, 1:-1] = x
    out = np.broadcast_to(b[:, None, None], (W.shape[0], h, w)).copy()
    for i in range(cin):
        for ky in range(3):
            for kx in range(3):
                out += W[:, i, ky, kx, None, None] * p[i, ky:ky + h, kx:kx + w]
    return out


def _pool(x: np.ndarray) -> np.ndarray:
    c, h, w = x.shape
    return x.reshape(c, h // 2, 2, w // 2, 2).max(axis=(2, 4))


def _up(x: np.ndarray) -> np.ndarray:
    return x.repeat(2, axis=1).repeat(2, axis=2)


class Net:
    """The trained network, run in NumPy (float64, fixed order): what Bruce's drives and the tests use."""
    name = "network"

    def __init__(self, layers: dict):
        self.layers = {k: np.asarray(v, float) for k, v in layers.items()}

    def logits(self, x: np.ndarray) -> np.ndarray:
        """Class scores (K, H, W) for one input (4, H, W)."""
        L = self.layers
        relu = lambda a: np.maximum(a, 0.0)                                    # noqa: E731
        h = relu(_conv3(x, L["stem.0.w"], L["stem.0.b"]))
        h = _pool(h)
        for j in range(2):
            h = relu(_conv3(h, L[f"down1.{j}.w"], L[f"down1.{j}.b"]))
        skip = h
        h = _pool(h)
        for j in range(2):
            h = relu(_conv3(h, L[f"down2.{j}.w"], L[f"down2.{j}.b"]))
        h = np.concatenate([_up(h), skip])
        for j in range(2):
            h = relu(_conv3(h, L[f"up1.{j}.w"], L[f"up1.{j}.b"]))
        Wh, bh = L["head.w"], L["head.b"]
        z = np.broadcast_to(bh[:, None, None], (Wh.shape[0],) + h.shape[1:]).copy()
        for i in range(h.shape[0]):
            z += Wh[:, i, None, None] * h[i]
        return _up(z)

    def log_probs(self, images: np.ndarray, dist: np.ndarray, theme: str) -> np.ndarray:
        x = inputs(images, dist)
        z = np.moveaxis(self.logits(x), 0, -1)
        return finish(z, theme, dist)

    def arrays(self) -> dict:
        return {f"net.{k}": v.astype(np.float32) for k, v in self.layers.items()}

    @classmethod
    def from_arrays(cls, a: dict) -> "Net":
        return cls({k[4:]: v for k, v in a.items() if k.startswith("net.")})

    def parameters(self) -> int:
        return int(sum(v.size for v in self.layers.values()))


def _torch_net():
    import torch

    class UNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            mk = lambda spec: torch.nn.ModuleList(torch.nn.Conv2d(a, b, 3, padding=1) for a, b in spec)  # noqa: E731
            self.stem, self.down1, self.down2, self.up1 = (mk(NET[k]) for k in ("stem", "down1", "down2", "up1"))
            self.head = torch.nn.Conv2d(*NET["head"], 1)

        def forward(self, x):
            F = torch.nn.functional
            h = F.relu(self.stem[0](x))
            h = F.max_pool2d(h, 2)
            for c in self.down1:
                h = F.relu(c(h))
            skip = h
            h = F.max_pool2d(h, 2)
            for c in self.down2:
                h = F.relu(c(h))
            h = torch.cat([F.interpolate(h, scale_factor=2, mode="nearest"), skip], 1)
            for c in self.up1:
                h = F.relu(c(h))
            return F.interpolate(self.head(h), scale_factor=2, mode="nearest")

    return UNet()


def net_of(module) -> Net:
    """A PyTorch U-Net's weights as a Net (float64)."""
    layers = {}
    for stage in ("stem", "down1", "down2", "up1"):
        for j, c in enumerate(getattr(module, stage)):
            layers[f"{stage}.{j}.w"] = c.weight.detach().double().numpy()
            layers[f"{stage}.{j}.b"] = c.bias.detach().double().numpy()
    layers["head.w"] = module.head.weight.detach().double().numpy()[:, :, 0, 0]
    layers["head.b"] = module.head.bias.detach().double().numpy()
    return Net(layers)


def fit_net(frames: dict, epochs: int = EPOCHS, batch: int = 32, lr: float = 3e-3, seed: int = 0, log=print) -> Net:
    """Train the U-Net (PyTorch, CPU): cross-entropy over the classes each pixel can be, Adam with a one-cycle
    schedule, frames mirrored left-right half the time."""
    import torch
    torch.manual_seed(seed)
    net = _torch_net()
    n = len(frames["labels"])
    steps = epochs * (n // batch)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    losses = []
    for ep in range(epochs):
        order = rng.permutation(n)
        total = 0.0
        for b0 in range(0, n - batch + 1, batch):
            idx = np.sort(order[b0:b0 + batch])
            img, dist, lab = frames["images"][idx], frames["dist"][idx], frames["labels"][idx]
            flip = rng.random(len(idx)) < 0.5
            img, dist, lab = (np.where(flip.reshape((-1,) + (1,) * (a.ndim - 1)), a[:, :, ::-1], a)
                              for a in (img, dist, lab))
            x = torch.tensor(inputs(img, dist), dtype=torch.float32)
            y = torch.tensor(lab.astype(np.int64))
            mask = _masks_torch(torch.tensor(frames["theme"][idx]), dist)
            z = net(x).masked_fill(~mask, -1e9)
            loss = torch.nn.functional.cross_entropy(z, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += float(loss.detach())
        losses.append(total / (n // batch))
        if log:
            log(f"  epoch {ep + 1}/{epochs}: loss {losses[-1]:.4f}  ({time.perf_counter() - t0:.0f} s)")
    out = net_of(net)
    out.losses = losses                                        # the training record (not saved with the weights)
    return out


# ---------------------------------------------------------------- how well each model sees
ASSET = Path(__file__).resolve().parents[1] / "assets" / "vision.npz"
BANDS = (0.0, 5.0, 10.0, 20.0, 40.0, math.inf)       # distance bands (metres along the bare ground)


REFERENCE_POSE = (50.0, 30.0, 1.2)            # the frame the committed models are checked on (the tests' desert map)


def reference_frame() -> cam.Frame:
    from ..terrain import solvable_map
    return cam.render(cam.Scene(solvable_map("desert", seed=3, size=96, max_grade=0.6)), REFERENCE_POSE)


def models_from(path: Path = ASSET) -> dict:
    """The three trained models, from their file."""
    a = arrays_from(path)
    return {"colour": ColourModel.from_arrays(a), "pixel": PixelModel.from_arrays(a), "net": Net.from_arrays(a)}


def arrays_from(path: Path = ASSET) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def logits_digest(net: Net, image: np.ndarray, dist: np.ndarray) -> str:
    """The network's scores for one frame, as a checksum: they come from sums and products alone, in a fixed order,
    so they are the same to the last bit on every machine."""
    return hashlib.sha256(np.ascontiguousarray(net.logits(inputs(image, dist)), "<f8").tobytes()).hexdigest()


def save_models(models: dict, path: Path = ASSET) -> dict:
    """Write the three models to one file, with the reference frame they are checked on; returns what the checks
    compare against (each model's labels for the frame, and the network's scores as a checksum)."""
    f = reference_frame()
    dist = encode_dist(f.dem_dist)
    arrays = {"reference.image": f.image, "reference.dist": dist, "reference.label": f.label}
    for m in models.values():
        arrays.update(m.arrays())
    np.savez_compressed(path, **arrays)
    loaded = models_from(path)                                 # what the file holds (the network's weights as float32)
    ref = {k: np.asarray(m.log_probs(f.image, dist, "desert").argmax(-1), np.uint8) for k, m in loaded.items()}
    return dict(pose=list(REFERENCE_POSE), net_logits_sha256=logits_digest(loaded["net"], f.image, dist),
                accuracy={k: float(np.mean(v == f.label)) for k, v in ref.items()},
                labels_sha256={k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in ref.items()})


def pixel_scores(model, frames: dict, every: int = 1) -> dict:
    """How well a model labels the pixels of `frames` (every `every`-th frame): accuracy over all pixels and over
    ground pixels (is it the right kind of ground?), each class's IoU, the ground kinds' confusion counts, accuracy by
    distance band, and calibration (on ground pixels: how often a pixel labelled with confidence p is right)."""
    K = len(cam.CLASSES)
    conf_mat = np.zeros((K, K), np.int64)
    band_right, band_all = np.zeros(len(BANDS) - 1), np.zeros(len(BANDS) - 1)
    cal_bins = np.linspace(0, 1, 11)
    cal_n, cal_conf, cal_right = np.zeros(10), np.zeros(10), np.zeros(10)
    for i in range(0, len(frames["labels"]), every):
        theme = THEMES[int(frames["theme"][i])]
        lp = model.log_probs(frames["images"][i], frames["dist"][i], theme)
        pred = lp.argmax(-1)
        lab = frames["labels"][i].astype(np.int64)
        np.add.at(conf_mat, (lab.ravel(), pred.ravel()), 1)
        ground = lab < cam.GROUND
        d = decode_dist(frames["dist"][i])
        band = np.searchsorted(BANDS, d, side="right") - 1
        for b in range(len(BANDS) - 1):
            m = ground & (band == b)
            band_all[b] += m.sum()
            band_right[b] += (pred[m] == lab[m]).sum()
        p = np.exp(lp.max(-1))[ground]
        right = (pred == lab)[ground]
        j = np.clip(np.digitize(p, cal_bins) - 1, 0, 9)
        np.add.at(cal_n, j, 1)
        np.add.at(cal_conf, j, p)
        np.add.at(cal_right, j, right)
    tp = np.diag(conf_mat).astype(float)
    iou = tp / np.maximum(conf_mat.sum(0) + conf_mat.sum(1) - tp, 1)
    g = slice(0, cam.GROUND)
    ground_all = conf_mat[g].sum()
    ece = float(np.sum(np.abs(cal_conf - cal_right)) / max(cal_n.sum(), 1))
    return dict(model=model.name, frames=int(len(range(0, len(frames["labels"]), every))),
                accuracy=float(tp.sum() / conf_mat.sum()),
                ground_accuracy=float(np.trace(conf_mat[g, g]) / max(ground_all, 1)),
                iou={c: round(float(v), 4) for c, v in zip(cam.CLASSES, iou)},
                by_distance=[dict(band=[BANDS[b], BANDS[b + 1]], pixels=int(band_all[b]),
                                  accuracy=float(band_right[b] / max(band_all[b], 1))) for b in range(len(BANDS) - 1)],
                calibration=dict(ece=ece, bins=[dict(n=int(n), confidence=float(c / n) if n else None,
                                                     accuracy=float(r / n) if n else None)
                                                for n, c, r in zip(cal_n, cal_conf, cal_right)]),
                confusion=conf_mat[g, g].tolist())


def train(directory: Path = DEFAULT_DIR, epochs: int = EPOCHS, log=print) -> dict:
    """Fit the colour model, the classic model and the network on the training frames; returns the models."""
    t0 = time.perf_counter()
    frames = load_split(directory, "train")
    if log:
        log(f"{len(frames['labels'])} training frames loaded ({time.perf_counter() - t0:.0f} s)")
    colour = ColourModel.fit(frames)
    if log:
        log(f"colour model fitted ({time.perf_counter() - t0:.0f} s)")
    pixel = fit_pixel_model(frames)
    if log:
        log(f"classic model fitted ({time.perf_counter() - t0:.0f} s)")
    net = fit_net(frames, epochs=epochs, log=log)
    if log:
        log(f"network trained: {net.parameters():,} parameters ({time.perf_counter() - t0:.0f} s)")
    return {"colour": colour, "pixel": pixel, "net": net}


# ---------------------------------------------------------------- from pixels to a map, and driving on it
LIDAR = 5.0                                   # metres: the short-range sensor that finds obstacles (robotnav.replan)
SURE = -30.0                                  # log-probability standing for "impossible" (instead of -inf)


class GroundBelief:
    """What Bruce believes while he drives with his camera: the heights (known, from the elevation data), the
    obstacles his short-range sensor has found, and for every cell a belief over its kind of ground, built up from
    what the camera has seen of it. Cells he has driven over he knows (he felt them). A move's believed time is the
    terrain's drive-time model with a cell's believed slowdown: the expected one under his belief ("expected") or
    that of the likeliest kind ("likeliest"). Ground never seen is taken at the slowdown the prior expects
    ("prior"), or at the fastest the map has ("fastest", the hopeful guess robotnav.replan makes about obstacles: it
    lures a planner off ground it has seen to be slow into ground it hasn't seen, which is often no better). Beliefs
    can now make a move cheaper as well as dearer (soft sand seen from afar may turn out firm up close); D* Lite copes
    with both, and its straight-line estimate stays below every move's believed time, since no slowdown is under 1.

    Ground comes in fields, though, and cells taken one by one forget that: seeing tall grass ahead, he turns towards
    ground he hasn't seen, priced at the average, which is likely more of the same field. With `spread`, ground he
    hasn't seen within that many cells of ground he has seen or felt is priced by the beliefs round it, mixed with
    the prior (which counts as `spread_weight` cells)."""

    def __init__(self, t, prior: np.ndarray, mode: str = "expected", temper: float = 1.0, hysteresis: float = 0.01,
                 unseen: str = "prior", spread: int = 0, spread_weight: float = 1.0):
        if mode not in ("expected", "likeliest"):
            raise ValueError("mode is 'expected' or 'likeliest'")
        if unseen not in ("prior", "fastest"):
            raise ValueError("unseen is 'prior' or 'fastest'")
        self.spread, self.spread_weight = int(spread), float(spread_weight)
        self.R, self.C, self.N = t.rows, t.cols, t.rows * t.cols
        self.cell, self.max_grade, self.uphill = float(t.cell_size), float(t.max_grade), float(t.uphill_penalty)
        self.H = [float(v) for v in np.asarray(t.height).ravel()]
        self.costs = np.asarray(t.surface_costs, float)                  # slowdown of each kind (surface index)
        self.log_prior = np.asarray(prior, float)
        self.mode, self.temper, self.hysteresis = mode, temper, hysteresis
        self.evidence = np.zeros((self.N, len(self.costs)))
        self.seen = np.zeros(self.N, bool)
        self.felt = np.zeros(self.N, bool)
        self.blk = [False] * self.N
        self.kind = np.full(self.N, -1)                                   # the kind of ground felt (-1: not driven)
        p = np.exp(self.log_prior - self.log_prior.max())
        self.p_prior = p / p.sum()
        guess = float(p @ self.costs / p.sum()) if unseen == "prior" else float(self.costs.min())
        self.S = [guess] * self.N                                         # ground never seen

    def cost(self, u: int, v: int, dr: int, dc: int) -> float:
        blk, C = self.blk, self.C
        if blk[u] or blk[v]:
            return math.inf
        if dr and dc:
            if blk[u + dr * C] or blk[u + dc]:
                return math.inf
            run = math.sqrt(2) * self.cell
        else:
            run = self.cell
        rise = self.H[v] - self.H[u]
        grade = rise / run
        if abs(grade) > self.max_grade:
            return math.inf
        return math.sqrt(run * run + rise * rise) * self.S[v] * (1.0 + self.uphill * max(grade, 0.0))

    def h(self, a: int, b: int) -> float:
        C = self.C
        dr, dc = abs(a // C - b // C), abs(a % C - b % C)
        return ((dr + dc) + (math.sqrt(2) - 2) * min(dr, dc)) * self.cell

    def posterior(self, cells: np.ndarray) -> np.ndarray:
        z = self.log_prior + self.evidence[cells]
        z = z - z.max(axis=1, keepdims=True)
        p = np.exp(z)
        return p / p.sum(axis=1, keepdims=True)

    def slowdown(self, cells: np.ndarray) -> np.ndarray:
        p = self.posterior(cells)
        return p @ self.costs if self.mode == "expected" else self.costs[p.argmax(1)]

    def observe(self, cells: np.ndarray, logp: np.ndarray) -> list:
        """Add one frame's evidence (each cell's mean log-probability over its pixels, per kind); returns the cells
        whose believed slowdown moved by more than the hysteresis."""
        if cells.size == 0:
            return []
        self.evidence[cells] += self.temper * logp
        self.seen[cells] = True
        live = cells[~self.felt[cells]]
        new = self.slowdown(live)
        old = np.array([self.S[c] for c in live])
        move = np.abs(new - old) > self.hysteresis * old
        for c, s in zip(live[move].tolist(), new[move].tolist()):
            self.S[c] = s
        return live[move].tolist()

    def feel(self, cell: int, kind: int) -> list:
        """Bruce drives onto `cell`: now he knows its kind."""
        self.felt[cell] = True
        self.kind[cell] = kind
        s = float(self.costs[kind])
        if self.S[cell] != s:
            self.S[cell] = s
            return [cell]
        return []

    def neighbourhood(self) -> list:
        """Price the ground he hasn't seen near ground he has seen or felt by the beliefs round it (see `spread`);
        returns the cells whose believed slowdown moved by more than the hysteresis."""
        if not self.spread:
            return []
        from scipy.ndimage import uniform_filter
        known = self.seen | self.felt
        K = len(self.costs)
        post = np.zeros((self.N, K))
        seen = np.nonzero(self.seen & ~self.felt)[0]
        if seen.size:
            post[seen] = self.posterior(seen)
        felt = np.nonzero(self.felt)[0]
        post[felt, self.kind[felt]] = 1.0
        size = 2 * self.spread + 1
        n = uniform_filter(known.reshape(self.R, self.C).astype(float), size=size, mode="constant").ravel() * size ** 2
        s = uniform_filter(post.reshape(self.R, self.C, K), size=(size, size, 1), mode="constant").reshape(self.N, K)
        cand = np.nonzero((n > 0.5) & ~known)[0]
        if cand.size == 0:
            return []
        p = (self.spread_weight * self.p_prior + s[cand] * size ** 2) / (self.spread_weight + n[cand])[:, None]
        new = p @ self.costs if self.mode == "expected" else self.costs[p.argmax(1)]
        old = np.array([self.S[c] for c in cand])
        move = np.abs(new - old) > self.hysteresis * old
        for c, v in zip(cand[move].tolist(), new[move].tolist()):
            self.S[c] = v
        return cand[move].tolist()

    def likeliest(self) -> np.ndarray:
        """Each cell's likeliest kind (surface index) under the belief; -1 where never seen nor felt."""
        out = np.full(self.N, -1)
        seen = np.nonzero(self.seen)[0]
        if seen.size:
            out[seen] = self.posterior(seen).argmax(1)
        return out


def frame_evidence(lp: np.ndarray, dem_cell: np.ndarray, dem_dist: np.ndarray, kinds: np.ndarray,
                   max_range: float = math.inf) -> tuple:
    """One frame's evidence per cell: the pixels a model thinks are ground (probability over 1/2), put on the cell
    the elevation data says they show, each giving the log-probability of every kind given that it is ground, and
    averaged per cell (pixels of one frame aren't independent looks). Returns (cells, mean log-probabilities)."""
    lp = lp.reshape(-1, lp.shape[-1])
    cell = dem_cell.ravel()
    p_ground = np.exp(lp[:, kinds]).sum(1)
    use = (cell >= 0) & (p_ground > 0.5) & (dem_dist.ravel() <= max_range)
    if not use.any():
        return np.zeros(0, np.int64), np.zeros((0, kinds.size))
    lk = np.maximum(lp[use][:, kinds] - np.log(p_ground[use])[:, None], SURE)
    cells, inv = np.unique(cell[use], return_inverse=True)
    total = np.zeros((cells.size, kinds.size))
    np.add.at(total, inv, lk)
    count = np.bincount(inv, minlength=cells.size)
    return cells.astype(np.int64), total / count[:, None]


def truth_log_probs(label: np.ndarray) -> np.ndarray:
    """Perfect labels as log-probabilities: 0 for the true class, SURE for the rest."""
    lp = np.full(label.shape + (len(cam.CLASSES),), SURE)
    np.put_along_axis(lp, label[..., None].astype(np.int64), 0.0, axis=-1)
    return lp


def kind_prior(theme: str, pixels: list) -> np.ndarray:
    """Log-prior over a theme's kinds (surface order) from the training pixels' shares."""
    names = theme_kinds(theme)
    n = np.array([pixels[cam.KINDS.index(k)] for k in names], float)
    return np.log(np.maximum(n, 1.0) / n.sum())


def see_drive(t, sc, start, goal, see, look: cam.Look, prior: np.ndarray, mode: str = "expected",
              temper: float = 1.0, max_range: float = math.inf, unseen: str = "prior", spread: int = 0,
              spread_weight: float = 1.0, max_steps: int | None = None) -> dict:
    """Drive from `start` to `goal` re-planning with D* Lite (robotnav.replan) as Bruce senses and sees.

    `see` is how he learns the ground: "blind" (only what he drives over), "map" (he knows every cell's kind from the
    start: the reference), "truth" (his camera, with every pixel labelled right), or a model (ColourModel, PixelModel,
    Net) labelling what his camera sees. Obstacles: his short-range sensor (LIDAR metres) finds them, as in
    robotnav.replan. The camera looks the way he is about to drive, one frame a step, and only pixels up to
    `max_range` metres away count; `mode`, `temper`, `unseen` and `spread` say how his belief prices the ground
    (GroundBelief). Returns the route and what it really took (the terrain's own drive time)."""
    from ..replan import DStarLite, disk
    from ..terrain import add_up
    theme = t.meta["theme"]
    C = t.cols
    si, gi = start[0] * C + start[1], goal[0] * C + goal[1]
    kinds = np.array([cam.KINDS.index(n) for n in t.surface_names])                # surface index -> class
    surface = np.asarray(t.surface).ravel()
    world = np.asarray(t.blocked).ravel()
    b = GroundBelief(t, prior, mode=mode, temper=temper, unseen=unseen, spread=spread, spread_weight=spread_weight)
    if see == "map":
        for x in range(b.N):
            b.felt[x] = True
            b.kind[x] = surface[x]
            b.S[x] = float(b.costs[surface[x]])
    offsets = disk(LIDAR)

    def sense(at: int) -> list:
        r, c = divmod(at, C)
        changed = []
        for dr, dc in offsets:
            nr, nc = r + dr, c + dc
            if 0 <= nr < t.rows and 0 <= nc < C:
                x = nr * C + nc
                if world[x] and not b.blk[x]:
                    b.blk[x] = True
                    changed.append(x)
        return changed

    def look_at(at: int, step: int, toward: int) -> list:
        if see in ("blind", "map"):
            return []
        r, c = divmod(at, C)
        tr, tc = divmod(toward, C)
        heading = math.atan2(tc - c, -(tr - r))
        f = cam.render(sc, (float(r), float(c), heading), replace(look, seed=look.seed + step))   # new noise each frame
        lp = truth_log_probs(f.label) if see == "truth" else see.log_probs(f.image, encode_dist(f.dem_dist), theme)
        cells, ev = frame_evidence(lp, f.dem_cell, f.dem_dist, kinds, max_range)
        return b.observe(cells, ev)

    t0 = time.perf_counter()
    sense(si)
    b.feel(si, int(surface[si]))
    b.neighbourhood()
    ds_ = DStarLite(b, si, gi)
    ds_.compute()
    plan = ds_.plan()
    driven, frames, work = [si], 0, 0
    limit = max_steps or 4 * b.N
    cur = si
    while plan and cur != gi and len(driven) <= limit:
        ahead = plan[1]
        changed = look_at(cur, len(driven), ahead)
        if see not in ("blind", "map"):
            frames += 1
            changed += b.neighbourhood()
        if changed:
            ds_.moved_and_sensed(sorted(set(changed)))
            work += ds_.compute()
            plan = ds_.plan()
            if not plan:
                break
        cur = plan[1]
        driven.append(cur)
        ds_.start = cur
        changed = sense(cur) + b.feel(cur, int(surface[cur])) + b.neighbourhood()
        if changed:
            ds_.moved_and_sensed(sorted(set(changed)))
            work += ds_.compute()
        plan = ds_.plan()
    found = cur == gi
    time_true = (add_up(t.edge_cost(divmod(a, C), divmod(z, C)) for a, z in zip(driven, driven[1:]))
                 if found else math.inf)
    seen = np.nonzero(b.seen)[0]
    right = int((b.likeliest()[seen] == surface[seen]).sum())
    return dict(found=found, time=time_true, driven=driven, frames=frames, work=work, seen=int(seen.size),
                seen_right=right, seconds=round(time.perf_counter() - t0, 2))


TRIP_SALT = 20261119
TRIP_RANGE = (40.0, 70.0)                     # metres between a trip's start and goal, as the crow flies


def pick_trips(region: np.ndarray, theme: str, seed: int, n: int = 2) -> list:
    """`n` trips across a map's largest drivable region, 40-70 m as the crow flies (long enough that seeing ahead
    matters): the same ones for the same map, every time."""
    rng = np.random.default_rng([TRIP_SALT, THEMES.index(theme), seed])
    cells = np.argwhere(region)
    out = []
    while len(out) < n:
        a, b = cells[rng.integers(len(cells))], cells[rng.integers(len(cells))]
        d = float(np.hypot(*(a - b)))
        if TRIP_RANGE[0] <= d <= TRIP_RANGE[1]:
            out.append((tuple(int(v) for v in a), tuple(int(v) for v in b)))
    return out


# ---------------------------------------------------------------- the study: drives on maps he has never seen
VARIANTS = ("map", "blind", "truth", "colour", "pixel", "net")
VARIANT_NAMES = {"map": "knows the ground", "blind": "no camera", "truth": "camera, perfect labels",
                 "colour": "camera + colour only", "pixel": "camera + classic ML", "net": "camera + network"}
SETTINGS = dict(mode="expected", temper=1.0, max_range=math.inf, unseen="prior")   # every drive's, but for CHOSEN's
# Each way of seeing's own settings, chosen on the validation drives (5 maps per theme, 2 trips each; mean extra time
# in brackets) among: every pixel a model labels, or only those within 20 m or 10 m; a cell's time from the slowdown
# its belief expects, or from its likeliest kind's; unseen ground priced by the ground within 3, 6 or 10 cells of it
# (GroundBelief's spread), or not. Perfect labels: spread 10 (2.0%; 4.3% without). Network and classic ML: pixels
# within 20 m, spread 6 (4.2% and 3.7%; 6.0% and 4.5% with all pixels and no spread). Colour only: all pixels, no
# spread (4.8%; spreading spread its mistakes: 5.4%). No camera: spread 3 (9.2%; 9.5% without). Unseen ground is
# priced at what the prior expects, not at the fastest (with perfect labels: 4.3% against 5.1%).
CHOSEN = {"blind": dict(spread=3), "truth": dict(spread=10), "colour": dict(),
          "pixel": dict(max_range=20.0, spread=6), "net": dict(max_range=20.0, spread=6)}
# The validation drives CHOSEN was picked from (choose), in the order they were made: (way of seeing, its settings on
# top of SETTINGS). First the references and how unseen ground is priced, with perfect labels; then each model's
# pixels and pricing; then spreading beliefs to the ground round them, once the first drives showed the need.
VALIDATION_RUNS = (
    ("map", {}), ("blind", {}), ("truth", {}), ("truth", {"unseen": "fastest"}),
    ("net", {}), ("net", {"mode": "likeliest"}), ("net", {"temper": 0.5}), ("net", {"max_range": 20.0}),
    *(("colour", dict(m, **r)) for m in ({}, {"mode": "likeliest"})
      for r in ({}, {"max_range": 20.0}, {"max_range": 10.0})),
    ("pixel", {}), ("pixel", {"max_range": 20.0}), ("pixel", {"max_range": 10.0}), ("pixel", {"mode": "likeliest"}),
    ("truth", {"spread": 3}), ("blind", {"spread": 3}), ("truth", {"spread": 6}), ("blind", {"spread": 6}),
    ("colour", {"spread": 3}), ("colour", {"spread": 6}), ("truth", {"spread": 10}),
    *((k, dict(m, **r, spread=6)) for k in ("net", "pixel", "colour") for m in ({}, {"mode": "likeliest"})
      for r in ({}, {"max_range": 20.0}) if (k, m, r) != ("colour", {}, {})),
    ("blind", {"spread": 10}), ("net", {"max_range": 20.0, "spread": 10}), ("pixel", {"max_range": 20.0, "spread": 10}),
    ("net", {"max_range": 20.0, "spread": 3}), ("pixel", {"max_range": 20.0, "spread": 3}),
)
_W: dict = {}


def _drive_map(task) -> list:
    """Every trip of one map, driven each way asked (one worker's share of the study)."""
    from dataclasses import replace as dc_replace

    from ..algorithms import astar
    entry, terrain_dir, model_path, variants, settings, chosen, trips, pixels = task
    if "models" not in _W or _W.get("path") != model_path:
        _W["models"], _W["path"] = (models_from(Path(model_path)) if model_path else {}), model_path
    arr = ds.load(terrain_dir, entry)
    t = ds.terrain_of(arr)
    sc = cam.Scene(t)
    prior = kind_prior(entry["theme"], pixels)
    rows = []
    for k, (s, g) in enumerate(pick_trips(arr["region"], entry["theme"], entry["seed"], trips)):
        best = astar(dc_replace(t, start=s, goal=g, checkpoints=[]))
        look = cam.random_look(np.random.default_rng([TRIP_SALT, entry["seed"], k]))
        for v in variants:
            see = v if v in ("map", "blind", "truth") else _W["models"][v]
            r = see_drive(t, sc, s, g, see, look, prior, **{**settings, **chosen.get(v, {})})
            rows.append(dict(map=entry["id"], theme=entry["theme"], trip=k, start=list(s), goal=list(g),
                             variant=v, optimal=best.cost, time=r["time"], extra=r["time"] / best.cost - 1.0,
                             frames=r["frames"], seen=r["seen"], seen_right=r["seen_right"], steps=len(r["driven"]) - 1,
                             seconds=r["seconds"]))
    return rows


def drive_study(split: str, variants=VARIANTS, settings: dict = SETTINGS, chosen: dict = CHOSEN,
                model_path: Path = ASSET, terrain_dir: Path = ds.DEFAULT_DIR, vision_dir: Path = DEFAULT_DIR,
                trips: int = 2, jobs: int = 2, maps: int | None = None, log=print) -> list:
    """Drive `trips` trips on every map of a split, each of the ways in `variants`, with the same light; returns one
    row per drive (its true time, against the quickest the whole map allows). Every drive uses `settings`, a model's
    own from `chosen` on top."""
    pixels = load_manifest(vision_dir)["pixels"]["train"]                # the kinds' shares: Bruce's prior
    entries = [e for e in ds.load_manifest(terrain_dir)["maps"] if e["split"] == split]
    if maps:
        entries = [e for th in THEMES for e in [e for e in entries if e["theme"] == th][:maps]]
    tasks = [(e, str(terrain_dir), str(model_path) if model_path else "", tuple(variants), settings, chosen, trips,
              pixels) for e in entries]
    rows, t0 = [], time.perf_counter()
    ctx = get_context("fork") if os.name == "posix" else get_context()
    with ctx.Pool(max(1, jobs)) as pool:
        for k, r in enumerate(pool.imap(_drive_map, tasks), 1):
            rows += r
            if log and (k % 9 == 0 or k == len(tasks)):
                log(f"  {k}/{len(tasks)} maps driven  ({time.perf_counter() - t0:.0f} s)")
    return rows


def choose(model_path: Path = ASSET, terrain_dir: Path = ds.DEFAULT_DIR, vision_dir: Path = DEFAULT_DIR,
           maps: int = 5, trips: int = 2, jobs: int = 2, log=print) -> dict:
    """Drive every run of VALIDATION_RUNS on the first `maps` validation maps of each theme, `trips` trips each, and
    pick each way of seeing's settings: those with the lowest mean extra time. Returns every run and the picks (which
    are CHOSEN)."""
    runs = []
    for v, over in VALIDATION_RUNS:
        rows = drive_study("val", variants=(v,), settings={**SETTINGS, **over}, chosen={}, model_path=model_path,
                           terrain_dir=terrain_dir, vision_dir=vision_dir, trips=trips, jobs=jobs, maps=maps, log=None)
        x = np.array([r["extra"] for r in rows])
        runs.append(dict(variant=v, settings=over, trips=len(x), mean=float(x.mean()), median=float(np.median(x)),
                         worst=float(x.max()),
                         per_theme={th: float(np.mean([r["extra"] for r in rows if r["theme"] == th])) for th in THEMES}))
        if log:
            log(f"  {VARIANT_NAMES[v]:<24} {json.dumps(over):<44} mean {100 * x.mean():5.2f}%  median "
                f"{100 * np.median(x):5.2f}%  worst {100 * x.max():5.1f}%")
    picks = {v: min((r for r in runs if r["variant"] == v), key=lambda r: r["mean"])["settings"]
             for v in VARIANTS if v != "map" and any(r["variant"] == v for r in runs)}
    return dict(runs=runs, chosen=picks)


def score_table(scores: list) -> str:
    """Pixel scores side by side: all pixels, ground pixels (the right kind?), ground by distance, calibration."""
    bands = [s["band"] for s in scores[0]["by_distance"]]
    head = [f"{a:.0f}-{b:.0f} m" if math.isfinite(b) else f"{a:.0f} m+" for a, b in bands]
    lines = [f"  {'':<22}{'all pixels':>11}{'ground':>9}   ground, by distance: " + "".join(f"{h:>9}" for h in head)
             + f"{'calibration':>13}"]
    for s in scores:
        lines.append(f"  {s['model']:<22}{100 * s['accuracy']:>10.1f}%{100 * s['ground_accuracy']:>8.1f}%   "
                     f"{'':<21}" + "".join(f"{100 * b['accuracy']:>8.1f}%" for b in s["by_distance"])
                     + f"{s['calibration']['ece']:>13.4f}")
    return "\n".join(lines)


def drive_table(summary: list, theme: str = "all") -> str:
    """Drive results for one theme (or all): extra time over the quickest route the whole map allows."""
    lines = [f"  {'':<26}{'trips':>6}{'mean':>8}{'median':>8}{'worst':>8}{'within 1%':>11}{'blind gap closed':>18}"
             f"{'frames':>8}"]
    for s in summary:
        if s["theme"] != theme:
            continue
        gap = "" if s["removed"] is None else f"{100 * s['removed']:.0f}%"
        lines.append(f"  {s['name']:<26}{s['trips']:>6}{100 * s['mean']:>7.1f}%{100 * s['median']:>7.1f}%"
                     f"{100 * s['worst']:>7.1f}%{100 * s['within_1']:>10.0f}%{gap:>18}{s['frames']:>8.0f}")
    return "\n".join(lines)


def summarize_drives(rows: list) -> list:
    """Per way of seeing (and per theme): mean, median and worst extra time over the quickest route, the share of
    trips within 1% of it, and how much of the extra time a blind drive costs that seeing took away."""
    out = []
    for theme in ("all",) + THEMES:
        sel = [r for r in rows if theme == "all" or r["theme"] == theme]
        by = {v: [r for r in sel if r["variant"] == v] for v in VARIANTS}
        mean = {v: float(np.mean([r["extra"] for r in rs])) for v, rs in by.items() if rs}
        for v, rs in by.items():
            if not rs:
                continue
            x = np.array([r["extra"] for r in rs])
            gain = None
            if {"blind", "map"} <= mean.keys() and mean["blind"] > mean["map"]:
                gain = float((mean["blind"] - mean[v]) / (mean["blind"] - mean["map"]))
            seen = sum(r["seen"] for r in rs)
            out.append(dict(theme=theme, variant=v, name=VARIANT_NAMES[v], trips=len(rs), mean=float(x.mean()),
                            median=float(np.median(x)), worst=float(x.max()), within_1=float(np.mean(x <= 0.01)),
                            removed=gain, frames=float(np.mean([r["frames"] for r in rs])),
                            seen_right=(sum(r["seen_right"] for r in rs) / seen) if seen else None,
                            seconds=float(np.mean([r["seconds"] for r in rs]))))
    return out
