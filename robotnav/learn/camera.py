"""Stage 4: Bruce's camera. A small forward camera on the rover, simulated by casting a ray for every pixel over the map.

Until stage 4 Bruce knows the whole map. From stage 4 he knows its heights (as he would from satellite elevation
data) but not its kinds of ground: he has to see them. This module is the camera he sees them with. For a position
and heading it renders what the camera would show, and for every pixel what is really there (the label), which cell
of the map it shows and how far away that is.

How a frame is made. Each pixel's ray is marched out from the camera's eye (1.2 m above the ground, tilted 10 degrees
down) in steps that grow with distance, against the ground (heights interpolated between cell centres), the water
(ponds and the sea are flat at their level) and the obstacles (trees as upright cylinders, rocks as domes). The
first thing a ray meets is what its pixel shows. Where it meets the ground or the water, the crossing between the
last two steps is found by linear interpolation, so the point, its cell and its distance are good to a few
centimetres. Each ray is also marched on to the bare ground, as the elevation data alone would place it: that is the
cell Bruce himself puts the pixel on (he doesn't know where the trees are), and a distance he knows before he sees
anything.

What the pixels look like. Each kind of ground has the two tones the viewer paints it in, mixed by a slow pattern
across the map, and a texture of its own drawn on the ground itself: ripples on soft sand, streaks in tall grass,
pebbles on rocky ground, blotches of leaves in the jungle. Every cell has a slight tint of its own, blended across
its edges. Fine texture fades with distance, as it would through a real lens: 20 m away, a pixel covers about a
metre of ground. The sun (direction and strength), the haze, the exposure, the white balance and the sensor noise
change from frame to frame (Look), so no model can lean on one light. All of this is made up for the study; what is
real is the problem it poses: firm and soft sand overlap in colour, so do grass and tall grass, and far ground is a
blur.

Everything is NumPy and deterministic: the same map, pose and look give the same frame, bit for bit.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..terrain import ROCK, TREE, WATER, Terrain
from .speeds import KINDS

CLASSES = KINDS + ("Rock", "Tree", "Water", "Sky")
GROUND = len(KINDS)                            # class ids below this are kinds of ground
ROCK_C, TREE_C, WATER_C, SKY_C = GROUND, GROUND + 1, GROUND + 2, GROUND + 3
ROCK_R = 0.55                                  # a rock: a dome this wide, sitting on its cell's centre

# The viewer's own colours for each map (robotnav/viewer3d.html, THEMES, before the console mutes them): two tones per
# kind of ground, the rocks, the trees (leaves and trunk), the water (shallow, deep) and the sky (horizon, zenith).
PALETTE = {
    "grassland": dict(surf={"Grass": ("#3a7a2a", "#74ab47"), "Tall grass": ("#285c22", "#40782e")},
                      rock="#8f8d86", leaf="#3d8a33", trunk="#5c4330", water=("#2fd0c4", "#08406c"),
                      sky=("#ffd3dc", "#4d8fe0")),
    "desert": dict(surf={"Firm sand": ("#c99a5b", "#e2bf86"), "Soft sand": ("#e4c48e", "#f4dcab"),
                         "Rocky ground": ("#9c6a43", "#ba8659"), "Oasis grass": ("#5f8a36", "#86ad4c")},
                   rock="#b5774a", leaf="#4f8a3f", trunk="#8f7254", water=("#14c4b6", "#0a4558"),
                   sky=("#ffd3cf", "#5295dc")),
    "island": dict(surf={"Grass": ("#4a8f37", "#78b250"), "Sand": ("#e6d39f", "#f0e2b8"),
                         "Jungle": ("#235c28", "#3a7631")},
                   rock="#55535a", leaf="#35803a", trunk="#8a7056", water=("#2fd0c4", "#08406c"),
                   sky=("#ffd3de", "#3d8de2")),
}
# Trees by map: (radius, height in metres, share of the height that is trunk). The desert's are cacti, except the
# palms round its oases.
TREES = {"grassland": (0.6, 3.6, 0.35), "desert": (0.3, 1.9, 0.0), "island": (0.35, 4.4, 0.75)}
PALM = (0.35, 4.2, 0.75)
# Each kind's texture: (how strongly it marks the ground, its wavelength in metres). Fine textures fade with distance.
TEXTURE = {"Grass": (0.16, 0.25), "Tall grass": (0.30, 0.35), "Firm sand": (0.07, 0.15), "Soft sand": (0.12, 0.5),
           "Rocky ground": (0.28, 0.2), "Oasis grass": (0.18, 0.3), "Sand": (0.06, 0.12), "Jungle": (0.40, 0.45)}


def _hex(s: str) -> np.ndarray:
    return np.array([int(s[i:i + 2], 16) / 255 for i in (1, 3, 5)])


# ---------------------------------------------------------------- the camera, the look of a frame, a frame
@dataclass(frozen=True)
class Camera:
    """A pinhole camera on the rover: `width` x `height` pixels across `hfov` degrees, `eye` metres above the
    ground, tilted `pitch` degrees (down when negative). Rays are marched from `near` to `far` metres in steps of
    `step` metres plus `growth` of the distance travelled."""
    width: int = 128
    height: int = 64
    hfov: float = 90.0
    pitch: float = -10.0
    eye: float = 1.2
    near: float = 0.25
    far: float = 100.0
    step: float = 0.06
    growth: float = 0.02

    @property
    def focal(self) -> float:
        return (self.width / 2) / math.tan(math.radians(self.hfov) / 2)

    @property
    def pixel_angle(self) -> float:
        """Radians across one pixel, at the centre of the image."""
        return 1.0 / self.focal

    def samples(self) -> np.ndarray:
        """The distances each ray is tested at."""
        out, t = [], self.near
        while t < self.far:
            out.append(t)
            t += self.step + self.growth * t
        return np.array(out)

    def rays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Each pixel's unit ray in the rover's level frame: (forward, right, up), arrays of height x width."""
        f = self.focal
        x = np.arange(self.width) - self.width / 2 + 0.5
        y = -(np.arange(self.height) - self.height / 2 + 0.5)
        xx, yy = np.meshgrid(x, y)
        n = np.sqrt(f * f + xx * xx + yy * yy)
        fwd, right, up = f / n, xx / n, yy / n
        p = math.radians(self.pitch)
        return fwd * math.cos(p) - up * math.sin(p), right, fwd * math.sin(p) + up * math.cos(p)


CAMERA = Camera()


@dataclass(frozen=True)
class Look:
    """How a frame is lit and exposed: the sun's direction (azimuth from north, elevation, radians), the share of
    ambient light, the haze (metres to fade most of the way), the exposure, the white balance and the sensor noise
    (standard deviation, 0-1 scale; `seed` draws it)."""
    sun_az: float = 2.4
    sun_el: float = 0.8
    ambient: float = 0.5
    fog: float = 110.0
    exposure: float = 1.0
    balance: tuple = (1.0, 1.0, 1.0)
    noise: float = 0.02
    seed: int = 0


def random_look(rng: np.random.Generator) -> Look:
    """A look drawn at random: any sun from 25 to 70 degrees up, haze, exposure +-20%, white balance +-6%, noise."""
    return Look(sun_az=float(rng.uniform(0, 2 * math.pi)), sun_el=float(rng.uniform(math.radians(25), math.radians(70))),
                ambient=float(rng.uniform(0.4, 0.6)), fog=float(rng.uniform(60, 160)),
                exposure=float(rng.uniform(0.8, 1.2)), balance=tuple(float(v) for v in rng.uniform(0.94, 1.06, 3)),
                noise=float(rng.uniform(0.01, 0.03)), seed=int(rng.integers(1 << 31)))


@dataclass
class Frame:
    """One camera frame. image: height x width x 3 bytes. label: each pixel's class (CLASSES). cell: the cell it
    shows (row * cols + col), -1 for sky. dist: metres to what it shows, inf for sky. dem_cell and dem_dist: the
    same on the bare ground (what the elevation data alone says the pixel shows), -1 / inf where the ray leaves the
    map first."""
    image: np.ndarray
    label: np.ndarray
    cell: np.ndarray
    dist: np.ndarray
    dem_cell: np.ndarray
    dem_dist: np.ndarray
    pose: tuple


# ---------------------------------------------------------------- deterministic noise, for textures and tints
def _hash01(ix: np.ndarray, iy: np.ndarray, seed: int) -> np.ndarray:
    """A number in [0, 1) for each integer lattice point (x, y), the same on any machine (32-bit integer mixing)."""
    m = 0xFFFFFFFF
    h = (np.asarray(ix, np.int64) * 374761393 + np.asarray(iy, np.int64) * 668265263 + (seed * 2246822519 & m)) & m
    h = h.astype(np.uint64)                       # 32-bit values times a 32-bit constant fit in 64 bits: no overflow
    h = ((h ^ (h >> np.uint64(13))) * np.uint64(3266489917)) & np.uint64(m)
    h = (h ^ (h >> np.uint64(16))) & np.uint64(m)
    return h / 4294967296.0


def value_noise(x: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    """Smooth noise in [0, 1): random values on the integer lattice, eased between them."""
    ix, iy = np.floor(x), np.floor(y)
    fx, fy = x - ix, y - iy
    fx, fy = fx * fx * (3 - 2 * fx), fy * fy * (3 - 2 * fy)
    ix, iy = ix.astype(np.int64), iy.astype(np.int64)
    a, b = _hash01(ix, iy, seed), _hash01(ix + 1, iy, seed)
    c, d = _hash01(ix, iy + 1, seed), _hash01(ix + 1, iy + 1, seed)
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


# ---------------------------------------------------------------- a map, ready to be filmed
def water_levels(t: Terrain) -> np.ndarray:
    """The water's surface over each water cell (the sea's level, or its pond's), -inf elsewhere. A stored map has
    no pond records (robotnav.learn.dataset), so a pond's level is put just above its deepest-lying shore: between
    its highest water cell and its lowest bank, as the generator fills them."""
    from scipy.ndimage import binary_dilation, label
    water = np.asarray(t.obstacle) == WATER
    level = np.full(water.shape, -np.inf)
    if t.sea_level is not None:
        level[water] = float(t.sea_level)
    for p in t.ponds or []:
        for r, c in p["cells"]:
            level[r, c] = float(p["level"])
    left = water & ~np.isfinite(level)
    if left.any():
        lab, n = label(left, structure=np.ones((3, 3)))
        H = np.asarray(t.height, float)
        for k in range(1, n + 1):
            pond = lab == k
            bank = binary_dilation(pond, structure=np.ones((3, 3))) & ~water
            top = float(H[pond].max())
            level[pond] = max(top + 0.02, float(H[bank].min()) - 0.1) if bank.any() else top + 0.02
    return level


class Scene:
    """What the camera needs of one map, worked out once: heights, obstacles, water levels, each cell's class and
    colours, the trees' sizes, and the map's own texture seed (so a patch of ground looks the same in every frame)."""

    def __init__(self, t: Terrain, seed: int | None = None):
        theme = t.meta.get("theme", "grassland")
        if theme not in PALETTE:
            raise ValueError(f"no camera palette for a {theme!r} map")
        pal = PALETTE[theme]
        self.t, self.theme = t, theme
        self.R, self.C = t.rows, t.cols
        self.seed = int(t.meta.get("seed") or 0) if seed is None else int(seed)
        self.H = np.asarray(t.height, dtype=float)
        self.obstacle = np.asarray(t.obstacle, dtype=np.int8)
        self.level = water_levels(t)
        self.kinds = np.array([KINDS.index(n) for n in t.surface_names])           # surface index -> class
        self.cls = self.kinds[np.asarray(t.surface)]                               # each cell's kind of ground
        self.dark = np.array([_hex(pal["surf"][n][0]) for n in t.surface_names])[np.asarray(t.surface)]
        self.light = np.array([_hex(pal["surf"][n][1]) for n in t.surface_names])[np.asarray(t.surface)]
        rr, cc = np.mgrid[0:self.R, 0:self.C]
        self.tint = 1.0 + 0.14 * (_hash01(rr, cc, self.seed * 7 + 1) - 0.5)         # +-7% per cell
        self.wind = 2 * math.pi * float(_hash01(np.array(3), np.array(5), self.seed))  # the ripples' direction
        radius, height, trunk = TREES[theme]
        self.tree_r = np.full((self.R, self.C), radius)
        self.tree_h = np.full((self.R, self.C), height)
        self.tree_trunk = np.full((self.R, self.C), trunk)
        if theme == "desert" and (self.obstacle == WATER).any():
            from scipy.ndimage import distance_transform_edt
            near = distance_transform_edt(self.obstacle != WATER) <= 4.5      # palms round the oases
            self.tree_r[near], self.tree_h[near], self.tree_trunk[near] = PALM
        self.rock = _hex(pal["rock"])
        self.leaf, self.trunk = _hex(pal["leaf"]), _hex(pal["trunk"])
        self.shallow, self.deep = (_hex(v) for v in pal["water"])
        self.horizon, self.zenith = (_hex(v) for v in pal["sky"])

    def ground(self, r: np.ndarray, c: np.ndarray) -> np.ndarray:
        """Bilinear height between cell centres at (r, c) (clamped to the map)."""
        return self.between(self.H, r, c)

    def between(self, A: np.ndarray, r: np.ndarray, c: np.ndarray) -> np.ndarray:
        """A per-cell array, read bilinearly between cell centres at (r, c) (clamped to the map)."""
        r = np.clip(r, 0, self.R - 1)
        c = np.clip(c, 0, self.C - 1)
        r0 = np.minimum(np.floor(r).astype(np.int64), self.R - 2)
        c0 = np.minimum(np.floor(c).astype(np.int64), self.C - 2)
        fr, fc = r - r0, c - c0
        return ((A[r0, c0] * (1 - fr) + A[r0 + 1, c0] * fr) * (1 - fc)
                + (A[r0, c0 + 1] * (1 - fr) + A[r0 + 1, c0 + 1] * fr) * fc)

    def slope(self, r: np.ndarray, c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The bilinear surface's gradient (d height / d row, d height / d col) at (r, c)."""
        r = np.clip(r, 0, self.R - 1)
        c = np.clip(c, 0, self.C - 1)
        r0 = np.minimum(np.floor(r).astype(np.int64), self.R - 2)
        c0 = np.minimum(np.floor(c).astype(np.int64), self.C - 2)
        fr, fc = r - r0, c - c0
        H = self.H
        dr = (H[r0 + 1, c0] - H[r0, c0]) * (1 - fc) + (H[r0 + 1, c0 + 1] - H[r0, c0 + 1]) * fc
        dc = (H[r0, c0 + 1] - H[r0, c0]) * (1 - fr) + (H[r0 + 1, c0 + 1] - H[r0 + 1, c0]) * fr
        return dr, dc


# ---------------------------------------------------------------- marching the rays
NOTHING, HIT_GROUND, HIT_WATER, HIT_TREE, HIT_ROCK = 0, 1, 2, 3, 4
CHUNK = 24                                     # samples tested at once per ray


def _surface_at(sc: Scene, pr, pc, pz):
    """At sample points: is each inside the map, under the bare ground, and what (if anything) it is inside."""
    inside = (pr >= 0) & (pr <= sc.R - 1) & (pc >= 0) & (pc <= sc.C - 1)
    g = sc.ground(pr, pc)
    rn = np.clip(np.rint(pr).astype(np.int64), 0, sc.R - 1)
    cn = np.clip(np.rint(pc).astype(np.int64), 0, sc.C - 1)
    ob = sc.obstacle[rn, cn]
    under = inside & (pz <= g)
    what = np.where(under, HIT_GROUND, NOTHING)
    water = inside & (ob == WATER) & (pz <= sc.level[rn, cn])
    what = np.where(water, HIT_WATER, what)
    if (ob == TREE).any() or (ob == ROCK).any():
        gc = sc.H[rn, cn]
        hd2 = (pr - rn) ** 2 + (pc - cn) ** 2
        tree = inside & (ob == TREE) & (hd2 <= sc.tree_r[rn, cn] ** 2) & (pz <= gc + sc.tree_h[rn, cn])
        rock = inside & (ob == ROCK) & (hd2 + (pz - gc) ** 2 <= ROCK_R * ROCK_R)
        what = np.where(tree, HIT_TREE, np.where(rock, HIT_ROCK, what))
    return inside, under, what


def march(sc: Scene, origin: tuple, dr: np.ndarray, dc: np.ndarray, dz: np.ndarray, cam: Camera = CAMERA):
    """March rays (unit directions dr, dc, dz in rows, cols, metres up) from `origin` (row, col, height).

    Returns (what, t_hit, t_dem): what each ray meets first (HIT_*, NOTHING if it leaves the map or goes past `far`),
    the distance to it, and the distance to the bare ground on its own (inf if none)."""
    r0, c0, z0 = origin
    T = cam.samples()
    n = dr.size
    what = np.zeros(n, np.int8)
    k_hit = np.full(n, -1)
    k_dem = np.full(n, -1)
    active = np.arange(n)
    for k0 in range(0, len(T), CHUNK):
        if active.size == 0:
            break
        tk = T[k0:k0 + CHUNK]
        pr = r0 + dr[active, None] * tk
        pc = c0 + dc[active, None] * tk
        pz = z0 + dz[active, None] * tk
        inside, under, w = _surface_at(sc, pr, pc, pz)
        hit = w != NOTHING
        first = np.where(hit.any(1), hit.argmax(1), -1)
        new = (k_hit[active] < 0) & (first >= 0)
        rays = active[new]
        k_hit[rays] = k0 + first[new]
        what[rays] = w[new, first[new]]
        g_first = np.where(under.any(1), under.argmax(1), -1)
        new = (k_dem[active] < 0) & (g_first >= 0)
        k_dem[active[new]] = k0 + g_first[new]
        gone = ~inside[:, -1] | (k_dem[active] >= 0)                    # left the map (it can't come back), or done
        active = active[~gone]
    t_hit = _refine(sc, origin, dr, dc, dz, T, k_hit, what, bare=False)
    t_dem = _refine(sc, origin, dr, dc, dz, T, k_dem, None, bare=True)
    return what, t_hit, t_dem


def _refine(sc, origin, dr, dc, dz, T, k, what, bare: bool) -> np.ndarray:
    """The distance where each ray crosses its surface, by linear interpolation between the samples either side
    (ground and water); obstacles are taken at the first sample inside them. inf where nothing was hit."""
    r0, c0, z0 = origin
    out = np.full(k.size, np.inf)
    hit = k >= 0
    if not hit.any():
        return out
    i = np.nonzero(hit)[0]
    kk = k[i]
    t1 = T[kk]
    out[i] = t1
    smooth = np.ones(i.size, bool) if bare else np.isin(what[i], (HIT_GROUND, HIT_WATER))
    smooth &= kk > 0
    j = i[smooth]
    if j.size:
        tb, ta = T[k[j]], T[k[j] - 1]

        def gap(t):
            r, c, z = r0 + dr[j] * t, c0 + dc[j] * t, z0 + dz[j] * t
            g = sc.ground(r, c)
            if bare:
                return z - g
            rn = np.clip(np.rint(r).astype(np.int64), 0, sc.R - 1)
            cn = np.clip(np.rint(c).astype(np.int64), 0, sc.C - 1)
            return z - np.maximum(g, sc.level[rn, cn])

        fa, fb = gap(ta), gap(tb)
        den = fa - fb
        frac = np.where(den > 0, fa / np.where(den > 0, den, 1.0), 1.0)
        out[j] = ta + (tb - ta) * np.clip(frac, 0.0, 1.0)
    return out


# ---------------------------------------------------------------- rendering
def render(sc: Scene, pose: tuple, look: Look = Look(), cam: Camera = CAMERA) -> Frame:
    """The frame Bruce's camera takes at `pose` = (row, col, heading): heading in radians, 0 looking north (up the
    map, towards row 0), pi/2 east."""
    r0, c0, heading = (float(v) for v in pose)
    fwd, right, up = cam.rays()
    ch, sh = math.cos(heading), math.sin(heading)
    dr = (-fwd * ch + right * sh).ravel()
    dc = (fwd * sh + right * ch).ravel()
    dz = up.ravel()
    z0 = float(sc.ground(np.array(r0), np.array(c0))) + cam.eye
    what, t_hit, t_dem = march(sc, (r0, c0, z0), dr, dc, dz, cam)

    n = dr.size
    shape = (cam.height, cam.width)
    hit = what != NOTHING
    tr = np.where(hit, t_hit, 0.0)
    hr, hc, hz = r0 + dr * tr, c0 + dc * tr, z0 + dz * tr
    rn = np.clip(np.rint(hr).astype(np.int64), 0, sc.R - 1)
    cn = np.clip(np.rint(hc).astype(np.int64), 0, sc.C - 1)
    cell = np.where(hit, rn * sc.C + cn, -1)
    label = np.full(n, SKY_C, np.int64)
    label = np.where(what == HIT_GROUND, sc.cls[rn, cn], label)
    label = np.where(what == HIT_WATER, WATER_C, label)
    label = np.where(what == HIT_TREE, TREE_C, label)
    label = np.where(what == HIT_ROCK, ROCK_C, label)

    color = _shade(sc, look, cam, what, hr, hc, hz, rn, cn, dr, dc, dz, t_hit)
    exp = look.exposure * np.asarray(look.balance)
    rng = np.random.default_rng(look.seed)
    color = color * exp + rng.normal(0.0, look.noise, color.shape)
    image = np.clip(np.rint(color * 255.0), 0, 255).astype(np.uint8).reshape(cam.height, cam.width, 3)

    dem = np.isfinite(t_dem)
    td = np.where(dem, t_dem, 0.0)
    dr_n = np.clip(np.rint(r0 + dr * td).astype(np.int64), 0, sc.R - 1)
    dc_n = np.clip(np.rint(c0 + dc * td).astype(np.int64), 0, sc.C - 1)
    return Frame(image=image, label=label.astype(np.uint8).reshape(shape), cell=cell.astype(np.int32).reshape(shape),
                 dist=np.where(hit, t_hit, np.inf).astype(np.float32).reshape(shape),
                 dem_cell=np.where(dem, dr_n * sc.C + dc_n, -1).astype(np.int32).reshape(shape),
                 dem_dist=np.where(dem, t_dem, np.inf).astype(np.float32).reshape(shape), pose=(r0, c0, heading))


def _shade(sc: Scene, look: Look, cam: Camera, what, hr, hc, hz, rn, cn, dr, dc, dz, dist) -> np.ndarray:
    """Each pixel's colour (0-1 RGB, before exposure and noise)."""
    n = what.size
    sun = np.array([-math.cos(look.sun_az) * math.cos(look.sun_el), math.sin(look.sun_az) * math.cos(look.sun_el),
                    math.sin(look.sun_el)])
    col = np.zeros((n, 3))
    normal = np.zeros((n, 3))
    normal[:, 2] = 1.0
    seed = sc.seed

    g = what == HIT_GROUND
    if g.any():
        x, y = hc[g], hr[g]
        gr, gc_ = sc.slope(y, x)
        nv = np.stack([-gr, -gc_, np.ones_like(gr)], 1)
        nv /= np.linalg.norm(nv, axis=1, keepdims=True)
        normal[g] = nv
        r, c = rn[g], cn[g]
        mix = value_noise(x / 3.0, y / 3.0, seed * 11 + 2) * 0.7 + value_noise(x / 1.1, y / 1.1, seed * 11 + 3) * 0.3
        base = sc.dark[r, c] * (1 - mix[:, None]) + sc.light[r, c] * mix[:, None]
        base *= sc.between(sc.tint, y, x)[:, None]                  # each cell's tint, blended across its edges
        # fine texture, faded where a pixel covers more ground than its wavelength
        view = np.stack([-dr[g], -dc[g], -dz[g]], 1)
        cos_i = np.clip(np.abs((view * nv).sum(1)), 0.05, 1.0)
        footprint = dist[g] * cam.pixel_angle / np.sqrt(cos_i)
        tex = np.zeros(x.size)
        kind = sc.cls[r, c]
        for k, name in enumerate(KINDS):
            m = kind == k
            if not m.any():
                continue
            amp, wave = TEXTURE[name]
            tex[m] = amp * _texture(name, x[m], y[m], sc.wind, seed) * np.exp(-(footprint[m] / wave) ** 2)
        col[g] = base * (1.0 + tex)[:, None]

    w = what == HIT_WATER
    if w.any():
        depth = np.clip(sc.level[rn[w], cn[w]] - sc.H[rn[w], cn[w]], 0, 3) / 3.0
        water = sc.shallow * (1 - depth[:, None]) + sc.deep * depth[:, None]
        fres = np.clip(1.0 - np.abs(dz[w]) * 3.0, 0.0, 1.0)[:, None] * 0.6         # the sky, mirrored at low angles
        col[w] = water * (1 - fres) + sc.horizon * fres

    tr = what == HIT_TREE
    if tr.any():
        r, c = rn[tr], cn[tr]
        frac = (hz[tr] - sc.H[r, c]) / sc.tree_h[r, c]
        trunk = frac < sc.tree_trunk[r, c]
        shade = 0.85 + 0.3 * value_noise(hc[tr] * 4.0, hz[tr] * 4.0, seed * 11 + 5)
        col[tr] = np.where(trunk[:, None], sc.trunk, sc.leaf) * shade[:, None]
        side = np.stack([hr[tr] - r, hc[tr] - c, np.zeros(r.size)], 1)
        side /= np.maximum(np.linalg.norm(side, axis=1, keepdims=True), 1e-9)
        normal[tr] = side

    rk = what == HIT_ROCK
    if rk.any():
        r, c = rn[rk], cn[rk]
        v = np.stack([hr[rk] - r, hc[rk] - c, hz[rk] - sc.H[r, c]], 1)
        normal[rk] = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)
        col[rk] = sc.rock * (0.8 + 0.4 * value_noise(hc[rk] * 6.0, hr[rk] * 6.0, seed * 11 + 7))[:, None]

    lit = what != NOTHING
    light = look.ambient + (1.0 - look.ambient) * 1.25 * np.clip((normal * sun).sum(1), 0.0, 1.0)
    col[lit] *= light[lit, None]
    col[w] = col[w] * 0.5 + col[w] / np.maximum(light[w, None], 1e-9) * 0.5     # water is lit more evenly
    haze = 1.0 - np.exp(-np.where(lit, dist, 0.0) / look.fog)
    col[lit] = col[lit] * (1 - haze[lit, None]) + sc.horizon * haze[lit, None]

    sky = ~lit
    if sky.any():
        e = np.clip(dz[sky] / 0.45, 0.0, 1.0)[:, None]
        col[sky] = sc.horizon * (1 - e) + sc.zenith * e
    return col


def _texture(name: str, x: np.ndarray, y: np.ndarray, wind: float, seed: int) -> np.ndarray:
    """A kind of ground's pattern on the ground itself, roughly in [-1, 1]."""
    s = seed * 11
    if name == "Soft sand":                                       # wind ripples, slightly wavy
        u = x * math.cos(wind) + y * math.sin(wind)
        return np.sin(2 * math.pi * u / 0.5 + 3.0 * value_noise(x / 2.0, y / 2.0, s + 21))
    if name == "Tall grass":                                      # streaks, combed one way by the wind
        u = x * math.cos(wind) + y * math.sin(wind)
        v = -x * math.sin(wind) + y * math.cos(wind)
        return 2 * value_noise(u * 1.2, v * 5.0, s + 22) - 1
    if name == "Rocky ground":                                    # pebbles: dark and pale specks
        p = value_noise(x * 5.0, y * 5.0, s + 23)
        return np.where(p > 0.72, -1.0, np.where(p < 0.22, 0.6, 0.0)) + (value_noise(x * 2.0, y * 2.0, s + 24) - 0.5)
    if name == "Jungle":                                          # blotches of leaves and shade
        return 2 * (0.6 * value_noise(x * 2.2, y * 2.2, s + 25) + 0.4 * value_noise(x * 6.0, y * 6.0, s + 26)) - 1
    if name in ("Grass", "Oasis grass"):                          # blades and clumps
        patch = value_noise(x * 1.5, y * 1.5, s + 27) if name == "Oasis grass" else 0.5
        return 2 * (0.7 * value_noise(x * 4.0, y * 4.0, s + 28) + 0.3 * patch) - 1
    return 2 * value_noise(x * 7.0, y * 7.0, s + 29) - 1          # sand: a fine grain


# ---------------------------------------------------------------- checking the geometry
def reference_hit(sc: Scene, origin: tuple, d: tuple, far: float = 100.0, step: float = 0.002,
                  bare: bool = False) -> tuple[int, float]:
    """One ray, tested every `step` metres with no growing steps and no interpolation: (what it meets first, the
    distance of the first sample inside it). The tests hold `march` to it."""
    r0, c0, z0 = origin
    dr, dc, dz = d
    block = 20000
    for k0 in range(0, int((far - 0.25) / step) + 1, block):
        t = 0.25 + step * np.arange(k0, k0 + block)
        inside, under, w = _surface_at(sc, (r0 + dr * t)[None], (c0 + dc * t)[None], (z0 + dz * t)[None])
        stop = (under if bare else w != NOTHING)[0] | ~inside[0]
        if stop.any():
            i = int(stop.argmax())
            if not inside[0, i]:
                return NOTHING, math.inf
            return (HIT_GROUND if bare else int(w[0, i])), float(t[i])
    return NOTHING, math.inf


def label_colors() -> np.ndarray:
    """A colour for each class (for pictures of labels): the kinds' light tones, then rock, tree (bark brown, apart
    from the jungle's green), water, sky."""
    tone = {}
    for pal in PALETTE.values():
        for name, (_, light) in pal["surf"].items():
            tone.setdefault(name, _hex(light))
    return np.array([tone[k] for k in KINDS] + [_hex("#8f8d86"), _hex("#7d5236"), _hex("#1aa5c0"), _hex("#cfe3ff")])
