"""Render the intro's bison logo (robotnav/assets/bison.webp): a wild bison charging head-on at the viewer, as a
black-metal relief.

1. Shape: every part (hump, legs, hooves, leg hair, beard, horns, face, mop, nose, eyes) is a closed outline in
   logo units (1000 x 860, centre line x = 500). Shaggy edges are curved, swept locks of uneven width; the head is
   symmetric, the legs are not (one reaching toward us, one tucked), so it reads as running.
2. Relief: each part becomes a height field (a rounded chamfer at its edge plus a soft pillow), with grooves
   between its locks; parts stack front over back.
3. Shading as black chrome: almost no body colour; light comes from reflections of a cool softbox upper left, the
   red sign above (seen through a smoothed surface, so only big forms catch it), a dim floor bounce and a horizon
   band, with Fresnel, a grazing rim and cavity darkening. The eyes get a red glint from the sign.

Needs numpy, scipy and Pillow (Pillow comes with matplotlib). Takes about 20 s.
Usage: python tools/bison_logo.py [OUT.webp]
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

CX = 500.0
U = 1.6                    # pixels per logo unit while rendering (then cropped and scaled to 600 px tall)
GROOVE = 2.5               # depth of the grooves between locks
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "robotnav" / "assets" / "bison.webp"


# ------------------------------------------------------------------------------------------------ outlines
def catmull(pts, n=12, closed=False):
    """A smooth curve through the control points."""
    P = np.asarray(pts, float)
    P = np.vstack([P[-1], P, P[0], P[1]]) if closed else np.vstack([P[0], P, P[-1]])
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        for t in np.linspace(0, 1, n, endpoint=False):
            t2, t3 = t * t, t * t * t
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    if not closed:
        out.append(P[-2])
    return np.array(out)


def resample(line, step):
    line = np.asarray(line, float)
    s = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(line, axis=0).T))])
    k = max(2, int(s[-1] / step) + 1)
    t = np.linspace(0, s[-1], k)
    return np.column_stack([np.interp(t, s, line[:, 0]), np.interp(t, s, line[:, 1])])


def locks(line, n, length, sweep, side=1.0, taper=(1.0, 1.0), curl=0.35, seed=0, jitter=0.25):
    """Replace an edge with n locks of hair: curved spikes whose tips lean `sweep` degrees off the outward normal
    (`side` flips which way is out); `taper` scales the lengths from the first lock to the last."""
    rng = np.random.default_rng(seed)
    L = resample(line, 1.0)
    s = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(L, axis=0).T))])
    out = []
    edges = np.linspace(0, s[-1], n + 1)
    if jitter > 0.3:                                     # uneven lock widths read as hair, even ones as a saw blade
        inner = edges[1:-1] + (rng.random(n - 1) - 0.5) * (s[-1] / n) * 0.55
        edges = np.concatenate([[0], np.sort(inner), [s[-1]]])
    for i in range(n):
        a, b = edges[i], edges[i + 1]
        ia, ib = np.searchsorted(s, a), min(len(L) - 1, np.searchsorted(s, b))
        pa, pb = L[ia], L[ib]
        mid = (pa + pb) / 2
        d = pb - pa; d /= np.linalg.norm(d) + 1e-9
        nrm = np.array([d[1], -d[0]]) * side
        f = taper[0] + (taper[1] - taper[0]) * (i / max(1, n - 1))
        ln = length * f * (1 - jitter + 2 * jitter * rng.random())
        ang = np.radians(sweep * (1 + 0.3 * (rng.random() - 0.5)))
        rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
        tip = mid + (rot @ nrm) * ln
        c1 = pa + (tip - pa) * 0.5 + nrm * ln * curl * 0.35    # the leading flank bulges,
        c2 = pb + (tip - pb) * 0.55 - nrm * ln * curl * 0.15   # the trailing one is hollow: a hooked lock
        out.extend(catmull([pa, c1, tip], 6)[:-1].tolist() + catmull([tip, c2, pb], 6)[:-1].tolist())
    out.append(L[-1])
    return np.array(out)


def mirror(pts):
    P = np.asarray(pts, float).copy()
    P[:, 0] = 2 * CX - P[:, 0]
    return P[::-1]


def sym(left_half):
    """A closed outline symmetric about the centre line, from its left half (top centre -> bottom centre)."""
    L = np.asarray(left_half, float)
    return np.vstack([L, mirror(L)[1:-1]])


def band(center, w0, w1=0.0, power=0.9, step=2.0):
    """A tapered band along a curve (horns, legs)."""
    C = resample(catmull(center, 16), step)
    d = np.gradient(C, axis=0); d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
    nrm = np.column_stack([-d[:, 1], d[:, 0]])
    t = np.linspace(0, 1, len(C))
    w = (w0 + (w1 - w0) * t ** power)[:, None] / 2
    return np.vstack([C + nrm * w, (C - nrm * w)[::-1]])


def xform(P, dx=0, dy=0, deg=0, sx=1, sy=1, about=(0, 0)):
    P = np.asarray(P, float) - about; a = np.radians(deg)
    P = P * [sx, sy]
    P = np.column_stack([P[:, 0] * np.cos(a) - P[:, 1] * np.sin(a), P[:, 0] * np.sin(a) + P[:, 1] * np.cos(a)])
    return P + about + [dx, dy]


def hoof(w, h):
    """A cloven hoof from the front, top centre at the origin: two toes with a split."""
    return catmull([(-w * 0.42, 0), (-w * 0.52, h * 0.55), (-w * 0.36, h), (-w * 0.08, h * 0.96), (0, h * 0.7),
                    (w * 0.08, h * 0.96), (w * 0.36, h), (w * 0.52, h * 0.55), (w * 0.42, 0)], 8, closed=True)


def parts():
    P = {}
    # the hump: a tall shaggy mound behind the lowered head, locks swept out and back by the run
    crest = catmull([(500, 40), (436, 46), (374, 66), (322, 98), (282, 140), (256, 192), (242, 250), (240, 310), (250, 370), (272, 420)], 12)
    crest = locks(crest, 11, 36, sweep=-50, side=1.0, taper=(0.75, 1.1), seed=13, jitter=0.5, curl=0.85)
    P["hump"] = sym(np.vstack([crest, [(330, 455), (420, 470), (500, 476)]]))
    # legs: the viewer's left reaching toward us, the viewer's right tucked with the knee up
    P["leg_l"] = band([(384, 572), (370, 620), (358, 668), (350, 712)], 90, 62, power=0.8)
    P["hoof_l"] = xform(hoof(84, 46), dx=347, dy=704, deg=6)
    P["leg_r"] = band([(622, 580), (636, 610), (646, 634), (650, 652)], 88, 66, power=0.8)
    P["hoof_r"] = xform(hoof(66, 30), dx=651, dy=646, deg=-14, sy=0.8)
    # the long hair hanging off the upper front legs
    side = locks(catmull([(360, 440), (310, 458), (288, 506), (290, 560)], 10), 3, 26, sweep=-48, side=-1.0, seed=4)
    bottom = locks(catmull([(290, 560), (328, 592), (378, 604), (420, 590)], 10), 5, 30, sweep=8, side=-1.0, seed=6)
    inner = catmull([(420, 590), (434, 540), (434, 480), (410, 444)], 10)
    P["chap_l"] = np.vstack([side, bottom[1:], inner[1:]])
    P["chap_r"] = mirror(xform(P["chap_l"], dy=-4))
    # the beard
    beard = catmull([(500, 506), (458, 512), (436, 542), (434, 586), (446, 628), (466, 664)], 12)
    bl = locks(beard[1:], 4, 24, sweep=28, side=1.0, taper=(0.7, 1.1), seed=8)
    P["beard"] = sym(np.vstack([[(500, 506)], bl, [(486, 700), (500, 718)]]))
    # horns: short and thick, out from under the mop, up and in
    P["horn_l"] = band([(372, 268), (330, 262), (298, 244), (282, 212), (284, 178), (300, 150), (316, 138)], 66, 2, power=1.0)
    P["horn_r"] = mirror(P["horn_l"])
    # the face: broad and flat, eyes far apart on its edges, cheeks lost in shaggy hair
    cheek = locks(catmull([(350, 296), (356, 350), (368, 400), (382, 446)], 10), 4, 14, sweep=40, side=1.0, seed=21, jitter=0.4)
    jaw = catmull([(382, 446), (392, 482), (404, 514), (428, 538), (462, 548), (500, 550)], 10)
    P["face"] = sym(np.vstack([[(500, 280), (400, 284), (356, 290)], cheek, jaw[1:]]))
    # the mop: a big shaggy cushion over the brow and the horn roots, its fringe hanging between the eyes
    mop_top = locks(catmull([(500, 150), (446, 154), (398, 170), (360, 198), (334, 236)], 12), 6, 24, sweep=-38, side=1.0, seed=11, jitter=0.45, curl=0.7)
    mop_side = locks(catmull([(334, 236), (326, 276), (336, 312)], 8), 2, 16, sweep=-20, side=1.0, seed=14, jitter=0.3)
    fringe = locks(catmull([(336, 312), (372, 322), (410, 330), (446, 346), (476, 362), (500, 378)], 10), 5, 18, sweep=-15, side=-1.0, seed=12, jitter=0.4, curl=0.5)
    P["mop"] = sym(np.vstack([mop_top, mop_side[1:], fringe[1:]]))
    # a wide nose pad with big nostrils; eyes deep under the fringe
    P["muzzle"] = sym(catmull([(500, 470), (452, 474), (418, 488), (406, 512), (424, 536), (462, 548), (500, 550)], 10))
    P["nostril_l"] = xform(catmull([(0, -16), (12, -6), (12, 8), (0, 14), (-12, 6), (-10, -8)], 6, closed=True), dx=452, dy=510, deg=-42)
    P["nostril_r"] = mirror(P["nostril_l"])
    P["eye_l"] = catmull([(366, 338), (382, 334), (398, 340), (402, 350), (388, 352), (372, 346)], 8, closed=True)
    P["eye_r"] = mirror(P["eye_l"])
    for k in ("face", "mop", "muzzle", "nostril_l", "nostril_r", "eye_l", "eye_r", "horn_l", "horn_r", "beard"):
        P[k] = xform(P[k], sx=1.1, sy=1.1, about=(500, 360))   # the head is the closest thing to us
    return P


ORDER = ["hump", "leg_l", "leg_r", "hoof_l", "hoof_r", "chap_l", "chap_r", "beard", "horn_l", "horn_r",
         "face", "muzzle", "eye_l", "eye_r", "mop", "nostril_l", "nostril_r"]
# base height, chamfer height, chamfer width, pillow height, pillow blur (units), which locks get grooves
SPEC = {
    "hump": (0, 9, 10, 46, 70, True), "leg_l": (18, 9, 10, 12, 16, False), "leg_r": (20, 9, 10, 12, 16, False),
    "hoof_l": (24, 7, 6, 6, 8, False), "hoof_r": (26, 7, 6, 6, 8, False), "chap_l": (26, 9, 10, 22, 30, True),
    "chap_r": (26, 9, 10, 22, 30, True), "beard": (36, 8, 10, 16, 22, True), "horn_l": (46, 14, 13, 6, 8, False),
    "horn_r": (46, 14, 13, 6, 8, False), "face": (40, 8, 12, 26, 60, True), "mop": (58, 10, 10, 22, 34, True),
    "muzzle": (52, 4, 26, 8, 26, False), "nostril_l": (58, -16, 5, 0, 0, False), "nostril_r": (58, -16, 5, 0, 0, False),
    "eye_l": (54, -2, 2, 0, 0, False), "eye_r": (54, -2, 2, 0, 0, False),
}
ROOT = {"hump": (500, 300), "chap_l": (362, 452), "chap_r": (638, 448), "beard": (500, 520), "mop": (500, 244), "face": (500, 330)}
REACH = {"hump": 64, "chap_l": 64, "chap_r": 64, "beard": 56, "mop": 40, "face": 40}


# ------------------------------------------------------------------------------------------------ relief
def notches_of(poly, max_angle=150.0):
    """Sharp concave vertices of a polygon: the roots between locks."""
    P = np.asarray(poly, float)
    area = 0.5 * np.sum(P[:, 0] * np.roll(P[:, 1], -1) - np.roll(P[:, 0], -1) * P[:, 1])
    out = []
    for i in range(len(P)):
        a, b, c = P[i - 1], P[i], P[(i + 1) % len(P)]
        u, v = a - b, c - b
        nu, nv = np.linalg.norm(u), np.linalg.norm(v)
        if nu < 1e-6 or nv < 1e-6: continue
        ang = np.degrees(np.arccos(np.clip(u @ v / (nu * nv), -1, 1)))
        cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        if cross * np.sign(area) < 0 and ang < max_angle:
            out.append(b)
    return np.array(out)


def curve(p0, p1, bend, n=12):
    """A gently bent path from p0 to p1 (bend: sideways offset of the middle, as a fraction of the length)."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0; L = np.linalg.norm(d) + 1e-9
    nrm = np.array([-d[1], d[0]]) / L
    t = np.linspace(0, 1, n)[:, None]
    return p0 + d * t + nrm * (bend * L * 4 * t * (1 - t))


def tube_map(paths, W, H):
    """Rounded tubes along paths ((points, width at start, width at end), in units), combined with max(), in [0, 1]."""
    M = np.zeros((H, W))
    for pts, w0, w1 in paths:
        pts = np.asarray(pts, float)
        s = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(pts, axis=0).T))])
        if s[-1] <= 0: continue
        t = np.linspace(0, s[-1], max(2, int(s[-1] * U / 1.2)))
        xs, ys = np.interp(t, s, pts[:, 0]) * U, np.interp(t, s, pts[:, 1]) * U
        for x, y, r in zip(xs, ys, (w0 + (w1 - w0) * (t / s[-1])) * U / 2):
            if r < 0.6: continue
            x0, x1 = max(0, int(x - r - 1)), min(W, int(x + r + 2))
            y0, y1 = max(0, int(y - r - 1)), min(H, int(y + r + 2))
            if x0 >= x1 or y0 >= y1: continue
            gx, gy = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1))
            prof = np.sqrt(np.clip(1 - ((gx - x) ** 2 + (gy - y) ** 2) / (r * r), 0, 1))
            np.maximum(M[y0:y1, x0:x1], prof, out=M[y0:y1, x0:x1])
    return M


def render():
    W, H = int(1000 * U), int(860 * U)
    rng = np.random.default_rng(21)
    P = parts()
    XX, YY = np.meshgrid(np.arange(W) / U, np.arange(H) / U)

    def mask_of(poly, ss=3):
        im = Image.new("L", (W * ss, H * ss), 0)
        ImageDraw.Draw(im).polygon([(x * U * ss, y * U * ss) for x, y in poly], fill=255)
        return np.asarray(im.resize((W, H), Image.BOX), float) / 255.0

    def grooves(k, inside):
        """One groove from every notch between two locks back toward the part's root; long flowing ones on the hump."""
        paths, root = [], np.array(ROOT[k], float)
        for i, q in enumerate(notches_of(P[k], 150)):
            d = root - q; L = np.linalg.norm(d)
            if L < 20: continue
            end = q + d / L * min(L * 0.8, REACH[k] * (0.6 + 0.8 * rng.random()))
            paths.append((curve(q + d / L * 2, end, 0.12 * (rng.random() - 0.5)), 6.0, 1.2))
        if k == "hump":
            for j in range(26):
                a_ = np.radians(-172 + j * (164 / 25)) + 0.04 * (rng.random() - 0.5)
                r0 = 120 + 30 * rng.random(); r1 = r0 + 80 + 50 * rng.random()
                p0 = root + r0 * np.array([np.cos(a_), np.sin(a_)])
                p1 = root + r1 * np.array([np.cos(a_ - 0.12 * np.sign(np.cos(a_))), np.sin(a_)])
                paths.append((curve(p0, p1, 0.10 * (rng.random() - 0.5)), 4.5, 1.0))
        return tube_map(paths, W, H) * inside

    Hf = np.zeros((H, W)); A = np.zeros((H, W))
    for k in ORDER:
        base, bh, bw, ph, pb, hairy = SPEC[k]
        m = mask_of(P[k]); inside = m > 0.5
        t = np.clip(ndi.distance_transform_edt(inside) / U / bw, 0, 1)
        h = base + bh * (1 - (1 - t) ** 2)                  # a firm rounded chamfer
        if ph:
            pil = ndi.gaussian_filter(inside.astype(float), pb * U)
            h = h + ph * pil / (pil[inside].max() + 1e-9)
        if hairy:
            h = h - GROOVE * grooves(k, inside)
        if k == "face":                                      # cheeks
            for cx_ in (420, 580):
                h = h + 5 * np.exp(-(((XX - cx_) / 40) ** 2 + ((YY - 410) / 60) ** 2)) * inside
        if k.startswith("eye"):                              # a glossy eyeball in its socket
            cy, cx = ndi.center_of_mass(inside)
            rr = np.hypot((np.arange(W)[None, :] - cx) / (13 * U), (np.arange(H)[:, None] - cy) / (7 * U))
            h = base - 8 + 9 * np.sqrt(np.clip(1 - rr ** 2, 0, 1))
        Hf = np.where(inside, h, Hf)
        A = np.maximum(A, m)

    v = np.array([0, 0, 1.0])
    def unit(a): a = np.asarray(a, float); return a / np.linalg.norm(a)
    def normals(Hm):
        gy, gx = np.gradient(Hm * U)
        nn = np.dstack([-gx, -gy, np.ones_like(Hm)]); return nn / np.linalg.norm(nn, axis=2, keepdims=True)
    def refl(nn):
        nd = np.clip(nn[..., 2], 0, 1); return 2 * nd[..., None] * nn - v, nd
    def lobe(r, dirv, lo, hi):
        c = np.einsum("ijk,k->ij", r, unit(dirv)); x = np.clip((c - lo) / (hi - lo), 0, 1); return x * x * (3 - 2 * x)
    def horizon(r):                                          # a bright band just above the reflected horizon
        up = -r[..., 1]
        return np.clip((up - 0.18) / 0.05, 0, 1) * np.clip(1 - (up - 0.23) / 0.45, 0, 1) ** 2
    r, ndv = refl(normals(ndi.gaussian_filter(Hf, 0.8 * U)))
    rc, _ = refl(normals(ndi.gaussian_filter(Hf, 3.2 * U)))
    F = 0.30 + 0.70 * (1 - ndv) ** 5
    cool, red = np.array([0.80, 0.86, 0.97]), np.array([1.0, 0.05, 0.03])
    env = (cool * (1.1 * lobe(r, [-0.55, -0.62, 0.56], 0.70, 0.97))[..., None]          # softbox upper left
           + cool * (0.22 * lobe(r, [0.5, -0.55, 0.67], 0.75, 0.97))[..., None]          # a weaker one upper right
           + red * (1.5 * lobe(rc, [0.0, -0.93, 0.36], 0.80, 0.98))[..., None]           # the red sign above
           + np.array([0.30, 0.33, 0.40]) * (0.30 * lobe(r, [0.65, 0.45, 0.6], 0.6, 0.95))[..., None]
           + np.array([0.05, 0.055, 0.065]) * np.clip(-r[..., 1], 0, 1)[..., None] ** 1.5
           + np.array([0.42, 0.45, 0.52]) * (0.55 * horizon(r))[..., None])
    col = 0.006 + F[..., None] * env * 1.15 + (0.22 * (1 - ndv) ** 2.2)[..., None] * np.array([0.55, 0.58, 0.66])
    cav = ndi.gaussian_filter(Hf, 4 * U) - Hf
    col *= np.clip(1 - 0.06 * np.clip(cav, 0, None), 0.25, 1)[..., None]
    for k in ("eye_l", "eye_r"):                             # a pin-point red glint from the sign
        cy, cx = ndi.center_of_mass(mask_of(P[k]) > 0.5)
        g = np.exp(-(((np.arange(W)[None, :] - (cx + 2.5 * U)) / (3.0 * U)) ** 2 + ((np.arange(H)[:, None] - (cy - 1.5 * U)) / (2.2 * U)) ** 2))
        col += g[..., None] * np.array([1.0, 0.10, 0.06]) * 1.4
    srgb = np.clip(col, 0, 1) ** (1 / 2.2)
    return Image.fromarray(np.dstack([srgb * 255, A * 255]).astype(np.uint8), "RGBA")


if __name__ == "__main__":
    im = render()
    a = np.asarray(im)[:, :, 3]
    ys, xs = np.nonzero(a > 2)
    crop = im.crop((max(0, xs.min() - 10), max(0, ys.min() - 10), min(im.width, xs.max() + 10), min(im.height, ys.max() + 10)))
    small = crop.resize((round(crop.width * 600 / crop.height), 600), Image.LANCZOS)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    small.save(OUT, "WEBP", quality=88, method=6, alpha_quality=90)
    print(f"wrote {OUT} ({small.width}x{small.height}, {OUT.stat().st_size // 1024} KB)")
