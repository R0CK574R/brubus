"""Make the intro's badge (robotnav/assets/bison.webp): a shield in polished silver with a bison's head, after the
project's own sketch.

The shield: flat across the top, straight sides, curving to a point; a bevelled silver rim all round, a bar across
the top under it, and a field of black enamel. On the field, the bison faces you: crescent horns sweeping up, a
spiky forelock between them whose lower edge is a scowling brow, slit eyes, a long face narrowing to the muzzle,
woolly curls down the cheeks and a pointed beard.

Everything is drawn as outlines in a 1000-unit box, turned into heights (rounded bevels from each pixel's distance
to its outline, parts stacked back to front, partings and eyes cut in), then shaded as silver reflecting a studio:
a grey dome, a big softbox overhead, a sharp key to the upper left, two tall strips behind and a dark floor. Drawn
at 3x and averaged down to 900 px, saved as WebP with alpha.

Needs numpy, scipy, matplotlib and Pillow. Takes about a minute and a half.
Usage: python tools/bison_logo.py [OUT.webp]
"""
import sys
from pathlib import Path

import numpy as np
from matplotlib.path import Path as Outline
from PIL import Image
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "robotnav" / "assets" / "bison.webp"
SIZE, SS = 900, 3                                    # the badge in pixels; drawn at 3x, averaged down
S = SIZE * SS
U = S / 1000.0                                       # pixels per design unit


# ------------------------------------------------------------------ outlines
def bez(p0, p1, p2, p3, n=64):
    t = np.linspace(0, 1, n)[:, None]
    p0, p1, p2, p3 = (np.asarray(p, float) for p in (p0, p1, p2, p3))
    return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3


def chain(*segs):
    return np.concatenate([bez(*s) if i == 0 else bez(*s)[1:] for i, s in enumerate(segs)])


def mirror(pts):
    pts = np.asarray(pts, float)
    return np.column_stack([1000 - pts[:, 0], pts[:, 1]])


def taper(center, w0, w1=0.0, bulge=0.0, power=1.0):
    """A stroke along a centre line, its width running from w0 to w1, fuller in the middle with `bulge`."""
    c = np.asarray(center, float)
    d = np.gradient(c, axis=0)
    d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
    nrm = np.column_stack([-d[:, 1], d[:, 0]])
    t = np.linspace(0, 1, len(c))
    w = (w0 + (w1 - w0) * t ** power) * (1 + bulge * np.sin(np.pi * t)) / 2
    return np.concatenate([c + nrm * w[:, None], (c - nrm * w[:, None])[::-1]])


def flames(cx, cy, rx, ry, tips, th0, th1, sharp, lean, n=500):
    """An arc of an ellipse (angles from straight up, + to the right) with pointed tips pushed out of it."""
    th = np.linspace(th0, th1, n)
    bump = np.zeros_like(th)
    for ta, amp in tips:
        bump = np.maximum(bump, amp * np.exp(-np.abs(th - ta) / sharp))
    x = cx + rx * np.sin(th) + bump * (np.sin(th) + lean * np.sign(th) * np.abs(np.sin(th)) ** 0.5)
    y = cy - ry * np.cos(th) - bump * np.cos(th)
    return np.column_stack([x, y])


SHIELD = chain(((150, 40), (150, 40), (850, 40), (850, 40)),
               ((850, 40), (850, 200), (850, 380), (850, 500)),
               ((850, 500), (850, 720), (700, 860), (500, 962)),
               ((500, 962), (300, 860), (150, 720), (150, 500)),
               ((150, 500), (150, 380), (150, 200), (150, 40)))
RIM, TOP = 46, 150                                   # the rim's width; the field starts below y = TOP

HORN = bez((404, 448), (334, 450), (270, 412), (290, 296))
MOP_TIPS = [(-1.18, 30), (-0.80, 44), (-0.40, 54), (0.0, 58), (0.40, 54), (0.80, 44), (1.18, 30)]
MOP_C, MOP_R = (500, 474), (128, 150)
MOP = np.concatenate([flames(*MOP_C, *MOP_R, MOP_TIPS, -1.40, 1.40, sharp=0.085, lean=0.3), [(500, 492)]])
FACE = chain(((500, 470), (462, 452), (424, 438), (392, 430)),     # the brow: a V down to the middle
             ((392, 430), (380, 500), (394, 580), (430, 630)),
             ((430, 630), (420, 672), (440, 722), (500, 728)),
             ((500, 728), (560, 722), (580, 672), (570, 630)),
             ((570, 630), (606, 580), (620, 500), (608, 430)),
             ((608, 430), (576, 438), (538, 452), (500, 470)))
EYE = chain(((414, 482), (434, 478), (462, 488), (484, 508)), ((484, 508), (460, 512), (432, 504), (414, 482)))
NOSTRIL = chain(((436, 666), (438, 690), (452, 710), (476, 716)), ((476, 716), (462, 700), (452, 684), (436, 666)))
HEAD_C, HEAD_R = (500, 566), (146, 160)
HEAD_TIPS = ([(s * a, amp) for s in (-1, 1) for a, amp in ((1.12, 30), (1.52, 34), (1.92, 36), (2.32, 36))]
             + [(np.pi - 0.5, 70), (np.pi - 0.24, 100), (np.pi, 118), (np.pi + 0.24, 100), (np.pi + 0.5, 70)])


def head_outline(n=900):
    """The shaggy head behind the face: round curls down the cheeks, hanging outward and down, and the beard's points
    fanning from the chin."""
    th = np.linspace(-np.pi, np.pi, n, endpoint=False)
    bump = np.zeros_like(th)
    for t, amp in HEAD_TIPS:
        dt = np.angle(np.exp(1j * (th - t)))
        if amp < 60:
            bump = np.maximum(bump, amp * np.sqrt(np.clip(1 - (dt / 0.19) ** 2, 0, 1)))
        else:
            bump = np.maximum(bump, amp * np.exp(-np.abs(dt) / 0.11))
    u = np.column_stack([np.sin(th), -np.cos(th)])
    flow = u + (0.9 * np.sin(th) ** 2)[:, None] * np.array([0.0, 1.0])
    below = np.clip(-np.cos(th), 0, None) ** 2
    flow = flow + (0.35 * below * np.sign(np.sin(th)) * np.abs(np.sin(th)) ** 0.5)[:, None] * np.array([1.0, 0.0])
    flow /= np.linalg.norm(flow, axis=1, keepdims=True)
    return np.array(HEAD_C) + u * np.array(HEAD_R) + flow * bump[:, None]


HEAD = head_outline()


# ------------------------------------------------------------------ outlines to heights
def resample(pts, step, closed=True):
    p = np.asarray(pts, float)
    if closed:
        p = np.vstack([p, p[:1]])
    seg = np.diff(p, axis=0)
    n = np.maximum(1, np.ceil(np.linalg.norm(seg, axis=1) / step).astype(int))
    out = [p[i] + seg[i] * (np.arange(n[i])[:, None] / n[i]) for i in range(len(seg))]
    return np.concatenate(out + ([] if closed else [p[-1:]]))


def shape(poly):
    """A closed outline's inside (tested at pixel centres) and each inside pixel's distance to the outline (units),
    measured to a dense sampling of the outline, so bevels come out smooth on curves and straight edges alike."""
    poly = np.asarray(poly, float)
    x0, y0 = np.maximum(np.floor(poly.min(0) * U).astype(int) - 2, 0)
    x1, y1 = np.minimum(np.ceil(poly.max(0) * U).astype(int) + 2, S)
    ys, xs = np.mgrid[y0:y1, x0:x1]
    pc = np.column_stack([(xs.ravel() + 0.5) / U, (ys.ravel() + 0.5) / U])
    inside = Outline(poly).contains_points(pc)
    d = np.zeros(len(pc), np.float32)
    d[inside] = cKDTree(resample(poly, 0.3 / U)).query(pc[inside], workers=-1)[0]
    mask, dist = np.zeros((S, S), bool), np.zeros((S, S), np.float32)
    mask[y0:y1, x0:x1] = inside.reshape(ys.shape)
    dist[y0:y1, x0:x1] = d.reshape(ys.shape)
    return mask, dist


def curve_dist(pts, reach):
    """Each pixel's distance (units) to an open curve, up to `reach`."""
    pts = np.asarray(pts, float)
    x0, y0 = np.maximum(np.floor((pts.min(0) - reach) * U).astype(int), 0)
    x1, y1 = np.minimum(np.ceil((pts.max(0) + reach) * U).astype(int), S)
    ys, xs = np.mgrid[y0:y1, x0:x1]
    pc = np.column_stack([(xs.ravel() + 0.5) / U, (ys.ravel() + 0.5) / U])
    d = cKDTree(resample(pts, 0.3 / U, closed=False)).query(pc, distance_upper_bound=reach, workers=-1)[0]
    out = np.full((S, S), reach, np.float32)
    out[y0:y1, x0:x1] = np.minimum(d, reach).reshape(ys.shape)
    return out


def dome(mask, d, r, top, soft=0.25):
    """Height inside a shape: a rounded bevel of radius r (units) rising to `top` (units), its crown softened."""
    d = d * mask
    if soft:
        d = np.minimum(d, ndi.gaussian_filter(d, soft * r * U))
    u = np.clip(d / r, 0, 1)
    return top * np.sqrt(1 - (1 - u) ** 2) * mask


def groove(pts, hw, depth, within):
    """A rounded groove of half-width hw (units) along a curve."""
    d = curve_dist(pts, hw * 1.5)
    return depth * np.clip(1 - (d / hw) ** 2, 0, 1) * within


def heights():
    """The badge's height field (units), the shield's mask, and where the field's black enamel shows."""
    shield, d_edge = shape(SHIELD)
    yy, xx = (np.mgrid[0:S, 0:S].astype(np.float32) + 0.5) / U
    field = shield & (d_edge > RIM) & (yy > TOP)
    # the rim: a flat-topped band with bevelled edges all round; under its top, a bar set off by a groove
    rim = shield & (d_edge <= RIM)
    bev = np.minimum(np.clip(d_edge / 12, 0, 1), np.clip((RIM - d_edge) / 12, 0, 1))
    h = np.where(rim, 13 * np.sqrt(1 - (1 - bev) ** 2), 0).astype(np.float32)
    bar = shield & ~rim & (yy <= TOP)
    bar_d = np.minimum.reduce([xx - (150 + RIM), (850 - RIM) - xx, yy - (40 + RIM), TOP - yy]).clip(0)
    h = np.where(bar, 11 * np.sqrt(1 - (1 - np.clip(bar_d / 10, 0, 1)) ** 2), h)
    relief = np.zeros((S, S), bool)

    def put(poly, base, top, r, soft=0.25):          # a part laid over what's behind it
        nonlocal h, relief
        mask, d = shape(poly)
        m = mask & field
        h = np.where(m, base + dome(m, d, r, top, soft), h)
        relief |= m
        return m

    for side in (-1, 1):                             # back to front: horns, the shaggy head, the face, the forelock
        put(taper(HORN if side < 0 else mirror(HORN), 120, 0, bulge=0.1, power=0.85), 0, 18, 40)
    head = put(HEAD, 2, 12, 40)
    tips = sorted(t for t, _ in HEAD_TIPS)
    for t0, t1 in zip(tips, tips[1:] + [tips[0] + 2 * np.pi]):
        tm = (t0 + t1) / 2                           # a parting into each notch between curls (the top is under the mop)
        if abs(np.angle(np.exp(1j * tm))) < 0.9:
            continue
        ux, uy = np.sin(tm), -np.cos(tm)
        notch = (HEAD_C[0] + ux * HEAD_R[0], HEAD_C[1] + uy * HEAD_R[1])
        inner = (HEAD_C[0] + ux * HEAD_R[0] * 0.72, HEAD_C[1] + uy * HEAD_R[1] * 0.72 - 20)
        h -= groove(bez(inner, inner, notch, notch, 24), 4.5, 3.5, head)
    face = put(FACE, 6, 18, 60, soft=0.14)
    mop = put(MOP, 9, 14, 30)
    for (t0, _), (t1, _) in zip(MOP_TIPS, MOP_TIPS[1:]):   # locks: a parting from each notch down toward the brow
        tm = (t0 + t1) / 2
        notch = (MOP_C[0] + MOP_R[0] * np.sin(tm), MOP_C[1] - MOP_R[1] * np.cos(tm))
        root = (500 + (notch[0] - 500) * 0.55, notch[1] + 0.55 * (452 - notch[1]))
        h -= groove(bez(notch, (notch[0] + (root[0] - notch[0]) * 0.4, notch[1] + 26), root, root), 5, 4, mop)
    for outline, depth, r in ((EYE, 7, 5), (NOSTRIL, 6, 6)):   # eyes and nostrils cut into the face
        for o in (outline, mirror(outline)):
            m, d = shape(o)
            h -= depth * dome(m, d, r, 1.0, 0)
    return h, shield, field, relief, xx, yy


def shade(h, shield, field, relief, xx, yy):
    """Linear colour: silver where the metal is, black enamel on the field, with the relief's shadow on it."""
    hp = ndi.gaussian_filter(h, 0.7) * U             # height in pixels, lightly smoothed
    gy, gx = np.gradient(hp)
    n = np.dstack([-gx, -gy, np.ones_like(hp)])
    n /= np.linalg.norm(n, axis=2, keepdims=True)
    eye = np.array([500 * U, 430 * U, 2600 * U], np.float32)
    v = np.dstack([eye[0] - xx * U, eye[1] - yy * U, eye[2] - hp])
    v /= np.linalg.norm(v, axis=2, keepdims=True)
    r = 2 * np.clip(np.sum(n * v, axis=2), 0, 1)[..., None] * n - v
    up = -r[..., 1]

    def lobe(d, w):
        d = np.asarray(d, np.float32) / np.linalg.norm(d)
        return np.clip(((r @ d) - (1 - w)) / w, 0, 1) ** 2

    sky = 0.2 + 0.38 * np.exp(-np.clip(up, 0, None) / 0.5)
    floor = 0.025 + 0.08 * np.exp(np.clip(up, None, 0) / 0.12)
    env = (np.where(up > 0, sky, floor) + 0.45 * np.exp(-((up - 0.03) / 0.03) ** 2)
           + 1.3 * lobe([0.0, -0.82, 0.57], 0.12) + 2.2 * lobe([-0.55, -0.55, 0.63], 0.05)
           + 1.0 * (lobe([-0.88, -0.2, -0.43], 0.12) + lobe([0.88, -0.2, -0.43], 0.12))
           + 0.22 * lobe([0.0, 0.0, 1.0], 0.5))
    metal = (shield & ~field) | relief
    ao = np.ones_like(h)                             # darker in crevices, where the surroundings stand higher
    for sg, k in ((2, 0.05), (6, 0.025)):
        ao *= np.clip(1 - k * np.clip(ndi.gaussian_filter(h, sg * U) - h, 0, None), 0.4, 1)
    silver = (env * ao)[..., None] * np.array([0.96, 0.95, 0.93], np.float32)
    sheen = 0.003 + 0.007 * np.clip(1 - (yy - TOP) / 700, 0, 1)
    shadow = ndi.gaussian_filter(np.roll(metal.astype(np.float32), (int(9 * U), int(6 * U)), (0, 1)), 9 * U)
    enamel = (sheen * (1 - 0.8 * shadow))[..., None] * np.array([1.0, 1.0, 1.05], np.float32)
    return np.where(metal[..., None], silver, enamel)


def badge() -> Image.Image:
    h, shield, field, relief, xx, yy = heights()
    col = shade(h, shield, field, relief, xx, yy)
    a = shield.astype(np.float32)
    block = lambda x: x.reshape(SIZE, SS, SIZE, SS, *x.shape[2:]).mean(axis=(1, 3))   # 3x3 pixels to one
    alpha = block(a)
    lin = block(col * a[..., None]) / np.maximum(alpha, 1e-6)[..., None]               # average in linear light
    rgb = np.clip(lin, 0, 1) ** (1 / 2.2)
    return Image.fromarray(np.dstack([rgb * 255, alpha * 255]).round().astype(np.uint8), "RGBA")


if __name__ == "__main__":
    out = badge()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.save(OUT, "WEBP", quality=90, method=6, alpha_quality=95)
    print(f"wrote {OUT} ({out.width}x{out.height}, {OUT.stat().st_size // 1024} KB)")
