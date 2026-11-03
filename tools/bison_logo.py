"""Make the intro's bison badge (robotnav/assets/bison.webp): a car-style emblem of a bison's head, head-on, in
black chrome.

1. Design: a few bold parts, like the pieces of a car emblem: the hump (an oval rising behind the head), the head
   (a woolly mop in big tufts, cheek locks, tapering to a broad nose), two short horns curving up out of the sides
   of the head, and the beard. A V across the forehead marks the mop's fringe over slanted eyes; two nostrils.
   Each part is separated from the ones behind it by a gap, and a ring frames it all.
2. Relief: every part stands proud of the badge's face with a rounded edge and a gentle dome; the ring is
   half-round.
3. Finish: polished black chrome for the ring, head, horns and beard, satin black for the hump, matte black for the
   face. Chrome is all reflection, so it is seen through a slightly perspective camera (flat faces sweep through the
   studio's gradient) and its edges catch a softbox overhead, a key light upper left, two rims behind and the red
   sign above.

Needs numpy, scipy and Pillow (Pillow comes with matplotlib). Takes about a minute.
Usage: python tools/bison_logo.py [OUT.webp] [SIZE]
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "robotnav" / "assets" / "bison.webp"
SIZE = int(sys.argv[2]) if len(sys.argv) > 2 else 900
N, SS = 1000, 3                  # design units across the badge; supersampling while rendering
RING = (480.0, 446.0)            # the ring's outer and inner radius
GAP = 10.0                       # between a part and the parts behind it


# ------------------------------------------------------------------------------------------- the design
def bez(p0, p1, p2, p3, n=24):
    t = np.linspace(0, 1, n)[:, None]
    p0, p1, p2, p3 = (np.asarray(v, float) for v in (p0, p1, p2, p3))
    return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t * t * p2 + t ** 3 * p3


def outline(*pts, sharp=()):
    """A closed smooth outline through the points (Catmull-Rom as cubic Beziers); indices in `sharp` are corners."""
    pts = [np.asarray(p, float) for p in pts]
    n, out = len(pts), [pts[0][None]]
    for i in range(n):
        p0, p1, p2, p3 = pts[i - 1], pts[i], pts[(i + 1) % n], pts[(i + 2) % n]
        t1 = (p2 - p0) / 6 if i not in sharp else (p2 - p1) / 3
        t2 = (p3 - p1) / 6 if (i + 1) % n not in sharp else (p2 - p1) / 3
        out.append(bez(p1, p1 + t1, p2 - t2, p2)[1:])
    return np.vstack(out)


def mirror(pts):
    return [(N - x, y) for x, y in pts]


def symmetric(left):
    """A left half, from the top of the centre line to the bottom, made whole."""
    return left + mirror(left[::-1])[1:-1]


HUMP = [(500, 124), (436, 130), (376, 156), (328, 200), (294, 258), (272, 326), (260, 400), (258, 476), (266, 546),
        (286, 608), (320, 658), (380, 690), (500, 698)]
HEAD = [(500, 226), (462, 228), (430, 236),                  # the top of the mop
        (416, 222), (394, 240),                              # a tuft
        (370, 240), (360, 262), (342, 274),                  # the corner tuft
        (344, 300), (338, 334), (340, 366),                  # the side, where the horn comes out
        (318, 392), (340, 410),                              # a cheek lock, swept down
        (326, 440), (352, 456),                              # another
        (372, 506), (396, 566), (418, 616),                  # tapering to the nose
        (430, 640), (456, 656), (500, 660)]                  # the nose
HEAD_CORNERS = (3, 4, 7, 11, 13, 25, 27, 31, 34, 35)
HORN = [(360, 348), (322, 346), (286, 334), (262, 306), (254, 268), (262, 232), (278, 206),
        (276, 236), (278, 268), (292, 292), (318, 308), (360, 312)]
BEARD = [(500, 676), (466, 678), (450, 692), (448, 730), (434, 762), (462, 762), (472, 796), (500, 846)]
EYE = [(372, 462), (392, 460), (414, 470), (430, 488), (408, 486), (386, 476)]
NOSTRIL = [(440, 620), (446, 606), (460, 598), (470, 600), (458, 616), (448, 630)]
FRINGE = ((356, 418), (400, 420), (460, 440), (500, 470))     # the V across the forehead (left half)


def fill(pts):
    im = Image.new("L", (N * SS, N * SS), 0)
    ImageDraw.Draw(im).polygon([(x * SS, y * SS) for x, y in pts], fill=255)
    return np.asarray(im) > 127


def groove(pts, w0, w1):
    """A cut along a curve, tapering from width w0 to w1."""
    pts = np.asarray(pts, float)
    s = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(pts, axis=0).T))])
    t = np.linspace(0, s[-1], max(2, int(s[-1] * 3)))
    im = Image.new("L", (N * SS, N * SS), 0)
    dr = ImageDraw.Draw(im)
    for x, y, w in zip(np.interp(t, s, pts[:, 0]), np.interp(t, s, pts[:, 1]), np.linspace(w0, w1, len(t))):
        r = w * SS / 2
        dr.ellipse([x * SS - r, y * SS - r, x * SS + r, y * SS + r], fill=255)
    return np.asarray(im) > 127


def grow(m, r):
    return ndi.binary_dilation(m, structure=np.ones((3, 3), bool), iterations=max(1, int(round(r * SS))))


def parts():
    """The badge's pieces as masks, each cut back by a gap where it passes behind another."""
    m = {"horn": fill(outline(*HORN, sharp=(6,))) | fill(outline(*mirror(HORN), sharp=(6,))),
         "head": fill(outline(*symmetric(HEAD), sharp=HEAD_CORNERS)),
         "beard": fill(outline(*symmetric(BEARD), sharp=(4, 7, 10))),
         "hump": fill(outline(*symmetric(HUMP)))}
    shown, front = {}, np.zeros_like(m["hump"])
    for k in ("horn", "head", "beard", "hump"):                 # front to back
        shown[k] = m[k] & ~grow(front, GAP)
        front |= m[k]
    holes = fill(outline(*EYE)) | fill(outline(*mirror(EYE))) | fill(outline(*NOSTRIL)) | fill(outline(*mirror(NOSTRIL)))
    v = bez(*FRINGE, 30)
    holes |= groove(v, 3, 11) | groove(np.column_stack([N - v[:, 0], v[:, 1]]), 3, 11)
    shown["head"] &= ~holes
    return shown


# ------------------------------------------------------------------------------------------- the render
def raised(mask, height, fillet, dome=0.0, dome_blur=30.0):
    """A part standing proud: a quarter-round edge into a flat top, plus a gentle dome."""
    d = ndi.distance_transform_edt(mask).astype(np.float32) / SS
    t = np.clip(d / fillet, 0, 1)
    h = height * np.sqrt(1 - (1 - t) ** 2)
    if dome:
        g = ndi.gaussian_filter(mask.astype(np.float32), dome_blur * SS)
        h = h + dome * g / (g[mask].max() + 1e-9) * t
    return np.where(mask, h, 0.0).astype(np.float32)


def unit(a):
    a = np.asarray(a, np.float32)
    return a / np.linalg.norm(a)


def render() -> Image.Image:
    yy, xx = (np.mgrid[0:N * SS, 0:N * SS].astype(np.float32) + 0.5) / SS
    P = parts()
    rr = np.hypot(xx - N / 2, yy - N / 2)
    r_out, r_in = RING
    mid, half = (r_out + r_in) / 2, (r_out - r_in) / 2
    ring = (rr <= r_out) & (rr >= r_in - 1)
    H = np.where(ring, 15.0 * np.sqrt(np.clip(1 - ((rr - mid) / half) ** 2, 0, 1)), 0.0).astype(np.float32)
    H = np.maximum(H, raised(P["hump"], 10.0, 9.0, dome=16.0, dome_blur=60))
    H = np.maximum(H, raised(P["beard"], 14.0, 10.0, dome=6.0, dome_blur=14))
    H = np.maximum(H, raised(P["head"], 16.0, 11.0, dome=10.0, dome_blur=45))
    H = np.maximum(H, raised(P["horn"], 16.0, 13.0, dome=3.0, dome_blur=8))
    polished = ring | P["head"] | P["horn"] | P["beard"]

    Hs = ndi.gaussian_filter(H, 0.4 * SS)
    gy, gx = np.gradient(Hs, 1.0 / SS)
    n = np.dstack([-gx, -gy, np.ones_like(Hs)])
    n /= np.linalg.norm(n, axis=2, keepdims=True)
    eye = (N / 2, 470.0, 1800.0)                              # a camera a little above the centre, close enough
    v = np.dstack([eye[0] - xx, eye[1] - yy, eye[2] - Hs])    # for flat faces to sweep through the reflections
    v /= np.linalg.norm(v, axis=2, keepdims=True)
    ndv = np.clip(np.sum(n * v, axis=2), 0, 1)
    r = 2 * ndv[..., None] * n - v
    up = -r[..., 1]

    def lobe(d, w):
        return np.clip(((r @ unit(d)) - (1 - w)) / w, 0, 1) ** 2

    def studio(spread):
        """What the metal reflects; spread > 1 blurs it (satin). A sky brightest near the horizon, a dark floor with
        a soft bounce, a horizon glow, a softbox overhead, a key upper left, two rims behind, the red sign above."""
        sky = 0.12 + 0.30 * np.exp(-np.clip(up, 0, None) / 0.35)
        floor = 0.012 + 0.07 * np.exp(np.clip(up, None, 0) / (0.10 * spread))
        white = (np.where(up > 0, sky, floor) + 0.35 * np.exp(-((up - 0.02) / (0.025 * spread)) ** 2)
                 + 1.1 * lobe([0.0, -0.80, 0.60], 0.10 * spread) + 1.6 * lobe([-0.55, -0.55, 0.63], 0.05 * spread)
                 + 0.9 * (lobe([-0.85, -0.15, -0.5], 0.10 * spread) + lobe([0.85, -0.15, -0.5], 0.10 * spread)))
        red = 1.6 * lobe([0.0, -0.96, 0.28], 0.05 * spread) * np.clip(1 - np.abs(r[..., 0]) * 2.5, 0, 1)
        return white[..., None] * np.array([0.82, 0.86, 0.95], np.float32) + red[..., None] * np.array([1.0, 0.07, 0.04], np.float32)

    fresnel = (1 - ndv) ** 5
    chrome = studio(1.0) * (0.42 + 0.58 * fresnel)[..., None]                       # black chrome
    satin = studio(2.6) * (0.30 + 0.70 * fresnel)[..., None] + 0.008               # the hump
    grain = ndi.gaussian_filter(np.random.default_rng(3).standard_normal(rr.shape).astype(np.float32), 0.7 * SS)
    matte = (0.006 + 0.003 * grain)[..., None] + studio(4.0) * (0.02 * (1 - ndv) ** 3)[..., None]   # the face
    col = np.where(polished[..., None], chrome, np.where(P["hump"][..., None], satin, matte))
    cav = np.clip(ndi.gaussian_filter(H, 3 * SS) - H, 0, None)                     # gaps and grooves shade themselves
    col *= np.clip(1 - 0.05 * cav, 0.3, 1)[..., None]
    alpha = np.clip(r_out + 0.5 - rr, 0, 1)
    rgb = np.clip(col, 0, 1) ** (1 / 2.2)
    return Image.fromarray(np.dstack([rgb * 255, alpha * 255]).astype(np.uint8), "RGBA")


if __name__ == "__main__":
    badge = render().resize((SIZE, SIZE), Image.LANCZOS)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    badge.save(OUT, "WEBP", quality=90, method=6, alpha_quality=95)
    print(f"wrote {OUT} ({badge.width}x{badge.height}, {OUT.stat().st_size // 1024} KB)")
