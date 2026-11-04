"""Make the intro's bison badge (robotnav/assets/bison.webp): a muscular bison charging three-quarters toward the
viewer, in black chrome, inside a round badge.

1. Build and run tools/bison_logo.cpp, which models the bison as a signed-distance sculpture (muscle masses over a
   deep chest, high hump and lean hindquarters; sculpted wool and locks) and ray-marches it as black chrome.
2. Make the badge: a half-round, polished black-chrome ring around a matte black face, lit by the same studio.
3. Set the bison into the face with a soft contact shadow, scale to 900 px and save as WebP with alpha.

Needs g++ with OpenMP, numpy, scipy and Pillow (Pillow comes with matplotlib). Takes a few minutes.
Usage: python tools/bison_logo.py [OUT.webp]
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

HERE = Path(__file__).resolve().parent
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "robotnav" / "assets" / "bison.webp"
SIZE = 900                       # the badge, in pixels
N, SS = 1000, 2                  # badge design units; supersampling
RING = (480.0, 446.0)            # the ring's outer and inner radius
FILL = 0.93                      # the bison's bounding-box diagonal, as a share of the face's diameter


def render_bison() -> Image.Image:
    with tempfile.TemporaryDirectory() as tmp:
        exe, pam = Path(tmp) / "bison_logo", Path(tmp) / "bison.pam"
        subprocess.run(["g++", "-O2", "-fopenmp", str(HERE / "bison_logo.cpp"), "-o", str(exe)], check=True)
        subprocess.run([str(exe), str(pam)], check=True)
        head, _, body = pam.read_bytes().partition(b"ENDHDR\n")
    kv = dict(line.split(b" ", 1) for line in head.split(b"\n")[1:] if b" " in line)
    w, h = int(kv[b"WIDTH"]), int(kv[b"HEIGHT"])
    im = Image.fromarray(np.frombuffer(body, np.uint8).reshape(h, w, 4).copy(), "RGBA")
    ys, xs = np.nonzero(np.asarray(im)[:, :, 3] > 4)
    return im.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))


def unit(a):
    a = np.asarray(a, np.float32)
    return a / np.linalg.norm(a)


def badge():
    """The ring and the face, in linear colour, with the badge's alpha and the radius at each pixel."""
    yy, xx = (np.mgrid[0:N * SS, 0:N * SS].astype(np.float32) + 0.5) / SS
    rr = np.hypot(xx - N / 2, yy - N / 2)
    r_out, r_in = RING
    mid, half = (r_out + r_in) / 2, (r_out - r_in) / 2
    ring = (rr <= r_out) & (rr >= r_in - 1)
    H = np.where(ring, 15.0 * np.sqrt(np.clip(1 - ((rr - mid) / half) ** 2, 0, 1)), 0.0).astype(np.float32)
    Hs = ndi.gaussian_filter(H, 0.4 * SS)
    gy, gx = np.gradient(Hs, 1.0 / SS)
    n = np.dstack([-gx, -gy, np.ones_like(Hs)])
    n /= np.linalg.norm(n, axis=2, keepdims=True)
    eye = (N / 2, 470.0, 1800.0)
    v = np.dstack([eye[0] - xx, eye[1] - yy, eye[2] - Hs])
    v /= np.linalg.norm(v, axis=2, keepdims=True)
    ndv = np.clip(np.sum(n * v, axis=2), 0, 1)
    r = 2 * ndv[..., None] * n - v
    up = -r[..., 1]

    def lobe(d, w):
        return np.clip(((r @ unit(d)) - (1 - w)) / w, 0, 1) ** 2

    sky = 0.12 + 0.30 * np.exp(-np.clip(up, 0, None) / 0.35)
    floor = 0.012 + 0.07 * np.exp(np.clip(up, None, 0) / 0.10)
    white = (np.where(up > 0, sky, floor) + 0.35 * np.exp(-((up - 0.02) / 0.025) ** 2)
             + 1.1 * lobe([0.0, -0.80, 0.60], 0.10) + 1.6 * lobe([-0.55, -0.55, 0.63], 0.05)
             + 0.9 * (lobe([-0.85, -0.15, -0.5], 0.10) + lobe([0.85, -0.15, -0.5], 0.10)))
    sign = 0.9 * lobe([0.0, -0.96, 0.28], 0.05) * np.clip(1 - np.abs(r[..., 0]) * 2.5, 0, 1)   # the off-white sign above
    chrome = (white[..., None] * np.array([0.82, 0.86, 0.95], np.float32) + sign[..., None] * np.array([1.0, 0.93, 0.8], np.float32))
    chrome *= (0.42 + 0.58 * (1 - ndv) ** 5)[..., None]
    grain = ndi.gaussian_filter(np.random.default_rng(3).standard_normal(rr.shape).astype(np.float32), 0.7 * SS)
    face = ((0.007 + 0.003 * grain) * (1.0 + 0.6 * np.clip(1 - rr / r_in, 0, 1)))[..., None] * np.ones(3, np.float32)
    col = np.where(ring[..., None], chrome, face)
    alpha = np.clip(r_out + 0.5 - rr, 0, 1)
    return col, alpha, rr


def compose(bison: Image.Image) -> Image.Image:
    col, alpha, rr = badge()
    scale = 2 * RING[1] * FILL * SS / np.hypot(bison.width, bison.height)
    w, h = round(bison.width * scale), round(bison.height * scale)
    b = np.asarray(bison.resize((w, h), Image.LANCZOS)).astype(np.float32) / 255.0
    x0, y0 = (N * SS - w) // 2, (N * SS - h) // 2
    a = np.zeros(rr.shape, np.float32)
    c = np.zeros(rr.shape + (3,), np.float32)
    a[y0:y0 + h, x0:x0 + w] = b[..., 3]
    c[y0:y0 + h, x0:x0 + w] = b[..., :3] ** 2.2
    shadow = ndi.gaussian_filter(np.roll(np.roll(a, 8 * SS, 0), 5 * SS, 1), 10 * SS)    # the bison stands off the face
    col = col * (1 - 0.85 * shadow * (rr < RING[1] - 1))[..., None]
    col = col * (1 - a[..., None]) + c * a[..., None]
    rgb = np.clip(col, 0, 1) ** (1 / 2.2)
    return Image.fromarray(np.dstack([rgb * 255, alpha * 255]).astype(np.uint8), "RGBA").resize((SIZE, SIZE), Image.LANCZOS)


if __name__ == "__main__":
    out = compose(render_bison())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.save(OUT, "WEBP", quality=90, method=6, alpha_quality=95)
    print(f"wrote {OUT} ({out.width}x{out.height}, {OUT.stat().st_size // 1024} KB)")
