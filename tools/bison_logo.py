"""Make the intro's bison logo (robotnav/assets/bison.webp).

1. Build and run tools/bison_logo.cpp, which models a charging bison as a signed-distance field and ray-marches
   it as black metal (about 1.5 minutes on two cores).
2. Crop to the bison, then add the dust it kicks up: billowing charcoal clouds from the hooves spreading out and
   back, lit from the upper left with a faint red from the sign above, flung dirt, and a thin veil of dust in
   front of the hooves. The cloud fades out before the picture's edges.
3. Scale to 720 px tall (twice its size on screen) and save as WebP with alpha.

Needs g++ with OpenMP, numpy, scipy and Pillow (Pillow comes with matplotlib).
Usage: python tools/bison_logo.py [OUT.webp]
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

HERE = Path(__file__).resolve().parent
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "robotnav" / "assets" / "bison.webp"
HEIGHT = 720


def render() -> Image.Image:
    with tempfile.TemporaryDirectory() as tmp:
        exe, pam = Path(tmp) / "bison_logo", Path(tmp) / "bison.pam"
        subprocess.run(["g++", "-O2", "-fopenmp", str(HERE / "bison_logo.cpp"), "-o", str(exe)], check=True)
        subprocess.run([str(exe), str(pam)], check=True)
        head, _, body = pam.read_bytes().partition(b"ENDHDR\n")
    kv = dict(line.split(b" ", 1) for line in head.split(b"\n")[1:] if b" " in line)
    w, h = int(kv[b"WIDTH"]), int(kv[b"HEIGHT"])
    return Image.fromarray(np.frombuffer(body, np.uint8).reshape(h, w, 4).copy(), "RGBA")


def crop(im: Image.Image) -> Image.Image:
    ys, xs = np.nonzero(np.asarray(im)[:, :, 3] > 3)
    return im.crop((xs.min() - 6, ys.min() - 6, xs.max() + 7, ys.max() + 7))


def _vnoise(x, y, seed):
    xi, yi = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    fx, fy = x - xi, y - yi
    fx, fy = fx * fx * (3 - 2 * fx), fy * fy * (3 - 2 * fy)
    def h(a, b):
        v = ((xi + a) * 374761393 + (yi + b) * 668265263 + seed * 2147483647) & 0xFFFFFFFF
        v = (v ^ (v >> 13)) * 1274126177 & 0xFFFFFFFF
        return ((v ^ (v >> 16)) & 0xFFFF) / 65535.0
    return (h(0, 0) * (1 - fx) + h(1, 0) * fx) * (1 - fy) + (h(0, 1) * (1 - fx) + h(1, 1) * fx) * fy


def _fbm(x, y, octaves, seed):
    s, a, n = 0.0, 0.5, 0.0
    for i in range(octaves):
        s = s + a * _vnoise(x, y, seed + i); n += a; x, y = x * 2.02 + 3.1, y * 2.02 + 1.7; a *= 0.5
    return s / n


def add_dust(bison: Image.Image) -> Image.Image:
    rng = np.random.default_rng(5)
    bw, bh = bison.size
    padx, pady = int(bw * 0.38), int(bh * 0.04)
    W, H = bw + 2 * padx, bh + pady
    Y, X = np.mgrid[0:H, 0:W].astype(np.float32)
    ground = H - 1
    hoof_l, hoof_r = (padx + bw * 0.27, H - pady - bh * 0.12), (padx + bw * 0.58, ground - bh * 0.015)
    blobs = [(hoof_r[0], ground - bh * 0.05, bh * 0.10), (hoof_l[0], hoof_l[1] + bh * 0.03, bh * 0.08),
             (padx * 0.55, ground - bh * 0.08, bh * 0.13), (W - padx * 0.55, ground - bh * 0.09, bh * 0.14),
             (padx * 1.05, ground - bh * 0.18, bh * 0.12), (W - padx * 1.0, ground - bh * 0.20, bh * 0.12),
             (W * 0.5, ground - bh * 0.06, bh * 0.11), (padx * 0.25, ground - bh * 0.03, bh * 0.08),
             (W - padx * 0.2, ground - bh * 0.03, bh * 0.08), (padx + bw * 0.08, ground - bh * 0.30, bh * 0.09),
             (padx + bw * 0.93, ground - bh * 0.32, bh * 0.09)]
    shape = np.zeros((H, W), np.float32)
    for cx, cy, r in blobs:                                   # where the cloud is: thick at the hooves, out and up behind
        shape = np.maximum(shape, np.exp(-((X - cx) ** 2 + ((Y - cy) * 1.25) ** 2) / (2 * r * r)))
    warp = _fbm(X / 60, Y / 60, 3, seed=11)
    n = _fbm(X / 34 + warp * 2.2, Y / 34 + warp * 1.6, 5, seed=3)
    dens = ndi.gaussian_filter(np.clip((shape * 1.15 - 0.25) + (n - 0.5) * 0.9, 0, 1), 1.2)
    dens *= np.clip((ground + 2 - Y) / (bh * 0.02), 0, 1)
    hgt = ndi.gaussian_filter(dens, 4.0) * 40                 # a billowy surface to light
    gy, gx = np.gradient(hgt)
    nrm = np.dstack([-gx, -gy, np.ones_like(hgt)]); nrm /= np.linalg.norm(nrm, axis=2, keepdims=True)
    key = np.clip(nrm @ np.array([-0.55, -0.6, 0.58]), 0, 1)
    top = np.clip(nrm @ np.array([0.0, -0.95, 0.3]), 0, 1)
    col = 0.035 + 0.16 * key[..., None] * np.array([0.78, 0.80, 0.86]) + 0.10 * top[..., None] ** 2 * np.array([1.0, 0.10, 0.06])
    edge = np.clip(np.minimum(X, W - 1 - X) / (padx * 0.55), 0, 1) * np.clip((H - 1 - Y) / (bh * 0.10), 0, 1)
    alpha = np.clip(dens * 0.92, 0, 0.92) * edge ** 2.0      # fade out well before the picture's edges
    rgb = (np.clip(col, 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)
    dust = Image.fromarray(np.dstack([rgb, (alpha * 255).astype(np.uint8)]), "RGBA")

    specks = Image.new("RGBA", (W, H), (0, 0, 0, 0)); dr = ImageDraw.Draw(specks)
    for (hx, hy), side in ((hoof_l, -1), (hoof_r, 1), (hoof_r, -1), (hoof_l, 1)):   # dirt flung out and up
        for _ in range(70):
            ang = np.radians(rng.uniform(15, 80)); dist = rng.gamma(2.0, bh * 0.06)
            x = hx + side * np.cos(ang) * dist * 1.6; y = hy - np.sin(ang) * dist * 0.9
            if not (0 <= x < W and 0 <= y < H): continue
            r = max(0.8, rng.gamma(1.6, 1.3) * (1.4 - min(1.0, dist / (bh * 0.35))))
            g = int(rng.uniform(20, 60))
            dr.ellipse([x - r, y - r * rng.uniform(0.6, 1.0), x + r, y + r], fill=(g, g, g + 4, 235))
            if r > 2.2 and rng.random() < 0.5:
                dr.ellipse([x - r * 0.45 - 0.5, y - r * 0.6, x - r * 0.05, y - r * 0.2], fill=(150, 150, 160, 200))

    out = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    out.alpha_composite(dust)
    out.alpha_composite(specks)
    out.alpha_composite(bison, (padx, 0))
    veil = alpha * 0.55 * np.clip((Y - (ground - bh * 0.07)) / (bh * 0.05), 0, 1)   # the hooves sit in the cloud
    out.alpha_composite(Image.fromarray(np.dstack([rgb, (veil * 255).astype(np.uint8)]), "RGBA"))
    return out


if __name__ == "__main__":
    logo = add_dust(crop(render()))
    logo = logo.resize((round(logo.width * HEIGHT / logo.height), HEIGHT), Image.LANCZOS)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    logo.save(OUT, "WEBP", quality=86, method=6, alpha_quality=90)
    print(f"wrote {OUT} ({logo.width}x{logo.height}, {OUT.stat().st_size // 1024} KB)")
