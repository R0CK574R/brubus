"""Prepare the badge for the viewer's intro: robotnav/assets/logo.webp from the logo's artwork.

    python tools/shark_logo.py path/to/logo.png

The artwork is the steel shark's head on a near-black ground. The intro shows it over its own backdrop, so the
ground is made clear: a pixel's opacity follows how bright it is (black is clear, the steel and the olive ring
are solid), which leaves the emblem's own shadows dark and lets the backdrop show around and between its parts.
The result is a 900 x 900 WebP with alpha."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

OUT = Path(__file__).resolve().parents[1] / "robotnav" / "assets" / "logo.webp"
CLEAR, SOLID = 15.0, 46.0          # brightness (0-255) at which the ground is fully clear, and the emblem solid
SIZE = 900


def prepare(src: Path) -> Image.Image:
    rgb = np.asarray(Image.open(src).convert("RGB"), np.float32)
    bright = rgb.max(axis=2)
    t = np.clip((bright - CLEAR) / (SOLID - CLEAR), 0.0, 1.0)
    alpha = t * t * (3 - 2 * t)
    out = np.dstack([rgb, alpha * 255]).round().astype(np.uint8)
    return Image.fromarray(out, "RGBA").resize((SIZE, SIZE), Image.LANCZOS)


if __name__ == "__main__":
    im = prepare(Path(sys.argv[1]))
    im.save(OUT, "WEBP", quality=75, alpha_quality=60, method=6)
    print(OUT, OUT.stat().st_size // 1024, "KB")
