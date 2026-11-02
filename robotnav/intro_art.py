"""Art for the viewer's intro: a blacked-out US flag (generated here as inline SVG) and the bison logo
(robotnav/assets/bison.webp, rendered by tools/bison_logo.py), both inlined so the page stays one file.

The flag uses the official proportions (hoist 1, fly 1.9, 13 stripes, a union 7 stripes deep and 0.76 long,
50 stars in rows of 6 and 5), every edge bent by one travelling wave, with fold shading laid over the cloth.
Its colours are near-blacks: on the page it reads as a shadow, tinted by the sign's red glow."""
from __future__ import annotations

import base64
import math
from pathlib import Path

BISON = Path(__file__).with_name("assets") / "bison.webp"

HOIST, FLY = 1000.0, 1900.0
UNION_H, UNION_W = HOIST * 7 / 13, HOIST * 0.76
STAR_STEP_Y, STAR_STEP_X, STAR_D = HOIST * 0.054, HOIST * 0.063, HOIST * 0.0616
STRIPE = HOIST / 13
WAVELENGTH, PHASE = 780.0, 0.6


def _amp(x: float) -> float:
    return 14 + 34 * (x / FLY) ** 1.1                       # the free end flaps more than the hoist


def _wave(x: float, y: float) -> tuple[float, float]:
    t = 2 * math.pi * x / WAVELENGTH + PHASE
    return x - 6 * (x / FLY) * math.cos(t), y + _amp(x) * math.sin(t) * (0.85 + 0.15 * y / HOIST)


def _num(v: float) -> str:
    return f"{v:.1f}".rstrip("0").rstrip(".")


def _path(points) -> str:
    return "M" + " L".join(f"{_num(x)} {_num(y)}" for x, y in points) + "Z"


def _band(x0: float, x1: float, y0: float, y1: float, n: int = 36) -> str:
    top = [_wave(x0 + (x1 - x0) * i / n, y0) for i in range(n + 1)]
    bottom = [_wave(x1 - (x1 - x0) * i / n, y1) for i in range(n + 1)]
    return _path(top + bottom)


def _star(cx: float, cy: float, r: float) -> str:
    pts = []
    for i in range(10):
        a = -math.pi / 2 + i * math.pi / 5
        rr = r if i % 2 == 0 else r * 0.382
        pts.append(_wave(cx + rr * math.cos(a), cy + rr * math.sin(a)))
    return _path(pts)


def star_centres() -> list[tuple[float, float]]:
    """The 50 stars: 9 rows, alternately 6 and 5 stars, on the official grid."""
    return [(STAR_STEP_X * c, STAR_STEP_Y * row) for row in range(1, 10)
            for c in (range(1, 12, 2) if row % 2 else range(2, 11, 2))]


def _folds(shadow: bool) -> str:
    """Shading along the fly, as two one-colour gradients (shadow, sheen) so only the opacity is interpolated."""
    stops = []
    for i in range(61):
        x = FLY * i / 60
        slope = math.cos(2 * math.pi * x / WAVELENGTH + PHASE) * (_amp(x) / 48)
        alpha = 0.55 * max(0.0, -slope) if shadow else 0.07 * max(0.0, slope)
        stops.append(f'<stop offset="{i / 60:.3f}" stop-color="{"#000" if shadow else "#fff"}" stop-opacity="{alpha:.3f}"/>')
    return "".join(stops)


def flag_svg(stripe_a: str = "#18181b", stripe_b: str = "#0b0b0d", union: str = "#0e0e10", star: str = "#26262b") -> str:
    """The waving flag as one <svg class="intro-flag">. Pass real colours to check the geometry."""
    box = f"-20 -60 {_num(FLY + 40)} {_num(HOIST + 120)}"
    out = [f'<svg class="intro-flag" viewBox="{box}" aria-hidden="true">',
           f'<defs><linearGradient id="flag-shade" gradientUnits="userSpaceOnUse" x1="0" y1="0" x2="{_num(FLY)}" y2="0">'
           f'{_folds(True)}</linearGradient><linearGradient id="flag-sheen" gradientUnits="userSpaceOnUse" x1="0" y1="0" '
           f'x2="{_num(FLY)}" y2="0">{_folds(False)}</linearGradient>'
           f'<clipPath id="flag-cloth"><path d="{_band(0, FLY, 0, HOIST)}"/></clipPath></defs>',
           '<g clip-path="url(#flag-cloth)">']
    for i in range(13):
        out.append(f'<path fill="{stripe_a if i % 2 == 0 else stripe_b}" d="{_band(0, FLY, i * STRIPE, (i + 1) * STRIPE + 0.6)}"/>')
    out.append(f'<path fill="{union}" d="{_band(0, UNION_W, 0, UNION_H)}"/>')
    out.append(f'<path fill="{star}" d="{"".join(_star(x, y, STAR_D / 2) for x, y in star_centres())}"/>')
    for shade in ("shade", "sheen"):
        out.append(f'<rect x="-20" y="-60" width="{_num(FLY + 40)}" height="{_num(HOIST + 120)}" fill="url(#flag-{shade})"/>')
    out.append("</g></svg>")
    return "".join(out)


def bison_uri() -> str:
    return "data:image/webp;base64," + base64.b64encode(BISON.read_bytes()).decode("ascii")
