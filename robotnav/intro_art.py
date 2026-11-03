"""Art for the viewer's intro: a blacked-out US flag (generated here as inline SVG) and the bison badge
(robotnav/assets/bison.webp, rendered by tools/bison_logo.py), both inlined so the page stays one file.

The flag is flat, at the official proportions (hoist 1, fly 1.9, 13 stripes, a union 7 stripes deep and 0.76
long, 50 stars in rows of 6 and 5), in near-blacks and greys only. It fills the screen: the page crops it like a
cover photo, keeping the union's corner. Portrait screens get it hung vertically, union top left, stripes
running down, stars upright."""
from __future__ import annotations

import base64
import math
from pathlib import Path

BISON = Path(__file__).with_name("assets") / "bison.webp"

HOIST, FLY = 1000.0, 1900.0
UNION_H, UNION_W = HOIST * 7 / 13, HOIST * 0.76
STAR_STEP_Y, STAR_STEP_X, STAR_D = HOIST * 0.054, HOIST * 0.063, HOIST * 0.0616
STRIPE = HOIST / 13
COLOURS = dict(stripe_a="#161616", stripe_b="#0b0b0b", union="#0e0e0e", star="#272727")   # greys only


def _num(v: float) -> str:
    return f"{v:.1f}".rstrip("0").rstrip(".")


def _star(cx: float, cy: float, r: float) -> str:
    pts = []
    for i in range(10):
        a = -math.pi / 2 + i * math.pi / 5
        rr = r if i % 2 == 0 else r * 0.382
        pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
    return "M" + " L".join(f"{_num(x)} {_num(y)}" for x, y in pts) + "Z"


def star_centres() -> list[tuple[float, float]]:
    """The 50 stars of the flag flown horizontally: 9 rows, alternately 6 and 5 stars, on the official grid."""
    return [(STAR_STEP_X * c, STAR_STEP_Y * row) for row in range(1, 10)
            for c in (range(1, 12, 2) if row % 2 else range(2, 11, 2))]


def flag_svg(vertical: bool = False, colours: dict | None = None) -> str:
    """The flat flag as one <svg class="intro-flag h|v">. Hung vertically, the union stays top left (the flag's
    own right): stripes become columns and the star rows become columns of upright stars."""
    c = {**COLOURS, **(colours or {})}
    w, h = (HOIST, FLY) if vertical else (FLY, HOIST)
    out = [f'<svg class="intro-flag {"v" if vertical else "h"}" viewBox="0 0 {_num(w)} {_num(h)}" '
           f'preserveAspectRatio="xMinYMin slice" aria-hidden="true">',
           f'<rect width="{_num(w)}" height="{_num(h)}" fill="{c["stripe_b"]}"/>']
    for i in range(0, 13, 2):                                   # the seven stripes that would be red
        x0, s = i * STRIPE, STRIPE
        rect = (f'x="{_num(x0)}" y="0" width="{_num(s)}" height="{_num(h)}"' if vertical
                else f'x="0" y="{_num(x0)}" width="{_num(w)}" height="{_num(s)}"')
        out.append(f'<rect {rect} fill="{c["stripe_a"]}"/>')
    uw, uh = (UNION_H, UNION_W) if vertical else (UNION_W, UNION_H)
    out.append(f'<rect width="{_num(uw)}" height="{_num(uh)}" fill="{c["union"]}"/>')
    stars = [(y, x) if vertical else (x, y) for x, y in star_centres()]
    out.append(f'<path fill="{c["star"]}" d="{"".join(_star(x, y, STAR_D / 2) for x, y in stars)}"/>')
    out.append("</svg>")
    return "".join(out)


def flags_svg() -> str:
    """Both hangings; the page shows the one that suits the screen."""
    return flag_svg(False) + flag_svg(True)


def bison_uri() -> str:
    return "data:image/webp;base64," + base64.b64encode(BISON.read_bytes()).decode("ascii")
