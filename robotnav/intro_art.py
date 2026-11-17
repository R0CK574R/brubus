"""Art for the viewer's intro: the backdrop (generated here as inline SVG) and the badge
(robotnav/assets/logo.webp, prepared by tools/shark_logo.py), both inlined so the page stays one file.

The backdrop is a sonar plot in the badge's own colours: near-black, with range rings and bearing lines in the
olive of the badge's ring, a scale of bearing ticks every 5 degrees, a faint glow at the centre and the corners
falling off to black. It is one square drawing centred on the screen; the page crops it like a cover photo, so
the same drawing serves a wide screen and a phone held upright."""
from __future__ import annotations

import base64
import math
from pathlib import Path

LOGO = Path(__file__).with_name("assets") / "logo.webp"

SIZE = 2000.0                                        # the drawing is SIZE x SIZE, centred on (SIZE/2, SIZE/2)
RING_STEP, RING_FIRST, RING_LAST = 125.0, 250.0, 1375.0
SCALE_R = 750.0                                      # the ring that carries the bearing ticks
COLOURS = dict(ground="#060807", line="#8c8a5c", glow="#1a1d14")   # near-black, and the badge's olive


def _num(v: float) -> str:
    return f"{v:.1f}".rstrip("0").rstrip(".")


def ring_radii() -> list[float]:
    return [RING_FIRST + RING_STEP * i for i in range(int((RING_LAST - RING_FIRST) / RING_STEP) + 1)]


def bearing_ticks() -> list[tuple[int, float]]:
    """(bearing in degrees, tick length) around the scale ring: every 5 degrees, longer at every 30."""
    return [(b, 34.0 if b % 30 == 0 else 14.0) for b in range(0, 360, 5)]


def backdrop_svg(colours: dict | None = None) -> str:
    """The backdrop as one <svg class="intro-back">."""
    c = {**COLOURS, **(colours or {})}
    m = SIZE / 2
    out = [f'<svg class="intro-back" viewBox="0 0 {_num(SIZE)} {_num(SIZE)}" preserveAspectRatio="xMidYMid slice" aria-hidden="true">',
           '<defs>'
           f'<radialGradient id="ib-glow"><stop offset="0" stop-color="{c["glow"]}"/><stop offset="0.55" stop-color="{c["glow"]}" stop-opacity="0"/></radialGradient>'
           '<radialGradient id="ib-edge"><stop offset="0.45" stop-color="#000" stop-opacity="0"/><stop offset="1" stop-color="#000" stop-opacity="0.9"/></radialGradient>'
           '</defs>',
           f'<rect width="{_num(SIZE)}" height="{_num(SIZE)}" fill="{c["ground"]}"/>',
           f'<circle cx="{_num(m)}" cy="{_num(m)}" r="{_num(m)}" fill="url(#ib-glow)"/>',
           f'<g fill="none" stroke="{c["line"]}">']
    for k, r in enumerate(ring_radii()):                                 # range rings, every fourth one stronger
        strong = k % 4 == 0
        out.append(f'<circle cx="{_num(m)}" cy="{_num(m)}" r="{_num(r)}" stroke-width="{"2" if strong else "1.2"}" stroke-opacity="{"0.42" if strong else "0.2"}"/>')
    spokes = []
    for b in range(0, 360, 30):                                          # bearing lines, clear of the badge
        s, co = math.sin(math.radians(b)), -math.cos(math.radians(b))
        spokes.append(f'M{_num(m + RING_FIRST * 2 * s)} {_num(m + RING_FIRST * 2 * co)}L{_num(m + RING_LAST * s)} {_num(m + RING_LAST * co)}')
    out.append(f'<path d="{"".join(spokes)}" stroke-width="1.2" stroke-opacity="0.13"/>')
    ticks = []
    for b, length in bearing_ticks():
        s, co = math.sin(math.radians(b)), -math.cos(math.radians(b))
        ticks.append(f'M{_num(m + SCALE_R * s)} {_num(m + SCALE_R * co)}L{_num(m + (SCALE_R + length) * s)} {_num(m + (SCALE_R + length) * co)}')
    out.append(f'<path d="{"".join(ticks)}" stroke-width="2" stroke-opacity="0.5"/>')
    out.append('</g>')
    out.append(f'<rect width="{_num(SIZE)}" height="{_num(SIZE)}" fill="url(#ib-edge)"/>')
    out.append('</svg>')
    return "".join(out)


def logo_uri() -> str:
    return "data:image/webp;base64," + base64.b64encode(LOGO.read_bytes()).decode("ascii")
