import re
import struct

from robotnav.intro_art import BISON, FLY, HOIST, STRIPE, UNION_H, UNION_W, flag_svg, star_centres


def test_flag_has_fifty_stars_in_rows_of_six_and_five():
    stars = star_centres()
    assert len(stars) == 50
    rows = {}
    for x, y in stars:
        rows.setdefault(round(y, 6), []).append(x)
    assert [len(rows[y]) for y in sorted(rows)] == [6, 5, 6, 5, 6, 5, 6, 5, 6]
    assert all(0 < x < UNION_W and 0 < y < UNION_H for x, y in stars)   # all inside the union


def test_flag_proportions():
    assert FLY / HOIST == 1.9
    assert UNION_H == 7 * STRIPE                                        # the union is seven stripes deep
    assert round(UNION_W / HOIST, 2) == 0.76


def test_flag_svg_is_thirteen_stripes_a_union_and_the_stars():
    svg = flag_svg()
    fills = re.findall(r'<path fill="(#[0-9a-f]{6})"', svg)
    assert len(fills) == 13 + 1 + 1                                     # stripes, union, one path for all the stars
    assert fills[:13] == ["#18181b", "#0b0b0d"] * 6 + ["#18181b"]       # dark and darker, starting and ending dark
    assert "nan" not in svg.lower() and "inf" not in svg.lower()
    assert svg.count("Z") == 1 + 13 + 1 + 50                            # clip, stripes, union, stars


def test_bison_logo_is_a_webp_with_alpha():
    data = BISON.read_bytes()
    assert data[:4] == b"RIFF" and data[8:12] == b"WEBP" and data[12:16] == b"VP8X"
    flags = data[20]
    assert flags & 0x10                                                 # it has an alpha channel
    w = 1 + int.from_bytes(data[24:27], "little")
    h = 1 + int.from_bytes(data[27:30], "little")
    assert (w, h) == (439, 600)                                         # the viewer's <img> says 439 x 600
    assert len(data) < 80_000
