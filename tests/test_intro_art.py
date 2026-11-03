import re

from robotnav.intro_art import BISON, COLOURS, FLY, HOIST, STRIPE, UNION_H, UNION_W, flag_svg, flags_svg, star_centres


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


def test_flag_is_flat_grey_and_fills_the_screen():
    for vertical in (False, True):
        svg = flag_svg(vertical)
        assert 'preserveAspectRatio="xMinYMin slice"' in svg             # covers the screen, keeps the union's corner
        assert svg.count("<rect") == 1 + 7 + 1                           # the ground, seven darker stripes, the union
        assert svg.count("Z") == 50                                      # the stars, and nothing waving
        for colour in set(re.findall(r'fill="(#[0-9a-f]{6})"', svg)):
            r, g, b = (int(colour[i:i + 2], 16) for i in (1, 3, 5))
            assert r == g == b and r < 0x30                              # near-black greys, no red or blue
    assert 'viewBox="0 0 1900 1000"' in flag_svg(False) and 'viewBox="0 0 1000 1900"' in flag_svg(True)
    assert flags_svg().count("<svg") == 2


def test_vertical_flag_keeps_the_union_top_left():
    svg = flag_svg(True)
    union = re.search(r'<rect width="([\d.]+)" height="([\d.]+)" fill="' + COLOURS["union"] + '"', svg)
    assert union and float(union.group(1)) < float(union.group(2))     # 7 stripes wide, 0.76 of the width tall


def test_bison_logo_is_a_webp_with_alpha():
    data = BISON.read_bytes()
    assert data[:4] == b"RIFF" and data[8:12] == b"WEBP" and data[12:16] == b"VP8X"
    assert data[20] & 0x10                                              # it has an alpha channel
    w = 1 + int.from_bytes(data[24:27], "little")
    h = 1 + int.from_bytes(data[27:30], "little")
    assert 300 <= w <= 900 and 400 <= h <= 1100
    assert len(data) < 200_000
