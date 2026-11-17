"""The intro's art: the sonar backdrop and the badge."""
import re

import pytest

from robotnav.intro_art import COLOURS, LOGO, RING_FIRST, RING_LAST, SCALE_R, SIZE, backdrop_svg, bearing_ticks, ring_radii


def test_backdrop_is_one_square_drawing_cropped_to_fill_the_screen():
    svg = backdrop_svg()
    assert svg.count("<svg") == 1 and 'class="intro-back"' in svg
    assert 'viewBox="0 0 2000 2000"' in svg and 'preserveAspectRatio="xMidYMid slice"' in svg   # centred, cropped like a cover photo


def test_range_rings_are_evenly_spaced_about_the_centre():
    radii = ring_radii()
    assert radii[0] == RING_FIRST and radii[-1] == RING_LAST and len(radii) == 10
    assert len({b - a for a, b in zip(radii, radii[1:])}) == 1
    circles = re.findall(r'<circle cx="([\d.]+)" cy="([\d.]+)" r="([\d.]+)" stroke-width', backdrop_svg())
    assert [float(r) for _, _, r in circles] == radii
    assert all(float(x) == float(y) == SIZE / 2 for x, y, _ in circles)
    assert RING_LAST < SIZE / 2 * 2 ** 0.5                              # the last ring still shows in the corners


def test_bearing_scale_has_a_tick_every_five_degrees():
    ticks = bearing_ticks()
    assert [b for b, _ in ticks] == list(range(0, 360, 5))
    assert all((length > 20) == (b % 30 == 0) for b, length in ticks)   # the long ones mark every 30 degrees
    assert RING_FIRST < SCALE_R < RING_LAST


def test_backdrop_is_dark_with_one_line_colour():
    svg = backdrop_svg()
    r, g, b = (int(COLOURS["ground"][i:i + 2], 16) for i in (1, 3, 5))
    assert max(r, g, b) < 16                                            # near-black
    assert set(re.findall(r'stroke="(#[0-9a-f]{6})"', svg)) == {COLOURS["line"]}
    assert "#b22234" not in svg.lower() and "red" not in svg            # no flag colours


def test_badge_is_a_webp_with_alpha():
    data = LOGO.read_bytes()
    assert data[:4] == b"RIFF" and data[8:12] == b"WEBP" and data[12:16] == b"VP8X"
    assert data[20] & 0x10                                              # it has an alpha channel
    w = 1 + int.from_bytes(data[24:27], "little")
    h = 1 + int.from_bytes(data[27:30], "little")
    assert w == h == 900                                                # the viewer's <img> says 900 x 900
    assert len(data) < 120_000


def test_badge_stands_clear_of_its_ground():
    Image = pytest.importorskip("PIL.Image")
    alpha = Image.open(LOGO).getchannel("A")
    assert alpha.getpixel((450, 400)) > 200                             # the head is solid
    for corner in ((10, 10), (889, 10), (10, 889), (889, 889)):
        assert alpha.getpixel(corner) == 0                              # the ground around it is clear
    left, top, right, bottom = alpha.getbbox()
    assert right - left > 600 and bottom - top > 600                    # and the badge fills its square
