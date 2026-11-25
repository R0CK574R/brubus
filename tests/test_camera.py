"""Stage 4's camera (robotnav.learn.camera): its rays meet what a fine march says they meet, its labels follow the
map, a flat field comes out exactly as geometry says, and the same inputs give the same frame."""
import math
from dataclasses import replace

import numpy as np
import pytest

from robotnav.learn import camera as cam
from robotnav.terrain import NONE, ROCK, TREE, WATER, Terrain, make_map, solvable_map


@pytest.fixture(scope="module")
def scenes():
    return {th: cam.Scene(solvable_map(th, seed=3, size=96, max_grade=0.6)) for th in ("grassland", "desert", "island")}


def _pose(sc, rng):
    while True:
        r, c = rng.uniform(12, 84, 2)
        if sc.obstacle[int(round(r)), int(round(c))] == NONE:
            return float(r), float(c), float(rng.uniform(0, 2 * math.pi))


def _rays(heading):
    fwd, right, up = cam.CAMERA.rays()
    ch, sh = math.cos(heading), math.sin(heading)
    return (-fwd * ch + right * sh).ravel(), (fwd * sh + right * ch).ravel(), up.ravel()


def test_rays_meet_what_a_fine_march_meets(scenes):
    """The fast march (steps growing with distance, the crossing interpolated) against one in 4 mm steps: the same
    thing first, and the ground or water at the same distance to 6 cm. Rays that only graze a tree or a ridge can
    slip between the fast march's steps, so a few may differ."""
    rng = np.random.default_rng(7)
    total = agree = bare = 0
    for sc in scenes.values():
        for _ in range(2):
            r0, c0, h = _pose(sc, rng)
            z0 = float(sc.ground(np.array(r0), np.array(c0))) + cam.CAMERA.eye
            dr, dc, dz = _rays(h)
            pick = rng.choice(dr.size, 12, replace=False)
            what, t_hit, t_dem = cam.march(sc, (r0, c0, z0), dr[pick], dc[pick], dz[pick])
            for j, i in enumerate(pick):
                d = (dr[i], dc[i], dz[i])
                w_ref, t_ref = cam.reference_hit(sc, (r0, c0, z0), d, step=0.004)
                g_ref, tg_ref = cam.reference_hit(sc, (r0, c0, z0), d, step=0.004, bare=True)
                total += 1
                if int(what[j]) == w_ref:
                    agree += 1
                    if w_ref in (cam.HIT_GROUND, cam.HIT_WATER):
                        assert abs(t_hit[j] - t_ref) < 0.06
                if math.isfinite(tg_ref) == math.isfinite(t_dem[j]):
                    bare += 1
                    if math.isfinite(tg_ref):
                        assert abs(t_dem[j] - tg_ref) < 0.06
    assert agree >= 0.94 * total and bare >= 0.95 * total


def test_labels_follow_the_map(scenes):
    rng = np.random.default_rng(3)
    seen = set()
    for sc in scenes.values():
        for _ in range(3):
            f = cam.render(sc, _pose(sc, rng), cam.random_look(rng))
            label, cell = f.label.ravel(), f.cell.ravel()
            seen.update(np.unique(label).tolist())
            ground = label < cam.GROUND
            assert np.array_equal(sc.cls.ravel()[cell[ground]], label[ground])           # the kind of ground it shows
            ob = sc.obstacle.ravel()
            assert (ob[cell[label == cam.WATER_C]] == WATER).all()
            assert (ob[cell[label == cam.TREE_C]] == TREE).all()
            assert (ob[cell[label == cam.ROCK_C]] == ROCK).all()
            sky = label == cam.SKY_C
            assert (cell[sky] == -1).all() and np.isinf(f.dist.ravel()[sky]).all()
            assert (f.dist.ravel()[~sky] > 0).all()
            assert np.array_equal(f.dem_cell.ravel()[ground], cell[ground])              # bare ground: the same cell
            assert f.image.shape == (cam.CAMERA.height, cam.CAMERA.width, 3) and f.image.dtype == np.uint8
    assert {cam.TREE_C, cam.SKY_C} <= seen and len(seen & set(range(cam.GROUND))) >= 4


def test_a_flat_field_comes_out_as_geometry_says():
    """Flat ground, grass west of the line between columns 19 and 20 and tall grass east of it, the camera on that
    line looking north: every ground pixel is where 1.2 m / sin(depression) puts it, the left half of the picture is
    grass, the right half tall grass, and what looks above the horizon sees sky."""
    n = 40
    surface = np.zeros((n, n), np.int8)
    surface[:, 20:] = 1
    t = Terrain(blocked=np.zeros((n, n), bool), start=(0, 0), goal=(1, 1), diagonal=True, height=np.zeros((n, n)),
                surface=surface, obstacle=np.zeros((n, n), np.int8), surface_costs=(1.0, 1.8),
                surface_names=("Grass", "Tall grass"), meta=dict(theme="grassland", seed=1))
    sc = cam.Scene(t)
    f = cam.render(sc, (30.0, 19.5, 0.0))
    _, _, dz = _rays(0.0)
    dz = dz.reshape(f.label.shape)
    ground = f.label < cam.GROUND
    assert ground.sum() > 0.4 * ground.size
    expect = cam.CAMERA.eye / -dz                       # flat ground: the crossing is exact (stored as 32-bit floats)
    assert np.allclose(f.dist[ground], expect[ground], rtol=1e-6, atol=0)
    w = cam.CAMERA.width
    left, right = f.label[:, : w // 2], f.label[:, w // 2:]
    assert (left[ground[:, : w // 2]] == 0).all() and (right[ground[:, w // 2:]] == 1).all()
    assert (f.label[dz > 0] == cam.SKY_C).all()


def test_the_same_frame_every_time(scenes):
    sc = scenes["desert"]
    pose = (50.0, 30.0, 1.2)
    look = cam.random_look(np.random.default_rng(11))
    a, b = cam.render(sc, pose, look), cam.render(sc, pose, look)
    for name in ("image", "label", "cell", "dist", "dem_cell", "dem_dist"):
        assert np.array_equal(getattr(a, name), getattr(b, name))
    other = cam.render(sc, pose, cam.random_look(np.random.default_rng(12)))
    assert not np.array_equal(a.image, other.image)                                   # another light ...
    assert np.array_equal(a.label, other.label) and np.array_equal(a.dist, other.dist)  # ... the same world


def test_pond_levels_without_their_records():
    """Stored maps keep no pond records; the levels worked out from the map alone put every water cell under water
    and land within a few centimetres of the generator's own."""
    t = make_map("desert", size=96, seed=1000)
    assert t.ponds
    true, derived = cam.water_levels(t), cam.water_levels(replace(t, ponds=[]))
    w = t.obstacle == WATER
    assert (derived[w] > t.height[w]).all()
    assert np.abs(derived[w] - true[w]).max() < 0.05
    assert np.isinf(derived[~w]).all()
