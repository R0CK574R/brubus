"""Stage 4 (robotnav.learn.vision): the frames are reproducible, a pixel can only be what its map and the geometry
allow, perfect labels give perfect cells, the belief behaves, and the seeing drive gets there. The trained models'
own tests are next (they read the committed weights; no PyTorch needed), then the page's copy of it all (stage 4b)."""
import json
import shutil

import numpy as np
import pytest

from robotnav.learn import camera as cam
from robotnav.learn import vision as V
from robotnav.terrain import make_map, solvable_map


def test_distance_bytes_round_trip():
    d = np.geomspace(0.25, 100.0, 500)
    back = V.decode_dist(V.encode_dist(d))
    assert np.all(np.abs(back / d - 1) <= 0.0121)                    # half a 2.4% step
    assert V.encode_dist(np.array([np.inf]))[0] == 0 and np.isinf(V.decode_dist(np.array([0]))[0])
    assert V.encode_dist(np.array([1e-3, 1e6])).tolist() == [1, 255]


def test_what_a_pixel_can_be():
    names = [c for c, ok in zip(cam.CLASSES, V.allowed("grassland")) if ok]
    assert names == ["Grass", "Tall grass", "Rock", "Tree", "Sky"]
    assert V.allowed("desert")[cam.WATER_C] and V.allowed("island")[cam.WATER_C]
    g = V.geometry(np.array([0, 7]))
    assert g[0, cam.SKY_C] and not g[0, :cam.GROUND].any() and g[0, cam.TREE_C]
    assert not g[1, cam.SKY_C] and g[1, :cam.GROUND].all()
    lp = V.finish(np.zeros((2, len(cam.CLASSES))), "grassland", np.array([0, 7]))
    assert np.allclose(np.exp(lp).sum(-1), 1.0)
    assert np.isneginf(lp[1, cam.SKY_C]) and np.isneginf(lp[1, cam.KINDS.index("Jungle")])


def test_frames_of_a_map_are_reproducible():
    from robotnav.learn import dataset as ds
    t = make_map("desert", size=48, seed=11)
    arrays = dict(height=t.height, surface=t.surface, obstacle=t.obstacle, surface_costs=np.asarray(t.surface_costs),
                  goals=np.array([[5, 5]]), meta=np.array(json.dumps(dict(
                      theme="desert", seed=11, cell_size=1.0, max_grade=0.6, uphill_penalty=3.0, sea_level=None,
                      surface_names=list(t.surface_names)))))
    a, b = V.render_map(arrays, "desert", 11, 3), V.render_map(arrays, "desert", 11, 3)
    for k in a:
        assert np.array_equal(a[k], b[k])
    assert a["images"].shape == (3, cam.CAMERA.height, cam.CAMERA.width, 3)
    assert ds.content_hash(a) == ds.content_hash(b)


@pytest.fixture(scope="module")
def desert():
    t = solvable_map("desert", seed=3, size=96, max_grade=0.6)
    return t, cam.Scene(t)


def test_perfect_labels_give_the_right_cells(desert):
    t, sc = desert
    kinds = np.array([cam.KINDS.index(n) for n in t.surface_names])
    f = cam.render(sc, (50.0, 30.0, 1.2))
    cells, ev = V.frame_evidence(V.truth_log_probs(f.label), f.dem_cell, f.dem_dist, kinds)
    ground = f.label < cam.GROUND
    assert set(cells.tolist()) == set(f.dem_cell[ground].tolist())
    assert np.array_equal(ev.argmax(1), np.asarray(t.surface).ravel()[cells])


def test_the_belief(desert):
    t, sc = desert
    prior = np.log(np.full(len(t.surface_names), 1 / len(t.surface_names)))
    b = V.GroundBelief(t, prior, mode="expected")
    assert b.S[0] == pytest.approx(float(np.mean(t.surface_costs)))  # unseen: what the prior expects
    assert V.GroundBelief(t, prior, unseen="fastest").S[0] == min(t.surface_costs)
    soft = list(t.surface_names).index("Soft sand")
    ev = np.full((1, len(t.surface_names)), V.SURE)
    ev[0, soft] = 0.0
    assert b.observe(np.array([5]), ev) == [5]
    assert b.S[5] == pytest.approx(t.surface_costs[soft], rel=1e-9) and b.likeliest()[5] == soft
    assert b.observe(np.array([5]), ev) == []                         # nothing new: no change to report
    assert b.feel(5, 0) == [5] and b.S[5] == t.surface_costs[0]      # driven over: known
    assert b.observe(np.array([5]), ev) == []                         # and felt ground stays known
    assert b.cost(0, 1, 0, 1) >= b.h(0, 1)                            # the estimate stays a lower bound


def test_ground_comes_in_fields(desert):
    """With `spread`, ground he hasn't seen is priced by the ground round it, the prior counting as one cell."""
    t, sc = desert
    prior = np.log(np.full(len(t.surface_names), 1 / len(t.surface_names)))
    soft = list(t.surface_names).index("Soft sand")
    ev = np.full((1, len(t.surface_names)), V.SURE)
    ev[0, soft] = 0.0
    c = 50 * t.cols + 50
    assert V.GroundBelief(t, prior).neighbourhood() == []             # off unless asked for
    b = V.GroundBelief(t, prior, spread=2)
    b.observe(np.array([c]), ev)
    changed = b.neighbourhood()
    assert len(changed) == 24 and c not in changed                    # the 5 x 5 round it, less the cell itself
    average = float(np.mean(t.surface_costs))
    assert b.S[c + 2] == pytest.approx((average + t.surface_costs[soft]) / 2) and b.S[c + 3] == pytest.approx(average)
    assert b.neighbourhood() == []                                    # nothing new: no change to report
    b.feel(c + 1, 0)                                                  # firm sand driven over, next to it
    b.neighbourhood()
    assert b.S[c + 2] < (average + t.surface_costs[soft]) / 2         # pulls its neighbours' price down


def test_the_seeing_drive_gets_there(desert):
    from dataclasses import replace

    from robotnav.algorithms import astar
    t, sc = desert
    s, g = (70, 20), (25, 60)
    best = astar(replace(t, start=s, goal=g, checkpoints=[])).cost
    prior = np.log(np.full(len(t.surface_names), 1 / len(t.surface_names)))
    look = cam.random_look(np.random.default_rng(1))
    runs = {see: V.see_drive(t, sc, s, g, see, look, prior) for see in ("map", "blind", "truth")}
    runs["fields"] = V.see_drive(t, sc, s, g, "truth", look, prior, spread=6)
    for see, r in runs.items():
        assert r["found"] and r["driven"][0] == s[0] * t.cols + s[1] and r["driven"][-1] == g[0] * t.cols + g[1]
        assert r["time"] >= best - 1e-9
        assert all(not t.blocked[divmod(x, t.cols)] for x in r["driven"])
    assert runs["truth"]["frames"] > 10 and runs["truth"]["seen_right"] == runs["truth"]["seen"]
    assert runs["blind"]["frames"] == 0 and runs["map"]["frames"] == 0


# ---------------------------------------------------------------- the committed models (robotnav/assets/vision.npz)
@pytest.fixture(scope="module")
def committed():
    return V.models_from(), V.arrays_from(), json.loads(V.ASSET.with_suffix(".card.json").read_text())


def test_the_committed_network_to_the_last_bit(committed):
    """Its scores for the reference frame come from sums and products in one fixed order, so they are the same on
    every machine: the checksum the card recorded when the network was saved."""
    models, a, card = committed
    net = models["net"]
    assert net.parameters() == card["network"]["parameters"] == 61_564
    assert V.logits_digest(net, a["reference.image"], a["reference.dist"]) == card["reference"]["net_logits_sha256"]


def test_the_committed_models_label_the_reference_frame(committed):
    models, a, card = committed
    for k, m in models.items():
        lp = m.log_probs(a["reference.image"], a["reference.dist"], "desert")
        assert np.allclose(np.exp(lp).sum(-1), 1.0)
        right = float(np.mean(lp.argmax(-1) == a["reference.label"]))
        assert right == pytest.approx(card["reference"]["accuracy"][k], abs=0.002) and right > 0.9


def test_the_reference_frame_is_what_the_camera_sees(committed):
    _, a, _ = committed
    f = V.reference_frame()
    assert np.mean(f.label == a["reference.label"]) > 0.995
    assert np.mean(np.abs(f.image.astype(int) - a["reference.image"]) <= 1) > 0.995


# ---------------------------------------------------------------- the page's copy (robotnav/eyes.js, stage 4b)
def test_numpy_random_numbers_step_by_step():
    """robotnav.learn.rng draws exactly what np.random.default_rng does, uniform and normal (the ziggurat's rare slow
    paths included): the recipe the page's camera follows for its sensor noise (rng.normal(0, noise, shape))."""
    from robotnav.learn import rng as RN
    for seed in (0, 1, 7, 123_456_789, 2**31 + 5, 2**40 + 3):
        mine, ref = RN.PCG64(seed), np.random.default_rng(seed)
        assert [mine.random() for _ in range(64)] == ref.random(64).tolist()
        assert [mine.normal(0.5, 2.0) for _ in range(3000)] == ref.normal(0.5, 2.0, 3000).tolist()


@pytest.mark.skipif(shutil.which("node") is None, reason="needs Node.js")
def test_the_pages_camera_and_drives_match_python(desert, committed, tmp_path):
    """robotnav/eyes.js, the page's copy of the camera, the models and the seeing drive, run under Node
    (tools/eyes_check.js): the camera's frames byte for byte, sensor noise included; the network's scores to the last
    bit (the card's checksum); the other models' to rounding, with the same likeliest class for every pixel; and a
    seeing drive's first steps, every way of seeing, step for step."""
    import base64
    import math
    import subprocess
    from pathlib import Path

    from robotnav.scene3d import map_core, page_look, vision_scene
    t, sc = desert
    models, a, card = committed
    vis = vision_scene([t])
    b64 = lambda x: base64.b64encode(np.ascontiguousarray(x).tobytes()).decode()   # noqa: E731
    dec = lambda s, dt: np.frombuffer(base64.b64decode(s), dtype=dt)               # noqa: E731
    poses, looks = [(50.0, 30.0, 1.2), (70.4, 20.7, 4.1)], [cam.random_look(np.random.default_rng(k)) for k in (1, 2)]
    s, g, prior, steps = (70, 20), (25, 60), np.array(vis["prior"]["desert"]), 12
    drives = {eye: V.see_drive(t, sc, s, g, models.get(eye, eye), looks[0], prior, max_steps=steps,
                               **{**V.SETTINGS, **V.CHOSEN.get(eye, {})}) for eye in V.VARIANTS}
    eyes = ("net", "pixel", "colour", "truth")
    cases = dict(vision=vis, uphill=t.uphill_penalty, maps={"desert": map_core(t)},
                 frames=[dict(map="desert", r=r, c=c, dir=[math.cos(h), math.sin(h)], look=page_look(lk), seed=lk.seed)
                         for (r, c, h), lk in zip(poses, looks)],
                 models=[dict(eye=e, theme="desert", image=b64(a["reference.image"]), dcode=b64(a["reference.dist"]),
                              label=b64(a["reference.label"])) for e in eyes],
                 drives=[dict(map="desert", si=s[0] * t.cols + s[1], gi=g[0] * t.cols + g[1], eye=eye, look=page_look(looks[0]),
                              settings=dict(vis["settings"], **vis["chosen"].get(eye, {}), max_steps=steps)) for eye in drives])
    (tmp_path / "cases.json").write_text(json.dumps(cases))
    check = Path(__file__).resolve().parents[1] / "tools" / "eyes_check.js"
    out = json.loads(subprocess.run(["node", str(check), str(tmp_path / "cases.json")], capture_output=True, text=True,
                                    check=True).stdout)
    for pose, lk, js in zip(poses, looks, out["frames"]):
        f = cam.render(sc, pose, lk)
        assert np.array_equal(dec(js["image"], np.uint8), f.image.ravel())
        assert np.array_equal(dec(js["label"], np.uint8), f.label.ravel())
        assert np.array_equal(dec(js["cell"], np.int32), f.cell.ravel())
        assert np.array_equal(dec(js["dist"], np.float32), f.dist.ravel())
        assert np.array_equal(dec(js["demCell"], np.int32), f.dem_cell.ravel())
        assert np.array_equal(dec(js["demDist"], np.float32), f.dem_dist.ravel())
        assert np.array_equal(dec(js["dcode"], np.uint8), V.encode_dist(f.dem_dist).ravel())
    assert out["models"][0]["logits"] == card["reference"]["net_logits_sha256"]
    for eye, js in zip(eyes, out["models"]):
        want = (V.truth_log_probs(a["reference.label"]) if eye == "truth"
                else models[eye].log_probs(a["reference.image"], a["reference.dist"], "desert"))
        got, fin = dec(js["lp"], "<f8").reshape(want.shape), np.isfinite(want)
        assert np.array_equal(np.isfinite(got), fin) and np.array_equal(got.argmax(-1), want.argmax(-1))
        assert np.abs(got[fin] - want[fin]).max() < 1e-9
    for (eye, py), js in zip(drives.items(), out["drives"]):
        assert js["driven"] == py["driven"] and len(py["driven"]) == steps + 1, eye
        assert (js["frames"], js["work"], js["seen"], js["seenRight"]) == (py["frames"], py["work"], py["seen"], py["seen_right"]), eye
