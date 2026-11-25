"""Stage 4 (robotnav.learn.vision): the frames are reproducible, a pixel can only be what its map and the geometry
allow, perfect labels give perfect cells, the belief behaves, and the seeing drive gets there. The trained models'
own tests are at the end (they read the committed weights; no PyTorch needed)."""
import json

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
