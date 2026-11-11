import math

import numpy as np
import pytest

from robotnav.airspace import AIR
from robotnav.sensing import SENSE, ekf_predict, ekf_start, ekf_update, ellipse, hazard, pd_at, risk, wilson


def test_detection_falls_with_range():
    assert pd_at(SENSE["r50"]) == pytest.approx(0.5)                    # one half at r50, by construction
    ranges = [1, 4, 8, 12, 16, 20, 24]
    pds = [pd_at(r) for r in ranges]
    assert all(a > b for a, b in zip(pds, pds[1:]))                     # the farther, the less surely
    assert pds[0] > 0.99 and pds[-1] < 0.01
    assert pd_at(AIR["sight"] + 0.1) == 0.0                              # nothing beyond a tower's reach
    assert [round(pd_at(r), 3) for r in (8, 16, 24)] == [0.743, 0.041, 0.001]   # the figures the console quotes


def test_hazard_adds_up_to_the_chance_of_no_detection():
    looks = [(3, pd_at(9.0)), (5, pd_at(14.0)), (2, pd_at(21.0))]        # (how many looks, at what probability)
    undetected = math.prod((1 - p) ** n for n, p in looks)
    steps = [(n * SENSE["look"], hazard(p)) for n, p in looks]
    assert risk(steps) == pytest.approx(1 - undetected, abs=1e-12)
    assert risk([]) == 0.0 and risk([(10.0, 0.0)]) == 0.0


def test_the_risk_figure_matches_looks_left_to_chance():
    rng = np.random.default_rng(3)
    legs = [(4.0, pd_at(13.0)), (2.0, pd_at(10.5)), (6.0, pd_at(17.0))]  # seconds in view, and the chance per look there
    n, hit = 20000, 0
    for _ in range(n):
        t, look, seen = 0.0, rng.random() * SENSE["look"], False
        for secs, p in legs:
            while look < t + secs:
                seen = seen or rng.random() < p
                look += SENSE["look"]
            t += secs
        hit += seen
    lo, hi = wilson(hit, n)
    assert lo <= risk([(s, hazard(p)) for s, p in legs]) <= hi


def test_wilson_interval():
    lo, hi = wilson(50, 100)
    assert (round(lo, 3), round(hi, 3)) == (0.404, 0.596)
    assert wilson(0, 1000)[0] == pytest.approx(0.0, abs=1e-12) and wilson(0, 1000)[1] < 0.005
    assert wilson(1000, 1000)[1] == 1.0


def _track(seed, steps=25):
    """Two towers watch a target crossing at 5 m/s; every look is a detection. Returns the errors and the filter's own
    idea of them (the normalised error squared, which should average 2 for a position that is two numbers)."""
    rng = np.random.default_rng(seed)
    towers = [(0.0, 0.0), (18.0, 4.0)]
    sb = math.radians(SENSE["sb_deg"])
    pos, vel = np.array([-10.0, 9.0]), np.array([4.0, 3.0])
    x = P = t0 = None
    errs, nees, raw = [], [], []
    for k in range(steps):
        t = k * 0.5
        truth = pos + vel * t
        tx, tz = towers[k % 2]
        d = truth - (tx, tz)
        rho = math.hypot(*d) + SENSE["sr"] * rng.standard_normal()
        beta = math.atan2(d[0], d[1]) + sb * rng.standard_normal()
        if x is None:
            x, P, t0 = ekf_start(tx, tz, rho, beta, t)
        else:
            x, P, t0 = ekf_predict(x, P, t0, t)
            x, P = ekf_update(x, P, tx, tz, rho, beta)
        raw.append(math.hypot(tx + rho * math.sin(beta) - truth[0], tz + rho * math.cos(beta) - truth[1]))   # the detection on its own
        e = x[:2] - truth
        errs.append(math.hypot(*e))
        nees.append(float(e @ np.linalg.inv(P[:2, :2]) @ e))
    return errs, nees, x, P, vel, raw


def test_tracker_closes_in_on_the_target():
    runs = [_track(seed) for seed in range(60)]
    late, alone = np.mean([np.mean(r[0][-10:]) for r in runs]), np.mean([np.mean(r[5][-10:]) for r in runs])
    assert late < 0.75 * alone                                           # the track beats any single detection
    assert np.mean([math.hypot(*(r[2][2:] - r[4])) for r in runs]) < 1.5 # and learns the velocity it was never told
    for r in runs[:5]:
        P = r[3]
        assert np.allclose(P, P.T) and np.all(np.linalg.eigvalsh(P) > 0) # a covariance stays a covariance


def test_tracker_knows_how_sure_it_is():
    # over many runs, the error measured in the filter's own covariance should average about 2 (two numbers)
    scores = [np.mean(_track(seed)[1][5:]) for seed in range(60)]
    assert 1.2 < np.mean(scores) < 3.0


def test_ellipse_of_a_plain_covariance():
    P = np.diag([4.0, 1.0, 0.0, 0.0])
    a, b, ang = ellipse(P)
    k = math.sqrt(5.991)
    assert (a, b, ang) == pytest.approx((2 * k, 1 * k, 0.0))
    a, b, ang = ellipse(np.diag([1.0, 9.0, 0.0, 0.0]))
    assert (a, b, ang) == pytest.approx((3 * k, 1 * k, math.pi / 2))
    Q = np.array([[2.0, 1.2], [1.2, 2.0]])                               # tilted 45 degrees
    P = np.zeros((4, 4)); P[:2, :2] = Q
    assert ellipse(P)[2] == pytest.approx(math.pi / 4)
