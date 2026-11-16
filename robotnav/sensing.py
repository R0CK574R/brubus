"""How the watchtowers detect and track, for the operator console.

A tower doesn't simply see or not see. It looks once a second, and each look detects a mover in clear view with a
probability that falls with range. The figure is the textbook one for a single look at a fluctuating (Swerling 1)
target:

    Pd = Pfa ** (1 / (1 + SNR)),    SNR = SNR50 * (r50 / r) ** 4

The signal-to-noise ratio falls with the fourth power of range (the radar equation), and SNR50 is chosen so that Pd
is one half at `r50` metres. Beyond a tower's reach, or without a line of sight, there is no detection.

A place's *hazard* is the rate that goes with that probability, -ln(1 - Pd) per look, summed over the towers. The
chance of crossing a route undetected is then exp(-hazard added up along the route), which is why the planner can
weigh it: hazard adds up along a path the way time does, so "time + w * hazard" has a shortest path, and sweeping w
gives every trade between a quick route and a safe one.

Each detection is a noisy range and bearing from its tower. An extended Kalman filter with a steady-velocity target,
shaken by random accelerations, turns them into a track: an estimate of where the vehicle is and a covariance that
says how sure the towers are. Two detections confirm a track; `drop` seconds without one and it is dropped.

A *trial* runs a plan once with the looks left to chance (`trial`), and many trials say how often it gets through.
The C++ core runs them on a thread pool (cpp/src/sensing.cpp); tests/test_cpp.py holds its field, its routes and
its trials to the plain Python here, number for number.

This module is the reference. The viewer (robotnav/viewer3d.html) carries the same arithmetic in JavaScript and takes
its constants from SENSE here; the browser test compares the two number for number.
"""
from __future__ import annotations

import math

import numpy as np

from .airspace import AIR

SENSE = dict(pfa=1e-4,      # false-alarm probability a look is tuned to
             r50=10.0,      # range (m) at which a look detects half the time
             look=1.0,      # seconds between one tower's looks
             sr=1.5,        # a detection's range error (m, one sigma)
             sb_deg=5.0,    # and its bearing error (degrees, one sigma)
             qa=1.5,        # how hard the target is assumed to manoeuvre (m/s^2, one sigma)
             drop=4.0,      # a track with no detection for this long is dropped (s)
             vel0=4.0,      # a new track's speed is unknown to about this much (m/s, one sigma)
             notice=5.0)    # how far off the tank notices a roadblock (m)
SNR50 = math.log(SENSE["pfa"]) / math.log(0.5) - 1


def pd_at(r: float) -> float:
    """One look at a mover r metres away in clear view: the probability it is detected."""
    if r > AIR["sight"]:
        return 0.0
    snr = SNR50 * (SENSE["r50"] / max(r, 0.5)) ** 4
    return min(0.999, SENSE["pfa"] ** (1 / (1 + snr)))


def hazard(pd: float) -> float:
    """The detection rate (per second) that matches a per-look probability."""
    return -math.log(1 - pd) / SENSE["look"]


def risk(steps) -> float:
    """The chance of being detected at least once on a route. `steps` is (seconds, hazard) for each step."""
    return 1 - math.exp(-sum(t * h for t, h in steps))


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """The 95% interval for a share of k in n (Wilson's score interval)."""
    q = k / n
    den = 1 + z * z / n
    mid = (q + z * z / (2 * n)) / den
    half = z * math.sqrt(q * (1 - q) / n + z * z / (4 * n * n)) / den
    return max(0.0, mid - half), min(1.0, mid + half)


# ---------------------------------------------------------------- the tracker: state (x, z, vx, vz)
def wrap_pi(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def ekf_start(tx: float, tz: float, rho: float, beta: float, t: float):
    """A track from one detection (range rho, bearing beta from the tower at (tx, tz)): where it puts the target, with
    the measurement's own spread; the velocity is unknown."""
    sr2, sb2 = SENSE["sr"] ** 2, math.radians(SENSE["sb_deg"]) ** 2
    sn, cs = math.sin(beta), math.cos(beta)
    J = np.array([[sn, rho * cs], [cs, -rho * sn]])           # how position moves with range and with bearing
    P = np.zeros((4, 4))
    P[:2, :2] = J @ np.diag([sr2, sb2]) @ J.T
    P[2, 2] = P[3, 3] = SENSE["vel0"] ** 2
    return np.array([tx + rho * sn, tz + rho * cs, 0.0, 0.0]), P, t


def ekf_predict(x, P, t0: float, t: float):
    """On to time t: x = F x, P = F P F' + Q, with white random accelerations of sigma `qa`."""
    dt = t - t0
    if dt <= 0:
        return x, P, t0
    F = np.eye(4)
    F[0, 2] = F[1, 3] = dt
    q = SENSE["qa"] ** 2
    Q = np.zeros((4, 4))
    for a, b in ((0, 2), (1, 3)):
        Q[a, a] = q * dt ** 4 / 4
        Q[a, b] = Q[b, a] = q * dt ** 3 / 2
        Q[b, b] = q * dt ** 2
    return F @ x, F @ P @ F.T + Q, t


def ekf_update(x, P, tx: float, tz: float, rho: float, beta: float):
    """Take in a range and bearing measured from the tower at (tx, tz)."""
    dx, dz = x[0] - tx, x[1] - tz
    r2 = max(dx * dx + dz * dz, 1e-6)
    r = math.sqrt(r2)
    H = np.array([[dx / r, dz / r, 0, 0], [dz / r2, -dx / r2, 0, 0]])
    y = np.array([rho - r, wrap_pi(beta - math.atan2(dx, dz))])
    S = H @ P @ H.T + np.diag([SENSE["sr"] ** 2, math.radians(SENSE["sb_deg"]) ** 2])
    K = P @ H.T @ np.linalg.inv(S)
    P2 = P - K @ H @ P
    return x + K @ y, (P2 + P2.T) / 2


def ellipse(P) -> tuple[float, float, float]:
    """The position's 95% ellipse: its two half-axes, and the long one's angle from x toward z."""
    a, b, d = P[0, 0], P[0, 1], P[1, 1]
    mid, dev = (a + d) / 2, math.sqrt((a - d) ** 2 / 4 + b * b)
    l1, l2 = mid + dev, max(mid - dev, 1e-9)
    c = math.sqrt(5.991)
    ang = (0.0 if a >= d else math.pi / 2) if abs(b) < 1e-12 else math.atan2(l1 - a, b)
    return c * math.sqrt(l1), c * math.sqrt(l2), ang


# ---------------------------------------------------------------- the field, risk-weighted routes and trials
def ground_field(t, air):
    """What the towers make of a mover a metre above each open ground cell: (pd, hazard), a look's chance of detection
    per tower, shape (rows, cols, towers), and the hazard, shape (rows, cols). 32-bit floats, as the viewer and the
    C++ core keep them."""
    from .airspace import ground_top
    R, C, T = t.rows, t.cols, len(air.towers)
    top = ground_top(t)
    pd = np.zeros((R, C, T), np.float32)
    haz = np.zeros((R, C), np.float32)
    sight2 = AIR["sight"] * AIR["sight"]
    for n, (tr, tc) in enumerate(air.towers):
        E = air.eyes[n]
        for r in range(R):
            for c in range(C):
                if t.blocked[r, c] or (r == tr and c == tc):
                    continue
                a = float(t.height[r, c]) + 1.0
                dr, dc, dy = r - tr, c - tc, a - E
                d2 = (dr * dr + dc * dc) + dy * dy
                if d2 > sight2:
                    continue
                steps = max(abs(dr), abs(dc))
                clear = True
                for j in range(1, steps):                # one sample per cell crossed, as the airspace takes them
                    f = j / steps
                    if top[math.floor(tr + f * dr + 0.5), math.floor(tc + f * dc + 0.5)] > E + f * (a - E):
                        clear = False
                        break
                if clear:
                    p = pd_at(math.sqrt(d2))
                    pd[r, c, n] = p
                    haz[r, c] = np.float32(float(haz[r, c]) + hazard(p))
    return pd, haz


class _RiskGrid:
    """The terrain with hazard priced in: a move costs its drive time plus w seconds per unit of hazard where it ends."""

    def __init__(self, t, haz, start, goal, w, avoid=None):
        self.t, self.haz, self.start, self.goal, self.w, self.avoid = t, haz, start, goal, w, avoid

    def neighbors(self, c):
        for n, cost in self.t.neighbors(c):
            if n == self.avoid:
                continue
            h = float(self.haz[n])
            yield n, (cost + self.w * cost * h if self.w and h else cost)

    def heuristic(self, a, b):
        return self.t.heuristic(a, b)

    def path_cost(self, path):
        return self.t.path_cost(path)


def risk_drive(t, haz, start, goal, w: float, avoid=None):
    """The tank's quickest route paying w seconds per unit of hazard, never through `avoid`: (cells, seconds per step),
    or ([], []) if there is none."""
    from .algorithms import _best_first
    path = _best_first(_RiskGrid(t, haz, tuple(start), tuple(goal), w, avoid), "risk", lambda g, h: g + h).path
    return path, ([0.0] + [t.edge_cost(a, b) for a, b in zip(path, path[1:])] if path else [])


def route_risk(haz, cells, steps) -> float:
    """The chance of being detected at least once on a route over a ground hazard field."""
    total = 0.0
    for k in range(1, len(cells)):           # added left to right, as the viewer and the C++ core add (sum() rounds otherwise)
        total += steps[k] * float(haz[cells[k]])
    return 1 - math.exp(-total)


def mulberry32(seed: int):
    """The small random generator the viewer and the C++ core share: uniform numbers in [0, 1)."""
    state = seed & 0xFFFFFFFF

    def rng() -> float:
        nonlocal state
        state = (state + 0x6D2B79F5) & 0xFFFFFFFF
        x = ((state ^ (state >> 15)) * (1 | state)) & 0xFFFFFFFF
        x = ((x + (((x ^ (x >> 7)) * (61 | x)) & 0xFFFFFFFF)) & 0xFFFFFFFF) ^ x
        return (x ^ (x >> 14)) / 4294967296.0
    return rng


def trial(t, pd, haz, cells, steps, w: float, seed: int, number: int):
    """One run of the tank's plan with the looks left to chance: (stuck, met a roadblock, detected, tracked, seconds).

    Half the runs meet a roadblock somewhere along the route (not near its ends); the tank notices it from `notice`
    metres away and plans again from there. Each tower's first look falls at a random moment and it looks every `look`
    seconds; a look in clear view detects with its probability; two detections within `drop` seconds make a track."""
    rng = mulberry32(seed + (number + 1) * 0x9E3779B9)
    legs = [(cells, steps, len(cells) - 1)]
    block = False
    if len(cells) > 14 and rng() < 0.5:
        j = 6 + math.floor(rng() * (len(cells) - 12))
        at = cells[j]
        frm = j - 1
        while frm > 0 and math.hypot(cells[frm - 1][0] - at[0], cells[frm - 1][1] - at[1]) <= SENSE["notice"]:
            frm -= 1
        nxt, nsteps = risk_drive(t, haz, cells[frm], cells[-1], w, at)
        block = True
        if not nxt:
            return True, True, False, False, math.inf
        legs = [(cells, steps, frm), (nxt, nsteps, len(nxt) - 1)]
    T = pd.shape[2]
    look = [rng() * SENSE["look"] for _ in range(T)]
    now, hits, last_hit, tracked = 0.0, 0, -math.inf, False
    for path, secs, to in legs:
        for k in range(1, to + 1):
            t1 = now + secs[k]
            for n in range(T):
                while look[n] < t1:
                    p = float(pd[path[k]][n])
                    if p > 0 and rng() < p:
                        hits += 1
                        if look[n] - last_hit <= SENSE["drop"]:
                            tracked = True
                        last_hit = look[n]
                    look[n] += SENSE["look"]
            now = t1
    return False, block, hits > 0, tracked, now
