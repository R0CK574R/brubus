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
             vel0=4.0)      # a new track's speed is unknown to about this much (m/s, one sigma)
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
