"""NumPy's random numbers, step by step: what np.random.default_rng(seed) does for random() and normal().

Bruce's camera (robotnav.learn.camera) draws its sensor noise with np.random.default_rng(look.seed).normal. The page's
camera (robotnav/viewer3d.html, stage 4b) draws the very same numbers in JavaScript, so its frames match Python's
byte for byte. This is the recipe both follow, written out from NumPy's source (numpy/random/bit_generator.pyx,
src/pcg64/pcg64.h, src/distributions/distributions.c):

    SeedSequence  spreads the seed's 32-bit words over a pool of four, with a multiply-xorshift hash, and draws eight
                  32-bit words from the pool: PCG64's starting state and its stream.
    PCG64         a 128-bit linear congruential generator; each 64-bit output xors the state's two halves and rotates
                  them by its top six bits ("XSL-RR").
    normal        the ziggurat method (Marsaglia & Tsang; NumPy's 256 layers): one 64-bit draw picks a layer, a sign
                  and a point; 99% of the time the point is inside and that is the number. The three tables come from
                  NumPy's source (robotnav/assets/normal-tables.json).

tests/test_vision.py holds this module to NumPy itself, draw for draw.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

TABLES = Path(__file__).resolve().parents[1] / "assets" / "normal-tables.json"
M32, M64, M128 = (1 << 32) - 1, (1 << 64) - 1, (1 << 128) - 1
INIT_A, MULT_A, INIT_B, MULT_B = 0x43B0D7E5, 0x931E8875, 0x8B51F9DD, 0x58F38DED
MIX_L, MIX_R = 0xCA01F9DD, 0x4973F715
MULT = (2549297995355413924 << 64) | 4865540595714422341      # PCG's default 128-bit multiplier
_T: dict = {}


def tables() -> dict:
    if not _T:
        _T.update(json.loads(TABLES.read_text()))
    return _T


def words(seed: int) -> list:
    """A non-negative integer as 32-bit words, lowest first (SeedSequence's entropy)."""
    if seed < 0:
        raise ValueError("expected a non-negative seed")
    out = [] if seed else [0]
    while seed:
        out.append(seed & M32)
        seed >>= 32
    return out


def seed_state(entropy: list) -> list:
    """SeedSequence(entropy).generate_state(4, np.uint64), for entropy given as 32-bit words."""
    hc = INIT_A

    def hashmix(v):
        nonlocal hc
        v ^= hc
        hc = (hc * MULT_A) & M32
        v = (v * hc) & M32
        return v ^ (v >> 16)

    def mix(x, y):
        r = (MIX_L * x - MIX_R * y) & M32
        return r ^ (r >> 16)

    pool = [hashmix(entropy[i] if i < len(entropy) else 0) for i in range(4)]
    for s in range(4):
        for d in range(4):
            if s != d:
                pool[d] = mix(pool[d], hashmix(pool[s]))
    for s in range(4, len(entropy)):
        for d in range(4):
            pool[d] = mix(pool[d], hashmix(entropy[s]))
    hb, out = INIT_B, []
    for i in range(8):
        v = pool[i % 4] ^ hb
        hb = (hb * MULT_B) & M32
        v = (v * hb) & M32
        out.append(v ^ (v >> 16))
    return [out[2 * i] | (out[2 * i + 1] << 32) for i in range(4)]


class PCG64:
    """np.random.default_rng(seed): PCG64 seeded through SeedSequence."""

    def __init__(self, seed: int):
        v = seed_state(words(seed))
        init, seq = (v[0] << 64) | v[1], (v[2] << 64) | v[3]
        self.state, self.inc = 0, ((seq << 1) | 1) & M128
        self._step()
        self.state = (self.state + init) & M128
        self._step()

    def _step(self) -> None:
        self.state = (self.state * MULT + self.inc) & M128

    def next64(self) -> int:
        self._step()
        s = self.state
        x, rot = ((s >> 64) ^ s) & M64, s >> 122
        return ((x >> rot) | (x << ((64 - rot) & 63))) & M64

    def random(self) -> float:
        return (self.next64() >> 11) * (1.0 / 9007199254740992.0)

    def standard_normal(self) -> float:
        T = tables()
        wi, ki, fi = T["wi"], T["ki"], T["fi"]
        while True:
            r = self.next64()
            idx = r & 0xFF
            r >>= 8
            sign = r & 1
            rabs = (r >> 1) & 0x000FFFFFFFFFFFFF
            x = rabs * wi[idx]
            if sign:
                x = -x
            if rabs < ki[idx]:
                return x                                          # 99.3% of the time
            if idx == 0:                                          # the tail
                while True:
                    xx = -T["inv_r"] * math.log1p(-self.random())
                    yy = -math.log1p(-self.random())
                    if yy + yy > xx * xx:
                        return -(T["r"] + xx) if (rabs >> 8) & 1 else T["r"] + xx
            elif (fi[idx - 1] - fi[idx]) * self.random() + fi[idx] < math.exp(-0.5 * x * x):
                return x

    def normal(self, loc: float, scale: float) -> float:
        return loc + scale * self.standard_normal()
