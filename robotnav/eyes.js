// What Bruce sees (README, stage 4b): his camera, the three ways of reading it, and his seeing drive, for the page.
//
// A port of robotnav.learn.camera (the camera) and robotnav.learn.vision (the models, the map he builds from what he
// sees, and see_drive), line for line: the same arithmetic in the same order, so it draws Python's frames byte for
// byte (its sensor noise too: NumPy's generator, step by step, as robotnav.learn.rng writes it out), the network's
// scores come out to the last bit, and the drives go where Python's go (tests/test_vision.py runs this file under
// Node against Python). What can't be made exact is where NumPy itself uses its own exp, log or hypot (the softmax,
// the beliefs, the classic model's edges): those agree to the last bit or so.
//
// The page runs it in a worker (a drive's frames take a tenth of a second each); robotnav/viewer3d.html inlines this
// file. It needs SCENE.vision (robotnav.scene3d.vision_scene) and a map from SCENE.maps.
"use strict";
const EYES = (() => {
  const NOTHING = 0, HIT_GROUND = 1, HIT_WATER = 2, HIT_TREE = 3, HIT_ROCK = 4;
  const ROCK = 1, TREE = 2, WATER = 3;                        // obstacles (robotnav.terrain)
  const NB_R = [-1, 1, 0, 0, -1, -1, 1, 1], NB_C = [0, 0, -1, 1, -1, 1, -1, 1];
  const SQRT2 = Math.SQRT2, TWO32 = 4294967296;
  let V = null, K = 0, GROUND = 0, ROCK_C = 0, TREE_C = 0, WATER_C = 0, SKY_C = 0, CW = 0, CH = 0, NPIX = 0;
  let FWD = null, RIGHT = null, UP = null, TS = null, NET = null, ZIG = null, COLOURS = null;

  // ---------------------------------------------------------------- setting up: constants, rays, weights
  function hexRGB(s) { return [1, 3, 5].map(i => parseInt(s.slice(i, i + 2), 16) / 255); }
  function decode(b64, Type) {
    const bin = typeof atob === "function" ? atob(b64) : Buffer.from(b64, "base64").toString("binary");
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    return new Type(bytes.buffer);
  }
  function init(vision) {
    V = vision;
    K = V.classes.length; GROUND = V.ground; ROCK_C = GROUND; TREE_C = GROUND + 1; WATER_C = GROUND + 2; SKY_C = GROUND + 3;
    const cam = V.camera, f = cam.focal, cp = cam.cos_pitch, sp = cam.sin_pitch;
    CW = cam.width; CH = cam.height; NPIX = CW * CH;
    FWD = new Float64Array(NPIX); RIGHT = new Float64Array(NPIX); UP = new Float64Array(NPIX);
    for (let r = 0; r < CH; r++) for (let c = 0; c < CW; c++) {   // Camera.rays: each pixel's ray, level frame
      const i = r * CW + c, x = c - CW / 2 + 0.5, y = -(r - CH / 2 + 0.5), n = Math.sqrt(f * f + x * x + y * y);
      const fw = f / n, up = y / n;
      FWD[i] = fw * cp - up * sp; RIGHT[i] = x / n; UP[i] = fw * sp + up * cp;
    }
    const ts = [];                                               // Camera.samples
    for (let t = cam.near; t < cam.far; t += cam.step + cam.growth * t) ts.push(t);
    TS = Float64Array.from(ts);
    const all = decode(V.net.data, Float32Array);                // the network's weights, as committed (32-bit)
    NET = {};
    let o = 0;
    for (const [name, shape] of V.net.layers) {
      const n = shape.reduce((a, b) => a * b, 1);
      NET[name] = Float64Array.from(all.subarray(o, o + n)); NET[name].shape = shape;
      o += n;
    }
    ZIG = { r: V.normal.r, inv: V.normal.inv_r, wi: Float64Array.from(V.normal.wi), ki: Float64Array.from(V.normal.ki), fi: Float64Array.from(V.normal.fi) };
    COLOURS = V.label_colors.map(c => c.map(v => Math.round(v * 255)));
  }

  // ---------------------------------------------------------------- NumPy's random numbers (robotnav.learn.rng)
  const M64 = (1n << 64n) - 1n, M128 = (1n << 128n) - 1n, MULT = (2549297995355413924n << 64n) | 4865540595714422341n;
  function seedState(words) {                                    // SeedSequence(words).generate_state(4, uint64)
    let hc = 0x43B0D7E5;
    const hashmix = v => { v = (v ^ hc) >>> 0; hc = Math.imul(hc, 0x931E8875) >>> 0; v = Math.imul(v, hc) >>> 0; return (v ^ (v >>> 16)) >>> 0; };
    const mix = (x, y) => { const r = (Math.imul(0xCA01F9DD, x) - Math.imul(0x4973F715, y)) >>> 0; return (r ^ (r >>> 16)) >>> 0; };
    const pool = [0, 1, 2, 3].map(i => hashmix(i < words.length ? words[i] : 0));
    for (let s = 0; s < 4; s++) for (let d = 0; d < 4; d++) if (s !== d) pool[d] = mix(pool[d], hashmix(pool[s]));
    for (let s = 4; s < words.length; s++) for (let d = 0; d < 4; d++) pool[d] = mix(pool[d], hashmix(words[s]));
    let hb = 0x8B51F9DD;
    const out = [];
    for (let i = 0; i < 8; i++) {
      let v = (pool[i % 4] ^ hb) >>> 0;
      hb = Math.imul(hb, 0x58F38DED) >>> 0; v = Math.imul(v, hb) >>> 0; out.push((v ^ (v >>> 16)) >>> 0);
    }
    return [0, 1, 2, 3].map(i => BigInt(out[2 * i]) | (BigInt(out[2 * i + 1]) << 32n));
  }
  class PCG {                                                    // np.random.default_rng(seed), for seeds below 2^53
    constructor(seed) {
      const w = [];
      for (let s = seed; w.length === 0 || s > 0; s = Math.floor(s / TWO32)) w.push(s % TWO32);
      const v = seedState(w), init = (v[0] << 64n) | v[1], seq = (v[2] << 64n) | v[3];
      this.s = 0n; this.inc = ((seq << 1n) | 1n) & M128;
      this.step(); this.s = (this.s + init) & M128; this.step();
    }
    step() { this.s = (this.s * MULT + this.inc) & M128; }
    next64() {
      this.step();
      const s = this.s, x = ((s >> 64n) ^ s) & M64, rot = s >> 122n;
      return ((x >> rot) | (x << ((64n - rot) & 63n))) & M64;
    }
    random() { return Number(this.next64() >> 11n) * (1 / 9007199254740992); }
    standardNormal() {                                           // NumPy's ziggurat, 256 layers
      const { wi, ki, fi } = ZIG;
      for (;;) {
        let r = this.next64();
        const idx = Number(r & 0xFFn);
        r >>= 8n;
        const sign = r & 1n, rabs = Number((r >> 1n) & 0x000FFFFFFFFFFFFFn);
        let x = rabs * wi[idx];
        if (sign) x = -x;
        if (rabs < ki[idx]) return x;                            // 99.3% of the time
        if (idx === 0) {
          for (;;) {
            const xx = -ZIG.inv * Math.log1p(-this.random()), yy = -Math.log1p(-this.random());
            if (yy + yy > xx * xx) return Math.floor(rabs / 256) % 2 ? -(ZIG.r + xx) : ZIG.r + xx;
          }
        } else if ((fi[idx - 1] - fi[idx]) * this.random() + fi[idx] < Math.exp(-0.5 * x * x)) return x;
      }
    }
  }

  // ---------------------------------------------------------------- the camera (robotnav.learn.camera)
  const mod32 = v => v - Math.floor(v / TWO32) * TWO32;
  function hash01(ix, iy, seed) {                                // _hash01: 32-bit integer mixing
    let h = mod32(ix * 374761393 + iy * 668265263 + mod32(seed * 2246822519));
    h = Math.imul((h ^ (h >>> 13)) >>> 0, 3266489917) >>> 0;
    h = (h ^ (h >>> 16)) >>> 0;
    return h / TWO32;
  }
  function vnoise(x, y, seed) {                                  // value_noise
    const ix = Math.floor(x), iy = Math.floor(y);
    let fx = x - ix, fy = y - iy;
    fx = fx * fx * (3 - 2 * fx); fy = fy * fy * (3 - 2 * fy);
    const a = hash01(ix, iy, seed), b = hash01(ix + 1, iy, seed), c = hash01(ix, iy + 1, seed), d = hash01(ix + 1, iy + 1, seed);
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy;
  }
  function rint(x) {                                             // NumPy's rint: halves to even
    const f = Math.floor(x), d = x - f;
    return d > 0.5 ? f + 1 : d < 0.5 ? f : (f % 2 === 0 ? f : f + 1);
  }
  const clip = (x, lo, hi) => (x < lo ? lo : x > hi ? hi : x);
  function makeScene(map) {                                      // Scene: one map, ready to be filmed
    const theme = map.key, pal = V.palette[theme], R = map.rows, C = map.cols, N = R * C, seed = map.seed;
    const sc = { theme, R, C, N, seed, H: Float64Array.from(map.heights), obstacle: Int8Array.from(map.obstacle),
                 surface: Int8Array.from(map.surface), costs: map.surfaces.map(s => s.cost), kindOf: map.kind.slice() };
    sc.blocked = Uint8Array.from(sc.obstacle, o => (o ? 1 : 0));
    sc.level = new Float64Array(N).fill(-Infinity);
    if (map.sea != null) for (let i = 0; i < N; i++) if (sc.obstacle[i] === WATER) sc.level[i] = map.sea;
    for (const p of map.ponds || []) for (const i of p.cells) sc.level[i] = p.level;
    sc.cls = Int8Array.from(sc.surface, s => sc.kindOf[s]);
    const dark = map.surfaces.map(s => hexRGB(pal.surf[s.name][0])), light = map.surfaces.map(s => hexRGB(pal.surf[s.name][1]));
    sc.dark = new Float64Array(N * 3); sc.light = new Float64Array(N * 3); sc.tint = new Float64Array(N);
    for (let i = 0; i < N; i++) {
      for (let k = 0; k < 3; k++) { sc.dark[i * 3 + k] = dark[sc.surface[i]][k]; sc.light[i * 3 + k] = light[sc.surface[i]][k]; }
      sc.tint[i] = 1.0 + 0.14 * (hash01((i / C) | 0, i % C, seed * 7 + 1) - 0.5);   // +-7% per cell
    }
    [sc.cosW, sc.sinW] = V.looks[theme].wind;
    const [radius, height, trunk] = V.trees[theme];
    sc.treeR = new Float64Array(N).fill(radius); sc.treeH = new Float64Array(N).fill(height); sc.treeT = new Float64Array(N).fill(trunk);
    const water = [];
    for (let i = 0; i < N; i++) if (sc.obstacle[i] === WATER) water.push(i);
    if (theme === "desert" && water.length) {                    // palms round the oases: water within 4.5 m
      for (let i = 0; i < N; i++) {
        const r = (i / C) | 0, c = i % C;
        let best = Infinity;
        for (const w of water) { const dr = ((w / C) | 0) - r, dc = (w % C) - c, d2 = dr * dr + dc * dc; if (d2 < best) best = d2; }
        if (best <= 20.25) [sc.treeR[i], sc.treeH[i], sc.treeT[i]] = V.palm;
      }
    }
    sc.rock = hexRGB(pal.rock); sc.leaf = hexRGB(pal.leaf); sc.trunk = hexRGB(pal.trunk);
    [sc.shallow, sc.deep] = pal.water.map(hexRGB); [sc.horizon, sc.zenith] = pal.sky.map(hexRGB);
    sc.kinds = [];                                               // the theme's kinds of ground (surface order) as classes
    for (let s = 0; s < map.surfaces.length; s++) sc.kinds.push(sc.kindOf[s]);
    sc.allowed = V.allowed[theme];
    return sc;
  }
  function between(sc, A, r, c) {                                // bilinear between cell centres, clamped to the map
    const R = sc.R, C = sc.C;
    r = clip(r, 0, R - 1); c = clip(c, 0, C - 1);
    const r0 = Math.min(Math.floor(r), R - 2), c0 = Math.min(Math.floor(c), C - 2), fr = r - r0, fc = c - c0;
    const i = r0 * C + c0;
    return (A[i] * (1 - fr) + A[i + C] * fr) * (1 - fc) + (A[i + 1] * (1 - fr) + A[i + C + 1] * fr) * fc;
  }
  function slope(sc, r, c) {                                     // the bilinear surface's gradient (d/d row, d/d col)
    const R = sc.R, C = sc.C, H = sc.H;
    r = clip(r, 0, R - 1); c = clip(c, 0, C - 1);
    const r0 = Math.min(Math.floor(r), R - 2), c0 = Math.min(Math.floor(c), C - 2), fr = r - r0, fc = c - c0, i = r0 * C + c0;
    return [(H[i + C] - H[i]) * (1 - fc) + (H[i + C + 1] - H[i + 1]) * fc, (H[i + 1] - H[i]) * (1 - fr) + (H[i + C + 1] - H[i + C]) * fr];
  }
  const SA = { inside: false, under: false, what: 0 };
  function surfaceAt(sc, pr, pc, pz) {                           // _surface_at, one sample
    const R = sc.R, C = sc.C, inside = pr >= 0 && pr <= R - 1 && pc >= 0 && pc <= C - 1, g = between(sc, sc.H, pr, pc);
    const rn = clip(rint(pr), 0, R - 1), cn = clip(rint(pc), 0, C - 1), cell = rn * C + cn, ob = sc.obstacle[cell];
    const under = inside && pz <= g;
    let what = under ? HIT_GROUND : NOTHING;
    if (inside && ob === WATER && pz <= sc.level[cell]) what = HIT_WATER;
    if (ob === TREE || ob === ROCK) {
      const gc = sc.H[cell], hd2 = (pr - rn) * (pr - rn) + (pc - cn) * (pc - cn);
      if (inside && ob === TREE && hd2 <= sc.treeR[cell] * sc.treeR[cell] && pz <= gc + sc.treeH[cell]) what = HIT_TREE;
      else if (inside && ob === ROCK && hd2 + (pz - gc) * (pz - gc) <= V.rock_r * V.rock_r) what = HIT_ROCK;
    }
    SA.inside = inside; SA.under = under; SA.what = what;
    return SA;
  }
  function gap(sc, r0, c0, z0, dr, dc, dz, t, bare) {
    const r = r0 + dr * t, c = c0 + dc * t, z = z0 + dz * t, g = between(sc, sc.H, r, c);
    if (bare) return z - g;
    const rn = clip(rint(r), 0, sc.R - 1), cn = clip(rint(c), 0, sc.C - 1);
    return z - Math.max(g, sc.level[rn * sc.C + cn]);
  }
  function refine(sc, r0, c0, z0, dr, dc, dz, k, what, bare) {   // _refine: the crossing, between two samples
    if (k < 0) return Infinity;
    if (!((bare || what === HIT_GROUND || what === HIT_WATER) && k > 0)) return TS[k];
    const tb = TS[k], ta = TS[k - 1], fa = gap(sc, r0, c0, z0, dr, dc, dz, ta, bare), fb = gap(sc, r0, c0, z0, dr, dc, dz, tb, bare);
    const den = fa - fb, frac = den > 0 ? fa / den : 1.0;
    return ta + (tb - ta) * clip(frac, 0.0, 1.0);
  }
  // A frame at (r0, c0) looking along `dir` (an index into V.headings, towards that neighbour, or [cos, sin] of the
  // heading), lit by `look`, its noise drawn from `seed`. Arrays per pixel, row by row: image (RGB bytes), label, cell,
  // dist, demCell, demDist.
  function render(sc, r0, c0, dir, look, seed) {
    const [ch, sh] = typeof dir === "number" ? V.headings[dir] : dir, n = NPIX, C = sc.C, R = sc.R;
    const dr = new Float64Array(n), dc = new Float64Array(n), dz = UP;
    for (let i = 0; i < n; i++) { dr[i] = -FWD[i] * ch + RIGHT[i] * sh; dc[i] = FWD[i] * sh + RIGHT[i] * ch; }
    const z0 = between(sc, sc.H, r0, c0) + V.camera.eye;
    const what = new Int8Array(n), tHit = new Float64Array(n), tDem = new Float64Array(n);
    for (let i = 0; i < n; i++) {                                // march: the first thing each ray meets, and the bare ground
      let kHit = -1, kDem = -1, w = NOTHING;
      for (let k = 0; k < TS.length; k++) {
        const t = TS[k], s = surfaceAt(sc, r0 + dr[i] * t, c0 + dc[i] * t, z0 + dz[i] * t);
        if (kHit < 0 && s.what !== NOTHING) { kHit = k; w = s.what; }
        if (s.under) { kDem = k; break; }
        if (!s.inside) break;                                    // it left the map: it can't come back
      }
      what[i] = w;
      tHit[i] = refine(sc, r0, c0, z0, dr[i], dc[i], dz[i], kHit, w, false);
      tDem[i] = refine(sc, r0, c0, z0, dr[i], dc[i], dz[i], kDem, w, true);
    }
    const label = new Uint8Array(n), cell = new Int32Array(n), hr = new Float64Array(n), hc = new Float64Array(n), hz = new Float64Array(n);
    const rn = new Int32Array(n), cn = new Int32Array(n);
    for (let i = 0; i < n; i++) {
      const hit = what[i] !== NOTHING, tr = hit ? tHit[i] : 0.0;
      hr[i] = r0 + dr[i] * tr; hc[i] = c0 + dc[i] * tr; hz[i] = z0 + dz[i] * tr;
      rn[i] = clip(rint(hr[i]), 0, R - 1); cn[i] = clip(rint(hc[i]), 0, C - 1);
      cell[i] = hit ? rn[i] * C + cn[i] : -1;
      const wv = what[i];
      label[i] = wv === HIT_GROUND ? sc.cls[rn[i] * C + cn[i]] : wv === HIT_WATER ? WATER_C : wv === HIT_TREE ? TREE_C : wv === HIT_ROCK ? ROCK_C : SKY_C;
    }
    const col = shade(sc, look, what, hr, hc, hz, rn, cn, dr, dc, dz, tHit);
    const rng = new PCG(seed), image = new Uint8Array(n * 3);
    for (let i = 0; i < n * 3; i++) {
      const v = col[i] * (look.exposure * look.balance[i % 3]) + (0.0 + look.noise * rng.standardNormal());
      image[i] = clip(rint(v * 255.0), 0, 255);
    }
    const demCell = new Int32Array(n), demDist = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const dem = isFinite(tDem[i]), td = dem ? tDem[i] : 0.0;
      demCell[i] = dem ? clip(rint(r0 + dr[i] * td), 0, R - 1) * C + clip(rint(c0 + dc[i] * td), 0, C - 1) : -1;
      demDist[i] = dem ? tDem[i] : Infinity;
    }
    const dist = new Float32Array(n);
    for (let i = 0; i < n; i++) dist[i] = what[i] !== NOTHING ? tHit[i] : Infinity;
    return { image, label, cell, dist, demCell, demDist, pose: [r0, c0, dir] };
  }
  function texture(kind, x, y, sc) {                             // _texture: a kind of ground's pattern on the ground
    const s = sc.seed * 11, cw = sc.cosW, sw = sc.sinW, name = V.kinds[kind];
    if (name === "Soft sand") return Math.sin(2 * Math.PI * (x * cw + y * sw) / 0.5 + 3.0 * vnoise(x / 2.0, y / 2.0, s + 21));
    if (name === "Tall grass") { const u = x * cw + y * sw, v = -x * sw + y * cw; return 2 * vnoise(u * 1.2, v * 5.0, s + 22) - 1; }
    if (name === "Rocky ground") {
      const p = vnoise(x * 5.0, y * 5.0, s + 23);
      return (p > 0.72 ? -1.0 : p < 0.22 ? 0.6 : 0.0) + (vnoise(x * 2.0, y * 2.0, s + 24) - 0.5);
    }
    if (name === "Jungle") return 2 * (0.6 * vnoise(x * 2.2, y * 2.2, s + 25) + 0.4 * vnoise(x * 6.0, y * 6.0, s + 26)) - 1;
    if (name === "Grass" || name === "Oasis grass") {
      const patch = name === "Oasis grass" ? vnoise(x * 1.5, y * 1.5, s + 27) : 0.5;
      return 2 * (0.7 * vnoise(x * 4.0, y * 4.0, s + 28) + 0.3 * patch) - 1;
    }
    return 2 * vnoise(x * 7.0, y * 7.0, s + 29) - 1;             // sand: a fine grain
  }
  function shade(sc, look, what, hr, hc, hz, rn, cn, dr, dc, dz, dist) {   // _shade: each pixel's colour, before exposure and noise
    const n = what.length, col = new Float64Array(n * 3), C = sc.C, sun = look.sun, seed = sc.seed, pa = V.camera.pixel_angle;
    const light = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const w = what[i];
      let n0 = 0, n1 = 0, n2 = 1;
      if (w === HIT_GROUND) {
        const x = hc[i], y = hr[i], [gr, gc] = slope(sc, y, x), a0 = -gr, a1 = -gc, nn = Math.sqrt(a0 * a0 + a1 * a1 + 1 * 1);
        n0 = a0 / nn; n1 = a1 / nn; n2 = 1 / nn;
        const cell = rn[i] * C + cn[i];
        const mix = vnoise(x / 3.0, y / 3.0, seed * 11 + 2) * 0.7 + vnoise(x / 1.1, y / 1.1, seed * 11 + 3) * 0.3;
        const tint = between(sc, sc.tint, y, x);
        const cosI = clip(Math.abs(-dr[i] * n0 + -dc[i] * n1 + -dz[i] * n2), 0.05, 1.0);
        const footprint = dist[i] * pa / Math.sqrt(cosI), kind = sc.cls[cell], [amp, wave] = V.texture[V.kinds[kind]];
        const q = footprint / wave, tex = amp * texture(kind, x, y, sc) * Math.exp(-(q * q));
        for (let k = 0; k < 3; k++) col[i * 3 + k] = (sc.dark[cell * 3 + k] * (1 - mix) + sc.light[cell * 3 + k] * mix) * tint * (1.0 + tex);
      } else if (w === HIT_WATER) {
        const cell = rn[i] * C + cn[i], depth = clip(sc.level[cell] - sc.H[cell], 0, 3) / 3.0;
        const fres = clip(1.0 - Math.abs(dz[i]) * 3.0, 0.0, 1.0) * 0.6;
        for (let k = 0; k < 3; k++) col[i * 3 + k] = (sc.shallow[k] * (1 - depth) + sc.deep[k] * depth) * (1 - fres) + sc.horizon[k] * fres;
      } else if (w === HIT_TREE) {
        const r = rn[i], c = cn[i], cell = r * C + c, frac = (hz[i] - sc.H[cell]) / sc.treeH[cell], trunk = frac < sc.treeT[cell];
        const sh = 0.85 + 0.3 * vnoise(hc[i] * 4.0, hz[i] * 4.0, seed * 11 + 5), base = trunk ? sc.trunk : sc.leaf;
        for (let k = 0; k < 3; k++) col[i * 3 + k] = base[k] * sh;
        const s0 = hr[i] - r, s1 = hc[i] - c, sn = Math.max(Math.sqrt(s0 * s0 + s1 * s1 + 0 * 0), 1e-9);
        n0 = s0 / sn; n1 = s1 / sn; n2 = 0 / sn;
      } else if (w === HIT_ROCK) {
        const r = rn[i], c = cn[i], cell = r * C + c, v0 = hr[i] - r, v1 = hc[i] - c, v2 = hz[i] - sc.H[cell];
        const vn = Math.max(Math.sqrt(v0 * v0 + v1 * v1 + v2 * v2), 1e-9);
        n0 = v0 / vn; n1 = v1 / vn; n2 = v2 / vn;
        const k6 = 0.8 + 0.4 * vnoise(hc[i] * 6.0, hr[i] * 6.0, seed * 11 + 7);
        for (let k = 0; k < 3; k++) col[i * 3 + k] = sc.rock[k] * k6;
      }
      light[i] = look.ambient + (1.0 - look.ambient) * 1.25 * clip(n0 * sun[0] + n1 * sun[1] + n2 * sun[2], 0.0, 1.0);
    }
    for (let i = 0; i < n; i++) {
      const w = what[i];
      if (w === NOTHING) {                                        // the sky
        const e = clip(dz[i] / 0.45, 0.0, 1.0);
        for (let k = 0; k < 3; k++) col[i * 3 + k] = sc.horizon[k] * (1 - e) + sc.zenith[k] * e;
        continue;
      }
      for (let k = 0; k < 3; k++) col[i * 3 + k] *= light[i];
      if (w === HIT_WATER) for (let k = 0; k < 3; k++) col[i * 3 + k] = col[i * 3 + k] * 0.5 + col[i * 3 + k] / Math.max(light[i], 1e-9) * 0.5;
      const haze = 1.0 - Math.exp(-dist[i] / look.fog);
      for (let k = 0; k < 3; k++) col[i * 3 + k] = col[i * 3 + k] * (1 - haze) + sc.horizon[k] * haze;
    }
    return col;
  }

  // ---------------------------------------------------------------- reading a frame (robotnav.learn.vision)
  function encodeDist(demDist) {                                 // metres as a byte on a log scale; 0 where no ground
    const D = V.dist, out = new Uint8Array(demDist.length);
    for (let i = 0; i < demDist.length; i++) {
      const d = demDist[i];
      if (!isFinite(d)) { out[i] = 0; continue; }
      const x = Math.log(clip(d, D.near, D.far) / D.near) / D.log;
      out[i] = 1 + rint(x * 254);
    }
    return out;
  }
  function sum12(a, o) {                                         // NumPy's order for 12 numbers (pairwise, 8 at a time)
    return (((a[o] + a[o + 1]) + (a[o + 2] + a[o + 3])) + ((a[o + 4] + a[o + 5]) + (a[o + 6] + a[o + 7]))) + a[o + 8] + a[o + 9] + a[o + 10] + a[o + 11];
  }
  function finish(z, theme, dcode) {                             // scores (pixel by pixel, K each) to log-probabilities, in place
    const ok = V.allowed[theme], ex = new Float64Array(K);
    for (let i = 0; i < NPIX; i++) {
      const none = dcode[i] === 0, o = i * K;
      let top = -Infinity;
      for (let k = 0; k < K; k++) {
        const g = k === SKY_C ? none : k < GROUND ? !none : true;
        if (!(ok[k] && g)) z[o + k] = -Infinity;
        if (z[o + k] > top) top = z[o + k];
      }
      for (let k = 0; k < K; k++) ex[k] = Math.exp(z[o + k] - top);
      const ls = Math.log(K === 12 ? sum12(ex, 0) : ex.reduce((s, v) => s + v, 0));
      for (let k = 0; k < K; k++) z[o + k] = z[o + k] - top - ls;
    }
    return z;
  }
  function conv3(x, cin, h, w, Wt, b) {                          // a 3 x 3 convolution, zero padding, NumPy's order of sums
    const cout = b.length, ph = h + 2, pw = w + 2, hw = h * w, p = new Float64Array(cin * ph * pw), out = new Float64Array(cout * hw);
    for (let i = 0; i < cin; i++) for (let y = 0; y < h; y++) p.set(x.subarray(i * hw + y * w, i * hw + y * w + w), i * ph * pw + (y + 1) * pw + 1);
    for (let o = 0; o < cout; o++) out.fill(b[o], o * hw, (o + 1) * hw);
    for (let i = 0; i < cin; i++) for (let ky = 0; ky < 3; ky++) for (let kx = 0; kx < 3; kx++) for (let o = 0; o < cout; o++) {
      const wv = Wt[((o * cin + i) * 3 + ky) * 3 + kx], ob = o * hw;
      for (let y = 0; y < h; y++) {
        const pr = i * ph * pw + (y + ky) * pw + kx, orow = ob + y * w;
        for (let xx = 0; xx < w; xx++) out[orow + xx] += wv * p[pr + xx];
      }
    }
    return out;
  }
  const relu = a => { for (let i = 0; i < a.length; i++) if (!(a[i] > 0)) a[i] = 0; return a; };
  function pool(x, c, h, w) {
    const out = new Float64Array(c * (h >> 1) * (w >> 1));
    for (let k = 0; k < c; k++) for (let y = 0; y < h >> 1; y++) for (let xx = 0; xx < w >> 1; xx++) {
      const i = k * h * w + 2 * y * w + 2 * xx;
      out[k * (h >> 1) * (w >> 1) + y * (w >> 1) + xx] = Math.max(x[i], x[i + 1], x[i + w], x[i + w + 1]);
    }
    return out;
  }
  function up(x, c, h, w) {
    const out = new Float64Array(c * h * w * 4), W2 = 2 * w;
    for (let k = 0; k < c; k++) for (let y = 0; y < 2 * h; y++) for (let xx = 0; xx < W2; xx++) out[k * 4 * h * w + y * W2 + xx] = x[k * h * w + (y >> 1) * w + (xx >> 1)];
    return out;
  }
  function netLogits(image, dcode) {                             // the U-Net (Net.logits): class scores K x height x width
    const h = CH, w = CW, hw = h * w, x = new Float64Array(4 * hw);
    for (let i = 0; i < hw; i++) { for (let k = 0; k < 3; k++) x[k * hw + i] = image[i * 3 + k] / 255.0; x[3 * hw + i] = dcode[i] / 255.0; }
    const L = name => NET[name], cv = (a, cin, hh, ww, nm) => relu(conv3(a, cin, hh, ww, L(nm + ".w"), L(nm + ".b")));
    let a = cv(x, 4, h, w, "stem.0");
    a = pool(a, 16, h, w);
    a = cv(a, 16, h / 2, w / 2, "down1.0"); a = cv(a, 24, h / 2, w / 2, "down1.1");
    const skip = a;
    a = pool(a, 24, h / 2, w / 2);
    a = cv(a, 24, h / 4, w / 4, "down2.0"); a = cv(a, 48, h / 4, w / 4, "down2.1");
    const u = up(a, 48, h / 4, w / 4), cat = new Float64Array(72 * hw / 4);
    cat.set(u); cat.set(skip, 48 * hw / 4);
    a = cv(cat, 72, h / 2, w / 2, "up1.0"); a = cv(a, 24, h / 2, w / 2, "up1.1");
    const Wh = L("head.w"), bh = L("head.b"), q = hw / 4, z = new Float64Array(K * q);
    for (let k = 0; k < K; k++) z.fill(bh[k], k * q, (k + 1) * q);
    for (let i = 0; i < 24; i++) for (let k = 0; k < K; k++) { const wv = Wh[k * 24 + i]; for (let j = 0; j < q; j++) z[k * q + j] += wv * a[i * q + j]; }
    return up(z, K, h / 2, w / 2);
  }
  function netLogProbs(frame, dcode, theme) {
    const lg = netLogits(frame.image, dcode), z = new Float64Array(NPIX * K);
    for (let k = 0; k < K; k++) for (let i = 0; i < NPIX; i++) z[i * K + k] = lg[k * NPIX + i];
    return finish(z, theme, dcode);
  }
  function colourLogProbs(frame, dcode, theme) {                 // ColourModel: a Gaussian per class over red, green, blue
    const M = V.colour, z = new Float64Array(NPIX * K), pr = M.prior[theme];
    for (let i = 0; i < NPIX; i++) {
      const x = [frame.image[i * 3] / 255.0, frame.image[i * 3 + 1] / 255.0, frame.image[i * 3 + 2] / 255.0];
      for (let k = 0; k < K; k++) {
        const m = M.mean[k], inv = M.inv[k], d = [x[0] - m[0], x[1] - m[1], x[2] - m[2]];
        let maha = 0;
        for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) maha += d[a] * inv[a][b] * d[b];
        z[i * K + k] = -0.5 * maha - 0.5 * M.logdet[k] + pr[k];
      }
    }
    return finish(z, theme, dcode);
  }
  function filter1d(src, dst, n, stride, size, reflect) {         // scipy's uniform_filter1d on one line: a running sum
    const s1 = size >> 1, s2 = size - s1 - 1, ext = new Float64Array(n + size - 1);
    for (let j = 0; j < n; j++) ext[s1 + j] = src[j * stride];
    for (let j = 0; j < s1; j++) ext[s1 - 1 - j] = reflect ? src[Math.min(j, n - 1) * stride] : 0;
    for (let j = 0; j < s2; j++) ext[s1 + n + j] = reflect ? src[Math.max(n - 1 - j, 0) * stride] : 0;
    let tmp = 0.0;
    for (let j = 0; j < size; j++) tmp += ext[j];
    dst[0] = tmp / size;
    for (let j = 1; j < n; j++) { tmp += ext[j - 1 + size] - ext[j - 1]; dst[j * stride] = tmp / size; }
  }
  function uniform2d(a, R, C, size, reflect) {                   // scipy's uniform_filter, rows first, then columns
    const t = new Float64Array(R * C), out = new Float64Array(R * C), col = new Float64Array(R), colOut = new Float64Array(R);
    for (let c = 0; c < C; c++) {
      for (let r = 0; r < R; r++) col[r] = a[r * C + c];
      filter1d(col, colOut, R, 1, size, reflect);
      for (let r = 0; r < R; r++) t[r * C + c] = colOut[r];
    }
    for (let r = 0; r < R; r++) filter1d(t.subarray(r * C, r * C + C), out.subarray(r * C, r * C + C), C, 1, size, reflect);
    return out;
  }
  function correlate3(src, dst, n, stride, anti) {               // scipy's correlate1d with [-1, 0, 1] or [1, 2, 1], reflect
    for (let j = 0; j < n; j++) {
      const m = src[Math.max(j - 1, 0) * stride], c = src[j * stride], p = src[Math.min(j + 1, n - 1) * stride];
      dst[j * stride] = anti ? c * 0.0 + (m - p) * -1.0 : c * 2.0 + (m + p) * 1.0;
    }
  }
  function sobel(a, R, C, alongCols) {                           // scipy's sobel(axis=-1) or (axis=-2), reflect
    const t = new Float64Array(R * C), out = new Float64Array(R * C), line = new Float64Array(Math.max(R, C)), res = new Float64Array(Math.max(R, C));
    const pass = (src, dst, overRows, anti) => {
      if (overRows) for (let c = 0; c < C; c++) {
        for (let r = 0; r < R; r++) line[r] = src[r * C + c];
        correlate3(line, res, R, 1, anti);
        for (let r = 0; r < R; r++) dst[r * C + c] = res[r];
      } else for (let r = 0; r < R; r++) correlate3(src.subarray(r * C, r * C + C), dst.subarray(r * C, r * C + C), C, 1, anti);
    };
    pass(a, t, !alongCols, true); pass(t, out, alongCols, false);
    return out;
  }
  function pixelLogProbs(frame, dcode, theme) {                  // PixelModel: logistic regression on 11 hand-made features
    const M = V.pixel, h = CH, w = CW, n = NPIX, F = new Array(11);
    const ch = k => Float64Array.from({ length: n }, (_, i) => frame.image[i * 3 + k] / 255.0);
    const rgb = [ch(0), ch(1), ch(2)];
    for (let k = 0; k < 3; k++) {
      F[k] = rgb[k];
      F[3 + k] = uniform2d(rgb[k], h, w, 5, true);
      const sq = uniform2d(rgb[k].map(v => v * v), h, w, 5, true);
      F[6 + k] = F[3 + k].map((m, i) => Math.sqrt(Math.max(sq[i] - m * m, 0.0)));
    }
    const lum = Float64Array.from({ length: n }, (_, i) => (rgb[0][i] + rgb[1][i] + rgb[2][i]) / 3);
    const sx = sobel(lum, h, w, true), sy = sobel(lum, h, w, false);
    F[9] = uniform2d(sx.map((v, i) => Math.sqrt(v * v + sy[i] * sy[i])), h, w, 5, true);   // (NumPy: hypot; this rounds like it more often)
    F[10] = Float64Array.from(dcode, v => v / 255.0);
    const z = new Float64Array(n * K), f = new Float64Array(11);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < 11; j++) f[j] = (F[j][i] - M.mean[j]) / M.scale[j];
      for (let k = 0; k < K; k++) {
        let s = M.b[k];
        for (let j = 0; j < 11; j++) s += f[j] * M.W[j][k];
        z[i * K + k] = s;
      }
    }
    return finish(z, theme, dcode);
  }
  function truthLogProbs(label) {
    const z = new Float64Array(NPIX * K).fill(V.sure);
    for (let i = 0; i < NPIX; i++) z[i * K + label[i]] = 0.0;
    return z;
  }
  function logProbs(eye, frame, dcode, theme) {
    return eye === "truth" ? truthLogProbs(frame.label) : eye === "net" ? netLogProbs(frame, dcode, theme)
      : eye === "colour" ? colourLogProbs(frame, dcode, theme) : pixelLogProbs(frame, dcode, theme);
  }
  function frameEvidence(lp, frame, kinds, maxRange) {           // frame_evidence: per cell, the mean log-probability of each kind
    const nk = kinds.length, sums = new Map();
    for (let i = 0; i < NPIX; i++) {
      const cell = frame.demCell[i];
      if (cell < 0) continue;
      let pg = 0.0;
      for (let j = 0; j < nk; j++) pg += Math.exp(lp[i * K + kinds[j]]);
      if (!(pg > 0.5 && frame.demDist[i] <= maxRange)) continue;
      let e = sums.get(cell);
      if (!e) sums.set(cell, (e = { s: new Float64Array(nk), n: 0 }));
      const lg = Math.log(pg);
      for (let j = 0; j < nk; j++) e.s[j] += Math.max(lp[i * K + kinds[j]] - lg, V.sure);
      e.n++;
    }
    const cells = [...sums.keys()].sort((a, b) => a - b), ev = new Float64Array(cells.length * nk);
    cells.forEach((c, q) => { const e = sums.get(c); for (let j = 0; j < nk; j++) ev[q * nk + j] = e.s[j] / e.n; });
    return { cells, ev };
  }

  // ---------------------------------------------------------------- his map (GroundBelief) and D* Lite on it
  class Belief {
    constructor(sc, prior, o) {
      this.sc = sc; this.R = sc.R; this.C = sc.C; this.N = sc.N; this.costs = sc.costs; this.nk = sc.costs.length;
      this.cell = 1.0; this.maxGrade = sc.maxGrade; this.uphill = sc.uphill; this.H = sc.H;
      this.logPrior = Float64Array.from(prior); this.mode = o.mode; this.temper = o.temper; this.hyst = o.hysteresis;
      this.spread = o.spread | 0; this.spreadW = o.spread_weight == null ? 1.0 : o.spread_weight;
      this.evidence = new Float64Array(this.N * this.nk); this.seen = new Uint8Array(this.N); this.felt = new Uint8Array(this.N);
      this.blk = new Uint8Array(this.N); this.kind = new Int8Array(this.N).fill(-1);
      const mx = Math.max(...this.logPrior), p = this.logPrior.map(v => Math.exp(v - mx)), ps = p.reduce((a, b) => a + b, 0);
      this.pPrior = p.map(v => v / ps);
      let dot = 0;
      for (let k = 0; k < this.nk; k++) dot += p[k] * this.costs[k];
      this.S = new Float64Array(this.N).fill(o.unseen === "fastest" ? Math.min(...this.costs) : dot / ps);
    }
    cost(u, v, dr, dc) {
      const blk = this.blk, C = this.C;
      if (blk[u] || blk[v]) return Infinity;
      let run;
      if (dr && dc) { if (blk[u + dr * C] || blk[u + dc]) return Infinity; run = SQRT2 * this.cell; } else run = this.cell;
      const rise = this.H[v] - this.H[u], grade = rise / run;
      if (Math.abs(grade) > this.maxGrade) return Infinity;
      return Math.sqrt(run * run + rise * rise) * this.S[v] * (1.0 + this.uphill * Math.max(grade, 0.0));
    }
    h(a, b) {
      const C = this.C, dr = Math.abs(((a / C) | 0) - ((b / C) | 0)), dc = Math.abs((a % C) - (b % C));
      return ((dr + dc) + (SQRT2 - 2) * Math.min(dr, dc)) * this.cell;
    }
    posterior(cell, out) {
      const nk = this.nk, o = cell * nk;
      let mx = -Infinity;
      for (let k = 0; k < nk; k++) { out[k] = this.logPrior[k] + this.evidence[o + k]; if (out[k] > mx) mx = out[k]; }
      let s = 0;
      for (let k = 0; k < nk; k++) { out[k] = Math.exp(out[k] - mx); s += out[k]; }
      for (let k = 0; k < nk; k++) out[k] /= s;
      return out;
    }
    price(p) {                                                   // a cell's believed slowdown from its belief
      if (this.mode === "expected") { let s = 0; for (let k = 0; k < this.nk; k++) s += p[k] * this.costs[k]; return s; }
      let best = 0;
      for (let k = 1; k < this.nk; k++) if (p[k] > p[best]) best = k;
      return this.costs[best];
    }
    observe(cells, ev) {
      const nk = this.nk, p = new Float64Array(nk), changed = [];
      cells.forEach((c, q) => { for (let k = 0; k < nk; k++) this.evidence[c * nk + k] += this.temper * ev[q * nk + k]; this.seen[c] = 1; });
      for (const c of cells) {
        if (this.felt[c]) continue;
        const nw = this.price(this.posterior(c, p)), old = this.S[c];
        if (Math.abs(nw - old) > this.hyst * old) { this.S[c] = nw; changed.push(c); }
      }
      return changed;
    }
    feel(cell, kind) {
      this.felt[cell] = 1; this.kind[cell] = kind;
      const s = this.costs[kind];
      if (this.S[cell] !== s) { this.S[cell] = s; return [cell]; }
      return [];
    }
    neighbourhood() {                                            // ground he hasn't seen, priced by the ground round it
      if (!this.spread) return [];
      const R = this.R, C = this.C, N = this.N, nk = this.nk, size = 2 * this.spread + 1, known = new Float64Array(N);
      const post = new Float64Array(N * nk), p = new Float64Array(nk);
      for (let i = 0; i < N; i++) {
        if (this.felt[i]) { known[i] = 1; post[i * nk + this.kind[i]] = 1.0; }
        else if (this.seen[i]) { known[i] = 1; this.posterior(i, p); for (let k = 0; k < nk; k++) post[i * nk + k] = p[k]; }
      }
      const n = uniform2d(known, R, C, size, false), s = [], one = new Float64Array(N);
      for (let k = 0; k < nk; k++) { for (let i = 0; i < N; i++) one[i] = post[i * nk + k]; s.push(uniform2d(one, R, C, size, false)); }
      const changed = [], area = size ** 2;
      for (let i = 0; i < N; i++) {
        if (!(n[i] * area > 0.5) || known[i]) continue;
        for (let k = 0; k < nk; k++) p[k] = (this.spreadW * this.pPrior[k] + s[k][i] * area) / (this.spreadW + n[i] * area);
        const nw = this.price(p), old = this.S[i];
        if (Math.abs(nw - old) > this.hyst * old) { this.S[i] = nw; changed.push(i); }
      }
      return changed;
    }
    likeliest(cell, p) { this.posterior(cell, p); let best = 0; for (let k = 1; k < this.nk; k++) if (p[k] > p[best]) best = k; return best; }
  }
  class DStar {                                                  // robotnav.replan.DStarLite, on a Belief
    constructor(b, start, goal) {
      const N = b.N;
      this.b = b; this.start = start; this.goal = goal; this.last = start;
      this.g = new Float64Array(N).fill(Infinity); this.rhs = new Float64Array(N).fill(Infinity);
      this.km = 0; this.inq = new Int32Array(N); this.stamp = 0; this.expanded = 0; this.arg = -1;
      this.k1 = []; this.k2 = []; this.ks = []; this.kv = [];
      this.rhs[goal] = 0;
      this.push(goal, b.h(start, goal), 0);
    }
    lessAt(i, j) { const a = this.k1, b = this.k2; return a[i] < a[j] || (a[i] === a[j] && (b[i] < b[j] || (b[i] === b[j] && this.ks[i] < this.ks[j]))); }
    swap(i, j) { const a = this.k1, b = this.k2, s = this.ks, v = this.kv; [a[i], a[j]] = [a[j], a[i]]; [b[i], b[j]] = [b[j], b[i]]; [s[i], s[j]] = [s[j], s[i]]; [v[i], v[j]] = [v[j], v[i]]; }
    push(u, k1, k2) {
      this.inq[u] = ++this.stamp;
      this.k1.push(k1); this.k2.push(k2); this.ks.push(this.stamp); this.kv.push(u);
      let i = this.kv.length - 1;
      while (i > 0) { const p = (i - 1) >> 1; if (!this.lessAt(i, p)) break; this.swap(i, p); i = p; }
    }
    popTop() {
      const last = this.kv.length - 1;
      this.swap(0, last); this.k1.pop(); this.k2.pop(); this.ks.pop(); this.kv.pop();
      for (let i = 0; ;) {
        const l = 2 * i + 1, r = l + 1;
        let m = i;
        if (l < this.kv.length && this.lessAt(l, m)) m = l;
        if (r < this.kv.length && this.lessAt(r, m)) m = r;
        if (m === i) break;
        this.swap(i, m); i = m;
      }
    }
    top() { while (this.kv.length && this.inq[this.kv[0]] !== this.ks[0]) this.popTop(); return this.kv.length > 0; }
    key(u) { const m = Math.min(this.g[u], this.rhs[u]); return [m + this.b.h(this.start, u) + this.km, m]; }
    updateVertex(u) {
      if (this.g[u] !== this.rhs[u]) { const k = this.key(u); this.push(u, k[0], k[1]); }
      else if (this.inq[u]) this.inq[u] = 0;
    }
    bestSucc(u) {
      const b = this.b, C = b.C, R = b.R, r = (u / C) | 0, c = u - r * C;
      let best = Infinity, arg = -1;
      for (let k = 0; k < 8; k++) {
        const dr = NB_R[k], dc = NB_C[k], nr = r + dr, nc = c + dc;
        if (nr < 0 || nc < 0 || nr >= R || nc >= C) continue;
        const s = u + dr * C + dc, v = b.cost(u, s, dr, dc) + this.g[s];
        if (v < best) { best = v; arg = s; }
      }
      this.arg = arg;
      return best;
    }
    compute() {
      const b = this.b, C = b.C, R = b.R, g = this.g, rhs = this.rhs, goal = this.goal, before = this.expanded;
      for (;;) {
        const live = this.top(), ks = this.key(this.start);
        const o1 = live ? this.k1[0] : Infinity, o2 = live ? this.k2[0] : Infinity;
        if (!((o1 < ks[0] || (o1 === ks[0] && o2 < ks[1])) || rhs[this.start] > g[this.start]) || !live) break;
        const u = this.kv[0], kn = this.key(u);
        if (o1 < kn[0] || (o1 === kn[0] && o2 < kn[1])) { this.push(u, kn[0], kn[1]); continue; }
        const r = (u / C) | 0, c = u - r * C;
        if (g[u] > rhs[u]) {
          g[u] = rhs[u]; this.inq[u] = 0; this.expanded++;
          for (let k = 0; k < 8; k++) {
            const dr = NB_R[k], dc = NB_C[k], pr = r - dr, pc = c - dc;
            if (pr < 0 || pc < 0 || pr >= R || pc >= C) continue;
            const p = u - dr * C - dc;
            if (p !== goal) { const v = b.cost(p, u, dr, dc) + g[u]; if (v < rhs[p]) rhs[p] = v; }
            this.updateVertex(p);
          }
        } else {
          const gOld = g[u];
          g[u] = Infinity; this.expanded++;
          for (let k = 0; k < 8; k++) {
            const dr = NB_R[k], dc = NB_C[k], pr = r - dr, pc = c - dc;
            if (pr < 0 || pc < 0 || pr >= R || pc >= C) continue;
            const p = u - dr * C - dc;
            if (p !== goal && rhs[p] === b.cost(p, u, dr, dc) + gOld) rhs[p] = this.bestSucc(p);
            this.updateVertex(p);
          }
          this.updateVertex(u);
        }
      }
      return this.expanded - before;
    }
    movedAndSensed(changed) {
      if (!changed.length) return;
      const b = this.b, C = b.C, R = b.R;
      this.km += b.h(this.last, this.start); this.last = this.start;
      const touched = new Set();
      for (const x of changed) {
        const r = (x / C) | 0, c = x - r * C;
        touched.add(x);
        for (let k = 0; k < 8; k++) { const nr = r + NB_R[k], nc = c + NB_C[k]; if (nr >= 0 && nc >= 0 && nr < R && nc < C) touched.add(nr * C + nc); }
      }
      for (const u of [...touched].sort((a, z) => a - z)) { if (u !== this.goal) this.rhs[u] = this.bestSucc(u); this.updateVertex(u); }
    }
    plan() {
      const out = [this.start], N = this.b.N;
      let u = this.start;
      while (u !== this.goal && out.length <= N) { if (this.bestSucc(u) === Infinity) return []; u = this.arg; out.push(u); }
      return u === this.goal ? out : [];
    }
  }

  // ---------------------------------------------------------------- the seeing drive (see_drive)
  // A generator: one record per step (where he is, what he saw on the way there, the plan he drove that step on, his
  // plan from there, what changed on his map), so the page can show the drive while it is still being driven. eye:
  // "map" (knows the ground), "blind", "truth" (perfect labels), "net", "colour" or "pixel". o: the settings
  // (SETTINGS with the eye's CHOSEN on top). The result's ms is his thinking (D* Lite and his map); seeMs, the camera
  // and the model.
  function* drive(sc, si, gi, eye, look, o) {
    let busy = 0, t0 = Date.now(), seeing = 0;                   // time spent driving (not waiting between steps), and seeing
    const C = sc.C, N = sc.N, b = new Belief(sc, V.prior[sc.theme], o), nk = sc.costs.length;
    const maxRange = o.max_range == null ? Infinity : o.max_range, kinds = sc.kinds, offs = [], L = V.lidar, n = Math.floor(L);
    for (let dr = -n; dr <= n; dr++) for (let dc = -n; dc <= n; dc++) if (dr * dr + dc * dc <= L * L) offs.push([dr, dc]);
    if (eye === "map") for (let x = 0; x < N; x++) { b.felt[x] = 1; b.kind[x] = sc.surface[x]; b.S[x] = sc.costs[sc.surface[x]]; }
    const sense = at => {                                        // the short-range sensor finds the obstacles
      const r = (at / C) | 0, c = at % C, changed = [];
      for (const [dr, dc] of offs) {
        const nr = r + dr, nc = c + dc;
        if (nr < 0 || nc < 0 || nr >= sc.R || nc >= C) continue;
        const x = nr * C + nc;
        if (sc.blocked[x] && !b.blk[x]) { b.blk[x] = 1; changed.push(x); }
      }
      return changed;
    };
    const shown = new Int16Array(N).fill(-1), p = new Float64Array(nk);   // what the map on screen shows per cell
    const mapDelta = cells => {
      const out = [];
      for (const x of cells) {
        const k = b.felt[x] ? b.kind[x] : b.seen[x] ? b.likeliest(x, p) : -1;
        if (k !== shown[x]) { shown[x] = k; out.push(x, k); }
      }
      return out;
    };
    const right = () => {
      let seen = 0, ok = 0;
      for (let x = 0; x < N; x++) if (b.seen[x]) { seen++; if (b.likeliest(x, p) === sc.surface[x]) ok++; }
      return [seen, ok];
    };
    const camera = eye !== "blind" && eye !== "map";
    sense(si); b.feel(si, sc.surface[si]); b.neighbourhood();
    const ds = new DStar(b, si, gi);
    ds.compute();                                                // (the first plan's work isn't counted, as in Python)
    let work = 0, plan = ds.plan(), cur = si, frames = 0;
    const driven = [si], limit = o.max_steps || 4 * N;
    const first = { k: 0, cell: si, plan, map: mapDelta(eye === "map" ? Array.from({ length: N }, (_, i) => i) : [si]), work, seen: 0, right: 0 };
    busy += Date.now() - t0; yield first; t0 = Date.now();
    while (plan.length && cur !== gi && driven.length <= limit) {
      const ahead = plan[1], at = cur;
      let changed = [], frame = null, observed = [];
      if (camera) {                                              // a frame towards the next step of the plan
        const r = (cur / C) | 0, c = cur % C, dr = ((ahead / C) | 0) - r, dc = (ahead % C) - c;
        const hk = NB_R.findIndex((v, k) => v === dr && NB_C[k] === dc);
        const ts = Date.now(), f = render(sc, r, c, hk, look, look.seed + driven.length), dcode = encodeDist(f.demDist);
        const lp = logProbs(eye, f, dcode, sc.theme);
        seeing += Date.now() - ts;                               // the camera and the model: the rest is thinking
        const ev = frameEvidence(lp, f, kinds, maxRange);
        changed = b.observe(ev.cells, ev.ev); observed = ev.cells;
        const pred = new Uint8Array(NPIX);
        for (let i = 0; i < NPIX; i++) { let best = 0; for (let k = 1; k < K; k++) if (lp[i * K + k] > lp[i * K + best]) best = k; pred[i] = best; }
        frame = { image: f.image, label: f.label, pred, heading: hk };
        frames++;
        changed = changed.concat(b.neighbourhood());
      }
      if (changed.length) {
        ds.movedAndSensed([...new Set(changed)].sort((a, z) => a - z)); work += ds.compute(); plan = ds.plan();
        if (!plan.length) break;
      }
      const via = camera ? plan : null;                          // the plan he drives this step on (after the frame)
      cur = plan[1]; driven.push(cur); ds.start = cur;
      changed = sense(cur).concat(b.feel(cur, sc.surface[cur]), b.neighbourhood());
      if (changed.length) { ds.movedAndSensed([...new Set(changed)].sort((a, z) => a - z)); work += ds.compute(); }
      plan = ds.plan();
      const [seen, ok] = right();
      const step = { k: driven.length - 1, cell: cur, from: at, frame, via, plan, map: mapDelta(observed.concat([cur])), work, seen, right: ok };
      busy += Date.now() - t0; yield step; t0 = Date.now();
    }
    const [seen, ok] = right();
    busy += Date.now() - t0;
    return { found: cur === gi, driven, frames, work, seen, seenRight: ok, ms: busy - seeing, seeMs: seeing };
  }

  return { init, makeScene, render, encodeDist, netLogits, logProbs, frameEvidence, Belief, DStar, drive, PCG, seedState,
           hash01, vnoise, uniform2d, sobel, colours: () => COLOURS, classes: () => V.classes };
})();

// In the page's worker: drive when asked, one step a message, and stop when a newer drive is asked for.
if (typeof importScripts === "function" && typeof self !== "undefined") {
  let scene = null, current = 0;
  self.onmessage = async ev => {
    const m = ev.data;
    if (m.type === "init") EYES.init(m.vision);
    else if (m.type === "map") { scene = EYES.makeScene(m.map); scene.maxGrade = m.map.max_grade; scene.uphill = m.uphill; }
    else if (m.type === "stop") current = m.id;
    else if (m.type === "drive") {
      current = m.id;
      const it = EYES.drive(scene, m.si, m.gi, m.eye, m.look, m.settings);
      for (;;) {
        const t = performance.now(), { value, done } = it.next();
        if (m.id !== current) return;
        if (done) { self.postMessage({ type: "done", id: m.id, result: value }); return; }
        const tr = value.frame ? [value.frame.image.buffer, value.frame.label.buffer, value.frame.pred.buffer] : [];
        self.postMessage({ type: "step", id: m.id, step: value, ms: performance.now() - t }, tr);
        await new Promise(res => setTimeout(res, 0));             // let a newer request in
        if (m.id !== current) return;
      }
    }
  };
}
if (typeof module !== "undefined" && module.exports) module.exports = EYES;
