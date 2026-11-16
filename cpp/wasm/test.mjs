// The WebAssembly build of the C++ core, checked under Node: it must print the very digest the native builds print
// (cpp/tests/golden.digest: every result on every preset trip of the three maps, bit for bit), and its Monte Carlo
// trials must not depend on how they are batched.
//   node cpp/wasm/test.mjs [robotnav_core.wasm]
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url)), root = join(here, "..", "..");
const wasm = readFileSync(process.argv[2] || join(root, "robotnav", "assets", "robotnav_core.wasm"));
const wasi = { clock_time_get: () => 0, fd_close: () => 0, fd_seek: () => 0, fd_write: () => 0 };   // the core needs no system
const { instance } = await WebAssembly.instantiate(wasm, { wasi_snapshot_preview1: wasi });
const x = instance.exports;
x._initialize();

// An array into the module's memory; the caller frees it.
function put(Type, values) {
    const at = x.rn_alloc(values.length * Type.BYTES_PER_ELEMENT);
    new Type(x.memory.buffer, at, values.length).set(values);
    return at;
}
// A map file (cpp/include/robotnav/mapio.hpp says how it reads) into the module.
function load(text) {
    const lines = text.split("\n");
    let i = 0;
    const head = {};
    const ponds = [], presets = [];
    for (; lines[i] !== "height"; i++) {
        const [key, ...rest] = lines[i].split(" ");
        if (key === "pond") ponds.push(rest.map(Number));
        else if (key === "preset") presets.push(rest.slice(0, 4).map(Number));
        else head[key] = rest.join(" ");
    }
    const R = +head.rows, C = +head.cols;
    const grid = () => { const out = []; for (let r = 0; r < R; r++) out.push(...lines[++i].trim().split(/\s+/).map(Number)); i++; return out; };
    const height = grid(), surface = grid(), obstacle = grid();
    x.rn_map_begin(R, C, +head.cell, +head.max_grade, +head.uphill, +head.short_tie);
    const name = put(Uint8Array, [...Buffer.from(head.theme), 0]);
    x.rn_map_theme(name, +head.seed);
    const h = put(Float64Array, height), s = put(Int32Array, surface), o = put(Int32Array, obstacle);
    x.rn_map_grids(h, s, o);
    const costs = head.surface_costs.split(" ").map(Number), c = put(Float64Array, costs);
    x.rn_map_surfaces(c, costs.length);
    if (head.sea !== "none") x.rn_map_sea(+head.sea);
    for (const [level, n, ...rc] of ponds) {
        const cells = []; for (let k = 0; k < n; k++) cells.push(rc[2 * k] * C + rc[2 * k + 1]);
        const p = put(Int32Array, cells); x.rn_map_pond(level, p, n); x.rn_free(p);
    }
    for (const [sr, sc, gr, gc] of presets) x.rn_map_preset(sr * C + sc, gr * C + gc);
    for (const p of [name, h, s, o, c]) x.rn_free(p);
    x.rn_map_done();
    return { R, C, presets };
}

let failures = 0;
const check = (ok, what) => { console.log(`${ok ? "ok  " : "FAIL"} ${what}`); if (!ok) failures++; };
let digest = "";
for (const theme of ["grassland", "desert", "island"]) {
    const map = load(readFileSync(join(root, "maps", `${theme}.map`), "utf8"));
    const at = x.rn_digest();
    digest += Buffer.from(x.memory.buffer, at, x.rn_text_length()).toString("latin1");
    // the tank's plan on the first trip, and its trials in one batch and in seven: the same counts
    const [sr, sc, gr, gc] = map.presets[0], out = x.rn_alloc(4 * 20000), res = x.rn_alloc(8 * 10);
    const n = x.rn_plan(0, sr * map.C + sc, gr * map.C + gc, 4, -1, out, 20000);
    const run = (first, count) => { x.rn_trials(0, out, n, 4, first, count, 20261111, -1, res); return [...new Float64Array(x.memory.buffer, res, 10)]; };
    const whole = run(0, 700), parts = Array(10).fill(0);
    for (let k = 0; k < 7; k++) run(k * 100, 100).forEach((v, j) => { parts[j] += v; });
    check(n > 1 && whole.slice(0, 9).every((v, j) => v === parts[j]) && Math.abs(whole[9] - parts[9]) < 1e-6, `${theme}: 700 trials give the same counts in one batch or seven (${whole[3]} detected, ${whole[1]} met a roadblock)`);
    x.rn_free(out); x.rn_free(res);
}
const golden = readFileSync(join(root, "cpp", "tests", "golden.digest"), "latin1").replace(/\r\n/g, "\n");
check(digest === golden, `the digest of the three maps is cpp/tests/golden.digest, byte for byte (${digest.split("\n").length - 1} lines)`);
if (digest !== golden) {
    const a = digest.split("\n"), b = golden.split("\n"), k = a.findIndex((l, j) => l !== b[j]);
    console.log(`first difference, line ${k + 1}:\n  wasm   ${a[k]}\n  golden ${b[k]}`);
}
process.exit(failures ? 1 : 0);
