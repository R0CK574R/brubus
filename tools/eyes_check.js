// The page's camera, models and seeing drive (robotnav/eyes.js), run under Node on cases written by Python, so
// tests/test_vision.py can hold them to robotnav.learn.camera and robotnav.learn.vision.
//   node tools/eyes_check.js cases.json > results.json
// cases.json: { vision, uphill, maps: {key: map}, frames: [...], models: [...], drives: [...] } (see the test).
"use strict";
const fs = require("fs"), crypto = require("crypto"), path = require("path");
const EYES = require(path.join(__dirname, "..", "robotnav", "eyes.js"));
const cases = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
EYES.init(cases.vision);
const scenes = {};
const sceneOf = key => scenes[key] || (scenes[key] = Object.assign(EYES.makeScene(cases.maps[key]),
  { maxGrade: cases.maps[key].max_grade, uphill: cases.uphill }));
const bytes = a => Buffer.from(a.buffer, a.byteOffset, a.byteLength);
const sha = a => crypto.createHash("sha256").update(bytes(a)).digest("hex");
const b64 = a => bytes(a).toString("base64");
const fromB64 = (s, Type) => { const b = Buffer.from(s, "base64"); return new Type(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength)); };
const out = { frames: [], models: [], drives: [] };
for (const c of cases.frames || []) {                       // a frame at a pose: every array, as bytes
  const t = Date.now(), f = EYES.render(sceneOf(c.map), c.r, c.c, c.dir, c.look, c.seed);
  out.frames.push({ ms: Date.now() - t, image: b64(f.image), label: b64(f.label), cell: b64(f.cell), dist: b64(f.dist),
                    demCell: b64(f.demCell), demDist: b64(f.demDist), dcode: b64(EYES.encodeDist(f.demDist)) });
}
for (const c of cases.models || []) {                       // a model on a given frame (image and distance bytes)
  const image = fromB64(c.image, Uint8Array), dcode = fromB64(c.dcode, Uint8Array), label = fromB64(c.label, Uint8Array);
  const t = Date.now(), r = { eye: c.eye };
  if (c.eye === "net") r.logits = sha(EYES.netLogits(image, dcode));
  r.lp = b64(EYES.logProbs(c.eye, { image, label }, dcode, c.theme));
  r.ms = Date.now() - t;
  out.models.push(r);
}
for (const c of cases.drives || []) {                       // a whole seeing drive
  const sc = sceneOf(c.map), it = EYES.drive(sc, c.si, c.gi, c.eye, c.look, c.settings), t = Date.now();
  let res;
  for (;;) { const s = it.next(); if (s.done) { res = s.value; break; } }
  out.drives.push(Object.assign(res, { ms: Date.now() - t, eye: c.eye }));
}
process.stdout.write(JSON.stringify(out));
