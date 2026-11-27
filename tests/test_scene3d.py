from robotnav.scene3d import write_viewer
from robotnav.terrain import solvable_map


def _page(tmp_path, **kw):
    t = solvable_map("grassland", seed=3, size=48)
    return write_viewer([t], tmp_path / "docs" / "index.html", scenes=[{"key": "grassland"}], **kw).read_text()


def test_viewer_is_a_full_standards_mode_document(tmp_path):
    page = _page(tmp_path)
    assert page.startswith("<!doctype html>\n")
    assert '<meta charset="utf-8">' in page[:200]               # the page has symbols like × and ⇄
    assert '"key":"grassland"' in page                          # the scene data went in
    assert "og:image" not in page                               # no link-preview tags unless it has a home


def test_hosted_viewer_has_link_previews(tmp_path):
    page = _page(tmp_path, site_url="https://example.github.io/demo/")
    head = page[:2500]
    assert '<meta property="og:image" content="https://example.github.io/demo/social.png">' in head
    assert '<meta name="twitter:card" content="summary_large_image">' in head
    assert '<link rel="canonical" href="https://example.github.io/demo/">' in head


def test_viewer_has_the_console_look(tmp_path):
    page = _page(tmp_path)
    assert "family=Inter" in page and "JetBrains+Mono" in page and "Orbitron" not in page   # words and numbers
    assert 'id="hud-compass-cv"' in page and 'id="hud-alt-cv"' in page               # the drone view's read-outs
    assert "M 155,150 C 155,50 115,30 100,30 85,30 45,50 45,150" in page             # MIL-STD-2525 friendly air frame
    assert "M 100,28 L 172,100 100,172 28,100 Z" in page                              # and the hostile ground one
    assert "UAV-01" in page and "TWR-" in page and "TRACKING" in page                # callsign tags


def test_viewer_has_the_operator_console(tmp_path):
    page = _page(tmp_path)
    for part in ("ops-ents", "ops-tasks", "ops-tl", "ops-log", "ops-pal", "ops-aar"):   # lists, timeline, log, command bar, report
        assert f'id="{part}"' in page
    assert '{ key: "balanced", name: "Balanced" }' in page and "const OPS_W = [0, 0.5, 1," in page   # three plans from a sweep of weights
    assert '"sense":{"pfa":0.0001,"r50":10.0' in page and "SCENE.sense" in page   # the sensing constants come from Python
    for fn in ("function pdAt(", "function ekfPredict(", "function ekfUpdate(", "function wilson(", "function opsTrial("):
        assert fn in page                                                        # detection, the tracker, the trials
    for status in ("SENT", "RUNNING", "DONE_OK", "DONE_NOT_OK"):                 # a task's life
        assert f'"{status}"' in page
    assert "re-planned: roadblock at" in page and "getUTCHours" in page          # the log's wording and its UTC clock
    assert 'w === "ops"' in page                                                 # the #ops link


def test_viewer_tasks_several_vehicles(tmp_path):
    page = _page(tmp_path)
    assert "const OPS_DRONES = 3, OPS_SEP = 3, OPS_W_FLEET = 4;" in page          # a tank and three drones, kept 3 m apart
    for fn in ("function hungarian(", "function opsAssign(", "function opsHold(", "function opsSurveyApprove("):
        assert fn in page                                                        # who goes where, and the hold on the ground
    assert 'id="ops-survey-btn"' in page and "Nearest-first happens to give the same here." in page


def test_the_cpp_core_rides_in_the_page_as_webassembly(tmp_path):
    import base64
    from robotnav.scene3d import CORE_WASM
    wasm = CORE_WASM.read_bytes()
    assert wasm[:4] == b"\0asm" and len(wasm) < 400_000                         # a WebAssembly module, small enough to carry
    for name in (b"rn_map_done", b"rn_astar", b"rn_plan", b"rn_trials", b"rn_digest"):
        assert name in wasm                                                     # what the page and the Node test call
    page = _page(tmp_path)
    assert base64.b64encode(wasm).decode() in page and "__CORE_WASM__" not in page
    assert "coreTrials(" in page and "WebAssembly.instantiate(" in page


def test_intro_art_is_inlined(tmp_path):
    page = _page(tmp_path)
    assert "__BACKDROP__" not in page and "__LOGO_URI__" not in page   # both placeholders were filled
    assert page.count('<svg class="intro-back"') == 1                # one backdrop, cropped to fit any screen
    assert 'src="data:image/webp;base64,' in page                  # the badge travels inside the page
    assert '<div class="intro-word">BRUBUS</div>' in page and '"Bruce · unmanned ground vehicle"' in page   # the project, and the tank's name


def test_viewer_carries_bruces_networks_exactly(tmp_path):
    """AI mode's data: each checkpoint network's weights decode to the committed file's doubles, bit for bit, and the
    hidden world's numbers are Python's."""
    import base64
    import json
    import re
    import numpy as np
    from robotnav.learn import speeds as sp
    from robotnav.scene3d import ai_scene, map_scene
    page = _page(tmp_path)
    scene = json.loads(re.search(r"const SCENE = (\{.*?\});\n", page, re.S).group(1))
    assert scene["ai"] == ai_scene()
    ai = scene["ai"]
    assert ai["kinds"] == list(sp.KINDS) and ai["slow"] == [sp.TRUE_SLOW[k] for k in sp.KINDS]
    assert ai["uphill"] == [sp.TRUE_UPHILL[k] for k in sp.KINDS] and (ai["side"], ai["rough"]) == (sp.SIDE, sp.ROUGH)
    card = json.loads(sp.ASSET.with_suffix(".card.json").read_text())
    career = card["careers"][-1]
    assert [n["trips"] for n in ai["nets"]] == [p["trips"] for p in career["curve"]] == [10, 30, 100, 300]
    assert [n["sizes"] for n in ai["nets"]] == [p["network"] for p in career["chosen"]]   # the sizes he chose
    for net in ai["nets"]:
        m = sp.load(sp.ASSET if net["trips"] == 300 else sp.checkpoint_path(net["trips"]))
        want = np.concatenate([m.mean, m.scale] + [np.concatenate([W.ravel(), b]) for W, b in m.layers])
        got = np.frombuffer(base64.b64decode(net["data"]), dtype="<f8")
        assert np.array_equal(got, want)                                          # every weight, exactly
    assert ai["study"][0]["trips"] == 0 and ai["study"][-1]["regret"] < ai["study"][0]["regret"] / 50
    t = solvable_map("island", seed=3, size=48)
    assert [sp.KINDS[k] for k in map_scene(t)["kind"]] == list(t.surface_names)   # each surface's kind of ground


def test_viewer_has_the_ai_mode(tmp_path):
    page = _page(tmp_path)
    assert 'id="ai"' in page and 'data-trips="300"' in page and 'aiBtn.textContent = "◆ AI"' in page   # panel, chips, button
    for fn in ("function aiPerMetre(", "function aiSense(", "function aiTrue(", "function aiRoutes(", "function setAi("):
        assert fn in page                                                         # his network, his senses, the truth
    assert 'w === "ai"' in page and "/^t(\\d{1,3})$/" in page                    # the #ai link, and /t30 for fewer trips


def test_the_tour_stops_at_the_ai_mode(tmp_path):
    page = _page(tmp_path)
    assert '[14, "ai"], [23, "aiTrue"], [34, "air"]' in page                   # on the grassland, before the air route
    assert "ai() {" in page and "aiTrue() {" in page and "setAi(true, true);" in page
    assert "setAir(false, true); setAi(false, true);" in page                  # each map's act starts on the ground


def test_viewer_carries_bruces_camera_exactly(tmp_path):
    """The camera mode's data (stage 4b): the network's weights decode to the committed 32-bit floats, bit for bit,
    layer by layer; everything else is vision_scene's, and each preset trip has the light the study's recipe draws."""
    import base64
    import json
    import re
    import numpy as np
    from robotnav.learn import camera as cam
    from robotnav.learn import vision as V
    from robotnav.scene3d import LOOKS, page_look, vision_scene
    from robotnav.terrain import THEMES
    page = _page(tmp_path)
    vis = json.loads(re.search(r"const SCENE = (\{.*?\});\n", page, re.S).group(1))["vision"]
    t = solvable_map("grassland", seed=3, size=48)
    assert vis == json.loads(json.dumps(vision_scene([t])))
    a, blob, at = V.arrays_from(), base64.b64decode(vis["net"]["data"]), 0
    for name, shape in vis["net"]["layers"]:
        w = np.asarray(a[f"net.{name}"], dtype="<f4")
        assert list(w.shape) == shape and blob[at:at + w.nbytes] == w.tobytes()
        at += w.nbytes
    assert at == len(blob) and sum(np.prod(s) for _, s in vis["net"]["layers"]) == 61_564   # every weight, nothing else
    looks = vis["looks"]["grassland"]["looks"]
    assert len(looks) == LOOKS == 21                                             # one a preset, one for trips you set
    for i in (0, 20):
        rng = np.random.default_rng([V.TRIP_SALT, THEMES.index("grassland"), 3, i])
        assert looks[i] == json.loads(json.dumps(page_look(cam.random_look(rng))))
    assert len(vis["normal"]["wi"]) == len(vis["normal"]["ki"]) == len(vis["normal"]["fi"]) == 256   # NumPy's ziggurat


def test_viewer_has_the_camera_mode(tmp_path):
    from robotnav.scene3d import eyes_js
    page = _page(tmp_path)
    assert 'id="eyes"' in page and 'data-eye="net"' in page and 'eyesBtn.textContent = "◉ Camera"' in page   # panel, chips, button
    assert 'id="ey-cam"' in page and 'id="ey-lab"' in page and 'id="ey-truth"' in page    # his camera, the labels, the truth
    assert eyes_js() in page and "/*__EYES__*/" not in page    # the camera, models and drive, as tested under Node
    for fn in ("function eyesRoutes(", "function eyesTick(", "function setEyes(", "new Worker(URL.createObjectURL("):
        assert fn in page                                                       # the trip, each frame, the mode, the worker
    assert 'w === "camera"' in page and 'pixel: "classic"' in page             # #desert/p19/camera, /classic for classic ML
