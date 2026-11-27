"""Export the three themed maps to one self-contained 3-D viewer (HTML + Three.js).

Python generates the maps, picks each map's 20 optional preset routes, and is the
reference implementation of every planner. When you drive a route, the browser runs
the same five planners plus the hill-blind shortest route (ported line for line), and
the tests check both give identical routes and times.
"""
from __future__ import annotations

import json
from html import escape
from pathlib import Path

import numpy as np

from .airspace import AIR, place_towers
from .learn import speeds as SP
from .sensing import SENSE
from .intro_art import backdrop_svg, logo_uri
from .terrain import SHORTEST_TIE, Terrain, preset_routes

TEMPLATE = Path(__file__).with_name("viewer3d.html")
EYES_JS = Path(__file__).with_name("eyes.js")                # the camera mode's camera, models and drive (stage 4b)
CORE_WASM = Path(__file__).with_name("assets") / "robotnav_core.wasm"
NAMES = {"grassland": "Grassland", "desert": "Desert", "island": "Island"}



def core_wasm() -> str:
    """The C++ core compiled to WebAssembly (tools/build_wasm.sh), as base64 for the page; "" if it isn't built."""
    import base64
    return base64.b64encode(CORE_WASM.read_bytes()).decode("ascii") if CORE_WASM.exists() else ""


def eyes_js() -> str:
    """robotnav/eyes.js, to ride in the page as the text of a script the page doesn't run itself (its worker does)."""
    text = EYES_JS.read_text()
    assert "</script" not in text.lower()
    return text


def map_scene(t: Terrain) -> dict:
    """A map for the page: its own data (map_core), the 20 optional paths and the air route's watchtowers."""
    return dict(map_core(t),
                presets=[dict(name=p["name"], start=list(p["start"]), goal=list(p["goal"]), time=float(p["time"]),
                              distance_m=float(p["distance_m"]), short_time=float(p["short_time"]),
                              saving=float(p["saving"]), path=[int(v) for cell in p["path"] for v in cell])
                         for p in preset_routes(t)],         # the 20 optional paths, quickest route each
                towers=[list(cell) for cell in place_towers(t)])   # the air route's watchtowers


def map_core(t: Terrain) -> dict:
    """A map's own data for the page, without the paths and towers that take a while to work out: what the planners
    and the camera need (tests/test_vision.py hands it to the page's camera under Node)."""
    theme = t.meta["theme"]
    return dict(
        key=theme, name=NAMES[theme], seed=t.meta["seed"], meta=t.meta,
        rows=t.rows, cols=t.cols, max_grade=t.max_grade, sea=t.sea_level,
        relief=round(float(t.height.max() - t.height.min()), 2),
        heights=[float(v) for v in t.height.ravel()],        # exact, so the browser plans like Python
        surface=[int(v) for v in t.surface.ravel()],
        obstacle=[int(v) for v in t.obstacle.ravel()],       # 0 open, 1 rock, 2 tree/cactus/palm, 3 water
        ponds=[dict(level=float(p["level"]), cells=[int(r * t.cols + c) for r, c in p["cells"]])
               for p in t.ponds],                            # desert oases: water level and cells (flat index)
        surfaces=[dict(name=n, cost=float(c)) for n, c in zip(t.surface_names, t.surface_costs)],
        kind=[SP.KINDS.index(n) for n in t.surface_names],   # each surface's kind of ground, for the AI mode
        start=[int(v) for v in t.start], goal=[int(v) for v in t.goal],
    )


def ai_scene() -> dict:
    """The AI mode's data (README, stage 3): the hidden true world, and Bruce's network after each checkpoint of the
    career he learned in (the shipped one, stage 3c), with the study's test-map results. Weights travel as base64
    little-endian doubles (mean, scale, then each layer's W row by row and b), so the browser has them exactly."""
    import base64
    card = json.loads(SP.ASSET.with_suffix(".card.json").read_text())
    career = card["careers"][-1]
    final = career["curve"][-1]["trips"]
    summary = {s["model"]: s for s in card["summary"] if s["theme"] == "all"}
    nets, study = [], [dict(trips=0, regret=summary["built-in"]["regret"], worst=summary["built-in"]["worst"],
                            within_1=summary["built-in"]["within_1"])]
    for point in career["curve"]:
        n = point["trips"]
        m = SP.load(SP.ASSET if n == final else SP.checkpoint_path(n))
        flat = np.concatenate([m.mean, m.scale] + [np.concatenate([W.ravel(), b]) for W, b in m.layers])
        nets.append(dict(trips=n, moves=point["moves"], sizes=[SP.FEATURES] + [int(W.shape[1]) for W, _ in m.layers],
                         data=base64.b64encode(np.asarray(flat, dtype="<f8").tobytes()).decode("ascii")))
        s = summary[f"{career['name']}, {n} trips"]
        study.append(dict(trips=n, regret=s["regret"], worst=s["worst"], within_1=s["within_1"]))
    return dict(kinds=list(SP.KINDS), slow=[SP.TRUE_SLOW[k] for k in SP.KINDS], uphill=[SP.TRUE_UPHILL[k] for k in SP.KINDS],
                brake_from=SP.BRAKE_FROM, brake=SP.BRAKE, side=SP.SIDE, rough=SP.ROUGH, nets=nets, study=study,
                trips_tested=summary["built-in"]["trips"])


LOOKS = 21                                         # per map: one light per preset trip, and one for trips you set


def page_look(lk) -> dict:
    """A camera light (robotnav.learn.camera.Look) as the page takes it: the sun as a direction, worked out here."""
    import math
    sun = [-math.cos(lk.sun_az) * math.cos(lk.sun_el), math.sin(lk.sun_az) * math.cos(lk.sun_el), math.sin(lk.sun_el)]
    return dict(sun=sun, ambient=lk.ambient, fog=lk.fog, exposure=lk.exposure, balance=list(lk.balance), noise=lk.noise,
                seed=lk.seed)


def vision_scene(maps: list[Terrain]) -> dict:
    """The camera mode's data (README, stage 4b): what the page needs to see as Bruce does in stage 4, from
    robotnav.learn.camera and robotnav.learn.vision. The camera's constants, the map colours, the three models (the
    network's weights as base64 little-endian 32-bit floats, as committed; the others' numbers as Python has them, the
    colour model's inverses and determinants included), what each map can show and what Bruce expects of its ground
    before he looks, each way of seeing's settings, a light for each trip, NumPy's normal-generator tables, and the
    study's results. Everything trigonometric is worked out here, so the page uses Python's own numbers."""
    import base64
    import math
    from .learn import camera as cam
    from .learn import rng as RN
    from .learn import vision as V
    from .replan import NB_C, NB_R
    from .terrain import THEMES, WATER
    card = json.loads(V.ASSET.with_suffix(".card.json").read_text())
    models = V.models_from()
    a = V.arrays_from()
    c = cam.CAMERA
    p = math.radians(c.pitch)
    layers = [f"{s}.{j}" for s, spec in V.NET.items() if s != "head" for j in range(len(spec))] + ["head"]
    blob, shapes = [], []
    for name in layers:
        for part in ("w", "b"):
            w = np.asarray(a[f"net.{name}.{part}"], dtype="<f4")
            blob.append(w.tobytes())
            shapes.append([f"{name}.{part}", list(w.shape)])
    colour, pixel = models["colour"], models["pixel"]
    looks = {}
    for t in maps:
        theme = t.meta["theme"]
        if theme not in cam.PALETTE:
            continue
        water = np.asarray(t.obstacle) == WATER
        level = cam.water_levels(t)
        assert np.isfinite(level[water]).all(), "the page's camera takes water levels from the sea and the ponds"
        out = [page_look(cam.random_look(np.random.default_rng([V.TRIP_SALT, THEMES.index(theme), int(t.meta["seed"]), i])))
               for i in range(LOOKS)]                        # a light for each preset trip, as the study draws them
        sc = cam.Scene(t)
        looks[theme] = dict(looks=out, wind=[math.cos(sc.wind), math.sin(sc.wind)])
    summary = {s["variant"]: s for s in card["test"]["drives"] if s["theme"] == "all"}
    zig = RN.tables()
    return dict(
        classes=list(cam.CLASSES), ground=cam.GROUND, kinds=list(cam.KINDS),
        camera=dict(width=c.width, height=c.height, eye=c.eye, near=c.near, far=c.far, step=c.step, growth=c.growth,
                    focal=c.focal, pixel_angle=c.pixel_angle, cos_pitch=math.cos(p), sin_pitch=math.sin(p)),
        dist=dict(near=V.NEAR, far=V.FAR, log=math.log(V.FAR / V.NEAR)),           # the distance byte's scale
        headings=[[math.cos(math.atan2(dc, -dr)), math.sin(math.atan2(dc, -dr))] for dr, dc in zip(NB_R, NB_C)],
        rock_r=cam.ROCK_R, trees=cam.TREES, palm=cam.PALM, texture=cam.TEXTURE, palette=cam.PALETTE,
        label_colors=[[float(v) for v in col] for col in cam.label_colors()],
        allowed={th: [bool(v) for v in V.allowed(th)] for th in THEMES},
        prior={th: [float(v) for v in V.kind_prior(th, card["data"]["train_pixels"])] for th in THEMES},
        net=dict(layers=shapes, data=base64.b64encode(b"".join(blob)).decode("ascii"),
                 sha256=card["reference"]["net_logits_sha256"]),
        colour=dict(mean=colour.mean.tolist(), inv=colour.inv.tolist(), logdet=colour.logdet.tolist(),
                    prior={th: colour.prior[th].tolist() for th in THEMES}),
        pixel=dict(mean=pixel.mean.tolist(), scale=pixel.scale.tolist(), W=pixel.W.tolist(), b=pixel.b.tolist()),
        settings=dict(V.SETTINGS, max_range=None, hysteresis=0.01), chosen=V.CHOSEN, sure=V.SURE, lidar=V.LIDAR,
        names=V.VARIANT_NAMES, looks=looks,
        normal=dict(r=zig["r"], inv_r=zig["inv_r"], wi=zig["wi"], ki=zig["ki"], fi=zig["fi"]),
        study=dict(trips=summary["net"]["trips"], mean={v: summary[v]["mean"] for v in V.VARIANTS},
                   ground={k: card["test"]["pixels"][k]["ground_accuracy"] for k in ("colour", "pixel", "net")}),
    )


TITLE = "Rover Hill Course"
DESCRIPTION = ("A rover finds the quickest path over hilly terrain: five pathfinding algorithms raced in 3D, "
               "the search replayed side by side, D* Lite re-planning with a limited sensor, and a drone that flies "
               "an air route out of enemy sight. Runs in the browser.")


def write_viewer(maps: list[Terrain], out: str | Path, scenes: list[dict] | None = None,
                 site_url: str | None = None) -> Path:
    """Write the viewer as a standalone HTML page.

    Pass `scenes` (from map_scene) if you already built them; picking presets takes a few seconds.
    Pass `site_url` (where the page will be hosted, ending in "/") to add link-preview tags, so a shared
    link shows the title, a description and social.png from that address.
    """
    scene = dict(cost=dict(uphill=maps[0].uphill_penalty, cell=maps[0].cell_size, short_tie=SHORTEST_TIE), air=AIR, sense=SENSE,
                 ai=ai_scene(), vision=vision_scene(maps), maps=scenes if scenes is not None else [map_scene(t) for t in maps])
    body = (TEMPLATE.read_text().replace("/*__SCENE__*/null", json.dumps(scene, separators=(",", ":")))
            .replace("<!--__BACKDROP__-->", backdrop_svg()).replace("__LOGO_URI__", logo_uri())
            .replace("__CORE_WASM__", core_wasm()).replace("/*__EYES__*/", eyes_js()))
    head = ["<!doctype html>", '<html lang="en">', '<meta charset="utf-8">',
            f'<meta name="description" content="{escape(DESCRIPTION)}">']
    if site_url:
        image = site_url + "social.png"
        head += [f'<link rel="canonical" href="{escape(site_url)}">',
                 '<meta property="og:type" content="website">',
                 f'<meta property="og:title" content="{escape(TITLE)}">',
                 f'<meta property="og:description" content="{escape(DESCRIPTION)}">',
                 f'<meta property="og:url" content="{escape(site_url)}">',
                 f'<meta property="og:image" content="{escape(image)}">',
                 '<meta property="og:image:width" content="1200">', '<meta property="og:image:height" content="630">',
                 '<meta name="twitter:card" content="summary_large_image">',
                 f'<meta name="twitter:title" content="{escape(TITLE)}">',
                 f'<meta name="twitter:description" content="{escape(DESCRIPTION)}">',
                 f'<meta name="twitter:image" content="{escape(image)}">']
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(head) + "\n" + body)
    return out
