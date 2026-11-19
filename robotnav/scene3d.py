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
CORE_WASM = Path(__file__).with_name("assets") / "robotnav_core.wasm"
NAMES = {"grassland": "Grassland", "desert": "Desert", "island": "Island"}



def core_wasm() -> str:
    """The C++ core compiled to WebAssembly (tools/build_wasm.sh), as base64 for the page; "" if it isn't built."""
    import base64
    return base64.b64encode(CORE_WASM.read_bytes()).decode("ascii") if CORE_WASM.exists() else ""


def map_scene(t: Terrain) -> dict:
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
        presets=[dict(name=p["name"], start=list(p["start"]), goal=list(p["goal"]), time=float(p["time"]),
                      distance_m=float(p["distance_m"]), short_time=float(p["short_time"]),
                      saving=float(p["saving"]), path=[int(v) for cell in p["path"] for v in cell])
                 for p in preset_routes(t)],                 # the 20 optional paths, quickest route each
        towers=[list(cell) for cell in place_towers(t)],     # the air route's watchtowers
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
                 ai=ai_scene(), maps=scenes if scenes is not None else [map_scene(t) for t in maps])
    body = (TEMPLATE.read_text().replace("/*__SCENE__*/null", json.dumps(scene, separators=(",", ":")))
            .replace("<!--__BACKDROP__-->", backdrop_svg()).replace("__LOGO_URI__", logo_uri())
            .replace("__CORE_WASM__", core_wasm()))
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
