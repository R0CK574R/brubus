"""Draw the README's pictures of the learning work (they need the data sets: python main.py dataset build, and for
the last one python main.py vision build).

    python tools/learn_figure.py            # docs/learn-labels.png: one test map, one goal, one trip
    python tools/learn_figure.py bench      # docs/learn-benchmark.png: cells searched against longer routes
    python tools/learn_figure.py speeds     # docs/learn-speeds.png: stage 3's learning curve (from its card)
    python tools/learn_figure.py vision     # docs/learn-vision.png: stage 4, what the camera sees and what it's worth

Left, the label: the true drive time from every cell to the goal. Middle, A*'s estimate of the same thing today,
the octile distance, on the same colour scale: it ignores the hills and the slow ground. Right, what that costs: the
cells A* searched to plan one trip, against the route itself, which is all a perfect estimate would search.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from robotnav import cpp  # noqa: E402
from robotnav.learn import dataset as ds  # noqa: E402
from robotnav.learn.labels import astar_table, cost_to_go, octile_table  # noqa: E402

MAP, GOAL, START = "grassland-1009", (73, 3), (14, 86)       # a test map: no model will ever train on it
OUT = ROOT / "docs" / "learn-labels.png"
BG, PANEL, INK, DIM, BLUE, AMBER = "#0b0e12", "#11151a", "#e7ecef", "#8f9aa4", "#5cc2ff", "#e0a83a"


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.colors import LightSource, ListedColormap

    for f in font_manager.findSystemFonts():
        if "Inter" in Path(f).name:
            font_manager.fontManager.addfont(f)
    plt.rcParams.update({"font.family": ["Inter", "DejaVu Sans"], "text.color": INK, "axes.labelcolor": DIM,
                         "xtick.color": DIM, "ytick.color": DIM})

    m = ds.load_manifest(ds.DEFAULT_DIR)
    entry = next(e for e in m["maps"] if e["id"] == MAP)
    assert entry["split"] == "test" and list(GOAL) in entry["goals"]
    arrays = ds.load(ds.DEFAULT_DIR, entry)
    t = ds.terrain_of(arrays)
    field = cpp.cost_to_go(t, GOAL) if cpp.available() else cost_to_go(t, GOAL)
    octile = octile_table(t, GOAL)
    if cpp.available():
        a, best = cpp.astar_table(t, START, GOAL, octile), cpp.astar_table(t, START, GOAL, field)
    else:
        a, best = astar_table(t, START, GOAL, octile), astar_table(t, START, GOAL, field)
    assert a.cost == best.cost

    blocked = t.blocked
    top = float(np.nanmax(np.where(np.isfinite(field), field, np.nan)))
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("#24282e")
    shade = LightSource(azdeg=315, altdeg=40).hillshade(t.height, vert_exag=3.0)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.6), dpi=110, facecolor=BG)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.83, bottom=0.04, wspace=0.08)

    def frame(ax, title, sub):
        ax.set_facecolor(PANEL)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#2a3038")
        ax.set_title(title, loc="left", fontsize=17, fontweight="semibold", color=INK, pad=30)
        ax.text(0, 1.035, sub, transform=ax.transAxes, fontsize=11.5, color=DIM, va="bottom")

    def marks(ax, start=True):
        ax.plot(GOAL[1], GOAL[0], marker="*", ms=20, mfc="white", mec=BG, mew=1.2, zorder=5)
        if start:
            ax.plot(START[1], START[0], marker="o", ms=11, mfc=BLUE, mec="white", mew=1.6, zorder=5)

    for ax, values, title, sub in (
            (axes[0], np.where(blocked | ~np.isfinite(field), np.nan, field), "The label: true time to the goal",
             "seconds of drive time from every cell, hills and slow ground included"),
            (axes[1], np.where(blocked, np.nan, octile), "Today's estimate: straight-line distance",
             f"what A* guesses (on average {100 * np.nanmean(octile[np.isfinite(field) & (field > 0)] / field[np.isfinite(field) & (field > 0)]):.0f}% of the truth here)")):
        im = ax.imshow(np.ma.masked_invalid(values), cmap=cmap, vmin=0, vmax=top, interpolation="nearest")
        ax.contour(np.where(np.isfinite(values), values, np.nan), levels=np.arange(20, top, 20), colors="white",
                   linewidths=0.6, alpha=0.45)
        frame(ax, title, sub)
        marks(ax, start=False)
    cb = fig.colorbar(im, ax=axes[:2], fraction=0.025, pad=0.01)
    cb.set_label("seconds to the goal", color=DIM)
    cb.outline.set_edgecolor("#2a3038")

    ax = axes[2]
    ax.imshow(shade, cmap="gray", vmin=-0.3, vmax=1.4, interpolation="nearest")
    ax.imshow(np.ma.masked_where(~blocked, np.ones_like(shade)), cmap=ListedColormap(["#1a1d21"]), interpolation="nearest")
    searched = np.zeros(blocked.shape)
    for r, c in a.explored_order if hasattr(a, "explored_order") else a.explored:
        searched[r, c] = 1
    ax.imshow(np.ma.masked_where(searched == 0, searched), cmap=ListedColormap([BLUE]), alpha=0.42, interpolation="nearest")
    path = np.array(a.path)
    ax.plot(path[:, 1], path[:, 0], color="white", lw=2.6)
    marks(ax)
    frame(ax, f"A* searched {a.nodes_expanded:,} cells",
          f"for a route of {len(a.path)} cells (white); with the true time as its estimate: {best.nodes_expanded}")
    fig.text(0.02, 0.955, f"One test map ({MAP}), never trained on. The goal is the star.", fontsize=12, color=DIM)
    fig.savefig(OUT, facecolor=BG)
    print(OUT, f"A* {a.nodes_expanded} cells, route {len(a.path)}, perfect {best.nodes_expanded}")


def bench() -> None:
    """Every planner of the stage-2 benchmark (python main.py learn eval) as a point: cells searched (log scale)
    against how much longer its routes are than the quickest, on the 1,440 trips of the test maps."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from robotnav.learn import evaluate as ev
    from robotnav.learn.model import ASSET

    for f in font_manager.findSystemFonts():
        if "Inter" in Path(f).name:
            font_manager.fontManager.addfont(f)
    plt.rcParams.update({"font.family": ["Inter", "DejaVu Sans"], "text.color": INK, "axes.labelcolor": DIM,
                         "xtick.color": DIM, "ytick.color": DIM})
    rows, setup = ev.evaluate(ASSET.read_text(), split="test", log=None)
    summary = {s["planner"]: s for s in ev.summarize(rows) if s["theme"] == "all"}
    style = {"Dijkstra": ("#8f9aa4", "o"), "A* straight-line": ("#8f9aa4", "o"), "A* climb-aware": ("#8f9aa4", "o"),
             "Weighted A* 1.25": ("#8f9aa4", "s"), "Weighted A* 1.5": ("#8f9aa4", "s"),
             "A* + network": (AMBER, "D"), "A* + network, capped": (AMBER, "D"),
             "A* + 8 landmarks": (BLUE, "o"), "A* perfect": ("#ffffff", "*")}
    nudge = {"Dijkstra": (0, 10, "left"), "A* straight-line": (10, -34, "left"), "A* climb-aware": (-10, -34, "right"),
             "Weighted A* 1.25": (8, 6, "left"), "Weighted A* 1.5": (8, 6, "left"), "A* + network": (10, 4, "left"),
             "A* + network, capped": (-12, 26, "right"), "A* + 8 landmarks": (8, 8, "left"),
             "A* perfect": (10, 4, "left")}
    fig, ax = plt.subplots(figsize=(11, 5.6), dpi=130, facecolor=BG)
    ax.set_facecolor(PANEL)
    for name, (colour, marker) in style.items():
        s = summary[name]
        x, y = s["cells"], 100 * s["extra_mean"]
        ax.scatter([x], [y], s=150 if marker == "*" else 70, color=colour, marker=marker, zorder=3,
                   edgecolor=BG, linewidth=1)
        label = name.replace("A* + 8 landmarks", "A* + landmarks").replace("A* perfect", "perfect estimate")
        ax.annotate(f"{label}\n{x:,.0f} cells, {y:.1f}% longer" if y > 0.005 else f"{label}\n{x:,.0f} cells, quickest",
                    (x, y), xytext=nudge[name][:2], textcoords="offset points", fontsize=9.5, color=colour,
                    ha=nudge[name][2])
    ax.set_xscale("log")
    ax.set_xlim(30, 9000)
    ax.set_xticks([50, 100, 300, 1000, 3000])
    ax.set_xticklabels(["50", "100", "300", "1,000", "3,000"])
    ax.minorticks_off()
    ax.set_ylim(-0.4, 4.0)
    ax.set_xlabel("cells searched per trip (mean, log scale)")
    ax.set_ylabel("routes longer than the quickest (mean %)")
    for side in ax.spines.values():
        side.set_color("#2a3038")
    ax.grid(True, color="#1f252c", linewidth=0.8)
    ax.set_title("A* on 45 maps no model has seen: fewer cells is better, lower is better", loc="left",
                 fontsize=13.5, color=INK, fontweight="semibold", pad=12)
    fig.tight_layout()
    out = ROOT / "docs" / "learn-benchmark.png"
    fig.savefig(out, facecolor=BG)
    print(out, f"landmarks set-up {sum(setup) / len(setup):.0f} ms per map")


def speeds() -> None:
    """Stage 3's learning curves, read from the card `python main.py speeds learn` writes: how much slower than the
    quickest Bruce's routes are on average and at worst, and how many are within 1% of it, after each number of
    trips he has driven, for stage 3 (planning with what he believes, the small network), stage 3b (exploring, the
    bigger network) and stage 3c (exploring, the size he chooses from his own log)."""
    import json
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.lines import Line2D
    from robotnav.learn.speeds import ASSET

    for f in font_manager.findSystemFonts():
        if "Inter" in Path(f).name:
            font_manager.fontManager.addfont(f)
    plt.rcParams.update({"font.family": ["Inter", "DejaVu Sans"], "text.color": INK, "axes.labelcolor": DIM,
                         "xtick.color": DIM, "ytick.color": DIM})
    card = json.loads(ASSET.with_suffix(".card.json").read_text())
    summary = {s["model"]: s for s in card["summary"] if s["theme"] == "all"}
    careers = {c["name"]: c for c in card["careers"]}
    first, last = card["careers"][0], card["careers"][-1]
    trips = [p["trips"] for p in last["curve"]]
    labels = ["0\n(built-in)"] + [str(n) for n in trips]
    x = np.arange(len(labels))
    series = [(first, "Stage 3: drives his missions, small network", DIM, (0, (4, 3)), "white"),
              (careers["exploring, bigger network"], "Stage 3b: explores, bigger network", AMBER, (0, (1, 2)), AMBER),
              (last, "Stage 3c: explores, chooses his network's size", BLUE, "-", BLUE)]
    labelled = (first, last)                                     # values written at the ends of these two

    def values(career, key):
        return [100 * summary["built-in"][key]] + [100 * summary[f"{career['name']}, {n} trips"][key] for n in trips]

    count = summary["built-in"]["trips"]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.3), dpi=130, facecolor=BG)
    fig.subplots_adjust(left=0.055, right=0.985, top=0.71, bottom=0.16, wspace=0.24)
    panels = ((axes[0], "regret", "Slower than the quickest route", f"mean of {count} trips (log scale)"),
              (axes[1], "worst", "The worst of those trips", "how much slower (log scale)"),
              (axes[2], "within_1", "Trips within 1% of the quickest", "share of the trips"))
    for ax, key, title, sub in panels:
        ax.set_facecolor(PANEL)
        for side in ax.spines.values():
            side.set_color("#2a3038")
        ax.grid(True, axis="y", color="#1f252c", linewidth=0.8)
        for career, _, color, dash, face in series:
            v = values(career, key)
            ax.plot(x, v, color=color, lw=2.2 if color == BLUE else 1.8, ls=dash, zorder=3 if color == BLUE else 2)
            ax.scatter(x, v, s=60 if color == BLUE else 44, color=color if face != "white" else PANEL, edgecolor=color,
                       linewidth=1.6, zorder=4 if color == BLUE else 3)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel("trips Bruce has driven and learned from")
        ax.set_title(title, loc="left", fontsize=13, color=INK, fontweight="semibold", pad=22)
        ax.text(0, 1.03, sub, transform=ax.transAxes, fontsize=10, color=DIM, va="bottom")
        if key == "within_1":
            ax.set_ylim(0, 108)
            ax.set_yticks([0, 25, 50, 75, 100])
            ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
            ax.axhline(100, color=DIM, lw=1, ls=(0, (1, 3)), zorder=1)
            ax.text(x[0] - 0.2, 101.5, "knowing the true speeds", fontsize=9.5, color=DIM, va="bottom")
        else:
            ax.set_yscale("log")
            ax.minorticks_off()
        ax.set_xlim(-0.35, len(x) - 0.25)
        for career, _, color, _, _ in series:                    # the last values, labelled where they end
            if career not in labelled:
                continue
            v = values(career, key)[-1]
            text = f"{v:.0f}%" if key == "within_1" else f"{v:.2f}%" if v < 1 else f"{v:.1f}%"
            if key == "within_1":                                # nearly equal: one above the line, one below
                up = color == BLUE
                ax.annotate(text, (x[-1], v), xytext=(-9, 7 if up else -9), textcoords="offset points", fontsize=10,
                            color=INK, ha="right", va="bottom" if up else "top")
            else:
                ax.annotate(text, (x[-1], v), xytext=(10, 0), textcoords="offset points", fontsize=10, color=INK,
                            ha="left", va="center")
    axes[0].set_ylim(0.02, 12)
    axes[0].set_yticks([0.03, 0.1, 0.3, 1, 3, 10])
    axes[0].set_yticklabels(["0.03%", "0.1%", "0.3%", "1%", "3%", "10%"])
    axes[1].set_ylim(0.3, 120)
    axes[1].set_yticks([0.3, 1, 3, 10, 30, 100])
    axes[1].set_yticklabels(["0.3%", "1%", "3%", "10%", "30%", "100%"])
    fig.text(0.055, 0.945, "Bruce learns his terrain speeds from his own driving", fontsize=15, color=INK,
             fontweight="semibold")
    fig.text(0.055, 0.9, f"routes planned with what he believes, driven in the simulator's hidden true world, on "
             f"{count} trips across maps he never drove", fontsize=11, color=DIM)
    handles = [Line2D([], [], color=color, lw=2, ls=dash, marker="o", markersize=7, markeredgewidth=1.6,
                      markerfacecolor=color if face != "white" else PANEL, markeredgecolor=color)
               for _, _, color, dash, face in series]
    fig.legend(handles, [name for _, name, _, _, _ in series], loc="upper right", bbox_to_anchor=(0.985, 0.99),
               frameon=False, fontsize=10.5, labelcolor=INK, handlelength=2.6, labelspacing=0.35)
    out = ROOT / "docs" / "learn-speeds.png"
    fig.savefig(out, facecolor=BG)
    print(out)


def vision() -> None:
    """Stage 4: one test frame (the picture, what each pixel really is, what the network says it is), how often each
    model gets the kind of ground right by distance, and the drives' extra time, read from the card (`python main.py
    vision eval`) and the camera frames (`python main.py vision build`)."""
    import json
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.patches import Patch
    from robotnav.learn import camera as cam
    from robotnav.learn import vision as V

    for f in font_manager.findSystemFonts():
        if "Inter" in Path(f).name:
            font_manager.fontManager.addfont(f)
    plt.rcParams.update({"font.family": ["Inter", "DejaVu Sans"], "text.color": INK, "axes.labelcolor": DIM,
                         "xtick.color": DIM, "ytick.color": DIM})
    card = json.loads(V.ASSET.with_suffix(".card.json").read_text())
    net = V.models_from()["net"]
    m = V.load_manifest()
    # the frame: of the test frames showing three kinds of ground or more, the one on which the network's share of
    # the ground right is the median: a typical frame, not a good one
    shots = []
    for e in (e for e in m["maps"] if e["split"] == "test"):
        a = V.load(V.DEFAULT_DIR, e)
        for i, lab in enumerate(a["labels"]):
            if sum(np.mean(lab == k) > 0.02 for k in range(cam.GROUND)) >= 3:
                pred = net.log_probs(a["images"][i], a["dist"][i], e["theme"]).argmax(-1)
                ground = lab < cam.GROUND
                shots.append((float(np.mean(pred[ground] == lab[ground])), e["id"], i, e, a, pred))
    shots.sort(key=lambda x: x[:3])
    right, _, i, e, a, pred = shots[len(shots) // 2]
    image, label = a["images"][i], a["labels"][i]
    colours = cam.label_colors()

    fig = plt.figure(figsize=(14.5, 9.0), dpi=130, facecolor=BG)
    grid = fig.add_gridspec(2, 6, height_ratios=[1.0, 1.25], left=0.05, right=0.985, top=0.86, bottom=0.08,
                            hspace=0.62, wspace=0.55)
    panels = [(fig.add_subplot(grid[0, 0:2]), image, "What the camera sees",
              f"a test frame of median difficulty ({e['id']}, never trained on)"),
             (fig.add_subplot(grid[0, 2:4]), colours[label], "What each pixel really is", "the renderer's exact labels"),
             (fig.add_subplot(grid[0, 4:6]), colours[pred], "What the network says",
              f"the right kind on {100 * right:.0f}% of this frame's ground")]
    for ax, img, title, sub in panels:
        ax.imshow(img, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ax.spines.values():
            side.set_color("#2a3038")
        ax.set_title(title, loc="left", fontsize=13, color=INK, fontweight="semibold", pad=20)
        ax.text(0, 1.03, sub, transform=ax.transAxes, fontsize=10, color=DIM, va="bottom")
    shown = sorted(set(np.unique(label).tolist()) | set(np.unique(pred).tolist()))
    fig.legend([Patch(facecolor=colours[k], edgecolor="none") for k in shown], [cam.CLASSES[k] for k in shown],
               loc="upper center", bbox_to_anchor=(0.52, 0.535), ncol=len(shown), frameon=False, fontsize=10,
               labelcolor=INK, handlelength=1.2, columnspacing=1.6)

    ax = fig.add_subplot(grid[1, 0:3])
    ax.set_facecolor(PANEL)
    for side in ax.spines.values():
        side.set_color("#2a3038")
    ax.grid(True, axis="y", color="#1f252c", linewidth=0.8)
    test = card["test"]["pixels"]
    bands = [b["band"] for b in test["net"]["by_distance"]]
    names = [f"{a_:.0f}-{b_:.0f} m" if b_ is not None else f"over {a_:.0f} m" for a_, b_ in bands]
    x = np.arange(len(bands))
    for key, name, colour in (("colour", "colour only", DIM), ("pixel", "classic ML", AMBER), ("net", "network", BLUE)):
        v = [100 * b["accuracy"] for b in test[key]["by_distance"]]
        ax.plot(x, v, color=colour, lw=2.2 if key == "net" else 1.8, zorder=3)
        ax.scatter(x, v, s=50, color=colour, edgecolor=PANEL, linewidth=1.5, zorder=4)
        ax.annotate(f"{name} ({100 * test[key]['ground_accuracy']:.1f}%)", (x[-1], v[-1]), xytext=(9, 0),
                    textcoords="offset points", fontsize=10, color=INK, va="center")
    cut = V.CHOSEN["net"].get("max_range")
    if cut:                                                      # how far Bruce trusts the two learned models
        at = next(j for j, (_, b_) in enumerate(bands) if b_ == cut) + 0.5
        ax.axvline(at, color=DIM, lw=1, ls=(0, (1, 3)), zorder=1)
        ax.text(at - 0.08, 33, f"the learned models'\npixels count up\nto {cut:.0f} m", fontsize=9.5, color=DIM,
                ha="right", va="bottom")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_xlim(-0.3, len(x) + 0.75)
    ax.set_ylim(30, 100)
    ax.set_yticks([40, 60, 80, 100])
    ax.set_yticklabels(["40%", "60%", "80%", "100%"])
    ax.set_xlabel("how far away the ground is")
    frames = sum(card["data"]["frames"]["test"].values())
    ax.set_title("The right kind of ground, by distance", loc="left", fontsize=13, color=INK, fontweight="semibold",
                 pad=20)
    ax.text(0, 1.03, f"ground pixels of the {frames} test frames (all of them, in brackets)", transform=ax.transAxes,
            fontsize=10, color=DIM, va="bottom")

    ax = fig.add_subplot(grid[1, 3:6])
    ax.set_facecolor(PANEL)
    for side in ax.spines.values():
        side.set_color("#2a3038")
    ax.grid(True, axis="x", color="#1f252c", linewidth=0.8)
    drives = {s["variant"]: s for s in card["test"]["drives"] if s["theme"] == "all"}
    order = [("blind", "no camera", "#4a535d"), ("colour", "camera + colour only", DIM),
             ("pixel", "camera + classic ML", AMBER), ("net", "camera + network", BLUE),
             ("truth", "camera, perfect labels", "#cfd6dc"), ("map", "knows the ground", "#cfd6dc")]
    y = np.arange(len(order))[::-1] * 1.5
    for yy, (key, name, colour) in zip(y, order):
        v = 100 * drives[key]["mean"]
        hatch = "////" if key == "map" else None
        ax.barh(yy, v, height=0.62, color=colour if key != "map" else PANEL, edgecolor=colour, linewidth=1.4,
                hatch=hatch, zorder=3)
        ax.text(v + 0.15, yy, f"{v:.1f}%", va="center", fontsize=10, color=INK)
        ax.text(0.05, yy + 0.42, name, va="bottom", fontsize=10, color=INK)
    ax.set_yticks([])
    ax.set_ylim(y[-1] - 0.6, y[0] + 1.05)
    top = max(100 * s["mean"] for s in drives.values())
    ax.set_xlim(0, top * 1.18)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.set_xlabel("slower than the quickest route the whole map allows (mean)")
    trips = drives["net"]["trips"]
    ax.set_title("Driving with what he sees", loc="left", fontsize=13, color=INK, fontweight="semibold", pad=20)
    ax.text(0, 1.03, f"{trips} trips across the 45 test maps, re-planning with D* Lite", transform=ax.transAxes,
            fontsize=10, color=DIM, va="bottom")

    fig.text(0.05, 0.955, "Bruce learns to see the ground", fontsize=15, color=INK, fontweight="semibold")
    fig.text(0.05, 0.92, "he knows the heights from elevation data, not what the ground is: soft sand or firm, grass or "
             "tall grass; his camera has to tell him", fontsize=11, color=DIM)
    out = ROOT / "docs" / "learn-vision.png"
    fig.savefig(out, facecolor=BG)
    print(out, e["id"], i, f"network right on {100 * right:.1f}% of the frame's ground")


if __name__ == "__main__":
    {"bench": bench, "speeds": speeds, "vision": vision}.get(sys.argv[1] if sys.argv[1:] else "", main)()
