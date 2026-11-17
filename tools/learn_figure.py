"""Draw docs/learn-labels.png, the README's picture of the learning data: one test map, one goal, one trip.

    python tools/learn_figure.py            # needs the data set: python main.py dataset build

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
BG, PANEL, INK, DIM, BLUE = "#0b0e12", "#11151a", "#e7ecef", "#8f9aa4", "#5cc2ff"


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


if __name__ == "__main__":
    main()
