"""Rough diagram mockups for the slides, to redraw in draw.io. Run: uv run python slides/mockups.py"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from evalsq.plots import INK  # noqa: E402,F401  (import applies the riso palette, paper and font rcParams)

OUT = Path(__file__).parent
# Plain draw.io flowchart look: (fill, edge) pairs. Default box, highlighted box, outcome box.
BLUE, ORANGE, GREEN, RED = ("#DAE8FC", "#6C8EBF"), ("#FFF2CC", "#D6B656"), ("#D5E8D4", "#82B366"), ("#FFF2CC", "#D6B656")
GREY = "#666666"


def canvas(w=10, h=3.2):
    fig, ax = plt.subplots(figsize=(w, h), facecolor="white")
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


def box(ax, x, y, text, w=1.8, h=0.8, color=BLUE, fs=11, dashed=False):
    fc, ec = color
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02",
                                fc=fc, ec=ec, lw=1.2, ls="--" if dashed else "-"))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=INK)


def arrow(ax, a, b, color=INK, rad=0.0, ls="-", text=None, toff=(0, 0.2)):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=14, color=color, lw=1.4,
                                 ls=ls, connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text((a[0] + b[0]) / 2 + toff[0], (a[1] + b[1]) / 2 + toff[1], text, ha="center", fontsize=9, color=color)


def save(fig, name):
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def pipeline():
    """Slide 3: ML lifecycle. Data -> Models -> Eval -> Deploy -> $ inside one iteration box, $ feeds the next
    iteration's data, the money loop arrives late, and a ? on the eval."""
    fig, ax = canvas(10, 4.6)
    y, w, d = 2.3, 1.5, 0.08
    xs = [1.2, 3.3, 5.4, 7.5]
    dx = 9.3
    ax.add_patch(FancyBboxPatch((0.25, 1.05), 9.6, 2.05, boxstyle="round,pad=0.05",
                                fc="none", ec=GREY, lw=1.0, ls=(0, (4, 3))))
    ax.text(0.35, 1.2, "one iteration", fontsize=9, color=GREY, va="center")
    for x, t in zip(xs, ["Data", "Models", "Eval", "Deploy"]):
        box(ax, x, y, t, w=w, color=RED if t == "Eval" else BLUE)
    for a, b in zip(xs + [dx], xs[1:] + [dx]):
        if a != b:
            arrow(ax, (a + w / 2 + d, y), (b - (0.4 if b == dx else w / 2) - d, y))
    box(ax, dx, y, "$", w=0.8, color=GREEN, fs=14)
    ax.text(xs[2], 2.75, "?", ha="center", fontsize=18, color=INK, weight="bold")
    arrow(ax, (dx, 2.72), (xs[0], 2.72), rad=0.22)
    ax.text(5.1, 3.75, "next iteration: new data, retrain, re-eval", ha="center", fontsize=10, color=INK)
    arrow(ax, (dx, 1.85), (xs[2], 1.85), color=GREY, rad=-0.2, ls="--")
    ax.text(7.2, 1.18, "real outcome: arrives L days late", ha="center", fontsize=10, color=GREY)
    save(fig, "mock_pipeline.png")


def timeline():
    """Slide 5: one month of the worked example. Decide at t on data up to t-L, trade, see P&L L days later."""
    fig, ax = canvas(10, 2.6)
    ax.plot([0.5, 9.5], [1.3, 1.3], color=INK, lw=1.2)
    ax.add_patch(FancyBboxPatch((0.6, 1.15), 2.8, 0.3, boxstyle="square,pad=0", fc=BLUE[0], ec=BLUE[1]))
    ax.text(2.0, 1.6, "labels we can see", ha="center", fontsize=9, color=INK)
    ax.add_patch(FancyBboxPatch((3.4, 1.15), 1.4, 0.3, boxstyle="square,pad=0", fc="#F5F5F5", ec=GREY))
    ax.text(4.1, 1.6, "L days: not in yet", ha="center", fontsize=9, color=GREY)
    ax.add_patch(FancyBboxPatch((4.8, 1.15), 3.0, 0.3, boxstyle="square,pad=0", fc=ORANGE[0], ec=ORANGE[1]))
    ax.text(6.3, 1.6, "month t: deployed model trades", ha="center", fontsize=9, color=INK)
    for x, t in [(3.4, "t-L\n"), (4.8, "t\nretrain + pick"), (7.8, "t+1\nnext pick"), (9.2, "t+1+L\nmonth t P&L seen")]:
        ax.plot([x, x], [1.1, 1.5], color=INK, lw=1)
        ax.text(x, 0.55, t, ha="center", fontsize=9)
    save(fig, "mock_timeline.png")


def bench_of_bench():
    """Slide 6: the question moves up one level. Metric -> pick -> $, and we score the metric by the $."""
    fig, ax = canvas(10, 3.4)
    for y, t in zip([2.9, 1.9, 0.9], ["accuracy", "AUC", "bear-day acc"]):
        box(ax, 1.3, y, t, w=2.0, h=0.6, fs=10)
        arrow(ax, (2.35, y), (3.9, 1.9))
    box(ax, 4.9, 1.9, "pick model\neach month", w=1.9, h=0.9, color=ORANGE, fs=10)
    arrow(ax, (5.9, 1.9), (7.0, 1.9))
    box(ax, 7.9, 1.9, "$ after\nN months", w=1.6, h=0.9, color=GREEN, fs=10)
    arrow(ax, (7.9, 1.4), (1.3, 0.55), color=GREY, rad=-0.25, ls="--")
    save(fig, "mock_bench_of_bench.png")


def heuristics_map():
    """Slide 7: the three checks, ordered familiar -> unusual, each tied to what it asks."""
    fig, ax = canvas(10, 2.6)
    items = [("H1  Validity", "does the score track $?", BLUE),
             ("H2  Temporal holdout", "does the best eval stay the best?", BLUE),
             ("H3  Complementarity", "what do the evals say about each other?", BLUE)]
    for i, (h, q, c) in enumerate(items):
        x = 1.7 + i * 3.3
        box(ax, x, 1.6, h, w=2.9, h=0.7, color=c, fs=11)
        ax.text(x, 0.85, q, ha="center", fontsize=9, color=INK)
    arrow(ax, (0.3, 0.3), (9.7, 0.3), color=GREY)
    ax.text(0.3, 0.05, "familiar", fontsize=8, color=GREY)
    ax.text(9.7, 0.05, "less common", fontsize=8, color=GREY, ha="right")
    save(fig, "mock_heuristics_map.png")


if __name__ == "__main__":
    pipeline()
    timeline()
    bench_of_bench()
    heuristics_map()
