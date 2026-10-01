"""Render slide 4 drift factorization (Huyen 2022 ordering) as PNG."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["mathtext.fontset"] = "cm"
EQ = r"$P(X, Y) = P(Y \mid X)\,P(X) = P(X \mid Y)\,P(Y)$"

for name, transparent in [("eq_drift.png", False), ("eq_drift_transparent.png", True)]:
    fig = plt.figure(figsize=(8, 1))
    fig.text(0.5, 0.5, EQ, fontsize=32, ha="center", va="center")
    fig.savefig(f"slides/{name}", dpi=300, bbox_inches="tight", pad_inches=0.15,
                transparent=transparent, facecolor="white")
    plt.close(fig)
