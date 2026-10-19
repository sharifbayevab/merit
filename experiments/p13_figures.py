"""P13: manuscript figures (PDF, single IEEE column width = 3.5 in).
fig_density.pdf  density index D vs accuracy gap to logistic regression (rules vs MR-Vote), all datasets
fig_smalln.pdf   sample-size curve (m objects per class)
fig_ablation.pdf 2x2 ablation: partition source (correlation / random) x vote (single / ensemble), paired to reference
Palette: validated categorical slots (blue, orange, aqua, yellow, magenta); each series also has its own marker and a
direct label (contrast WARN for aqua/yellow/magenta on the light surface).
"""
import glob, json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, "..", "results")
OUT = os.path.join(HERE, "..", "manuscript", "figs"); os.makedirs(OUT, exist_ok=True)
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "font.family": "serif", "pdf.fonttype": 42, "lines.linewidth": 1.4})

# ---------------------------------------------------------------- density
dp = pd.read_csv(os.path.join(R, "density_points.csv"), index_col=0)
fig, axs = plt.subplots(1, 2, figsize=(3.5, 1.75), sharey=True)
for ax, col, color, mk, title in ((axs[0], "gap_rules", ORANGE, "o", "Feature-level rules"), (axs[1], "gap_mr", BLUE, "s", "MERIT")):
    dev = dp.index.str.startswith("dev:")
    ax.axhline(0, color=INK2, lw=0.8)
    ax.scatter(dp.D[~dev], dp[col][~dev], s=14, color=color, marker=mk, edgecolor="white", linewidth=0.5, label="held-out", zorder=3)
    ax.scatter(dp.D[dev], dp[col][dev], s=14, facecolor="white", edgecolor=color, marker=mk, linewidth=0.9, label="development", zorder=3)
    r, p = spearmanr(dp.D, dp[col])
    ax.set_title(title, color=INK, loc="left")
    ax.text(0.97, 0.05, rf"$\rho={r:.2f}$" + ("\n$p<0.001$" if p < 0.001 else f"\n$p={p:.2f}$"), transform=ax.transAxes,
            ha="right", va="bottom", fontsize=6.8, color=INK)
    ax.set_xlabel("Density index $D$")
axs[0].set_ylabel("Bal. acc. minus\nlogistic regression")
axs[0].legend(loc="upper right", fontsize=6, frameon=False, handletextpad=0.2)
fig.tight_layout(pad=0.4); fig.savefig(os.path.join(OUT, "fig_density.pdf")); plt.close(fig)

# ---------------------------------------------------------------- sample-size curve
sn = pd.read_csv(os.path.join(R, "smalln_points.csv"))
mu = sn.groupby("m")[["HYB-ENS6NF", "TICL2bal-CGf", "TPFNbal-CGf", "LRbal", "TR2m"]].mean()
SER = [("HYB-ENS6NF", "MERIT", BLUE, "s"), ("TICL2bal-CGf", "TabICL v2", ORANGE, "^"), ("TPFNbal-CGf", "TabPFN v2", AQUA, "v"),
       ("LRbal", "Log. reg.", YELLOW, "o"), ("TR2m", "Feature-level rules", MAGENTA, "D")]
fig, ax = plt.subplots(figsize=(3.5, 2.1))
for col, lab, color, mk in SER:
    ax.plot(mu.index, mu[col], color=color, marker=mk, markersize=4.5, markeredgecolor="white", markeredgewidth=0.5, label=lab)
ax.set_xscale("log", base=2); ax.set_xticks(mu.index); ax.set_xticklabels([str(int(m)) for m in mu.index])
ax.set_xlabel("Training objects per class $m$"); ax.set_ylabel("Mean balanced accuracy")
ax.legend(loc="lower right", fontsize=6.3, frameon=False, ncol=1)
fig.tight_layout(pad=0.4); fig.savefig(os.path.join(OUT, "fig_smalln.pdf")); plt.close(fig)

# ---------------------------------------------------------------- ablation 2x2 (paired to the single k-means partition)
ab = pd.DataFrame([json.loads(l) for f in glob.glob(os.path.join(R, "spark", "p9", "p9_*.jsonl")) for l in open(f)])
ab = ab[ab.bacc.notna()]; at = ab.pivot_table(index="ds", columns="method", values="bacc", aggfunc="mean")
cells = [("RAND20", "random, single"), ("RENS24", "random, vote (24)"), ("ENS6", "correlation, vote (6)")]
rng = np.random.default_rng(0)
fig, ax = plt.subplots(figsize=(3.5, 1.35))
for i, (m, lab) in enumerate(cells):
    d = (at[m] - at["K20"]).values
    b = rng.choice(d, (10000, len(d))).mean(1); lo, hi = np.percentile(b, [2.5, 97.5])
    color = BLUE if m == "ENS6" else ORANGE
    ax.plot([lo, hi], [i, i], color=color, lw=1.6, solid_capstyle="round")
    ax.plot(d.mean(), i, marker="o", color=color, markersize=5, markeredgecolor="white", markeredgewidth=0.6)
ax.axvline(0, color=INK2, lw=0.8)
ax.set_yticks(range(len(cells))); ax.set_yticklabels([c[1] for c in cells]); ax.invert_yaxis()
ax.set_xlabel("Difference to one correlation partition")
ax.grid(axis="y", visible=False)
fig.tight_layout(pad=0.4); fig.savefig(os.path.join(OUT, "fig_ablation.pdf")); plt.close(fig)
print("figures:", sorted(os.listdir(OUT)))
