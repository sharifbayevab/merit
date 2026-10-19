"""P22: fig_main.pdf (MERIT minus each method, 26 held-out tables, 95% bootstrap CI, Holm-significant filled)
and fig_speed.pdf (single-query latency and per-object batch time, log scale, widest table)."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); MAN = os.path.join(HERE, "..", "manuscript")
BLUE, ORANGE, INK, INK2, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e6e5e0"
plt.rcParams.update({"font.size": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "font.family": "serif", "pdf.fonttype": 42})
N = json.load(open(os.path.join(MAN, "numbers.json")))
NAMES = {"TICL2bal-CGf": "TabICL v2", "TPFNbal-CGf": "TabPFN v2", "HYB-K20": "MERIT-1", "HYB-L1K20": "MERIT-S", "WIDE8kbal": "TabPFN-Wide",
         "LRbal": "Logistic regression", "RSEbal": "Random-subspace LR", "XGBbal": "XGBoost", "TR2m": "Feature-level rules",
         "KTSP": "k-TSP", "RFbal": "Random forest"}
rows = [(NAMES[m], v) for m, v in N["test"].items() if m in NAMES]
rows.sort(key=lambda r: r[1]["diff"])
fig, ax = plt.subplots(figsize=(3.5, 2.45))
for i, (name, v) in enumerate(rows):
    lo, hi = v["ci"]; sig = v["p_holm"] < 0.05; col = BLUE if sig else INK2
    ax.plot([lo, hi], [i, i], color=col, lw=1.5, solid_capstyle="round")
    ax.plot(v["diff"], i, "o", ms=4.6, color=col if sig else "white", markeredgecolor=col, markeredgewidth=1.1)
ax.axvline(0, color=INK2, lw=0.8)
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows]); ax.grid(axis="y", visible=False)
ax.set_xlabel("MERIT minus method, balanced accuracy")
ax.text(0.99, 0.03, "filled: Holm-adjusted $p<0.05$", transform=ax.transAxes, ha="right", fontsize=6.3, color=INK2)
fig.tight_layout(pad=0.4); fig.savefig(os.path.join(MAN, "figs", "fig_main.pdf")); plt.close(fig)

S = N["speed"]["TCGA-TCGA-BRCA_Gene-Expression-Quantification"]
order = ["MERIT-1 (CPU)", "Logistic regression (CPU)", "MERIT (CPU)", "TabPFN v2 (GPU)", "TabICL v2 (GPU)", "TabICL v2 (CPU)",
         "TabPFN-Wide (GPU)", "TabPFN v2 (CPU)"]
order = [m for m in order if m in S][::-1]
fig, ax = plt.subplots(figsize=(3.5, 2.2))
for i, m in enumerate(order):
    col = BLUE if m.startswith("MERIT") else ORANGE if m.startswith("Tab") else INK2
    ax.plot([S[m]["batch_ms"], S[m]["single_ms"]], [i, i], color=GRID, lw=2, zorder=1)
    ax.plot(S[m]["single_ms"], i, "o", ms=5, color=col, zorder=3)
    ax.plot(S[m]["batch_ms"], i, "D", ms=4.2, color="white", markeredgecolor=col, markeredgewidth=1.1, zorder=3)
ax.set_xscale("log"); ax.set_yticks(range(len(order))); ax.set_yticklabels(order); ax.grid(axis="y", visible=False)
ax.set_xlabel("Time per object (ms, log scale)")
ax.plot([], [], "o", color=INK2, label="single query"); ax.plot([], [], "D", color="white", markeredgecolor=INK2, label="batch, per object")
ax.legend(loc="upper right", fontsize=6.3, frameon=False)
fig.tight_layout(pad=0.4); fig.savefig(os.path.join(MAN, "figs", "fig_speed.pdf")); plt.close(fig)
print("ok")
