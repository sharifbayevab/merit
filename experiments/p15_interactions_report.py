"""P15: summarise P14 (module interactions), add numbers to numbers.json, write a table and two figures.
fig_synergy.pdf    dataset-level mean rule synergy vs gap MERIT-1 minus additive module model (and vs LR)
fig_case.pdf       one heavy rule on the case dataset: test-fold module scores, class colours, rule boundary
tab_interactions.tex  per-group summary
"""
import glob, json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, "..", "results")
MAN = os.path.join(HERE, "..", "manuscript")
BLUE, ORANGE, INK, INK2, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e6e5e0"
plt.rcParams.update({"font.size": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.titlesize": 8,
                     "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "font.family": "serif", "pdf.fonttype": 42})

rows = [json.loads(l) for f in glob.glob(os.path.join(R, "spark", "p14", "p14_*.jsonl")) for l in open(f)]
df = pd.DataFrame(rows)
g = df.groupby("ds").agg(merit1=("ba_merit1", "mean"), modlr=("ba_modlr", "mean"), syn=("synergy", "mean"),
                         pair=("ba_pair", "mean"), single=("ba_single", "mean"), pos=("frac_pos_syn", "mean"), n=("split", "count"))
g = g[g.n >= 10]
p4 = [json.loads(l) for f in glob.glob(os.path.join(R, "spark", "p4_*.jsonl")) + glob.glob(os.path.join(R, "mac_extra", "p4_*.jsonl")) for l in open(f)]
p4 = pd.DataFrame(p4); p4 = p4[p4.bacc.notna() & (p4.ds != "OpenML-1458")]
lr = p4[p4.method == "LRbal"].groupby("ds").bacc.mean()
g = g.join(lr.rename("lr"), how="inner")
g["gap_add"] = g.merit1 - g.modlr; g["gap_lr"] = g.merit1 - g.lr

N = json.load(open(os.path.join(MAN, "numbers.json")))
r1, p1 = spearmanr(g.syn, g.gap_add); r2, p2 = spearmanr(g.syn, g.gap_lr)
d = g.gap_add
N["interactions"] = {"n": int(len(g)), "mean_syn": float(g.syn.mean()), "median_syn": float(g.syn.median()),
                     "frac_pos_rules": float(g.pos.mean()), "ba_pair": float(g.pair.mean()), "ba_single": float(g.single.mean()),
                     "merit1": float(g.merit1.mean()), "modlr": float(g.modlr.mean()),
                     "gap_add": float(d.mean()), "gap_add_wins": int((d > 0).sum()), "p_gap_add": float(wilcoxon(d).pvalue),
                     "rho_syn_gapadd": float(r1), "p_syn_gapadd": float(p1), "rho_syn_gaplr": float(r2), "p_syn_gaplr": float(p2)}
hi = g[g.syn >= g.syn.median()]; lo = g[g.syn < g.syn.median()]
N["interactions"]["hi_gap_add"] = float(hi.gap_add.mean()); N["interactions"]["lo_gap_add"] = float(lo.gap_add.mean())
N["interactions"]["hi_gap_lr"] = float(hi.gap_lr.mean()); N["interactions"]["lo_gap_lr"] = float(lo.gap_lr.mean())
g.to_csv(os.path.join(R, "interactions_points.csv"))

# table: top datasets by synergy
top = g.sort_values("syn", ascending=False)
lines = [r"\begin{tabular}{lccccc}", r"\toprule", r"Dataset & Synergy & MERIT-1 & Additive & Log.\ reg. & MERIT-1 $-$ additive \\", r"\midrule"]
def nm(s): return s.replace("dev:", "").replace("TCGA-TCGA-", "TCGA-").replace("_Gene-Expression-Quantification", "").replace("_", r"\_").replace("GEO-", "").replace("-Methylation", " methyl.")
for ds, r in list(top.iterrows())[:6] + [("...", None)] + list(top.iterrows())[-3:]:
    if r is None: lines.append(r"\multicolumn{6}{c}{$\cdots$} \\"); continue
    lines.append(rf"{nm(ds)} & {r.syn:+.3f} & {r.merit1:.3f} & {r.modlr:.3f} & {r.lr:.3f} & {r.gap_add:+.3f} \\")
lines += [r"\midrule", rf"Mean over {len(g)} datasets & {g.syn.mean():+.3f} & {g.merit1.mean():.3f} & {g.modlr.mean():.3f} & {g.lr.mean():.3f} & {g.gap_add.mean():+.3f} \\",
          r"\bottomrule", r"\end{tabular}"]
open(os.path.join(MAN, "tables", "tab_interactions.tex"), "w").write("\n".join(lines) + "\n")

# figure 1: synergy vs gap to additive model
fig, ax = plt.subplots(figsize=(3.5, 2.0))
dev = g.index.str.startswith("dev:")
ax.axhline(0, color=INK2, lw=0.8); ax.axvline(0, color=INK2, lw=0.8)
ax.scatter(g.syn[~dev], g.gap_add[~dev], s=16, color=BLUE, edgecolor="white", linewidth=0.5, label="held-out", zorder=3)
ax.scatter(g.syn[dev], g.gap_add[dev], s=16, facecolor="white", edgecolor=BLUE, linewidth=0.9, label="development", zorder=3)
ax.set_xlabel("Mean rule synergy (pair minus best single module)"); ax.set_ylabel("MERIT-1 minus\nadditive module model")
ax.text(0.97, 0.05, rf"$\rho={r1:.2f}$, $p={p1:.3f}$" if p1 >= 0.001 else rf"$\rho={r1:.2f}$, $p<0.001$", transform=ax.transAxes, ha="right", fontsize=6.8)
ax.legend(loc="upper left", fontsize=6, frameon=False)
fig.tight_layout(pad=0.4); fig.savefig(os.path.join(MAN, "figs", "fig_synergy.pdf")); plt.close(fig)

# figure 2: case study = dataset with the largest synergy among those where MERIT-1 beats LR
cand = g[(g.gap_lr > 0)].sort_values("syn", ascending=False)
case_ds = cand.index[0]
case = next(r["case"] for r in rows if r["ds"] == case_ds and r["split"] == 0 and "case" in r)
Bm, yb = np.array(case["B"]), np.array(case["yte"]); Am, ya = np.array(case["A"]), np.array(case["ytr"])
w, t = case["w"], case["t"]
fig, ax = plt.subplots(figsize=(3.5, 2.4))
for Xs, ys, alpha, lab in ((Am, ya, 0.35, "train"), (Bm, yb, 1.0, "test")):
    ax.scatter(Xs[ys == 1, 0], Xs[ys == 1, 1], s=10 if lab == "train" else 16, color=ORANGE, alpha=alpha, marker="o", edgecolor="white", linewidth=0.3,
               label=f"target class ({lab})")
    ax.scatter(Xs[ys == 0, 0], Xs[ys == 0, 1], s=10 if lab == "train" else 16, color=BLUE, alpha=alpha, marker="s", edgecolor="white", linewidth=0.3,
               label=f"other ({lab})")
xs = np.linspace(min(Am[:, 0].min(), Bm[:, 0].min()), max(Am[:, 0].max(), Bm[:, 0].max()), 50)
if abs(w[1]) > 1e-9:
    ax.plot(xs, (t - w[0] * xs) / w[1], color=INK, lw=1.2, ls="--", label="rule boundary")
ys_all = np.concatenate([Am[:, 1], Bm[:, 1]]); ax.set_ylim(ys_all.min() - 0.3, ys_all.max() + 0.3)
ax.set_xlabel(f"module score $s_{{{case['S'][0]}}}$"); ax.set_ylabel(f"module score $s_{{{case['S'][1]}}}$")
ax.legend(fontsize=5.6, frameon=False, loc="best", ncol=1)
fig.tight_layout(pad=0.4); fig.savefig(os.path.join(MAN, "figs", "fig_case.pdf")); plt.close(fig)
N["interactions"]["case"] = {"ds": case_ds, "modules": case["S"], "ba_pair": case["ba_pair"], "ba_single": case["ba_single"],
                             "w": case["w"], "t": case["t"], "gap_lr": float(g.loc[case_ds, "gap_lr"]), "syn": float(g.loc[case_ds, "syn"])}
json.dump(N, open(os.path.join(MAN, "numbers.json"), "w"), indent=1)
print(json.dumps(N["interactions"], indent=1))
print(g.sort_values("syn", ascending=False)[["syn", "merit1", "modlr", "lr", "gap_add", "gap_lr"]].round(3).to_string())
