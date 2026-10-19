"""P12: generate manuscript tables (LaTeX) and a numbers file from the result logs.

Sources: results/spark/p4_*.jsonl (Spark benchmark), results/mac_extra/p4_*.jsonl (LRbal, XGBbal),
results/p7/p7_*.jsonl (HYB-ENS), results/spark/p9 (ablations), results/spark/p11 (sample-size curve),
results/p6_*.json (complexity), results/spark/p8 (NeuroRules).
Outputs: ../manuscript/tables/*.tex and ../manuscript/numbers.json (every number quoted in the text).
"""
import glob, json, os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "..", "results")
OUT = os.path.join(HERE, "..", "manuscript", "tables"); os.makedirs(OUT, exist_ok=True)
NUM = {}


def load(pats):
    rows = []
    for p in pats:
        for f in glob.glob(os.path.join(R, p)):
            rows += [json.loads(l) for l in open(f)]
    return pd.DataFrame(rows)


def holm(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0
    for i, j in enumerate(o):
        run = max(run, (m - i) * ps[j]); adj[j] = min(1.0, run)
    return adj


def boot_ci(d, B=10000, seed=0):
    rng = np.random.default_rng(seed); d = np.asarray(d)
    b = rng.choice(d, (B, len(d))).mean(1)
    return np.percentile(b, 2.5), np.percentile(b, 97.5)


# ---------------------------------------------------------------- benchmark table
df = load(["spark/p4_*.jsonl", "mac_extra/p4_*.jsonl", "p7/p7_*.jsonl"])
_nf = load(["spark/p9/p9_*.jsonl"]); _nf = _nf[_nf.method == "ENS6NF"].assign(method="MERIT")  # final MERIT (noise floor)
df = pd.concat([df, _nf[["ds", "split", "method", "bacc"]]], ignore_index=True)
df = df[df.ds != "OpenML-1458"]                         # ARCENE duplicate of dev:arcene
dens = df[df.method == "DENSITY"].groupby("ds").agg(D=("D", "mean"), n=("n", "first"), d=("d", "first"), C=("C", "first"))
res = df[df.bacc.notna()]
cnt = res.pivot_table(index="ds", columns="method", values="bacc", aggfunc="count")
tab = res.pivot_table(index="ds", columns="method", values="bacc", aggfunc="mean")
NAMES = {"MERIT": "MERIT (ours)", "HYB-K20": "MERIT-1 (ours)", "HYB-L1K20": "MERIT-S (ours)",
         "TICL2bal-CGf": "TabICL v2", "TPFNbal-CGf": "TabPFN v2", "WIDE8kbal": "TabPFN-Wide",
         "LRbal": "Logistic regression", "XGBbal": "XGBoost", "RSEbal": "Random-subspace LR", "RFbal": "Random forest",
         "KTSP": "$k$-TSP", "TR2m": "Feature-level rules"}
M = list(NAMES)
for part in ("test", "dev"):
    sel = ~tab.index.str.startswith("dev:") if part == "test" else tab.index.str.startswith("dev:")
    t, c = tab[sel], cnt[sel]
    ok = (c[M] >= 10).all(1)
    t = t.loc[ok, M]
    NUM[f"{part}_n_datasets"] = int(len(t))
    ranks = t.rank(axis=1, ascending=False).mean()
    base = t["MERIT"]
    rows, ps = [], []
    for m in M:
        r = {"m": m, "mean": t[m].mean(), "rank": ranks[m]}
        if m != "MERIT":
            d = (base - t[m]).values
            r["diff"] = d.mean(); r["ci"] = boot_ci(d); r["wins"] = int((d > 0).sum()); r["ties"] = int((d == 0).sum())
            r["p"] = wilcoxon(d).pvalue if np.any(d != 0) else 1.0; ps.append(r["p"])
        rows.append(r)
    adj = holm(ps); k = 0
    for r in rows:
        if "p" in r: r["p_holm"] = adj[k]; k += 1
    rows.sort(key=lambda r: -r["mean"])
    NUM[part] = {r["m"]: {kk: (float(v) if not isinstance(v, tuple) else [float(v[0]), float(v[1])]) for kk, v in r.items() if kk != "m"} for r in rows}
    lines = [r"\begin{tabular}{lcccc}", r"\toprule",
             r"Method & Mean bal.\ acc. & Mean rank & MERIT $-$ method [95\% CI] & Wins / $p_{\mathrm{Holm}}$ \\", r"\midrule"]
    for r in rows:
        if r["m"] == "MERIT":
            lines.append(rf"\textbf{{{NAMES[r['m']]}}} & \textbf{{{r['mean']:.3f}}} & {r['rank']:.2f} & -- & -- \\")
        else:
            p = r["p_holm"]; ps_ = "$<$0.001" if p < 0.001 else f"{p:.3f}"
            lines.append(rf"{NAMES[r['m']]} & {r['mean']:.3f} & {r['rank']:.2f} & {r['diff']:+.3f} [{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}] & {r['wins']}/{len(t)} / {ps_} \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(OUT, f"tab_main_{part}.tex"), "w").write("\n".join(lines) + "\n")

# ---------------------------------------------------------------- datasets table
def short(ds):
    x = ds.replace("dev:", "")
    rep = {"FusionAI-NTv2-GeneFusion": ("FusionAI NT-v2", "DNA embedding"), "gut-cirrhosis": ("Gut cirrhosis", "metagenomic markers"),
           "TDC-AMES": ("TDC AMES", "fingerprint"), "TDC-BBB-Martins": ("TDC BBB", "fingerprint"), "TDC-CYP3A4-Veith": ("TDC CYP3A4", "fingerprint")}
    if x in rep: return rep[x]
    if x.startswith("TCGA-TCGA-"): return (x.split("-")[2].split("_")[0].join(["TCGA-", ""]), "RNA-seq")
    if x.startswith("GEO-GSE") and "Methylation" in x: return (x.split("-")[1], "DNA methylation")
    if x.startswith("GEO-"): return (x.split("-")[1], "microarray")
    if x.startswith("MGYS"): return (x, "metagenomic function")
    if x.startswith("OpenML-"): return (x, "microarray")
    return (x.replace("_", r"\_"), "mass spectrometry" if x == "arcene" else "microarray")
info = []
for ds in sorted(dens.index, key=lambda s: (not s.startswith("dev:"), s)):
    r = dens.loc[ds]; nm, typ = short(ds)
    info.append((nm, typ, "dev" if ds.startswith("dev:") else "test", int(r.n), int(r.d), int(r.C), r.D))
lines = [r"\begin{tabular}{llcrrrr}", r"\toprule", r"Dataset & Data type & Split & $n$ & $p$ & $C$ & $D$ \\", r"\midrule"]
for name, typ, sp, n, d, C, D in info:
    lines.append(rf"{name} & {typ} & {sp} & {n} & {d} & {C} & {D:+.3f} \\")
lines += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(OUT, "tab_datasets.tex"), "w").write("\n".join(lines) + "\n")
NUM["n_all_datasets"] = len(info)

# ---------------------------------------------------------------- density index
j = tab.join(dens).dropna(subset=["TR2m", "LRbal", "MERIT", "D"])
for part, sel in (("all", slice(None)), ("test", ~j.index.str.startswith("dev:"))):
    jj = j[sel]
    r1, p1 = spearmanr(jj.D, jj.TR2m - jj.LRbal); r2, p2 = spearmanr(jj.D, jj["MERIT"] - jj.LRbal)
    NUM[f"density_{part}"] = {"n": int(len(jj)), "rho_rules": float(r1), "p_rules": float(p1), "rho_mr": float(r2), "p_mr": float(p2)}
j[["D"]].assign(gap_rules=j.TR2m - j.LRbal, gap_mr=j["MERIT"] - j.LRbal).to_csv(os.path.join(R, "density_points.csv"))

# density gate (tau chosen on dev splits, evaluated on test): compact rules if D < tau else MR-Single
dsplit = df[df.method == "DENSITY"].groupby(["ds", "split"]).D.mean()
rs = res.groupby(["ds", "split", "method"]).bacc.mean().unstack()
jj = rs.join(dsplit, how="inner").dropna(subset=["TR2m", "HYB-K20", "LRbal", "D"])
dev = jj.index.get_level_values(0).str.startswith("dev:")
best = max(np.linspace(-0.05, 0.3, 36), key=lambda tau: (np.where(jj.D[dev] < tau, jj.TR2m[dev], jj["HYB-K20"][dev]).mean(), tau))
g = pd.Series(np.where(jj.D < best, jj.TR2m, jj["HYB-K20"]), index=jj.index)
per = pd.DataFrame({"GATE": g[~dev], "MR": jj["HYB-K20"][~dev], "LR": jj.LRbal[~dev], "R": jj.TR2m[~dev]}).groupby(level=0).mean()
NUM["density_gate"] = {"tau": float(best), "share_rules": float((jj.D[~dev] < best).mean()), "gate": float(per.GATE.mean()),
                       "mr_single": float(per.MR.mean()), "lr": float(per.LR.mean()), "rules": float(per.R.mean()),
                       "gate_minus_mr": float((per.GATE - per.MR).mean()), "p_gate_mr": float(wilcoxon(per.GATE - per.MR).pvalue),
                       "gate_minus_lr": float((per.GATE - per.LR).mean()), "p_gate_lr": float(wilcoxon(per.GATE - per.LR).pvalue)}

# ---------------------------------------------------------------- ablations (35 datasets)
ab = load(["spark/p9/p9_*.jsonl"]); ab = ab[ab.bacc.notna()]
at = ab.pivot_table(index="ds", columns="method", values="bacc", aggfunc="mean")
AB = [("K20", "Single k-means partition, $K{=}20$ (reference)"),
      ("K5", "$K{=}5$"), ("K10", "$K{=}10$"), ("K40", "$K{=}40$"), ("K80", "$K{=}80$"),
      ("RAND20", "Random modules instead of correlation modules"),
      ("ORD1", "One-dimensional rules"), ("ORD3", "Three-dimensional rules"),
      ("THROPT", "Accuracy-maximising thresholds"), ("C001", "Within-module $C{=}0.01$"), ("C1", "Within-module $C{=}1$"),
      ("ENS2", "Vote over 2 k-means partitions"), ("ENS4", "Vote over 4 k-means partitions"),
      ("ENS6", "Vote over 6 k-means partitions"), ("ENS12", "Vote over 12 k-means partitions"),
      ("RENS6", "Vote over 6 random partitions"), ("RENS24", "Vote over 24 random partitions")]
lines = [r"\begin{tabular}{lccc}", r"\toprule", r"Variant & Mean & $-$ reference & Wins / $p$ \\", r"\midrule"]
NUM["ablation"] = {"n": int(len(at))}
for m, lab in AB:
    if m == "K20":
        lines.append(rf"{lab} & {at[m].mean():.3f} & -- & -- \\"); continue
    d = at[m] - at["K20"]; p = wilcoxon(d).pvalue
    NUM["ablation"][m] = {"mean": float(at[m].mean()), "diff": float(d.mean()), "wins": int((d > 0).sum()), "p": float(p)}
    lines.append(rf"{lab} & {at[m].mean():.3f} & {d.mean():+.3f} & {int((d > 0).sum())}/{len(at)} / {p:.3f} \\")
    if m in ("K80", "C1", "ENS12"): lines.append(r"\midrule")
lines += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(OUT, "tab_ablation.tex"), "w").write("\n".join(lines) + "\n")
for a, b in [("RENS6", "ENS6"), ("RENS24", "ENS6"), ("RENS24", "RAND20"), ("ENS6", "RAND20")]:
    d = at[a] - at[b]; NUM["ablation"][f"{a}-{b}"] = {"diff": float(d.mean()), "wins": int((d > 0).sum()), "p": float(wilcoxon(d).pvalue)}

# ---------------------------------------------------------------- sample-size curve
sn = load(["spark/p11/*.jsonl"]); sn = sn[sn.bacc.notna()]
c = sn.groupby(["ds", "m", "method"]).bacc.agg(["mean", "count"]).reset_index(); c = c[c["count"] >= 10]
w = c.pivot_table(index=["ds", "m"], columns="method", values="mean")
SM = ["HYB-ENS6NF", "TICL2bal-CGf", "TPFNbal-CGf", "LRbal", "TR2m"]
w = w[SM].dropna()
lines = [r"\begin{tabular}{rccccc}", r"\toprule",
         r"$m$ & MERIT & TabICL v2 & TabPFN v2 & Log.\ reg. & Feature-level rules \\", r"\midrule"]
NUM["small_n"] = {}
for m in sorted(w.index.get_level_values("m").unique()):
    x = w.xs(m, level="m"); mu = x.mean()
    NUM["small_n"][int(m)] = {"n": int(len(x)), **{k: float(v) for k, v in mu.items()},
                              "mr_minus_lr": float((x["HYB-ENS6NF"] - x.LRbal).mean()), "p_mr_lr": float(wilcoxon(x["HYB-ENS6NF"] - x.LRbal).pvalue),
                              "wins_mr_lr": int(((x["HYB-ENS6NF"] - x.LRbal) > 0).sum()),
                              "ticl_minus_lr": float((x["TICL2bal-CGf"] - x.LRbal).mean()), "p_ticl_lr": float(wilcoxon(x["TICL2bal-CGf"] - x.LRbal).pvalue)}
    best_ = mu.idxmax()
    cells = [(rf"\textbf{{{mu[k]:.3f}}}" if k == best_ else f"{mu[k]:.3f}") for k in SM]
    lines.append(rf"{int(m)} & " + " & ".join(cells) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(OUT, "tab_smalln.tex"), "w").write("\n".join(lines) + "\n")
w.reset_index().to_csv(os.path.join(R, "smalln_points.csv"), index=False)

# ---------------------------------------------------------------- complexity (dev, P6)
cx = []
for f in sorted(glob.glob(os.path.join(R, "p6_*.json"))):
    for r in json.load(open(f)):
        for m in ("LR", "LASSO", "TR2m", "HYB-K20", "HYB-L1K20"):
            cx.append({"ds": r["ds"], "m": m, "bacc": r[m][0], "feat": r[m][1]["n_feat"],
                       "mods": r[m][1].get("n_modules", np.nan), "rules": r[m][1].get("n_rules", np.nan)})
cx = pd.DataFrame(cx); g = cx.groupby(["m", "ds"]).agg(bacc=("bacc", "mean"), feat=("feat", "median"), mods=("mods", "median"), rules=("rules", "median")).groupby(level=0)
cs = pd.DataFrame({"bacc": g.bacc.mean(), "feat": g.feat.median(), "mods": g.mods.median(), "rules": g.rules.median()})
CN = {"LASSO": "Lasso logistic regression", "TR2m": "Feature-level rules", "HYB-L1K20": "MERIT-S (ours)", "LR": "Logistic regression (all features)", "HYB-K20": "MERIT-1 (ours)"}
lines = [r"\begin{tabular}{lcccc}", r"\toprule", r"Model & Bal.\ acc. & Features used & Modules & Rules \\", r"\midrule"]
NUM["complexity"] = {}
for m in ("LASSO", "TR2m", "HYB-L1K20", "LR", "HYB-K20"):
    r = cs.loc[m]; NUM["complexity"][m] = {k: float(v) for k, v in r.items()}
    mods = "--" if np.isnan(r.mods) else f"{r.mods:.0f}"; rules = "--" if np.isnan(r.rules) else f"{r.rules:.0f}"
    lines.append(rf"{CN[m]} & {r.bacc:.3f} & {r.feat:.0f} & {mods} & {rules} \\")
lines += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(OUT, "tab_complexity.tex"), "w").write("\n".join(lines) + "\n")

# ---------------------------------------------------------------- NeuroRules on test, P7 negative results on dev
nr = load(["spark/p8/p8_*.jsonl"]); nr = nr[nr.bacc.notna()]
nr0 = nr[nr.get("prune", 0).fillna(0) == 0].groupby("ds").bacc.mean(); nrp = nr[nr.get("prune", 0).fillna(0) > 0]
pr = nrp.groupby("ds").agg(p=("bacc", "mean"), u=("bacc_unpruned", "mean"))
tt = tab[~tab.index.str.startswith("dev:")]
k = nr0.index.intersection(tt.index)
d = tt.loc[k, "HYB-K20"] - nr0[k]
NUM["neurorules"] = {"n": int(len(k)), "nr": float(nr0[k].mean()), "mr_single": float(tt.loc[k, "HYB-K20"].mean()),
                     "wins_mr": int((d > 0).sum()), "p": float(wilcoxon(d).pvalue),
                     "pruned": float(pr.p.mean()), "unpruned": float(pr.u.mean()), "n_pruned": int(len(pr))}
p7 = load(["p7/p7_dev_*.jsonl"]); p7 = p7[p7.bacc.notna()]
t7 = p7.pivot_table(index="ds", columns="method", values="bacc", aggfunc="mean")
NUM["p7_dev"] = {m: float(t7[m].mean()) for m in t7.columns}
NUM["p7_dev"]["HYB-K20"] = float(tab.loc[t7.index, "HYB-K20"].mean())

json.dump(NUM, open(os.path.join(HERE, "..", "manuscript", "numbers.json"), "w"), indent=1)
print(json.dumps({k: NUM[k] for k in ("test_n_datasets", "dev_n_datasets", "density_test", "density_gate", "neurorules")}, indent=1))
print("test table:", {m: round(v["mean"], 3) for m, v in NUM["test"].items()})
