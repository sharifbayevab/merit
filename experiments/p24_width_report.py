"""P24: report for P23 (TabPFN v2 / TabICL v2 input-width sensitivity) on the 26 held-out tables of the main comparison.
Reads results/spark/p23/*.jsonl and the P4/P9 results used by p12_tables.py. Prints, per input, the mean balanced
accuracy, MERIT minus model [95% bootstrap CI], wins and Wilcoxon p; also a per-table oracle over widths (optimistic)."""
import glob, json, os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, "..", "results")


def load(pats):
    rows = []
    for p in pats:
        for f in glob.glob(os.path.join(R, p)):
            rows += [json.loads(l) for l in open(f)]
    return pd.DataFrame(rows)


def boot_ci(d, B=10000, seed=0):
    b = np.random.default_rng(seed).choice(d, (B, len(d))).mean(1)
    return np.percentile(b, 2.5), np.percentile(b, 97.5)


base = load(["spark/p4_*.jsonl", "mac_extra/p4_*.jsonl"])
nf = load(["spark/p9/p9_*.jsonl"]); nf = nf[nf.method == "ENS6NF"].assign(method="MERIT")
w = load(["spark/p23/*.jsonl"])
df = pd.concat([base[["ds", "split", "method", "bacc"]], nf[["ds", "split", "method", "bacc"]], w[["ds", "split", "method", "bacc"]]])
df = df[df.bacc.notna() & ~df.ds.str.startswith("dev:") & (df.ds != "OpenML-1458")].drop_duplicates(["ds", "split", "method"])
cnt = df.pivot_table(index="ds", columns="method", values="bacc", aggfunc="count")
tab = df.pivot_table(index="ds", columns="method", values="bacc", aggfunc="mean")
MAIN = ["MERIT", "TICL2bal-CGf", "TPFNbal-CGf", "LRbal"]
tables = tab.index[(cnt[MAIN] >= 10).all(1)]
print("tables in main comparison:", len(tables))
wm = sorted(w.method.unique())
out = {}
for m in ["TPFNbal-CGf", "TICL2bal-CGf"] + wm:
    ok = [t for t in tables if m in cnt and cnt.loc[t, m] >= 10] if m in cnt else []
    if not ok:
        continue
    d = (tab.loc[ok, "MERIT"] - tab.loc[ok, m]).values
    lo, hi = boot_ci(d)
    p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
    out[m] = dict(n=len(ok), mean=float(tab.loc[ok, m].mean()), merit=float(tab.loc[ok, "MERIT"].mean()), diff=float(d.mean()),
                  ci=[float(lo), float(hi)], wins=int((d > 0).sum()), p=float(p))
    print(f"{m:16s} n={len(ok):2d} mean={out[m]['mean']:.3f} MERIT={out[m]['merit']:.3f} diff={d.mean():+.3f} [{lo:+.3f},{hi:+.3f}] wins={out[m]['wins']} p={p:.3f}")
for model, b in (("PFN", "TPFNbal-CGf"), ("ICL", "TICL2bal-CGf")):
    cols = [b] + [m for m in wm if m.startswith(model)]
    ok = [t for t in tables if all(c in cnt and cnt.loc[t, c] >= 10 for c in cols)]
    if len(ok) == len(tables):
        best = tab.loc[ok, cols].max(1); d = (tab.loc[ok, "MERIT"] - best).values; lo, hi = boot_ci(d)
        out[f"{model}-oracle"] = dict(n=len(ok), mean=float(best.mean()), diff=float(d.mean()), ci=[float(lo), float(hi)])
        print(f"{model}-oracle (best width per table, optimistic): mean={best.mean():.3f} diff={d.mean():+.3f} [{lo:+.3f},{hi:+.3f}]")
json.dump(out, open(os.path.join(R, "p24_width.json"), "w"), indent=1)
