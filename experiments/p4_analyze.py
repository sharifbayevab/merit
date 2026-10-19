"""Analyse P4: held-out TabBench-Bio (+ dev sets on the same protocol).
Tables: mean bacc per dataset x method, average rank, paired dataset-level comparisons
(mean diff, bootstrap 95% CI, Wilcoxon p, wins), and P3: density index D vs gap(rules - LR)."""
import glob, json, os, sys
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, spearmanr

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", sys.argv[1] if len(sys.argv) > 1 else "spark")
EXTRA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "mac_extra")
rows = [json.loads(l) for f in glob.glob(os.path.join(RES, "p4_*.jsonl")) + glob.glob(os.path.join(EXTRA, "p4_*.jsonl")) for l in open(f)]
df = pd.DataFrame(rows)
# OpenML-1458 is ARCENE, identical to the dev set dev:arcene -> excluded from the held-out part
df = df[df.ds != "OpenML-1458"]
dens = df[df.method == "DENSITY"].groupby("ds").agg(D=("D", "mean"), n=("n", "first"), d=("d", "first"), C=("C", "first"))
res = df[(df.method != "DENSITY") & df.bacc.notna()]
tab = res.pivot_table(index="ds", columns="method", values="bacc", aggfunc="mean")
cnt = res.pivot_table(index="ds", columns="method", values="bacc", aggfunc="count")
errs = df[df.get("error").notna()] if "error" in df else pd.DataFrame()
if len(errs):
    print("errors:\n", errs.groupby(["method"]).error.first().str[:120].to_string())
full = cnt.min(1) >= 10
tab = tab[full.reindex(tab.index).fillna(False)]
tab = tab.dropna(axis=1, thresh=int(0.8 * len(tab)))
tab["split"] = np.where(tab.index.str.startswith("dev:"), "dev", "test")
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 100)
print(tab.join(dens[["D", "n", "d", "C"]]).round(3).to_string())

for part in ["test", "dev"]:
    t = tab[tab.split == part].drop(columns="split").dropna()
    if not len(t):
        continue
    print(f"\n=== {part}: {len(t)} datasets with all methods ===")
    print("mean:", t.mean().round(3).sort_values(ascending=False).to_dict())
    print("avg rank:", t.rank(axis=1, ascending=False).mean().round(2).sort_values().to_dict())
    rng = np.random.default_rng(0)
    for a, b in [("HYB-K20", "LR"), ("HYB-K20", "LRbal"), ("LRbal", "LR"), ("HYB-L1K20", "LRbal"), ("HYB-K20", "WIDE8k"), ("HYB-K20", "TR2m"), ("HYB-L1K20", "LR"),
                 ("HYB-L1K20", "TR2m"), ("TR2m", "LR"), ("HYB-K20", "XGB"), ("HYB-K20", "XGBbal"), ("HYB-K20", "RFbal"), ("HYB-K20", "RSEbal"), ("HYB-K20", "WIDE8kbal"), ("HYB-K20", "TPFNbal-CGf"), ("HYB-K20", "TICL2bal-CGf"), ("HYB-K20", "TICL2-CGf"), ("HYB-K20", "TPFN-CGf"), ("KTSP", "TR2m")]:
        if a not in t or b not in t:
            continue
        d = (t[a] - t[b]).values
        boots = [rng.choice(d, len(d)).mean() for _ in range(5000)]
        p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
        print(f"{a:10s} - {b:9s} {d.mean():+.3f} [{np.percentile(boots, 2.5):+.3f}, {np.percentile(boots, 97.5):+.3f}] "
              f"p={p:.3f} wins {int((d > 0).sum())}/{len(d)}")

# P3: density index predicts where rules are safe
j = tab.join(dens).dropna(subset=["TR2m", "LR", "D"])
j["gap_rules"] = j.TR2m - j.LR
print("\n=== P3 density index (train-fold) vs gap(TR2m - LR) ===")
for part in ["all", "test"]:
    jj = j if part == "all" else j[j.split == "test"]
    r, p = spearmanr(jj.D, jj.gap_rules)
    print(f"{part}: n={len(jj)} spearman={r:+.2f} p={p:.3f}")
    if "HYB-K20" in jj:
        r2, p2 = spearmanr(jj.D, jj["HYB-K20"] - jj.LR)
        print(f"   D vs gap(HYB-K20 - LR): spearman={r2:+.2f} p={p2:.3f}")
print(j[["D", "gap_rules"]].sort_values("D").round(3).to_string())
