"""Analyse P1: mean balanced accuracy per (reduction, head), bottleneck decomposition.

Reads ../results/p1*_*.jsonl (Mac CPU heads and Spark GPU heads are merged; duplicates
of the same (ds, split, red, head) keep the first record).
"""
import glob, json, os, sys
import numpy as np
import pandas as pd

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
rows, seen = [], set()
for f in sorted(glob.glob(os.path.join(RES, "p1*_*.jsonl"))):
    for l in open(f):
        r = json.loads(l)
        key = (r["ds"], r["split"], r["red"], r["head"])
        if key in seen or "bacc" not in r:
            continue
        seen.add(key)
        rows.append(r)
df = pd.DataFrame(rows)
df["cell"] = df["red"] + ":" + df["head"]
print("records", len(df), "datasets", df.ds.nunique())

# 1) main table: mean bacc per dataset x cell
tab = df.pivot_table(index="cell", columns="ds", values="bacc", aggfunc="mean")
n_splits = df.pivot_table(index="cell", columns="ds", values="bacc", aggfunc="count")
tab["mean"] = tab.mean(1)
tab["rank"] = tab.drop(columns="mean").rank(ascending=False).mean(1)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
print("\n=== mean balanced accuracy (15 splits) ===")
print(tab.sort_values("mean", ascending=False).round(3).to_string())
print("\nmin #splits per cell:", int(n_splits.min().min()))

# 2) bottleneck decomposition
def m(cell):
    return tab.loc[cell].drop(["mean", "rank"]) if cell in tab.index else None

print("\n=== H1 reduction gap (same head, different reduction) ===")
for head in ["LR", "TR2", "TR2m", "TPFN", "EBM", "XGB"]:
    line = []
    for red in ["ALL", "F500", "V500", "CGf", "CGm", "CGp"]:
        c = f"{red}:{head}"
        if c in tab.index:
            line.append(f"{red}={tab.loc[c, 'mean']:.3f}")
    print(f"{head:5s}", "  ".join(line))

print("\n=== H2 head gap on CGf (Mirzaev representatives) ===")
for head in ["LR", "RF", "XGB", "EBM", "TR1", "TR2", "TR3", "TR1m", "TR2m", "TR3m", "TR2h", "TR2hm", "TPFN", "TICL"]:
    c = f"CGf:{head}"
    if c in tab.index:
        print(f"{head:6s} mean={tab.loc[c, 'mean']:.3f} rank={tab.loc[c, 'rank']:.2f}")

print("\n=== H3 rule-selection optimism (train quality of chosen rules - test quality) ===")
tr = df[df["head"].str.startswith("TR")]
if len(tr):
    g = tr.groupby(["red", "head"]).agg(train_q=("rule_train_q", "mean"), test_q=("rule_test_q", "mean"),
                                         ens_train=("train_bacc", "mean"), ens_test=("bacc", "mean"),
                                         n_rules=("n_rules", "mean"), n_feat=("n_feat_used", "mean"))
    g["rule_optimism"] = g.train_q - g.test_q
    g["ens_optimism"] = g.ens_train - g.ens_test
    print(g.round(3).to_string())

# 3) paired comparisons against TR2 on CGf
print("\n=== paired split-level differences vs CGf:TR2 (mean, 95% bootstrap CI, wins/datasets) ===")
base = df[df.cell == "CGf:TR2"].set_index(["ds", "split"]).bacc
rng = np.random.default_rng(0)
for c in ["CGf:LR", "F500:LR", "ALL:LR", "CGf:TR2m", "CGf:TR3", "CGf:TR3m", "CGf:EBM", "F500:TPFN", "CGf:TPFN", "ALL:TICL", "CGf:TICL", "ALL:XGB"]:
    s = df[df.cell == c].set_index(["ds", "split"]).bacc
    j = base.index.intersection(s.index)
    if not len(j):
        continue
    d = (s[j] - base[j])
    per_ds = d.groupby(level=0).mean()
    boots = [per_ds.sample(len(per_ds), replace=True, random_state=int(rng.integers(1e9))).mean() for _ in range(2000)]
    print(f"{c:10s} {per_ds.mean():+.3f} [{np.percentile(boots, 2.5):+.3f}, {np.percentile(boots, 97.5):+.3f}]  "
          f"wins {int((per_ds > 0).sum())}/{len(per_ds)}")

if "--save" in sys.argv:
    tab.round(4).to_csv(os.path.join(RES, "p1_table.csv"))
