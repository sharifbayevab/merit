"""P26: report for P25 (classical p >> n baselines) against MERIT on the 26 held-out tables and the 8 dev sets.
Prints mean balanced accuracy, mean rank among {MERIT, TabICL, TabPFN, LR, P25 methods}, MERIT minus method [95% CI],
wins, Wilcoxon p (and Holm over the four new comparisons). Writes results/p26_classic.json."""
import glob, json, os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, "..", "results")
L = lambda p: [json.loads(l) for f in glob.glob(os.path.join(R, p)) for l in open(f)]
df = pd.concat([pd.DataFrame(L("spark/p4_*.jsonl") + L("mac_extra/p4_*.jsonl") + L("spark/p25/*.jsonl")),
                pd.DataFrame(L("spark/p9/p9_*.jsonl")).query("method=='ENS6NF'").assign(method="MERIT")])
df = df[df.bacc.notna() & (df.ds != "OpenML-1458")].drop_duplicates(["ds", "split", "method"])
t = df.pivot_table(index="ds", columns="method", values="bacc", aggfunc="mean")
c = df.pivot_table(index="ds", columns="method", values="bacc", aggfunc="count")
NEW = ["SVMbal", "ENbal", "SLDA", "PAM"]
REF = ["MERIT", "TICL2bal-CGf", "TPFNbal-CGf", "LRbal"]
out = {}
for part in ("test", "dev"):
    sel = ~t.index.str.startswith("dev:") if part == "test" else t.index.str.startswith("dev:")
    tabs = t.index[sel & (c[REF] >= 10).all(1).values]
    new = [m for m in NEW if m in c and (c.loc[tabs, m] >= 10).all()]
    missing = {m: int((c.loc[tabs, m].fillna(0) < 10).sum()) for m in NEW if m not in new and m in c}
    cols = REF + new
    rk = t.loc[tabs, cols].rank(axis=1, ascending=False).mean()
    print(f"== {part}: {len(tabs)} tables; incomplete: {missing}")
    rows, ps = {}, []
    for m in cols:
        r = {"mean": float(t.loc[tabs, m].mean()), "rank": float(rk[m])}
        if m != "MERIT":
            d = (t.loc[tabs, "MERIT"] - t.loc[tabs, m]).values
            b = np.random.default_rng(0).choice(d, (10000, len(d))).mean(1)
            r.update(diff=float(d.mean()), ci=[float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))],
                     wins=int((d > 0).sum()), p=float(wilcoxon(d).pvalue))
        rows[m] = r
    pn = [rows[m]["p"] for m in new]
    if pn:
        o = np.argsort(pn); adj = np.empty(len(pn)); run = 0
        for i, j in enumerate(o):
            run = max(run, (len(pn) - i) * pn[j]); adj[j] = min(1, run)
        for m, a in zip(new, adj):
            rows[m]["p_holm4"] = float(a)
    for m, r in sorted(rows.items(), key=lambda kv: -kv[1]["mean"]):
        extra = "" if m == "MERIT" else f" diff={r['diff']:+.3f} [{r['ci'][0]:+.3f},{r['ci'][1]:+.3f}] wins={r['wins']} p={r['p']:.3f}" + \
            (f" p_holm4={r['p_holm4']:.3f}" if "p_holm4" in r else "")
        print(f"{m:14s} mean={r['mean']:.3f} rank={r['rank']:.2f}{extra}")
    out[part] = rows
json.dump(out, open(os.path.join(R, "p26_classic.json"), "w"), indent=1)
