"""P17: properties + speed table from P16 timings (largest dataset for the table; all datasets into numbers.json)."""
import json, os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, "..", "results"); MAN = os.path.join(HERE, "..", "manuscript")
t = pd.DataFrame([json.loads(l) for l in open(os.path.join(R, "p16_timing.jsonl"))])
t = t[t.get("single_ms").notna()] if "single_ms" in t else t
N = json.load(open(os.path.join(MAN, "numbers.json")))
N["speed"] = {ds: {r.model: {"fit_s": float(r.fit_s), "batch_ms": float(r.batch_ms), "single_ms": float(r.single_ms)}
                   for r in g.itertuples()} for ds, g in t.groupby("ds")}
big = "TCGA-TCGA-BRCA_Gene-Expression-Quantification"
g = t[t.ds == big].set_index("model")
PROPS = [  # model name in timing, display, features used, pre-training, hardware, readable
    ("MERIT (CPU)", "MERIT", "all (30{,}000)", "none", "CPU", "rules over modules"),
    ("MERIT-1 (CPU)", "MERIT-1", "all (30{,}000)", "none", "CPU", "rules over modules"),
    ("Logistic regression (CPU)", "Logistic regression", "all (30{,}000)", "none", "CPU", "weights"),
    ("TabPFN v2 (GPU)", "TabPFN v2", "200 representatives", "synthetic tables", "GPU", "no"),
    ("TabPFN v2 (CPU)", "TabPFN v2", "200 representatives", "synthetic tables", "CPU", "no"),
    ("TabICL v2 (GPU)", "TabICL v2", "200 representatives", "synthetic tables", "GPU", "no"),
    ("TabICL v2 (CPU)", "TabICL v2", "200 representatives", "synthetic tables", "CPU", "no"),
    ("TabPFN-Wide (GPU)", "TabPFN-Wide", "8000 filtered", "synthetic tables", "GPU", "no")]
lines = [r"\begin{tabular}{llllrrr}", r"\toprule",
         r"Model & Features used & Pre-training & Readable & Fit (s) & Latency (ms) & Batch (ms/obj.) \\", r"\midrule"]
for key, disp, feat, pre, hw, rd in PROPS:
    if key not in g.index: continue
    r = g.loc[key]
    lat = f"{r.single_ms:.1f}" if r.single_ms < 100 else f"{r.single_ms:.0f}"
    bt = f"{r.batch_ms:.2f}" if r.batch_ms < 1 else f"{r.batch_ms:.1f}"
    lines.append(rf"{disp} ({hw}) & {feat} & {pre} & {rd} & {r.fit_s:.1f} & {lat} & {bt} \\")
lines += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(MAN, "tables", "tab_properties.tex"), "w").write("\n".join(lines) + "\n")
json.dump(N, open(os.path.join(MAN, "numbers.json"), "w"), indent=1)
pd.set_option("display.width", 200)
print(t.pivot_table(index="model", columns="ds", values=["single_ms", "fit_s"]).round(2).to_string())
