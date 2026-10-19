"""P19: simulation table (Section Theoretical Analysis) from P18 results."""
import glob, json, os
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, "..", "results"); MAN = os.path.join(HERE, "..", "manuscript")
df = pd.DataFrame([json.loads(l) for f in glob.glob(os.path.join(R, "p18", "*.jsonl")) for l in open(f)])
order = ["sparse", "1", "2", "5", "10", "20"]
t = df.groupby("setting")[["bayes", "oracle30", "rules", "lr", "merit", "merit_nf"]].mean().reindex(order)
lab = {"sparse": "5 single features", "1": "1 module", "2": "2 modules", "5": "5 modules", "10": "10 modules", "20": "20 modules"}
lines = [r"\begin{tabular}{lcccccc}", r"\toprule",
         r"Signal in & Bayes & Best 30 features & Feature-level rules & Log.\ reg. & MERIT, no floor & MERIT \\", r"\midrule"]
for k, r in t.iterrows():
    vals = [r.rules, r.lr, r.merit, r.merit_nf]; best = max(vals)
    cells = [(rf"\textbf{{{v:.3f}}}" if v == best else f"{v:.3f}") for v in vals]
    lines.append(rf"{lab[k]} & {r.bayes:.3f} & {r.oracle30:.3f} & " + " & ".join(cells) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(MAN, "tables", "tab_sim.tex"), "w").write("\n".join(lines) + "\n")
N = json.load(open(os.path.join(MAN, "numbers.json"))); N["sim"] = {k: {c: float(v) for c, v in r.items()} for k, r in t.iterrows()}
json.dump(N, open(os.path.join(MAN, "numbers.json"), "w"), indent=1); print(t.round(3))
