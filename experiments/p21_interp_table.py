"""P21: interpretation table + numbers for the uterus case study."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, "..", "results", "p20"); MAN = os.path.join(HERE, "..", "manuscript")
a = json.load(open(os.path.join(R, "p20_OpenML-1138.json"))); b = json.load(open(os.path.join(R, "p20b_OpenML-1138.json")))
coh = a["coherence"]; nk = sum(c["n_sig"] > 0 for c in coh["kmeans"]); nr = sum(c["n_sig"] > 0 for c in coh["random"])
tm = {x["term"]: x["p_adj"] for x in b["tissue_merit"]["Human_Gene_Atlas"]}; tl = {x["term"]: x["p_adj"] for x in b["tissue_lasso"]["Human_Gene_Atlas"]}
am = {x["term"]: x["p_adj"] for x in b["tissue_merit"]["ARCHS4_Tissues"]}; al = {x["term"]: x["p_adj"] for x in b["tissue_lasso"]["ARCHS4_Tissues"]}
N = json.load(open(os.path.join(MAN, "numbers.json")))
N["interp"] = {"coherent_kmeans": nk, "coherent_random": nr, "n_modules": len(coh["kmeans"]),
               "uterus_merit": tm.get("Uterus"), "uterus_lasso": tl.get("Uterus"), "smooth_merit": tm.get("SmoothMuscle"),
               "endometrium_merit": am.get("ENDOMETRIUM (BULK)"), "endometrium_lasso": al.get("ENDOMETRIUM (BULK)"),
               "jac_merit": b["jaccard_merit_top100"], "jac_lasso": b["jaccard_lasso"], "n_lasso_genes": len(a["lasso_genes"]),
               "top_go": [(x["name"], x["p"]) for x in (a["enr_top"] or [])[:3]]}
json.dump(N, open(os.path.join(MAN, "numbers.json"), "w"), indent=1)
rows = [
 ("1", "HOXD10, ZMYND11, FKBP1B, ITGB5, XIST", "Müllerian HOX code; female-specific XIST"),
 ("2", "SNCAIP, PRKD3, PRRX1, SSPN, RBPJ", "mesenchymal and smooth-muscle program"),
 ("3", "PPP3CB, EPYC, COL12A1, CPXM1, CSGALNACT2", "extracellular matrix, collagen fibrils"),
 ("4", "NEBL, CD44, ARF6, EPB41L5", "cytoskeleton and adhesion (partner module)"),
]
lines = [r"\begin{tabular}{clp{0.42\columnwidth}}", r"\toprule", r"Module & Top genes by weight & Interpretation \\", r"\midrule"]
for m, g, i in rows: lines.append(rf"{m} & {g} & {i} \\")
lines += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(MAN, "tables", "tab_interp.tex"), "w").write("\n".join(lines) + "\n")
print(json.dumps(N["interp"], indent=1, default=str))
