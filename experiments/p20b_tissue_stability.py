"""P20b: (a) tissue identity of MERIT genes via Enrichr tissue atlases (ARCHS4_Tissues, Human_Gene_Atlas);
(b) Jaccard stability of MERIT top-100 genes vs lasso-selected genes across the 10 training folds."""
import json, os, sys, warnings, itertools
os.environ.setdefault("OMP_NUM_THREADS", "6")
import numpy as np, requests
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegressionCV
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "..", "results", "p20")
os.environ.setdefault("TB_DIR", os.path.join(HERE, "..", "data_tb"))
import p4_bench; p4_bench.TB = os.environ["TB_DIR"]
from p4_bench import get_data, get_splits
from p20_interpret import gene_importance, mygene_symbols

DS = sys.argv[1] if len(sys.argv) > 1 else "OpenML-1138"


def enrichr(genes, libs=("ARCHS4_Tissues", "Human_Gene_Atlas")):
    r = requests.post("https://maayanlab.cloud/Enrichr/addList", files={"list": (None, "\n".join(genes)), "description": (None, DS)}, timeout=60)
    uid = r.json()["userListId"]; out = {}
    for lib in libs:
        e = requests.get("https://maayanlab.cloud/Enrichr/enrich", params={"userListId": uid, "backgroundType": lib}, timeout=60).json()[lib]
        out[lib] = [{"term": x[1], "p": x[2], "p_adj": x[6], "genes": x[5]} for x in e[:10]]
    return out


def main():
    X, y, g = get_data(DS)
    names = np.load(os.path.join(os.environ["TB_DIR"], DS + ".npz"))["feature_names"]
    med = np.nanmedian(X, 0); X = np.where(np.isnan(X), med, X); keep = X.std(0) > 1e-8; X, names = X[:, keep].astype(float), names[keep]
    target = int(np.argmin(np.bincount(y))); yb = (y == target).astype(int)
    q = [str(n).split(".")[0] for n in names]
    sym_map = mygene_symbols(q, "reporter" if not q[0].startswith("ENSG") else "ensembl.gene"); sym = np.array([sym_map.get(i, "") for i in q])
    merit_sets, lasso_sets = [], []
    for si, (tr, te) in enumerate(get_splits(yb, g)):
        Ztr = (X[tr] - X[tr].mean(0)) / (X[tr].std(0) + 1e-9)
        imp, _ = gene_importance(Ztr, yb[tr]); merit_sets.append(set(np.argsort(-imp)[:100].tolist()))
        las = LogisticRegressionCV(Cs=[0.01, 0.03, 0.1, 0.3, 1], cv=3, penalty="l1", solver="liblinear", class_weight="balanced", max_iter=3000).fit(Ztr, yb[tr])
        lasso_sets.append(set(np.where(np.abs(las.coef_.ravel()) > 1e-8)[0].tolist()))
        print("split", si, len(lasso_sets[-1]), flush=True)
    jac = lambda A, B: len(A & B) / max(len(A | B), 1)
    jm = [jac(a, b) for a, b in itertools.combinations(merit_sets, 2)]; jl = [jac(a, b) for a, b in itertools.combinations(lasso_sets, 2)]
    # size-matched control for MERIT: lasso top-k by |coef| with k = 100 is not available when lasso selects < 100; report sizes
    full = json.load(open(os.path.join(OUT, f"p20_{DS}.json")))
    tissue_merit = enrichr(full["top100"]); tissue_lasso = enrichr(full["lasso_genes"])
    res = {"jaccard_merit_top100": float(np.mean(jm)), "jaccard_lasso": float(np.mean(jl)),
           "lasso_sizes": [len(s) for s in lasso_sets], "tissue_merit": tissue_merit, "tissue_lasso": tissue_lasso}
    json.dump(res, open(os.path.join(OUT, f"p20b_{DS}.json"), "w"), indent=1)
    print("Jaccard MERIT top-100:", round(np.mean(jm), 3), " lasso:", round(np.mean(jl), 3), "lasso sizes", res["lasso_sizes"])
    for lib in tissue_merit:
        print("MERIT", lib, [(x["term"], f'{x["p_adj"]:.1e}') for x in tissue_merit[lib][:5]])
        print("LASSO", lib, [(x["term"], f'{x["p_adj"]:.1e}') for x in tissue_lasso[lib][:5]])


if __name__ == "__main__":
    main()
