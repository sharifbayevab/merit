"""P20: interpretation case study with gene names (plan: ../06_plan_interpretation.md).

Dataset: OpenML-1138 (uterus vs other tissues, HG-U133 probes). Steps:
 1. MERIT (6 partitions, noise-floor rule selection) fitted on the full capped table (for reading) and on the 10
    training folds of the P4 protocol (for stability).
 2. Gene importance = sum over partitions and selected rules of alpha * |w_col| / sd_col * |beta_gene|.
 3. Stability: frequency of each probe in the top-50 over the 10 training folds; stable = frequency >= 0.7.
 4. Probe -> gene symbol via mygene.info; enrichment via g:Profiler (GO:BP, KEGG, REAC, HPA; custom background).
 5. Module coherence: share of k-means modules vs size-matched random modules with >= 1 significant GO:BP term.
 6. Lasso comparison.
Outputs: ../results/p20/*.json, ../manuscript/tables/tab_interp.tex, numbers.json["interp"].
"""
import json, os, sys, time, warnings
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
import requests
warnings.filterwarnings("ignore")
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from methods import ThresholdRules
from p4_bench import get_data, get_splits, clean

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "..", "results", "p20"); os.makedirs(OUT, exist_ok=True)
MAN = os.path.join(HERE, "..", "manuscript")
DS = sys.argv[1] if len(sys.argv) > 1 else "OpenML-1138"
os.environ.setdefault("TB_DIR", os.path.join(HERE, "..", "data_tb"))
import p4_bench; p4_bench.TB = os.environ["TB_DIR"]
PARTS = [(10, 0), (20, 0), (40, 0), (10, 1), (20, 1), (40, 1)]


def fit_partition(Z, y, K, seed):
    lab = MiniBatchKMeans(K, random_state=seed, n_init=3, batch_size=4096).fit(Z.T / np.sqrt(len(Z))).labels_
    groups = [np.where(lab == k)[0] for k in range(K) if (lab == k).any()]
    def scores(Za, ya, Zb):
        return np.column_stack([LogisticRegression(C=0.1, max_iter=3000).fit(Za[:, g], ya).decision_function(Zb[:, g]) for g in groups])
    oof = np.zeros((len(y), len(groups)))
    for a, b in StratifiedKFold(3, shuffle=True, random_state=seed).split(Z, y):
        oof[b] = scores(Z[a], y[a], Z[b])
    betas = [LogisticRegression(C=0.1, max_iter=3000).fit(Z[:, g], y).coef_.ravel() for g in groups]
    sd = oof.std(0) + 1e-9
    r = ThresholdRules(k=2, thr="mid", seed=seed, qmin="auto").fit((oof - oof.mean(0)) / sd, y)
    return groups, betas, sd, r


def gene_importance(Z, y):
    imp = np.zeros(Z.shape[1]); rules_out = []
    for K, seed in PARTS:
        groups, betas, sd, r = fit_partition(Z, y, K, seed)
        _, chosen, alphas = r.models[0]
        for (q, S, w, t), a in zip(chosen, alphas):
            for col, wc in zip(S, w):
                imp[groups[col]] += a * abs(wc) / sd[col] * np.abs(betas[col])
            rules_out.append({"K": K, "seed": seed, "q": float(q), "alpha": float(a), "modules": [int(c) for c in S],
                              "w": [float(v) for v in w], "t": float(t),
                              "top_genes": [[int(i) for i in groups[c][np.argsort(-np.abs(betas[c]))[:8]]] for c in S],
                              "top_signs": [[float(np.sign(v)) for v in betas[c][np.argsort(-np.abs(betas[c]))[:8]]] for c in S]})
    return imp, rules_out


def mygene_symbols(ids, scope):
    out = {}
    for i in range(0, len(ids), 1000):
        chunk = ids[i:i + 1000]
        r = requests.post("https://mygene.info/v3/query", data={"q": ",".join(chunk), "scopes": scope, "fields": "symbol",
                                                                "species": "human", "size": 1}, timeout=120)
        for h in r.json():
            if "symbol" in h and h["query"] not in out: out[h["query"]] = h["symbol"]
    return out


def gprofiler(genes, background, sources=("GO:BP", "KEGG", "REAC", "HPA")):
    body = {"organism": "hsapiens", "query": sorted(set(genes)), "sources": list(sources), "user_threshold": 0.05,
            "significance_threshold_method": "g_SCS", "domain_scope": "custom", "background": sorted(set(background)), "no_evidences": True}
    for attempt in range(3):
        try:
            r = requests.post("https://biit.cs.ut.ee/gprofiler/api/gost/profile/", json=body, timeout=300)
            return [{"source": x["source"], "name": x["name"], "p": x["p_value"], "n_hit": x["intersection_size"], "term_size": x["term_size"]}
                    for x in r.json().get("result", [])]
        except Exception:
            time.sleep(5)
    return None


def main():
    X, y, g = get_data(DS)
    names = np.load(os.path.join(os.environ["TB_DIR"], DS + ".npz"))["feature_names"]
    med = np.nanmedian(X, 0); X = np.where(np.isnan(X), med, X); keep = X.std(0) > 1e-8; X, names = X[:, keep].astype(float), names[keep]
    target = int(np.argmin(np.bincount(y)))  # minority class = uterus / tumour-normal minority
    yb = (y == target).astype(int)
    Z = (X - X.mean(0)) / (X.std(0) + 1e-9)
    # 1-3: stability over the 10 training folds
    freq = np.zeros(X.shape[1])
    splits = get_splits(yb, g)
    for si, (tr, te) in enumerate(splits):
        Xtr = X[tr]; Ztr = (Xtr - Xtr.mean(0)) / (Xtr.std(0) + 1e-9)
        imp, _ = gene_importance(Ztr, yb[tr]); freq[np.argsort(-imp)[:50]] += 1
        print("split", si, flush=True)
    freq /= len(splits)
    imp_full, rules = gene_importance(Z, yb)
    # 4: symbols
    ids = [str(n) for n in names]
    scope = "reporter" if not ids[0].startswith("ENSG") else "ensembl.gene"
    q = [i.split(".")[0] for i in ids]
    sym_map = mygene_symbols(q, scope); sym = np.array([sym_map.get(i, "") for i in q])
    background = [s for s in sym if s]
    stable = np.where(freq >= 0.7)[0]
    stable_genes = sorted({sym[i] for i in stable if sym[i]})
    top100 = [sym[i] for i in np.argsort(-imp_full)[:150] if sym[i]][:100]
    enr_stable = gprofiler(stable_genes, background); enr_top = gprofiler(top100, background)
    # 5: module coherence, K=20 partition on full data vs random modules of the same sizes
    groups, betas, sd, r = fit_partition(Z, yb, 20, 0)
    rng = np.random.default_rng(0); perm = rng.permutation(X.shape[1]); sizes = [len(gg) for gg in groups]
    rand_groups = np.split(perm, np.cumsum(sizes)[:-1])
    coh = {"kmeans": [], "random": []}
    for kind, gs in (("kmeans", groups), ("random", rand_groups)):
        for gg in gs:
            genes = [sym[i] for i in gg if sym[i]]
            res = gprofiler(genes, background, sources=("GO:BP",)) if len(genes) >= 5 else []
            best = min([x["p"] for x in res], default=1.0) if res is not None else None
            coh[kind].append({"size": len(gg), "n_sig": 0 if not res else len(res), "best_p": best})
    # 6: lasso
    las = LogisticRegressionCV(Cs=[0.01, 0.03, 0.1, 0.3, 1], cv=3, penalty="l1", solver="liblinear", class_weight="balanced", max_iter=3000).fit(Z, yb)
    las_idx = np.where(np.abs(las.coef_.ravel()) > 1e-8)[0]
    las_genes = sorted({sym[i] for i in las_idx if sym[i]})
    enr_lasso = gprofiler(las_genes, background)
    res = {"ds": DS, "target": int(target), "n": int(len(y)), "p": int(X.shape[1]), "n_mapped": int(len(background)),
           "stable_probes": int(len(stable)), "stable_genes": stable_genes, "top100": top100,
           "freq_top": [[str(names[i]), sym[i], float(freq[i]), float(imp_full[i])] for i in np.argsort(-imp_full)[:40]],
           "enr_stable": enr_stable, "enr_top": enr_top, "enr_lasso": enr_lasso, "lasso_genes": las_genes,
           "overlap_lasso_stable": sorted(set(las_genes) & set(stable_genes)), "coherence": coh,
           "rules": [{**rr, "top_symbols": [[sym[i] for i in gl] for gl in rr["top_genes"]]} for rr in rules]}
    json.dump(res, open(os.path.join(OUT, f"p20_{DS}.json"), "w"), indent=1)
    ks = [c["n_sig"] > 0 for c in coh["kmeans"]]; rs = [c["n_sig"] > 0 for c in coh["random"]]
    print("stable genes:", len(stable_genes), stable_genes[:30])
    print("enrichment (stable):", [(x["source"], x["name"], f'{x["p"]:.1e}') for x in (enr_stable or [])[:15]])
    print("enrichment (top100):", [(x["source"], x["name"], f'{x["p"]:.1e}') for x in (enr_top or [])[:15]])
    print("coherent modules: kmeans", sum(ks), "/", len(ks), " random", sum(rs), "/", len(rs))
    print("lasso genes:", len(las_genes), "overlap with stable:", len(res["overlap_lasso_stable"]))
    print("lasso enrichment:", [(x["source"], x["name"], f'{x["p"]:.1e}') for x in (enr_lasso or [])[:8]])


if __name__ == "__main__":
    main()
