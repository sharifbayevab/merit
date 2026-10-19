"""P1e: honest hybrid. (a) K in {20, 50} chosen by inner 3-fold CV of the whole pipeline (SM-TR2m);
(b) sparse modules: within-module L1 logistic (C=0.5) at K=20, reporting accuracy and the mean
number of non-zero features per module (interpretability)."""
import json, os, sys, warnings
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
warnings.filterwarnings("ignore")
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from common import load, splits, balanced_acc
from methods import ThresholdRules

def mod_fit_scores(Za, ya, Zb, lab, K, pen):
    S, nnz = [], []
    for k in range(K):
        idx = np.where(lab == k)[0]
        if len(idx) == 0: continue
        m = (LogisticRegression(C=0.1, max_iter=3000) if pen == "l2" else
             LogisticRegression(C=0.5, penalty="l1", solver="liblinear", max_iter=3000) if len(np.unique(ya)) == 2 else
             LogisticRegression(C=0.5, penalty="l1", solver="saga", max_iter=3000)).fit(Za[:, idx], ya)
        nnz.append(int((np.abs(m.coef_).sum(0) > 1e-8).sum()))
        s = m.decision_function(Zb[:, idx]); S.append(s[:, None] if s.ndim == 1 else s)
    return np.hstack(S), nnz

def pipeline(Ztr, ytr, Zte, K, pen, seed):
    lab = MiniBatchKMeans(K, random_state=seed, n_init=3, batch_size=4096).fit(Ztr.T / np.sqrt(len(Ztr))).labels_
    oof = None
    for a, b in StratifiedKFold(3, shuffle=True, random_state=seed).split(Ztr, ytr):
        s, _ = mod_fit_scores(Ztr[a], ytr[a], Ztr[b], lab, K, pen)
        if oof is None: oof = np.zeros((len(ytr), s.shape[1]))
        oof[b] = s
    Ste, nnz = mod_fit_scores(Ztr, ytr, Zte, lab, K, pen)
    m, sd = oof.mean(0), oof.std(0) + 1e-9
    r = ThresholdRules(k=2, thr="mid").fit((oof - m) / sd, ytr)
    return r.predict((Ste - m) / sd), nnz, r

def main(ds):
    X, y = load(ds); res = []
    for si, (tr, te) in enumerate(splits(y)):
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        Ztr, Zte = (X[tr] - mu) / sd, (X[te] - mu) / sd
        inner = {K: [] for K in (20, 50)}
        for a, b in StratifiedKFold(3, shuffle=True, random_state=100 + si).split(Ztr, y[tr]):
            for K in inner:
                p, _, _ = pipeline(Ztr[a], y[tr][a], Ztr[b], K, "l2", si)
                inner[K].append(balanced_acc(y[tr][b], p))
        Kbest = max(inner, key=lambda K: np.mean(inner[K]))
        p, _, _ = pipeline(Ztr, y[tr], Zte, Kbest, "l2", si)
        rec = {"ds": ds, "split": si, "Kbest": Kbest, "NESTED": balanced_acc(y[te], p)}
        p, nnz, r = pipeline(Ztr, y[tr], Zte, 20, "l1", si)
        rec["L1K20"] = balanced_acc(y[te], p); rec["l1_nnz_per_module"] = float(np.mean(nnz))
        used = {int(f) for _, ch, _ in r.models for _, S, _, _ in ch for f in S}
        rec["l1_n_rules"] = int(sum(len(ch) for _, ch, _ in r.models)); rec["l1_modules_cols_used"] = len(used)
        res.append(rec); print(ds, si, rec, flush=True)
    json.dump(res, open(os.path.join(os.path.dirname(__file__), "..", "results", f"p1e_{ds}.json"), "w"))

if __name__ == "__main__":
    main(sys.argv[1])
