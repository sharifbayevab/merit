"""P1d: supervised module scores. Modules = k-means on feature profiles (train fold).
Inside each module: ridge logistic direction; out-of-fold (3-fold) module scores on the training
fold avoid leakage; second stage = LR over module scores (SM-LR) and threshold rules over
module scores (SM-TR2m, SM-TR3m)."""
import json, os, sys, warnings
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
warnings.filterwarnings("ignore")
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from common import load, splits, balanced_acc
from methods import ThresholdRules

def module_scores(Za, ya, Zb, lab, K, C=0.1):
    cls = np.unique(ya); S = []
    for k in range(K):
        idx = np.where(lab == k)[0]
        if len(idx) == 0: continue
        m = LogisticRegression(C=C, max_iter=3000).fit(Za[:, idx], ya)
        s = m.decision_function(Zb[:, idx])
        S.append(s[:, None] if s.ndim == 1 else s)
    return np.hstack(S)

def main(ds, K):
    X, y = load(ds); res = []
    for si, (tr, te) in enumerate(splits(y)):
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        Ztr, Zte = (X[tr] - mu) / sd, (X[te] - mu) / sd
        lab = MiniBatchKMeans(K, random_state=si, n_init=3, batch_size=4096).fit(Ztr.T / np.sqrt(len(tr))).labels_
        oof = None
        for a, b in StratifiedKFold(3, shuffle=True, random_state=si).split(Ztr, y[tr]):
            s = module_scores(Ztr[a], y[tr][a], Ztr[b], lab, K)
            if oof is None: oof = np.zeros((len(tr), s.shape[1]))
            oof[b] = s
        Ste = module_scores(Ztr, y[tr], Zte, lab, K)
        m_s, s_s = oof.mean(0), oof.std(0) + 1e-9
        A, B = (oof - m_s) / s_s, (Ste - m_s) / s_s
        rec = {"ds": ds, "split": si, "K": K}
        rec["SM-LR"] = balanced_acc(y[te], LogisticRegressionCV(Cs=[1e-3, 1e-2, 1e-1, 1], cv=3, max_iter=5000).fit(A, y[tr]).predict(B))
        for k in (2, 3):
            r = ThresholdRules(k=k, thr="mid", cand=(None, 60, 25)).fit(A, y[tr])
            rec[f"SM-TR{k}m"] = balanced_acc(y[te], r.predict(B))
            rec[f"SM-TR{k}m_nfeat_modules"] = len({int(f) for _, ch, _ in r.models for _, S, _, _ in ch for f in S})
        res.append(rec)
    return res

if __name__ == "__main__":
    ds, K = sys.argv[1], int(sys.argv[2])
    res = main(ds, K)
    json.dump(res, open(os.path.join(os.path.dirname(__file__), "..", "results", f"p1d_{ds}_K{K}.json"), "w"))
    print(ds, K, {k: round(float(np.mean([r[k] for r in res])), 3) for k in res[0] if k.startswith("SM")})
