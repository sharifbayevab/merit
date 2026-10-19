"""P1c: can a module (feature-group) backbone recover dense signal?
Modules: k-means (K clusters) on standardized feature profiles of ALL features (train fold);
module score = mean of standardized members. Head: LR on K module scores.
Compare with LR on all features and rules (from P1)."""
import json, os, sys, warnings
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
warnings.filterwarnings("ignore")
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LogisticRegressionCV
from common import load, splits, balanced_acc

def modules(Ztr, K, seed):
    P = Ztr.T / np.sqrt(len(Ztr))          # feature profiles, unit norm (standardized)
    km = MiniBatchKMeans(K, random_state=seed, n_init=3, batch_size=4096).fit(P)
    return km.labels_

def main(ds):
    X, y = load(ds); out = []
    for si, (tr, te) in enumerate(splits(y)):
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        Ztr, Zte = (X[tr] - mu) / sd, (X[te] - mu) / sd
        rec = {"ds": ds, "split": si}
        for K in [20, 50, 200, 500]:
            lab = modules(Ztr, K, si)
            M = np.zeros((X.shape[1], K)); M[np.arange(X.shape[1]), lab] = 1; M /= M.sum(0, keepdims=True) + 1e-9
            m = LogisticRegressionCV(Cs=[1e-3, 1e-2, 1e-1, 1, 10], cv=3, max_iter=5000).fit(Ztr @ M, y[tr])
            rec[f"mod{K}"] = balanced_acc(y[te], m.predict(Zte @ M))
        out.append(rec)
    return out

if __name__ == "__main__":
    ds = sys.argv[1]
    res = main(ds)
    json.dump(res, open(os.path.join(os.path.dirname(__file__), "..", "results", f"p1c_{ds}.json"), "w"))
    print(ds, {k: round(np.mean([r[k] for r in res]), 3) for k in res[0] if k.startswith("mod")})
