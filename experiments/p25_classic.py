"""P25: classical p >> n baselines (reviewer point 8) on the P4 protocol (same splits, cleaning, standardisation).

SVMbal  linear SVM, balanced class weights, C in {1e-4..1} by inner 3-fold CV (balanced accuracy)
ENbal   elastic-net logistic regression (l1_ratio 0.5), balanced class weights, C in {0.01, 0.1, 1} by inner 3-fold CV
SLDA    shrinkage LDA, Ledoit-Wolf shrinkage towards a scaled identity, equal class priors; the p x p covariance is never
        formed (Woodbury identity over the n x n Gram matrix)
PAM     nearest shrunken centroids, shrinkage threshold by inner 3-fold CV (balanced accuracy)
Output: results/p25/p25_<ds>.jsonl.  Usage: python p25_classic.py <ds> [methods]
"""
import json, os, sys, time, warnings
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
warnings.filterwarnings("ignore")
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestCentroid
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from common import balanced_acc
from p4_bench import get_data, get_splits, clean

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results", "p25"); os.makedirs(OUT, exist_ok=True)
METHODS = ["SVMbal", "ENbal", "SLDA", "PAM"]


class ShrinkLDA:
    """LDA with Sigma = (1 - a) S + a mu I, S the pooled within-class covariance, a by Ledoit-Wolf; equal priors."""

    def fit(self, X, y):
        self.classes_ = np.unique(y); n, p = X.shape
        self.M = np.vstack([X[y == c].mean(0) for c in self.classes_])
        Xc = X - self.M[np.searchsorted(self.classes_, y)]
        G = Xc @ Xc.T / n                                    # n x n; S = Xc.T Xc / n shares its nonzero spectrum
        mu = np.trace(G) / p
        tr_S2 = np.sum(G ** 2)
        d2 = tr_S2 / p - mu ** 2                              # ||S - mu I||_F^2 / p
        r = np.sum(Xc ** 2, 1)
        b2 = min(d2, (np.sum(r ** 2) / n - tr_S2) / (n * p))  # Ledoit-Wolf estimate of E||x x^T - S||^2 / (n p)
        a = b2 / d2 if d2 > 0 else 1.0
        self.a, self.mu = a, mu
        # Sigma^{-1} v = (1/c0) [v - Xc^T (c0 n / (1 - a) I + Xc Xc^T)^{-1} Xc v],  c0 = a mu
        c0 = a * mu
        if (1 - a) < 1e-12:
            self.W = self.M.T / c0
        else:
            K = np.linalg.inv(c0 * n / (1 - a) * np.eye(n) + Xc @ Xc.T)
            V = self.M.T
            self.W = (V - Xc.T @ (K @ (Xc @ V))) / c0          # p x C
        self.b = -0.5 * np.sum(self.M.T * self.W, 0)
        return self

    def predict(self, X):
        return self.classes_[np.argmax(X @ self.W + self.b, 1)]


def bacc_scorer(est, X, y):
    return balanced_acc(y, est.predict(X))


def run(m, Ztr, ytr, Zte, si):
    cv = StratifiedKFold(3, shuffle=True, random_state=si)
    if m == "SVMbal":
        g = GridSearchCV(LinearSVC(class_weight="balanced", max_iter=5000, dual=True), {"C": [1e-4, 1e-3, 1e-2, 1e-1, 1]},
                         cv=cv, scoring=bacc_scorer, n_jobs=1)
        return g.fit(Ztr, ytr).predict(Zte)
    if m == "ENbal":
        g = GridSearchCV(LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=0.5, class_weight="balanced",
                                            max_iter=1000, tol=1e-3), {"C": [0.01, 0.1, 1]}, cv=cv, scoring=bacc_scorer, n_jobs=1)
        return g.fit(Ztr, ytr).predict(Zte)
    if m == "SLDA":
        return ShrinkLDA().fit(Ztr, ytr).predict(Zte)
    if m == "PAM":
        g = GridSearchCV(NearestCentroid(), {"shrink_threshold": [None, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0]}, cv=cv,
                         scoring=bacc_scorer, n_jobs=1, error_score=0.0)
        return g.fit(Ztr, ytr).predict(Zte)
    raise ValueError(m)


def main(ds, methods):
    out = os.path.join(OUT, f"p25_{ds.replace('/', '_').replace(':', '_')}.jsonl")
    done = set()
    if os.path.exists(out):
        for l in open(out):
            r = json.loads(l); done.add((r["split"], r["method"]))
    X, y, g = get_data(ds)
    for si, (tr, te) in enumerate(get_splits(y, g)):
        if si >= int(os.environ.get("MAX_SPLITS", "99")):
            break
        Xtr, Xte = clean(X[tr], X[te])
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        Ztr, Zte = (Xtr - mu) / sd, (Xte - mu) / sd
        for m in methods:
            if (si, m) in done:
                continue
            t0 = time.time(); rec = {"ds": ds, "split": si, "method": m}
            try:
                rec["bacc"] = balanced_acc(y[te], np.asarray(run(m, Ztr, y[tr], Zte, si)).astype(int))
            except Exception as e:
                rec["error"] = f"{type(e).__name__}: {str(e)[:200]}"
            rec["t"] = time.time() - t0
            open(out, "a").write(json.dumps(rec) + "\n")
        print(ds, si, flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2].split(",") if len(sys.argv) > 2 else METHODS)
