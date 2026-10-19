"""P6: accuracy vs complexity on the 8 dev sets (P4 protocol: 5-fold x 2, seed 0).
Complexity = number of original features the fitted model actually uses, plus #rules / #modules.
Models: LR-all (dense), LASSO-LR (L1, C by inner CV), TR2m on CGf, HYB-K20 (L2 modules),
HYB-L1K20 (L1 modules). For hybrids, used features = union of non-zero members of the modules
referenced by the selected rules."""
import json, os, sys, warnings
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
warnings.filterwarnings("ignore")
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from common import load, balanced_acc
from methods import Reduction, ThresholdRules
from p4_bench import get_splits

def mod_models(Za, ya, lab, K, pen):
    ms, cols = [], []
    binary = len(np.unique(ya)) == 2
    for k in range(K):
        idx = np.where(lab == k)[0]
        if len(idx) == 0: continue
        m = (LogisticRegression(C=0.1, max_iter=3000) if pen == "l2" else
             LogisticRegression(C=0.5, penalty="l1", solver="liblinear" if binary else "saga", max_iter=3000)).fit(Za[:, idx], ya)
        ms.append((idx, m))
    return ms

def scores(ms, Z):
    out, owner = [], []
    for j, (idx, m) in enumerate(ms):
        s = m.decision_function(Z[:, idx]); s = s[:, None] if s.ndim == 1 else s
        out.append(s); owner += [j] * s.shape[1]
    return np.hstack(out), np.array(owner)

def hybrid(Ztr, ytr, Zte, K, pen, seed):
    lab = MiniBatchKMeans(K, random_state=seed, n_init=3, batch_size=4096).fit(Ztr.T / np.sqrt(len(Ztr))).labels_
    oof = None
    for a, b in StratifiedKFold(3, shuffle=True, random_state=seed).split(Ztr, ytr):
        s, _ = scores(mod_models(Ztr[a], ytr[a], lab, K, pen), Ztr[b])
        if oof is None: oof = np.zeros((len(ytr), s.shape[1]))
        oof[b] = s
    ms = mod_models(Ztr, ytr, lab, K, pen)
    Ste, owner = scores(ms, Zte)
    mu, sd = oof.mean(0), oof.std(0) + 1e-9
    r = ThresholdRules(k=2, thr="mid", seed=seed).fit((oof - mu) / sd, ytr)
    p = r.predict((Ste - mu) / sd)
    used_cols = {int(f) for _, ch, _ in r.models for _, S, _, _ in ch for f in S}
    used_mods = {int(owner[c]) for c in used_cols}
    feats = set()
    for j in used_mods:
        idx, m = ms[j]
        feats |= set(idx[np.abs(m.coef_).sum(0) > 1e-8].tolist())
    return p, {"n_rules": int(sum(len(ch) for _, ch, _ in r.models)), "n_modules": len(used_mods), "n_feat": len(feats)}

def main(ds):
    X, y = load(ds); res = []
    for si, (tr, te) in enumerate(get_splits(y, None)):
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        Ztr, Zte = (X[tr] - mu) / sd, (X[te] - mu) / sd
        rec = {"ds": ds, "split": si}
        m = LogisticRegressionCV(Cs=[1e-3, 1e-2, 1e-1, 1, 10], cv=3, max_iter=3000).fit(Ztr, y[tr])
        rec["LR"] = (balanced_acc(y[te], m.predict(Zte)), {"n_feat": X.shape[1]})
        binary = len(np.unique(y)) == 2
        m = LogisticRegressionCV(Cs=[0.01, 0.03, 0.1, 0.3, 1, 3], cv=3, penalty="l1",
                                 solver="liblinear" if binary else "saga", max_iter=3000).fit(Ztr, y[tr])
        rec["LASSO"] = (balanced_acc(y[te], m.predict(Zte)), {"n_feat": int((np.abs(m.coef_).sum(0) > 1e-8).sum())})
        R = Reduction("CGf").fit(X[tr], y[tr]); r = ThresholdRules(k=2, thr="mid", seed=si).fit(R.transform(X[tr]), y[tr])
        used = {int(f) for _, ch, _ in r.models for _, S, _, _ in ch for f in S}
        rec["TR2m"] = (balanced_acc(y[te], r.predict(R.transform(X[te]))),
                       {"n_rules": int(sum(len(ch) for _, ch, _ in r.models)), "n_feat": len(used)})
        for name, pen in [("HYB-K20", "l2"), ("HYB-L1K20", "l1")]:
            p, c = hybrid(Ztr, y[tr], Zte, 20, pen, si); rec[name] = (balanced_acc(y[te], p), c)
        res.append(rec)
    json.dump(res, open(os.path.join(os.path.dirname(__file__), "..", "results", f"p6_{ds}.json"), "w"))
    print(ds, {k: (round(float(np.mean([r[k][0] for r in res])), 3),
                   {kk: round(float(np.mean([r[k][1][kk] for r in res])), 1) for kk in res[0][k][1]})
               for k in res[0] if k not in ("ds", "split")}, flush=True)

if __name__ == "__main__":
    main(sys.argv[1])
