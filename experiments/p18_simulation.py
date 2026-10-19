"""P18: controlled simulation for the theory section.

Model: two classes with equal priors, x | y ~ N(mu_y, Sigma), p = 2000 features in K0 = 20 true modules of 100
features, Sigma block-diagonal with equicorrelation rho_w inside a module. The mean difference delta is spread
UNIFORMLY over all features of r signal modules (dense inside modules), scaled so that the Mahalanobis distance is
Delta. r controls module-level density; 'sparse' puts the whole signal on 5 single features instead.
Bayes balanced accuracy = Phi(Delta/2). Best accuracy with any s features = Phi(Delta_S/2) (Proposition 1).
Methods (same code as the benchmark): feature-level rules (TR2m on CGf), class-balanced LR, MERIT-1 (K=20),
MERIT (6 partitions). n_train = 50 per class, n_test = 2000, 10 replicates per setting.
"""
import json, os, sys, warnings
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
warnings.filterwarnings("ignore")
from scipy.stats import norm
from sklearn.linear_model import LogisticRegressionCV
from common import balanced_acc
from methods import Reduction, ThresholdRules
from p9_ablation import hybrid_decision, to_pred

HERE = os.path.dirname(os.path.abspath(__file__))
P, K0, M0, RHO, DELTA = 2000, 20, 100, 0.6, 3.0


def make_delta(setting, rng):
    d = np.zeros(P)
    if setting == "sparse":
        idx = rng.choice(P, 5, replace=False); d[idx] = 1.0
    else:
        r = int(setting)
        for k in rng.choice(K0, r, replace=False):
            d[k * M0:(k + 1) * M0] = 1.0
    # Mahalanobis norm under block equicorrelation: Sigma_k^{-1} 1 = 1 / (1 - rho + rho m) per full block
    return d


def mahal(d):
    tot = 0.0
    for k in range(K0):
        b = d[k * M0:(k + 1) * M0]
        if not b.any(): continue
        S = (1 - RHO) * np.eye(M0) + RHO * np.ones((M0, M0))
        tot += b @ np.linalg.solve(S, b)
    return np.sqrt(tot)


def sample(n_per, d, rng):
    L = np.linalg.cholesky((1 - RHO) * np.eye(M0) + RHO * np.ones((M0, M0)))
    X, y = [], []
    for c in (0, 1):
        Z = rng.standard_normal((n_per, P))
        Xc = np.hstack([Z[:, k * M0:(k + 1) * M0] @ L.T for k in range(K0)]) + (d if c == 1 else 0)
        X.append(Xc); y += [c] * n_per
    return np.vstack(X), np.array(y)


def main(setting):
    out = os.path.join(HERE, "..", "results", "p18", f"p18_{setting}.jsonl"); os.makedirs(os.path.dirname(out), exist_ok=True)
    if os.path.exists(out): os.remove(out)
    for rep in range(10):
        rng = np.random.default_rng(1000 + rep)
        d = make_delta(setting, rng); d = d * DELTA / mahal(d)
        Xtr, ytr = sample(50, d, rng); Xte, yte = sample(1000, d, rng)
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9; Ztr, Zte = (Xtr - mu) / sd, (Xte - mu) / sd
        rec = {"setting": setting, "rep": rep, "bayes": float(norm.cdf(DELTA / 2))}
        # Proposition 1: best accuracy with the 30 most informative features (oracle subset)
        imp = np.abs(d); top = np.argsort(-imp)[:30]
        sub = np.zeros(P); sub[top] = d[top]
        dS = np.sqrt(sum(sub[k * M0:(k + 1) * M0][sub[k * M0:(k + 1) * M0] != 0] @ np.linalg.solve(
            ((1 - RHO) * np.eye(M0) + RHO * np.ones((M0, M0)))[np.ix_(np.nonzero(sub[k * M0:(k + 1) * M0])[0], np.nonzero(sub[k * M0:(k + 1) * M0])[0])],
            sub[k * M0:(k + 1) * M0][sub[k * M0:(k + 1) * M0] != 0]) for k in range(K0) if sub[k * M0:(k + 1) * M0].any()))
        rec["oracle30"] = float(norm.cdf(dS / 2))
        R = Reduction("CGf").fit(Xtr, ytr)
        rec["rules"] = balanced_acc(yte, ThresholdRules(k=2, thr="mid", seed=rep).fit(R.transform(Xtr), ytr).predict(R.transform(Xte)))
        rec["lr"] = balanced_acc(yte, LogisticRegressionCV(Cs=[1e-3, 1e-2, 1e-1, 1, 10], cv=3, max_iter=3000, class_weight="balanced").fit(Ztr, ytr).predict(Zte))
        rec["merit1"] = balanced_acc(yte, to_pred(*hybrid_decision(Ztr, ytr, Zte, K=20, seed=rep)))
        dec, cl = 0, None
        for i in range(6):
            cl, dd = hybrid_decision(Ztr, ytr, Zte, K=[10, 20, 40][i % 3], seed=rep * 100 + i // 3)
            dec = dec + dd
        rec["merit"] = balanced_acc(yte, to_pred(cl, dec))
        dec, cl = 0, None
        for i in range(6):
            cl, dd = hybrid_decision(Ztr, ytr, Zte, K=[10, 20, 40][i % 3], seed=rep * 100 + i // 3, qmin="auto")
            dec = dec + dd
        rec["merit_nf"] = balanced_acc(yte, to_pred(cl, dec))
        open(out, "a").write(json.dumps(rec) + "\n"); print(rec, flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
