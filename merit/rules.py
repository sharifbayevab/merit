"""Two-dimensional threshold rules over a small set of input scores.

For every class (one-vs-rest; a single task for two classes) and every pair of inputs:
  discriminant function  = shrinkage Fisher direction w = (S + lam I)^-1 (m1 - m0),
  threshold rule         = 1[w'x > t], t = midpoint of the projected class means,
  rule quality q         = training balanced accuracy of the rule.
Rules are ranked by q. A rule is admitted while q exceeds the selection noise floor
  tau = 1/2 + 1/4 sqrt(1/n1 + 1/n0) sqrt(2 ln N)     (N = number of candidate rules),
and is skipped when its outputs correlate above rho with an admitted rule. At most M rules
are kept per class; votes are weighted by alpha = 1/2 ln(q / (1 - q)).
"""
import itertools
import numpy as np
from sklearn.feature_selection import f_classif


class ThresholdRules:
    def __init__(self, k=2, M=31, rho=0.9, n_cand=60, max_subsets=3000, lam=0.1, noise_floor=True, seed=0):
        self.k, self.M, self.rho, self.n_cand = k, M, rho, n_cand
        self.max_subsets, self.lam, self.noise_floor, self.seed = max_subsets, lam, noise_floor, seed

    def _subsets(self, F):
        top = np.argsort(-F)[: min(self.n_cand, len(F))]
        subs = list(itertools.combinations(top, self.k))
        if len(subs) > self.max_subsets:
            rng = np.random.default_rng(self.seed)
            subs = [subs[i] for i in rng.choice(len(subs), self.max_subsets, replace=False)]
        return subs

    def _fit_rule(self, Z, yb, S):
        A = Z[:, S]
        m1, m0 = A[yb == 1].mean(0), A[yb == 0].mean(0)
        w = np.linalg.solve(np.atleast_2d(np.cov(A.T)) + self.lam * np.eye(len(S)), m1 - m0)
        t = float(0.5 * (m1 + m0) @ w)
        p = A @ w > t
        q = 0.5 * (np.mean(p[yb == 1]) + np.mean(~p[yb == 0]))
        return w, t, q

    def fit(self, Z, y):
        self.classes_ = np.unique(y)
        tasks = [self.classes_[1]] if len(self.classes_) == 2 else list(self.classes_)
        self.models_ = []  # per task: (class, [(q, S, w, t), ...], alphas)
        for c in tasks:
            yb = (y == c).astype(int)
            F = np.nan_to_num(f_classif(Z, yb)[0])
            rules = []
            for S in self._subsets(F):
                S = np.array(S)
                w, t, q = self._fit_rule(Z, yb, S)
                rules.append((q, S, w, t))
            rules.sort(key=lambda r: -r[0])
            tau = None
            if self.noise_floor:
                n1, n0 = max(int(yb.sum()), 1), max(int((1 - yb).sum()), 1)
                tau = 0.5 + 0.25 * np.sqrt(1.0 / n1 + 1.0 / n0) * np.sqrt(2.0 * np.log(max(len(rules), 2)))
            chosen, outs = [], []
            for q, S, w, t in rules:
                if tau is not None and q < tau and chosen:
                    break
                o = (Z[:, S] @ w > t).astype(float) * 2 - 1
                if any(abs(np.corrcoef(o, p)[0, 1]) > self.rho for p in outs if o.std() > 0 and p.std() > 0):
                    continue
                chosen.append((q, S, w, t)); outs.append(o)
                if len(chosen) >= self.M:
                    break
            qc = np.clip([r[0] for r in chosen], 0.501, 0.999)
            self.models_.append((c, chosen, 0.5 * np.log(qc / (1 - qc))))
        return self

    def decision(self, Z):
        """Weighted vote in [-1, 1] per task (column)."""
        cols = []
        for _, chosen, a in self.models_:
            R = np.stack([(Z[:, S] @ w > t).astype(float) * 2 - 1 for _, S, w, t in chosen], 1)
            cols.append(R @ a / max(a.sum(), 1e-9))
        return np.stack(cols, 1)
