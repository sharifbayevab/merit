"""MERIT: Module-level Ensembles of Rules for Interpretable Tabular classification.

Pipeline for one partition (K modules, seed s), fitted on the training sample only:
  1. standardise features; k-means (K clusters) on the feature profiles z_i / sqrt(m) groups
     strongly correlated features into modules;
  2. inside every module an L2 logistic regression (C = 0.1) gives a module score;
     training-sample scores are out-of-fold (3-fold), test scores come from a refit on all objects;
  3. two-dimensional threshold rules over the standardised module scores (merit.rules.ThresholdRules).
The final decision sums the rule votes of several partitions (default: K = 10, 20, 40 x 2 seeds).
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from .rules import ThresholdRules


class MERIT(BaseEstimator, ClassifierMixin):
    """Parameters
    ----------
    Ks : module counts of the partitions in the vote, cycled over seeds.
    n_seeds : number of seeds per module count (partitions = len(Ks) * n_seeds).
    C : inverse L2 strength of the module logistic regressions.
    n_folds : folds for the out-of-fold module scores.
    random_state : base seed; partition i uses seed random_state + i // len(Ks).
    """

    def __init__(self, Ks=(10, 20, 40), n_seeds=2, C=0.1, n_folds=3, max_rules=31, rho=0.9, lam=0.1,
                 noise_floor=True, random_state=0):
        self.Ks, self.n_seeds, self.C, self.n_folds = Ks, n_seeds, C, n_folds
        self.max_rules, self.rho, self.lam, self.noise_floor, self.random_state = max_rules, rho, lam, noise_floor, random_state

    # ------------------------------------------------------------------ helpers
    def _scores(self, Za, ya, Zb, modules):
        out = []
        for idx in modules:
            s = LogisticRegression(C=self.C, max_iter=3000).fit(Za[:, idx], ya).decision_function(Zb[:, idx])
            out.append(s[:, None] if s.ndim == 1 else s)
        return np.hstack(out)

    def _fit_partition(self, Z, y, K, seed):
        lab = MiniBatchKMeans(K, random_state=seed, n_init=3, batch_size=4096).fit(Z.T / np.sqrt(len(Z))).labels_
        modules = [np.where(lab == k)[0] for k in range(K) if (lab == k).any()]
        oof = None
        for a, b in StratifiedKFold(self.n_folds, shuffle=True, random_state=seed).split(Z, y):
            s = self._scores(Z[a], y[a], Z[b], modules)
            if oof is None:
                oof = np.zeros((len(y), s.shape[1]))
            oof[b] = s
        regs = [LogisticRegression(C=self.C, max_iter=3000).fit(Z[:, idx], y) for idx in modules]
        mu, sd = oof.mean(0), oof.std(0) + 1e-9
        k = min(2, oof.shape[1])
        rules = ThresholdRules(k=k, M=self.max_rules, rho=self.rho, lam=self.lam, noise_floor=self.noise_floor,
                               seed=seed).fit((oof - mu) / sd, y)
        return dict(K=K, seed=seed, modules=modules, regs=regs, mu=mu, sd=sd, rules=rules)

    def _module_scores(self, Z, p):
        out = [r.decision_function(Z[:, idx]) for idx, r in zip(p["modules"], p["regs"])]
        S = np.hstack([s[:, None] if s.ndim == 1 else s for s in out])
        return (S - p["mu"]) / p["sd"]

    # ------------------------------------------------------------------ API
    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        self.classes_, y = np.unique(y, return_inverse=True)
        self.x_mean_, self.x_std_ = X.mean(0), X.std(0) + 1e-9
        Z = (X - self.x_mean_) / self.x_std_
        Ks = list(self.Ks)
        self.partitions_ = [self._fit_partition(Z, y, Ks[i % len(Ks)], self.random_state + i // len(Ks))
                            for i in range(len(Ks) * self.n_seeds)]
        return self

    def decision_function(self, X):
        Z = (np.asarray(X, dtype=np.float64) - self.x_mean_) / self.x_std_
        dec = sum(p["rules"].decision(self._module_scores(Z, p)) for p in self.partitions_)
        return dec[:, 0] if len(self.classes_) == 2 else dec

    def predict(self, X):
        d = self.decision_function(X)
        return self.classes_[(d > 0).astype(int)] if len(self.classes_) == 2 else self.classes_[np.argmax(d, 1)]

    def explain(self, feature_names=None, top=5, partition=0):
        """Readable rules of one partition: each rule names two modules and their most weighted features."""
        p = self.partitions_[partition]
        names = np.asarray(feature_names) if feature_names is not None else None
        n_out = [1 if r.coef_.shape[0] == 1 else r.coef_.shape[0] for r in p["regs"]]
        col2mod = np.repeat(np.arange(len(p["modules"])), n_out)

        def describe(col):
            m = col2mod[col]; idx = p["modules"][m]
            w = p["regs"][m].coef_[col - int(np.sum(n_out[:m]))]
            o = np.argsort(-np.abs(w))[:top]
            feats = [(str(names[idx[j]]) if names is not None else int(idx[j]), float(w[j])) for j in o]
            return dict(module=int(m), size=len(idx), top_features=feats)

        out = []
        for c, chosen, a in p["rules"].models_:
            for (q, S, w, t), al in zip(chosen, a):
                out.append(dict(target_class=self.classes_[c].item() if hasattr(self.classes_[c], 'item') else self.classes_[c], quality=float(q), weight=float(al),
                                direction=[float(v) for v in w], threshold=float(t), modules=[describe(int(s)) for s in S]))
        return out
