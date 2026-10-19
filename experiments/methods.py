"""Feature reductions and classification heads for the P1 swap matrix.

Reductions (fit on the training fold only):
  ALL   - all features (only for heads that can take them)
  F500  - top-500 features by ANOVA F (supervised filter)
  V500  - top-500 features by variance (unsupervised filter)
  CGf   - correlation groups + representative = max-F feature of the group
          (Mirzaev-style "subsets of strongly coupled features" + "representative feature")
  CGm   - same groups, representative = medoid (max mean |r| inside the group; label-free)
  CGp   - same groups, first principal component of the group (GOTabPFN-lite meta-feature)

Groups: prefilter top-P features by F (P=2000), average-linkage clustering on 1-|r|,
cut at distance 1-R_CUT, keep the R groups whose representative has the largest F.

Heads: LR, RF, XGB, EBM, TR1/TR2/TR3 (Mirzaev-style k-dimensional threshold-rule
ensemble, our reimplementation), TPFN (TabPFN v2), TICL (TabICL v2).
"""
import itertools
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier

P_PREFILTER = 2000
R_CUT = 0.7
R_KEEP = 200


# ----------------------------------------------------------------------------- reductions
class Reduction:
    def __init__(self, kind):
        self.kind = kind

    def fit(self, X, y):
        self.mu = X.mean(0)
        self.sd = X.std(0) + 1e-9
        Z = (X - self.mu) / self.sd
        k = self.kind
        if k == "ALL":
            self.cols = np.arange(X.shape[1])
            return self
        F, _ = f_classif(Z, y)
        F = np.nan_to_num(F)
        if k == "F500":
            self.cols = np.argsort(-F)[:500]
            return self
        if k == "V500":
            self.cols = np.argsort(-X.var(0))[:500]
            return self
        pre = np.argsort(-F)[:min(P_PREFILTER, X.shape[1])]
        Zp = Z[:, pre]
        C = np.corrcoef(Zp.T)
        C = np.nan_to_num(C)
        D = 1.0 - np.abs(C)
        np.fill_diagonal(D, 0.0)
        D = np.clip((D + D.T) / 2, 0, None)
        L = linkage(squareform(D, checks=False), method="average")
        lab = fcluster(L, t=1.0 - R_CUT, criterion="distance")
        groups = []
        for g in np.unique(lab):
            members = np.where(lab == g)[0]
            if k == "CGm":
                sub = np.abs(C[np.ix_(members, members)]).mean(1)
                rep = members[np.argmax(sub)]
            else:
                rep = members[np.argmax(F[pre[members]])]
            groups.append((F[pre[rep]], pre[members], pre[rep]))
        groups.sort(key=lambda t: -t[0])
        groups = groups[:R_KEEP]
        self.n_groups_total = int(len(np.unique(lab)))
        self.group_sizes = [len(g[1]) for g in groups]
        if k in ("CGf", "CGm"):
            self.cols = np.array([g[2] for g in groups])
        else:  # CGp: PCA-1 per group, sign aligned with the representative
            self.cols = None
            self.pcs = []
            for _, members, rep in groups:
                if len(members) == 1:
                    self.pcs.append((members, np.ones(1)))
                    continue
                Zm = Z[:, members]
                _, _, Vt = np.linalg.svd(Zm - Zm.mean(0), full_matrices=False)
                v = Vt[0]
                j = list(members).index(rep)
                if v[j] < 0:
                    v = -v
                self.pcs.append((members, v))
        return self

    def transform(self, X):
        Z = (X - self.mu) / self.sd
        if self.kind == "CGp":
            return np.stack([Z[:, m] @ v for m, v in self.pcs], 1)
        return Z[:, self.cols]


# ----------------------------------------------------------------------------- threshold rules
def _best_threshold(s, yb):
    """Threshold on a 1-D score maximizing balanced accuracy for rule 1[s > t]."""
    o = np.argsort(s)
    ss, yy = s[o], yb[o]
    npos, nneg = yy.sum(), len(yy) - yy.sum()
    # predicting positive for everything above cut i (exclusive)
    pos_above = npos - np.concatenate([[0], np.cumsum(yy)])
    neg_below = np.concatenate([[0], np.cumsum(1 - yy)])
    bacc = 0.5 * (pos_above / max(npos, 1) + neg_below / max(nneg, 1))
    i = int(np.argmax(bacc))
    if i == 0:
        t = ss[0] - 1e-6
    elif i == len(ss):
        t = ss[-1] + 1e-6
    else:
        t = 0.5 * (ss[i - 1] + ss[i])
    return t, float(bacc[i])


class ThresholdRules:
    """k-dimensional threshold-rule ensemble (reimplementation of the Mirzaev/Fazilov family).

    For every class (one-vs-rest) and every k-subset of candidate features:
      discriminant function = shrinkage Fisher direction on the subset,
      threshold rule        = 1[score > t], t maximizing training balanced accuracy.
    Rules are ranked by training quality; a rule is skipped if its training outputs
    correlate > RHO with an already chosen rule (removing "groups of highly correlated
    discriminant functions"); at most M rules are kept ("preferred rules"); votes are
    weighted by 0.5*log(q/(1-q)) (generalized recognition operator).
    """

    def __init__(self, k=2, M=31, rho=0.9, cand=(None, 60, 25), max_subsets=3000, lam=0.1, seed=0,
                 select="train", thr="opt", qmin=None):
        self.k, self.M, self.rho, self.lam, self.thr = k, M, rho, lam, thr
        self.qmin = qmin  # keep only rules whose training quality exceeds qmin (None: always M rules)
        self.cand = cand[k - 1]
        self.max_subsets, self.seed, self.select = max_subsets, seed, select

    def _subsets(self, F):
        d = len(F)
        top = np.argsort(-F)[: (d if self.cand is None else min(self.cand, d))]
        subs = list(itertools.combinations(top, self.k))
        if len(subs) > self.max_subsets:
            rng = np.random.default_rng(self.seed)
            subs = [subs[i] for i in rng.choice(len(subs), self.max_subsets, replace=False)]
        return subs

    def _fit_rule(self, Z, yb, S):
        A = Z[:, S]
        m1, m0 = A[yb == 1].mean(0), A[yb == 0].mean(0)
        cov = np.atleast_2d(np.cov(A.T)) + self.lam * np.eye(len(S))
        w = np.linalg.solve(cov, m1 - m0)
        s = A @ w
        t, q = _best_threshold(s, yb)
        if self.thr == "mid":  # classical Fisher cut: midpoint of the projected class means
            t = float(0.5 * (m1 + m0) @ w)
            p = s > t
            q = 0.5 * (np.mean(p[yb == 1]) + np.mean(~p[yb == 0]))
        return w, t, q

    def fit(self, Z, y):
        self.classes = np.unique(y)
        tasks = [self.classes[1]] if len(self.classes) == 2 else list(self.classes)
        self.models = []
        self.diag = {"train_q_sel": [], "n_rules": []}
        for c in tasks:
            yb = (y == c).astype(int)
            F, _ = f_classif(Z, yb)
            F = np.nan_to_num(F)
            rules = []
            if self.select == "loo_half":
                # honest selection: rule quality is measured on a fixed held-out half of the training fold
                rng = np.random.default_rng(self.seed + int(c))
                a, b = [], []
                for v in (0, 1):
                    iv = rng.permutation(np.where(yb == v)[0])
                    a += list(iv[: len(iv) // 2]); b += list(iv[len(iv) // 2:])
                a, b = np.array(a), np.array(b)
            for S in self._subsets(F):
                S = np.array(S)
                if self.select == "loo_half":
                    w, t, _ = self._fit_rule(Z[a], yb[a], S)
                    pb = (Z[b][:, S] @ w > t).astype(int)
                    q = 0.5 * (np.mean(pb[yb[b] == 1] == 1) + np.mean(pb[yb[b] == 0] == 0))
                    w, t, _ = self._fit_rule(Z, yb, S)
                else:
                    w, t, q = self._fit_rule(Z, yb, S)
                rules.append((q, S, w, t))
            rules.sort(key=lambda r: -r[0])
            qmin = self.qmin
            if qmin == "auto":  # selection noise floor (Proposition 3): chance + expected max of N null qualities
                n1, n0 = max(int(yb.sum()), 1), max(int((1 - yb).sum()), 1)
                qmin = 0.5 + 0.25 * np.sqrt(1.0 / n1 + 1.0 / n0) * np.sqrt(2.0 * np.log(max(len(rules), 2)))
            chosen, outs = [], []
            for q, S, w, t in rules:
                if qmin is not None and q < qmin and chosen:
                    break
                o = (Z[:, S] @ w > t).astype(float) * 2 - 1
                if any(abs(np.corrcoef(o, p)[0, 1]) > self.rho for p in outs if o.std() > 0 and p.std() > 0):
                    continue
                chosen.append((q, S, w, t))
                outs.append(o)
                if len(chosen) >= self.M:
                    break
            alphas = [0.5 * np.log(np.clip(q, 0.501, 0.999) / (1 - np.clip(q, 0.501, 0.999))) for q, *_ in chosen]
            self.models.append((c, chosen, np.array(alphas)))
            self.diag["train_q_sel"].append(float(np.mean([r[0] for r in chosen])))
            self.diag["n_rules"].append(len(chosen))
        return self

    def rule_outputs(self, Z):
        return [np.stack([(Z[:, S] @ w > t).astype(float) * 2 - 1 for _, S, w, t in ch], 1)
                for _, ch, _ in self.models]

    def decision(self, Z):
        return np.stack([R @ a / max(a.sum(), 1e-9) for R, (_, _, a) in zip(self.rule_outputs(Z), self.models)], 1)

    def predict(self, Z):
        s = self.decision(Z)
        if len(self.classes) == 2:
            return np.where(s[:, 0] > 0, self.classes[1], self.classes[0])
        return self.classes[np.argmax(s, 1)]


# ----------------------------------------------------------------------------- heads
def make_head(name, n_classes, seed=0):
    if name == "LR":
        return LogisticRegressionCV(Cs=[1e-3, 1e-2, 1e-1, 1, 10], cv=3, max_iter=5000, n_jobs=1)
    if name == "RF":
        return RandomForestClassifier(n_estimators=500, n_jobs=2, random_state=seed)
    if name == "XGB":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.1, subsample=0.8,
                             colsample_bytree=0.5, n_jobs=2, random_state=seed, verbosity=0)
    if name == "EBM":
        from interpret.glassbox import ExplainableBoostingClassifier
        return ExplainableBoostingClassifier(interactions=(10 if n_classes == 2 else 0), outer_bags=4,
                                             n_jobs=2, random_state=seed)
    if name in ("TR1", "TR2", "TR3"):
        return ThresholdRules(k=int(name[-1]), seed=seed)
    if name in ("TR1m", "TR2m", "TR3m"):
        return ThresholdRules(k=int(name[2]), seed=seed, thr="mid")
    if name == "TR2h":
        return ThresholdRules(k=2, seed=seed, select="loo_half")
    if name == "TR2hm":
        return ThresholdRules(k=2, seed=seed, select="loo_half", thr="mid")
    if name == "TPFN":
        from tabpfn import TabPFNClassifier
        import os
        dev = os.environ.get("PFN_DEVICE", "cuda")
        kw = dict(memory_saving_mode=False) if dev == "mps" else {}
        return TabPFNClassifier(device=dev, random_state=seed, ignore_pretraining_limits=True, **kw)
    if name == "TICL":
        from tabicl import TabICLClassifier
        import os
        return TabICLClassifier(device=os.environ.get("ICL_DEVICE", "cuda"), random_state=seed)
    raise ValueError(name)


# ----------------------------------------------------------------------------- P2 additions
class KTSP:
    """k-Top Scoring Pairs (Tan & Geman 2005): rule 1[x_i < x_j], k disjoint pairs, majority vote.
    Scores on the top-`cand` features by F; k chosen from {1,3,...,9} by 3-fold CV. One-vs-rest for C>2."""

    def __init__(self, cand=200, ks=(1, 3, 5, 7, 9), seed=0):
        self.cand, self.ks, self.seed = cand, ks, seed

    @staticmethod
    def _pairs(Z, yb, cols, kmax):
        A = Z[:, cols]
        P1 = (A[yb == 1][:, :, None] < A[yb == 1][:, None, :]).mean(0)
        P0 = (A[yb == 0][:, :, None] < A[yb == 0][:, None, :]).mean(0)
        S = P1 - P0
        iu = np.triu_indices(len(cols), 1)
        sc = np.abs(S[iu])
        order = np.argsort(-sc)
        used, out = set(), []
        for o in order:
            i, j = iu[0][o], iu[1][o]
            if i in used or j in used:
                continue
            used |= {i, j}
            out.append((cols[i], cols[j], np.sign(S[i, j]), sc[o]))
            if len(out) >= kmax:
                break
        return out

    def _vote(self, Z, pairs, k):
        v = [s * ((Z[:, i] < Z[:, j]).astype(float) * 2 - 1) for i, j, s, _ in pairs[:k]]
        return np.mean(v, 0)

    def fit(self, Z, y):
        from sklearn.model_selection import StratifiedKFold
        self.classes = np.unique(y)
        tasks = [self.classes[1]] if len(self.classes) == 2 else list(self.classes)
        self.models = []
        for c in tasks:
            yb = (y == c).astype(int)
            F = np.nan_to_num(f_classif(Z, yb)[0])
            # CV for k
            acc = {k: [] for k in self.ks}
            skf = StratifiedKFold(3, shuffle=True, random_state=self.seed)
            for a, b in skf.split(Z, yb):
                Fa = np.nan_to_num(f_classif(Z[a], yb[a])[0])
                pr = self._pairs(Z[a], yb[a], np.argsort(-Fa)[:self.cand], max(self.ks))
                for k in self.ks:
                    p = self._vote(Z[b], pr, k) > 0
                    acc[k].append(0.5 * (np.mean(p[yb[b] == 1]) + np.mean(~p[yb[b] == 0])))
            kbest = max(self.ks, key=lambda k: np.mean(acc[k]))
            pr = self._pairs(Z, yb, np.argsort(-F)[:self.cand], kbest)
            self.models.append((c, pr, kbest))
        return self

    def decision(self, Z):
        return np.stack([self._vote(Z, pr, k) for _, pr, k in self.models], 1)

    def predict(self, Z):
        s = self.decision(Z)
        if len(self.classes) == 2:
            return np.where(s[:, 0] > 0, self.classes[1], self.classes[0])
        return self.classes[np.argmax(s, 1)]


def augment(Z, y, n_per=20, alpha=0.4, noise=0.1, seed=0):
    """Within-class mixup + small Gaussian noise in the reduced space (candidate points for the teacher)."""
    rng = np.random.default_rng(seed)
    out = []
    for c in np.unique(y):
        Zc = Z[y == c]
        m = len(Zc) * n_per
        i, j = rng.integers(0, len(Zc), m), rng.integers(0, len(Zc), m)
        lam = rng.beta(alpha, alpha, m)[:, None]
        out.append(lam * Zc[i] + (1 - lam) * Zc[j] + noise * rng.standard_normal((m, Z.shape[1])))
    return np.vstack(out)


class TeacherRules:
    """Hybrid: a strong teacher (TabPFN/TabICL/LR) labels augmented points; threshold rules are
    fitted and selected on real + teacher-labelled points (only real points if teacher disagrees
    with a real label). Final rules keep the Mirzaev form (k features, one threshold, weighted vote)."""

    def __init__(self, teacher, k=2, n_per=20, conf=0.0, seed=0, thr="opt"):
        self.teacher, self.k, self.n_per, self.conf, self.seed, self.thr = teacher, k, n_per, conf, seed, thr

    def fit(self, Z, y):
        self.teacher.fit(Z, y)
        A = augment(Z, y, self.n_per, seed=self.seed)
        # predict in chunks: TabPFN memory grows with the number of test points (TOX_171 hit >100 GB unified memory)
        P = np.vstack([self.teacher.predict_proba(A[i:i + 256]) for i in range(0, len(A), 256)])
        cls = np.asarray(self.teacher.classes_)
        keep = P.max(1) >= self.conf
        ya = cls[P.argmax(1)][keep]
        self.n_aug = int(keep.sum())
        self.rules = ThresholdRules(k=self.k, seed=self.seed, thr=self.thr).fit(np.vstack([Z, A[keep]]),
                                                                                 np.concatenate([y, ya]))
        self.classes, self.models, self.diag = self.rules.classes, self.rules.models, self.rules.diag
        return self

    def rule_outputs(self, Z):
        return self.rules.rule_outputs(Z)

    def predict(self, Z):
        return self.rules.predict(Z)
