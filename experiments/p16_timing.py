"""P16: training time and response latency (Spark GB10: 20 CPU cores, unified-memory GPU), run on an otherwise idle box.
For each dataset (split 0 of the P4 protocol) and model:
  fit_s        wall time of fit on the training fold (includes the feature grouping for TFM inputs)
  batch_ms     wall time of predicting the whole test fold, per object
  single_ms    median latency of predicting ONE object (20 repeats after 3 warm-up calls)
Models: MERIT (6 partitions), MERIT-1 (K=20), LRbal (CPU); TabPFN v2 and TabICL v2 on the 200 group representatives
(GPU and CPU); TabPFN-Wide on 8000 filtered features (GPU, separate venv -> run with --wide).
Usage: python p16_timing.py [--wide]
"""
import json, os, sys, time, warnings
os.environ.setdefault("OMP_NUM_THREADS", "8")
import numpy as np
warnings.filterwarnings("ignore")
import torch
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegressionCV
from p4_bench import get_data, get_splits, clean
from methods import Reduction, ThresholdRules, make_head
from p9_ablation import hybrid_decision  # noqa (reference implementation)
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

HERE = os.path.dirname(os.path.abspath(__file__))
DATASETS = ["dev:colon", "OpenML-1138", "TCGA-TCGA-BRCA_Gene-Expression-Quantification"]


class MeritModel:
    """Deployable MERIT: fit stores module models and rules; predict = standardise -> module scores -> rule votes."""
    def __init__(self, parts):
        self.parts = parts  # list of (K, seed)

    def fit(self, Xtr, ytr):
        self.mu, self.sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        Z = (Xtr - self.mu) / self.sd
        self.models = []
        for K, seed in self.parts:
            lab = MiniBatchKMeans(K, random_state=seed, n_init=3, batch_size=4096).fit(Z.T / np.sqrt(len(Z))).labels_
            groups = [np.where(lab == k)[0] for k in range(K) if (lab == k).any()]
            def scores(Za, ya, Zb):
                out = []
                for idx in groups:
                    s = LogisticRegression(C=0.1, max_iter=3000).fit(Za[:, idx], ya).decision_function(Zb[:, idx])
                    out.append(s[:, None] if s.ndim == 1 else s)
                return np.hstack(out)
            oof = None
            for a, b in StratifiedKFold(3, shuffle=True, random_state=seed).split(Z, ytr):
                s = scores(Z[a], ytr[a], Z[b])
                if oof is None: oof = np.zeros((len(ytr), s.shape[1]))
                oof[b] = s
            mods = [LogisticRegression(C=0.1, max_iter=3000).fit(Z[:, idx], ytr) for idx in groups]
            m_, s_ = oof.mean(0), oof.std(0) + 1e-9
            r = ThresholdRules(k=2, thr="mid", seed=seed).fit((oof - m_) / s_, ytr)
            # compact deployment form: one weight matrix for all module scores
            W = np.zeros((Z.shape[1], oof.shape[1])); b0 = np.zeros(oof.shape[1]); c = 0
            for idx, m in zip(groups, mods):
                co = m.coef_; k = co.shape[0]
                W[np.ix_(idx, range(c, c + k))] = co.T; b0[c:c + k] = m.intercept_; c += k
            self.models.append((W, b0, m_, s_, r))
        self.classes = np.unique(ytr)
        return self

    def predict(self, X):
        Z = (X - self.mu) / self.sd; dec = 0
        for W, b0, m_, s_, r in self.models:
            dec = dec + r.decision(((Z @ W + b0) - m_) / s_)
        return np.where(dec[:, 0] > 0, self.classes[1], self.classes[0]) if len(self.classes) == 2 else self.classes[np.argmax(dec, 1)]


class TFMOnReps:
    def __init__(self, kind, device):
        self.kind, self.device = kind, device

    def fit(self, Xtr, ytr):
        os.environ["PFN_DEVICE"] = self.device; os.environ["ICL_DEVICE"] = self.device
        self.R = Reduction("CGf").fit(Xtr, ytr)
        self.h = make_head(self.kind, len(np.unique(ytr)), 0).fit(self.R.transform(Xtr), ytr)
        self.prior = np.bincount(ytr); return self

    def predict(self, X):
        P = self.h.predict_proba(self.R.transform(X)); cl = np.asarray(self.h.classes_)
        return cl[np.argmax(P / self.prior[cl], 1)]


class Wide:
    def fit(self, Xtr, ytr):
        from tabpfnwide.classifier import TabPFNWideClassifier
        self.mu, self.sd = Xtr.mean(0), Xtr.std(0) + 1e-9; Z = (Xtr - self.mu) / self.sd
        self.cols = np.argsort(-np.nan_to_num(f_classif(Z, ytr)[0]))[:8000]
        self.h = TabPFNWideClassifier(model_name="wide-v2-8k", device="cuda").fit(Z[:, self.cols], ytr)
        self.prior = np.bincount(ytr); return self

    def predict(self, X):
        P = np.asarray(self.h.predict_proba(((X - self.mu) / self.sd)[:, self.cols])); cl = np.arange(P.shape[1])
        return cl[np.argmax(P / self.prior[cl], 1)]


class LRb:
    def fit(self, Xtr, ytr):
        self.mu, self.sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        self.h = LogisticRegressionCV(Cs=[1e-3, 1e-2, 1e-1, 1, 10], cv=3, max_iter=3000, class_weight="balanced").fit((Xtr - self.mu) / self.sd, ytr)
        return self

    def predict(self, X):
        return self.h.predict((X - self.mu) / self.sd)


def timeit(model, Xtr, ytr, Xte):
    t0 = time.perf_counter(); model.fit(Xtr, ytr); fit = time.perf_counter() - t0
    if torch.cuda.is_available(): torch.cuda.synchronize()
    t0 = time.perf_counter(); model.predict(Xte); batch = (time.perf_counter() - t0) / len(Xte) * 1000
    for _ in range(3): model.predict(Xte[:1])
    lat = []
    for i in range(20):
        t0 = time.perf_counter(); model.predict(Xte[i % len(Xte):i % len(Xte) + 1]); lat.append((time.perf_counter() - t0) * 1000)
    return fit, batch, float(np.median(lat))


def main(wide):
    out = os.path.join(HERE, "..", "results", "p16_timing.jsonl")
    models = {"TabPFN-Wide (GPU)": Wide} if wide else {
        "MERIT (CPU)": lambda: MeritModel([(10, 0), (20, 0), (40, 0), (10, 1), (20, 1), (40, 1)]),
        "MERIT-1 (CPU)": lambda: MeritModel([(20, 0)]),
        "Logistic regression (CPU)": LRb,
        "TabPFN v2 (GPU)": lambda: TFMOnReps("TPFN", "cuda"), "TabICL v2 (GPU)": lambda: TFMOnReps("TICL", "cuda"),
        "TabPFN v2 (CPU)": lambda: TFMOnReps("TPFN", "cpu"), "TabICL v2 (CPU)": lambda: TFMOnReps("TICL", "cpu")}
    for ds in DATASETS:
        X, y, g = get_data(ds); tr, te = get_splits(y, g)[0]
        Xtr, Xte = clean(X[tr], X[te])
        for name, mk in models.items():
            try:
                fit, batch, single = timeit(mk(), Xtr, y[tr], Xte)
                rec = {"ds": ds, "n_train": int(len(tr)), "p": int(Xtr.shape[1]), "model": name, "fit_s": fit, "batch_ms": batch, "single_ms": single}
            except Exception as e:
                rec = {"ds": ds, "model": name, "error": f"{type(e).__name__}: {str(e)[:200]}"}
            open(out, "a").write(json.dumps(rec) + "\n"); print(rec, flush=True)


if __name__ == "__main__":
    main("--wide" in sys.argv)
