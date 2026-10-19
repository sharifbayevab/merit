"""P4: held-out benchmark on TabBench-Bio classification sets (settings frozen on the 8 dev sets).

Protocol: rows capped at N_CAP by stratified subsampling (seed 0); classes with < 10 rows dropped;
5-fold x 2 repeats (StratifiedGroupKFold when the dataset has patient/site groups);
NaN -> train-fold median; constant columns dropped. Metric: balanced accuracy.
Methods (frozen): LR-all, XGB-all, TR2m on CGf (Mirzaev), KTSP, HYB-K20 (L2 modules + TR2m),
HYB-L1K20 (sparse modules), HYB-K50, TPFN on CGf, WIDE8k on F-top-8000.
P3 density index (train fold only): D = innerCV bacc(LR all) - innerCV bacc(LR top-10 by F).
Usage: python p4_bench.py <bio_id|dev:name> [methods,comma,sep]
"""
import json, os, sys, time, warnings
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
warnings.filterwarnings("ignore")
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, RepeatedStratifiedKFold
from common import load, balanced_acc
from methods import Reduction, ThresholdRules, KTSP, make_head
from p1e_nested import pipeline as hyb_pipeline

HERE = os.path.dirname(os.path.abspath(__file__))
TB = os.environ.get("TB_DIR", os.path.join(HERE, "..", "data_tb"))
N_CAP = 600
ALL_METHODS = ["LR", "XGB", "TR2m", "KTSP", "HYB-K20", "HYB-L1K20", "HYB-K50", "TPFN-CGf", "WIDE8k"]


def get_data(name):
    if name.startswith("dev:"):
        X, y = load(name[4:])
        return X.astype(np.float32), y, None
    z = np.load(os.path.join(TB, name.replace("/", "_") + ".npz"), allow_pickle=False)
    X, y, g = z["X"].astype(np.float32), z["y"], z["groups"]
    g = g if len(g) == len(y) and len(np.unique(g)) < len(y) else None
    _, y = np.unique(y, return_inverse=True)
    keep = np.isin(y, np.where(np.bincount(y) >= 10)[0])
    X, y = X[keep], y[keep]; g = g[keep] if g is not None else None
    _, y = np.unique(y, return_inverse=True)
    if len(y) > N_CAP:
        rng = np.random.default_rng(0)
        idx = np.concatenate([rng.choice(np.where(y == c)[0], max(10, int(round(N_CAP * np.mean(y == c)))), replace=False)
                              for c in np.unique(y)])
        X, y = X[idx], y[idx]; g = g[idx] if g is not None else None
    return X, y, g


def get_splits(y, g):
    if g is None:
        return list(RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=0).split(np.zeros(len(y)), y))
    out = []
    for r in range(2):
        out += list(StratifiedGroupKFold(5, shuffle=True, random_state=r).split(np.zeros(len(y)), y, g))
    return out


def clean(Xtr, Xte):
    med = np.nanmedian(Xtr, 0); med = np.where(np.isnan(med), 0, med)
    Xtr = np.where(np.isnan(Xtr), med, Xtr); Xte = np.where(np.isnan(Xte), med, Xte)
    keep = Xtr.std(0) > 1e-8
    return Xtr[:, keep].astype(np.float64), Xte[:, keep].astype(np.float64)


def lr():
    return LogisticRegressionCV(Cs=[1e-3, 1e-2, 1e-1, 1, 10], cv=3, max_iter=3000)


def density_index(Z, y, seed):
    acc = {"all": [], "top10": []}
    for a, b in StratifiedKFold(3, shuffle=True, random_state=seed).split(Z, y):
        F = np.nan_to_num(f_classif(Z[a], y[a])[0]); top = np.argsort(-F)[:10]
        acc["all"].append(balanced_acc(y[b], lr().fit(Z[a], y[a]).predict(Z[b])))
        acc["top10"].append(balanced_acc(y[b], lr().fit(Z[a][:, top], y[a]).predict(Z[b][:, top])))
    return float(np.mean(acc["all"]) - np.mean(acc["top10"])), float(np.mean(acc["all"]))


def run_method(m, Xtr, ytr, Xte, Ztr, Zte, C, si):
    if m == "LR":
        return lr().fit(Ztr, ytr).predict(Zte)
    if m == "LRbal":  # same LR, class_weight="balanced" (fair baseline under balanced accuracy)
        return LogisticRegressionCV(Cs=[1e-3, 1e-2, 1e-1, 1, 10], cv=3, max_iter=3000, class_weight="balanced").fit(Ztr, ytr).predict(Zte)
    if m == "XGB":
        return make_head("XGB", C, si).fit(Ztr, ytr).predict(Zte)
    if m == "RFbal":  # random forest, balanced class weights per bootstrap
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=500, class_weight="balanced_subsample", n_jobs=2,
                                      random_state=si).fit(Ztr, ytr).predict(Zte)
    if m == "RSEbal":  # random-subspace ensemble: 50 balanced LRs, each on 500 random features, soft vote
        from sklearn.linear_model import LogisticRegression
        rng = np.random.default_rng(si); P = 0
        for b in range(50):
            cols = rng.choice(Ztr.shape[1], min(500, Ztr.shape[1]), replace=False)
            h = LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced").fit(Ztr[:, cols], ytr)
            P = P + h.predict_proba(Zte[:, cols])
        return h.classes_[np.argmax(P, 1)]
    if m == "XGBbal":  # sample weights inversely proportional to class frequency
        w = (len(ytr) / (C * np.bincount(ytr)))[ytr]
        return make_head("XGB", C, si).fit(Ztr, ytr, sample_weight=w).predict(Zte)
    if m == "TPFNbal-CGf":  # prior-corrected TabPFN: argmax p(c|x) / p(c)
        R = Reduction("CGf").fit(Xtr, ytr); A, B = R.transform(Xtr), R.transform(Xte)
        h = make_head("TPFN", C, si).fit(A, ytr)
        return np.asarray(h.classes_)[np.argmax(h.predict_proba(B) / np.bincount(ytr)[np.asarray(h.classes_)], 1)]
    if m == "TICL2-CGf":  # TabICL v2 on CGf representatives; returns raw and prior-corrected predictions
        R = Reduction("CGf").fit(Xtr, ytr); A, B = R.transform(Xtr), R.transform(Xte)
        h = make_head("TICL", C, si).fit(A, ytr)
        P = np.vstack([h.predict_proba(B[i:i + 256]) for i in range(0, len(B), 256)]); cl = np.asarray(h.classes_)
        return cl[np.argmax(P, 1)], cl[np.argmax(P / np.bincount(ytr)[cl], 1)]
    if m in ("TR2m", "KTSP", "TPFN-CGf"):
        R = Reduction("CGf").fit(Xtr, ytr); A, B = R.transform(Xtr), R.transform(Xte)
        h = ThresholdRules(k=2, thr="mid", seed=si) if m == "TR2m" else KTSP(seed=si) if m == "KTSP" else make_head("TPFN", C, si)
        return h.fit(A, ytr).predict(B)
    if m.startswith("HYB"):
        K = 50 if m == "HYB-K50" else 20
        p, nnz, r = hyb_pipeline(Ztr, ytr, Zte, K, "l1" if "L1" in m else "l2", si)
        return p
    if m == "WIDE8k":
        from tabpfnwide.classifier import TabPFNWideClassifier
        F = np.nan_to_num(f_classif(Ztr, ytr)[0]); cols = np.argsort(-F)[:8000]
        return TabPFNWideClassifier(model_name="wide-v2-8k", device="cuda").fit(Ztr[:, cols], ytr).predict(Zte[:, cols])
    if m == "WIDE8kbal":
        from tabpfnwide.classifier import TabPFNWideClassifier
        F = np.nan_to_num(f_classif(Ztr, ytr)[0]); cols = np.argsort(-F)[:8000]
        h = TabPFNWideClassifier(model_name="wide-v2-8k", device="cuda").fit(Ztr[:, cols], ytr)
        P = np.asarray(h.predict_proba(Zte[:, cols])); cl = np.asarray(getattr(h, "classes_", np.arange(P.shape[1])))
        return cl[np.argmax(P / np.bincount(ytr)[cl], 1)]
    raise ValueError(m)


def main(name, methods):
    tag = name.replace("/", "_").replace(":", "_")
    out = os.path.join(os.environ.get("P4_OUT", os.path.join(HERE, "..", "results")), f"p4_{tag}.jsonl")
    done = set()
    if os.path.exists(out):
        for l in open(out):
            r = json.loads(l); done.add((r["split"], r["method"]))
    X, y, g = get_data(name); C = len(np.unique(y))
    for si, (tr, te) in enumerate(get_splits(y, g)):
        Xtr, Xte = clean(X[tr], X[te])
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        Ztr, Zte = (Xtr - mu) / sd, (Xte - mu) / sd
        if (si, "DENSITY") not in done and not os.environ.get("SKIP_DENSITY"):
            D, a_all = density_index(Ztr, y[tr], si)
            open(out, "a").write(json.dumps({"ds": name, "split": si, "method": "DENSITY", "D": D, "inner_lr_all": a_all,
                                             "n": int(len(y)), "d": int(X.shape[1]), "C": C, "grouped": g is not None}) + "\n")
        for m in methods:
            if (si, m) in done:
                continue
            t0 = time.time(); rec = {"ds": name, "split": si, "method": m}
            try:
                out_ = run_method(m, Xtr, y[tr], Xte, Ztr, Zte, C, si)
                if isinstance(out_, tuple):  # raw + balanced variants from one fit
                    rec["bacc"] = balanced_acc(y[te], np.asarray(out_[0]).astype(int))
                    rec2 = dict(rec, method=m.replace("-", "bal-", 1), bacc=balanced_acc(y[te], np.asarray(out_[1]).astype(int)))
                    open(out, "a").write(json.dumps(rec2) + "\n")
                else:
                    rec["bacc"] = balanced_acc(y[te], np.asarray(out_).ravel().astype(int))
            except Exception as e:
                rec["error"] = f"{type(e).__name__}: {str(e)[:200]}"
            rec["t"] = time.time() - t0
            open(out, "a").write(json.dumps(rec) + "\n")
        print(name, si, flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2].split(",") if len(sys.argv) > 2 else ALL_METHODS)
