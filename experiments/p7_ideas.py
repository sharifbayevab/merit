"""P7: Mirzaev ideas inside SOTA models (dev sets first; P4 protocol and data loading).

Methods (all TFMs prior-corrected = balanced; predictions in chunks of 256 rows):
  MOD-TPFN / MOD-TICL : out-of-fold supervised module scores (K=20, L2) -> TabPFN v2 / TabICL v2   (I1)
  GSB-TPFN            : group-stratified feature bagging, 4 bags x 500 groups -> TabPFN           (I2)
  RSB-TPFN            : random-subset bagging, 4 bags x 500 features from the same top-2000 pool   (I2 control)
  ORD-TICL / RND-TICL : TabICL on the 500 top-F features, group-contiguous vs random column order  (I3)
  HYB-ENS             : hybrid rules averaged over module partitions K in {10,20,40} x 2 seeds   (I4, CPU)
  FUSE-TICL           : late fusion of HYB-K20 rule decision and MOD-TICL log-probabilities       (I5)
Usage: python p7_ideas.py <dataset> <methods,comma,sep>
"""
import json, os, sys, time, warnings
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
warnings.filterwarnings("ignore")
from scipy.cluster.hierarchy import linkage, fcluster, leaves_list
from scipy.spatial.distance import squareform
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from common import balanced_acc
from methods import ThresholdRules, make_head
from p4_bench import get_data, get_splits, clean
from p6_interp import mod_models, scores

HERE = os.path.dirname(os.path.abspath(__file__))


def tfm(kind, C, seed):
    return make_head("TPFN" if kind == "PFN" else "TICL", C, seed)


def balanced_proba(h, A, B, ytr):
    h.fit(A, ytr)
    P = np.vstack([h.predict_proba(B[i:i + 256]) for i in range(0, len(B), 256)])
    cl = np.asarray(h.classes_)
    P = P / np.bincount(ytr)[cl]
    return cl, P / P.sum(1, keepdims=True)


def module_features(Ztr, ytr, Zte, K, seed):
    lab = MiniBatchKMeans(K, random_state=seed, n_init=3, batch_size=4096).fit(Ztr.T / np.sqrt(len(Ztr))).labels_
    oof = None
    for a, b in StratifiedKFold(3, shuffle=True, random_state=seed).split(Ztr, ytr):
        s, _ = scores(mod_models(Ztr[a], ytr[a], lab, K, "l2"), Ztr[b])
        if oof is None: oof = np.zeros((len(ytr), s.shape[1]))
        oof[b] = s
    Ste, _ = scores(mod_models(Ztr, ytr, lab, K, "l2"), Zte)
    mu, sd = oof.mean(0), oof.std(0) + 1e-9
    return (oof - mu) / sd, (Ste - mu) / sd


def groups(Ztr, ytr, P=2000, r_cut=0.7):
    F = np.nan_to_num(f_classif(Ztr, ytr)[0]); pre = np.argsort(-F)[:min(P, Ztr.shape[1])]
    C = np.nan_to_num(np.corrcoef(Ztr[:, pre].T)); D = np.clip(1 - np.abs(C), 0, None); np.fill_diagonal(D, 0)
    L = linkage(squareform((D + D.T) / 2, checks=False), "average")
    lab = fcluster(L, 1 - r_cut, "distance")
    gs = [pre[lab == g] for g in np.unique(lab)]
    gs.sort(key=lambda m: -F[m].max())
    return gs, F, pre, L


def hybrid_decision(Ztr, ytr, Zte, K, seed):
    A, B = module_features(Ztr, ytr, Zte, K, seed)
    r = ThresholdRules(k=2, thr="mid", seed=seed).fit(A, ytr)
    return r.classes, r.decision(B), r.decision(A)


def run(m, Ztr, ytr, Zte, C, si):
    rng = np.random.default_rng(si)
    if m in ("MOD-TPFN", "MOD-TICL"):
        A, B = module_features(Ztr, ytr, Zte, 20, si)
        cl, P = balanced_proba(tfm(m[4:7], C, si), A, B, ytr)
        return cl[P.argmax(1)]
    if m in ("GSB-TPFN", "RSB-TPFN"):
        gs, F, pre, _ = groups(Ztr, ytr)
        top = gs[:500]
        Ps = []
        for b in range(4):
            if m == "GSB-TPFN":
                cols = np.array([g[np.argmax(F[g])] if b == 0 else rng.choice(g) for g in top])
            else:
                cols = rng.choice(pre, min(500, len(pre)), replace=False)
            cl, P = balanced_proba(tfm("PFN", C, si + b), Ztr[:, cols], Zte[:, cols], ytr)
            Ps.append(P)
        return cl[np.mean(Ps, 0).argmax(1)]
    if m in ("ORD-TICL", "RND-TICL"):
        F = np.nan_to_num(f_classif(Ztr, ytr)[0]); cols = np.argsort(-F)[:500]
        if m == "ORD-TICL":  # group-contiguous order: leaves of the average-linkage tree on |r|
            Cm = np.nan_to_num(np.corrcoef(Ztr[:, cols].T)); D = np.clip(1 - np.abs(Cm), 0, None); np.fill_diagonal(D, 0)
            cols = cols[leaves_list(linkage(squareform((D + D.T) / 2, checks=False), "average"))]
        else:
            cols = rng.permutation(cols)
        cl, P = balanced_proba(tfm("ICL", C, si), Ztr[:, cols], Zte[:, cols], ytr)
        return cl[P.argmax(1)]
    if m == "HYB-ENS":
        dec, cl = 0, None
        for K in (10, 20, 40):
            for s in (0, 1):
                cl, d, _ = hybrid_decision(Ztr, ytr, Zte, K, si * 10 + s)
                dec = dec + d
        return np.where(dec[:, 0] > 0, cl[1], cl[0]) if len(cl) == 2 else cl[np.argmax(dec, 1)]
    if m == "FUSE-TICL":
        A, B = module_features(Ztr, ytr, Zte, 20, si)
        r = ThresholdRules(k=2, thr="mid", seed=si).fit(A, ytr)
        cl, P = balanced_proba(tfm("ICL", C, si), A, B, ytr)
        d = r.decision(B)
        rule_logit = np.hstack([-d, d]) if len(cl) == 2 else d
        z = np.log(P + 1e-6) + 2.0 * rule_logit  # fixed equal-scale fusion (rule votes lie in [-1, 1])
        return cl[z.argmax(1)]
    raise ValueError(m)


def main(name, methods):
    tag = name.replace("/", "_").replace(":", "_")
    out = os.path.join(os.environ.get("P7_OUT", os.path.join(HERE, "..", "results")), f"p7_{tag}.jsonl")
    done = set()
    if os.path.exists(out):
        for l in open(out):
            r = json.loads(l); done.add((r["split"], r["method"]))
    X, y, g = get_data(name); C = len(np.unique(y))
    for si, (tr, te) in enumerate(get_splits(y, g)):
        todo = [m for m in methods if (si, m) not in done]
        if not todo: continue
        Xtr, Xte = clean(X[tr], X[te]); mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        Ztr, Zte = (Xtr - mu) / sd, (Xte - mu) / sd
        for m in todo:
            t0 = time.time(); rec = {"ds": name, "split": si, "method": m}
            try:
                rec["bacc"] = balanced_acc(y[te], np.asarray(run(m, Ztr, y[tr], Zte, C, si)).astype(int))
            except Exception as e:
                rec["error"] = f"{type(e).__name__}: {str(e)[:200]}"
            rec["t"] = time.time() - t0
            open(out, "a").write(json.dumps(rec) + "\n")
        print(name, si, flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2].split(","))
