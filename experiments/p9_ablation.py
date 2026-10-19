"""P9: ablations of the module-rule hybrid (CPU, P4 protocol, dev + held-out sets).

Base: K=20 k-means modules, within-module L2 logistic (C=0.1), out-of-fold module scores,
2-D threshold rules with midpoint thresholds.
Ablations:
  K5 K10 K20 K40 K80        number of modules (single partition)
  RAND20                    random assignment of features to 20 modules (control: is correlation structure needed?)
  ORD1 ORD3                 rule order 1 and 3 instead of 2
  THROPT                    accuracy-maximising thresholds instead of midpoints
  C001 C1                   within-module C = 0.01 / 1
  ENS1 ENS2 ENS4 ENS6 ENS12 number of partitions in the vote (K cycles through 10, 20, 40; seeds added)
  RENS6 RENS12 RENS24       same vote over RANDOM partitions (support sets in the sense of Zhuravlev's estimate-calculation algorithms)
Usage: python p9_ablation.py <dataset> <ablations,comma,sep>
"""
import json, os, sys, time, warnings
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
warnings.filterwarnings("ignore")
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from common import balanced_acc
from methods import ThresholdRules
from p4_bench import get_data, get_splits, clean

HERE = os.path.dirname(os.path.abspath(__file__))


def module_scores(Za, ya, Zb, lab, K, C):
    out = []
    for k in range(K):
        idx = np.where(lab == k)[0]
        if len(idx) == 0: continue
        s = LogisticRegression(C=C, max_iter=3000).fit(Za[:, idx], ya).decision_function(Zb[:, idx])
        out.append(s[:, None] if s.ndim == 1 else s)
    return np.hstack(out)


def hybrid_decision(Ztr, ytr, Zte, K=20, seed=0, C=0.1, order=2, thr="mid", rand=False, qmin=None):
    if rand:
        lab = np.random.default_rng(seed).integers(0, K, Ztr.shape[1])
    else:
        lab = MiniBatchKMeans(K, random_state=seed, n_init=3, batch_size=4096).fit(Ztr.T / np.sqrt(len(Ztr))).labels_
    oof = None
    for a, b in StratifiedKFold(3, shuffle=True, random_state=seed).split(Ztr, ytr):
        s = module_scores(Ztr[a], ytr[a], Ztr[b], lab, K, C)
        if oof is None: oof = np.zeros((len(ytr), s.shape[1]))
        oof[b] = s
    Ste = module_scores(Ztr, ytr, Zte, lab, K, C)
    mu, sd = oof.mean(0), oof.std(0) + 1e-9
    order = min(order, oof.shape[1])  # K=1 binary gives a single module score -> one-dimensional rules
    r = ThresholdRules(k=order, thr=thr, seed=seed, cand=(None, 60, 25), qmin=qmin).fit((oof - mu) / sd, ytr)
    return r.classes, r.decision((Ste - mu) / sd)


def to_pred(cl, dec):
    return np.where(dec[:, 0] > 0, cl[1], cl[0]) if len(cl) == 2 else cl[np.argmax(dec, 1)]


def run(ab, Ztr, ytr, Zte, si):
    if ab.startswith("K"):
        return to_pred(*hybrid_decision(Ztr, ytr, Zte, K=int(ab[1:]), seed=si))
    if ab == "K20NF":  # noise-floor selection of preferred rules
        return to_pred(*hybrid_decision(Ztr, ytr, Zte, K=20, seed=si, qmin="auto"))
    if ab == "ENS6NF":
        dec, cl = 0, None
        for i in range(6):
            cl, d = hybrid_decision(Ztr, ytr, Zte, K=[10, 20, 40][i % 3], seed=si * 100 + i // 3, qmin="auto")
            dec = dec + d
        return to_pred(cl, dec)
    if ab == "RAND20":
        return to_pred(*hybrid_decision(Ztr, ytr, Zte, rand=True, seed=si))
    if ab in ("ORD1", "ORD3"):
        return to_pred(*hybrid_decision(Ztr, ytr, Zte, order=int(ab[-1]), seed=si))
    if ab == "THROPT":
        return to_pred(*hybrid_decision(Ztr, ytr, Zte, thr="opt", seed=si))
    if ab in ("C001", "C1"):
        return to_pred(*hybrid_decision(Ztr, ytr, Zte, C=0.01 if ab == "C001" else 1.0, seed=si))
    if ab.startswith("RENS"):  # AVO-style: vote over random support partitions of the feature set
        n = int(ab[4:]); Ks = [10, 20, 40]; dec, cl = 0, None
        for i in range(n):
            cl, d = hybrid_decision(Ztr, ytr, Zte, K=Ks[i % 3], seed=si * 1000 + i, rand=True)
            dec = dec + d
        return to_pred(cl, dec)
    if ab.startswith("ENS"):
        n = int(ab[3:]); Ks = [10, 20, 40]; dec, cl = 0, None
        for i in range(n):
            cl, d = hybrid_decision(Ztr, ytr, Zte, K=Ks[i % 3], seed=si * 100 + i // 3)
            dec = dec + d
        return to_pred(cl, dec)
    raise ValueError(ab)


def main(name, abls):
    tag = name.replace("/", "_").replace(":", "_")
    out = os.path.join(os.environ.get("P9_OUT", os.path.join(HERE, "..", "results", "p9")), f"p9_{tag}.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    done = set()
    if os.path.exists(out):
        for l in open(out):
            r = json.loads(l); done.add((r["split"], r["method"]))
    X, y, g = get_data(name)
    for si, (tr, te) in enumerate(get_splits(y, g)):
        todo = [a for a in abls if (si, a) not in done]
        if not todo: continue
        Xtr, Xte = clean(X[tr], X[te]); mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        Ztr, Zte = (Xtr - mu) / sd, (Xte - mu) / sd
        cum = sorted(int(a[4:]) for a in todo if a.startswith("RENS"))
        if cum:  # cumulative AVO-style vote: one pass over max(n) random partitions, snapshots at each requested n
            t0 = time.time(); dec, cl = 0, None; Ks = [10, 20, 40]
            try:
                for i in range(max(cum)):
                    cl, d = hybrid_decision(Ztr, y[tr], Zte, K=Ks[i % 3], seed=si * 1000 + i, rand=True)
                    dec = dec + d
                    if i + 1 in cum:
                        open(out, "a").write(json.dumps({"ds": name, "split": si, "method": f"RENS{i + 1}",
                                                         "bacc": balanced_acc(y[te], np.asarray(to_pred(cl, dec)).astype(int)),
                                                         "t": time.time() - t0}) + "\n")
            except Exception as e:
                open(out, "a").write(json.dumps({"ds": name, "split": si, "method": f"RENS{max(cum)}", "error": str(e)[:200]}) + "\n")
            todo = [a for a in todo if not a.startswith("RENS")]
        for a in todo:
            t0 = time.time(); rec = {"ds": name, "split": si, "method": a}
            try:
                rec["bacc"] = balanced_acc(y[te], np.asarray(run(a, Ztr, y[tr], Zte, si)).astype(int))
            except Exception as e:
                rec["error"] = f"{type(e).__name__}: {str(e)[:200]}"
            rec["t"] = time.time() - t0
            open(out, "a").write(json.dumps(rec) + "\n")
        print(name, si, flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2].split(","))
