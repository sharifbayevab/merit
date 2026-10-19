"""P11: sample-size curve. Same test folds as P4; the training fold is subsampled to m objects per class
(m in {5, 10, 20, 40}, capped by availability; stratified, seed = split). No method is fine-tuned on the data:
TFMs are used as released (in-context only), all predictions class-balanced.
Methods: HYB-K20, HYB-ENS6 (Mirzaev rules over OOF module scores; vote over 6 k-means partitions),
TR2m (original rules on CGf representatives), LRbal, TPFNbal-CGf, TICL2bal-CGf.
Usage: python p11_small_n.py <dataset>
"""
import json, os, sys, time, warnings
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
warnings.filterwarnings("ignore")
import torch
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(float(os.environ.get("MEMFRAC", "0.10")))
from sklearn.linear_model import LogisticRegressionCV
from common import balanced_acc
from methods import Reduction, ThresholdRules, make_head
from p4_bench import get_data, get_splits, clean
from p9_ablation import hybrid_decision, to_pred

HERE = os.path.dirname(os.path.abspath(__file__))
MS = [5, 10, 20, 40, 80]


def tfm_bal(kind, A, B, ytr, C, seed):
    h = make_head(kind, C, seed).fit(A, ytr)
    P = np.vstack([h.predict_proba(B[i:i + 256]) for i in range(0, len(B), 256)])
    cl = np.asarray(h.classes_)
    return cl[np.argmax(P / np.bincount(ytr)[cl], 1)]


def run(m, Xtr, ytr, Xte, Ztr, Zte, C, si):
    if m == "HYB-K20":
        return to_pred(*hybrid_decision(Ztr, ytr, Zte, K=20, seed=si))
    if m.startswith("MR-K"):  # single partition with K modules (K=1: one module over all features)
        return to_pred(*hybrid_decision(Ztr, ytr, Zte, K=int(m[4:]), seed=si))
    if m == "HYB-ENS6NF":  # final MERIT: six partitions, noise-floor rule selection
        dec, cl = 0, None
        for i in range(6):
            K = [10, 20, 40][i % 3]
            cl, d = hybrid_decision(Ztr, ytr, Zte, K=K, seed=si * 100 + i // 3, qmin="auto")
            dec = dec + d
        return to_pred(cl, dec)
    if m == "HYB-ENS6":
        dec, cl = 0, None
        for i in range(6):
            K = [10, 20, 40][i % 3]
            cl, d = hybrid_decision(Ztr, ytr, Zte, K=K, seed=si * 100 + i // 3)
            dec = dec + d
        return to_pred(cl, dec)
    if m == "LRbal":
        return LogisticRegressionCV(Cs=[1e-3, 1e-2, 1e-1, 1, 10], cv=3, max_iter=3000, class_weight="balanced").fit(Ztr, ytr).predict(Zte)
    R = Reduction("CGf").fit(Xtr, ytr); A, B = R.transform(Xtr), R.transform(Xte)
    if m == "TR2m":
        return ThresholdRules(k=2, thr="mid", seed=si).fit(A, ytr).predict(B)
    if m == "TPFNbal-CGf":
        return tfm_bal("TPFN", A, B, ytr, C, si)
    if m == "TICL2bal-CGf":
        return tfm_bal("TICL", A, B, ytr, C, si)
    raise ValueError(m)


def main(name, methods):
    out = os.path.join(HERE, "..", "results", "p11", f"p11_{name.replace('/', '_').replace(':', '_')}.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    done = set()
    if os.path.exists(out):
        for l in open(out):
            r = json.loads(l); done.add((r["split"], r["m"], r["method"]))
    X, y, g = get_data(name)
    for si, (tr, te) in enumerate(get_splits(y, g)):
        rng = np.random.default_rng(si)
        for m in MS:
            idx = np.concatenate([rng.permutation(tr[y[tr] == c])[:m] for c in np.unique(y[tr])])
            if np.bincount(y[idx]).min() < 5 or len(np.unique(y[idx])) < len(np.unique(y)):
                continue
            todo = [me for me in methods if (si, m, me) not in done]
            if not todo: continue
            Xtr, Xte = clean(X[idx], X[te]); mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
            Ztr, Zte = (Xtr - mu) / sd, (Xte - mu) / sd
            ytr = y[idx]; _, ytr_enc = np.unique(ytr, return_inverse=True)  # labels already 0..C-1
            C = len(np.unique(ytr))
            for me in todo:
                t0 = time.time(); rec = {"ds": name, "split": si, "m": m, "n_train": int(len(idx)), "method": me}
                try:
                    rec["bacc"] = balanced_acc(y[te], np.asarray(run(me, Xtr, ytr, Xte, Ztr, Zte, C, si)).astype(int))
                except Exception as e:
                    rec["error"] = f"{type(e).__name__}: {str(e)[:200]}"
                rec["t"] = time.time() - t0
                open(out, "a").write(json.dumps(rec) + "\n")
        print(name, si, flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2].split(",") if len(sys.argv) > 2 else
         ["TR2m", "LRbal", "HYB-K20", "HYB-ENS6", "TPFNbal-CGf", "TICL2bal-CGf"])
