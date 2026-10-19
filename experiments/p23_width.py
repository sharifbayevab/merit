"""P23: input-width sensitivity of TabPFN v2 / TabICL v2 on the held-out TabBench-Bio tables (reviewer point 3).

Same splits, cleaning and prior correction as P4 (TPFNbal-CGf / TICL2bal-CGf). Inputs:
  CG{k}: correlation-group representatives as in P4, but keeping k groups; prefilter = max(2000, 4k) top-F features
         (with fewer than k groups, all groups are kept; the realised width is logged).
  F{k}:  the k features with the largest ANOVA F.
k in {500, 1000, 2000}. Methods: PFN-CG1000, ICL-F2000, ...  Output: results/p23/p23_<ds>.jsonl (balanced accuracy).
Usage: python p23_width.py <ds> <PFN|ICL> [k,k,...]
"""
import json, os, sys, time, warnings
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
warnings.filterwarnings("ignore")
import torch
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(float(os.environ.get("GPU_FRAC", "0.35")))
from sklearn.feature_selection import f_classif
import methods
from methods import Reduction, make_head
from common import balanced_acc
from p4_bench import get_data, get_splits, clean

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results", "p23"); os.makedirs(OUT, exist_ok=True)


def columns(kind, k, Xtr, ytr):
    if kind == "F":
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        F = np.nan_to_num(f_classif((Xtr - mu) / sd, ytr)[0])
        return None, np.argsort(-F)[:k]
    methods.P_PREFILTER, methods.R_KEEP = max(2000, 4 * k), k
    R = Reduction("CGf").fit(Xtr, ytr)
    return R, None


def run(model, kind, k, Xtr, ytr, Xte, C, si):
    R, cols = columns(kind, k, Xtr, ytr)
    if R is not None:
        A, B = R.transform(Xtr), R.transform(Xte)
    else:
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        A, B = ((Xtr - mu) / sd)[:, cols], ((Xte - mu) / sd)[:, cols]
    h = make_head("TPFN" if model == "PFN" else "TICL", C, si).fit(A, ytr)
    step = 64 if A.shape[1] > 1000 else 128
    P = np.vstack([h.predict_proba(B[i:i + step]) for i in range(0, len(B), step)]); cl = np.asarray(h.classes_)
    return cl[np.argmax(P / np.bincount(ytr)[cl], 1)], A.shape[1]


def main(ds, model, ks):
    out = os.path.join(OUT, f"p23_{ds.replace('/', '_').replace(':', '_')}.jsonl")
    done = set()
    if os.path.exists(out):
        for l in open(out):
            r = json.loads(l); done.add((r["split"], r["method"]))
    X, y, g = get_data(ds); C = len(np.unique(y))
    for si, (tr, te) in enumerate(get_splits(y, g)):
        if si >= int(os.environ.get("MAX_SPLITS", "99")):
            break
        Xtr, Xte = clean(X[tr], X[te])
        for k in ks:
            for kind in ("CG", "F"):
                m = f"{model}-{kind}{k}"
                if (si, m) in done or k > Xtr.shape[1]:
                    continue
                t0 = time.time(); rec = {"ds": ds, "split": si, "method": m}
                try:
                    pred, w = run(model, kind, k, Xtr, y[tr], Xte, C, si)
                    rec.update(bacc=balanced_acc(y[te], np.asarray(pred).astype(int)), width=int(w))
                except Exception as e:
                    rec["error"] = f"{type(e).__name__}: {str(e)[:200]}"
                rec["t"] = time.time() - t0
                open(out, "a").write(json.dumps(rec) + "\n")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        print(ds, model, si, flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], [int(v) for v in (sys.argv[3] if len(sys.argv) > 3 else "500,1000,2000").split(",")])
