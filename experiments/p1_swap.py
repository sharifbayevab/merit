"""P1: reduction x head swap matrix on 8 HDLSS datasets (15 CV splits each).

Question: where does a Mirzaev-style pipeline (correlation groups -> representative
features -> k-dim threshold rules) lose accuracy relative to strong heads
(LR, XGB, EBM, TabPFN v2, TabICL v2)? Reduction gap vs head gap vs rule-selection optimism.
Usage: python p1_swap.py <dataset>   (results appended to ../results/p1_<dataset>.jsonl, resumable)
"""
import json, os, sys, time, warnings
os.environ.setdefault("OMP_NUM_THREADS", "2"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
import numpy as np
warnings.filterwarnings("ignore")
from common import load, splits, balanced_acc
from methods import Reduction, make_head, ThresholdRules

CONFIG = {
    "ALL":  ["LR", "RF", "XGB", "TICL"],
    "F500": ["LR", "RF", "XGB", "EBM", "TR1", "TR2", "TR3", "TR2h", "TR1m", "TR2m", "TR3m", "TR2hm", "TPFN", "TICL"],
    "V500": ["LR", "TR2", "TR2m", "TPFN"],
    "CGf":  ["LR", "RF", "XGB", "EBM", "TR1", "TR2", "TR3", "TR2h", "TR1m", "TR2m", "TR3m", "TR2hm", "TPFN", "TICL"],
    "CGm":  ["LR", "TR2", "TR2m", "TPFN"],
    "CGp":  ["LR", "TR2", "TR2m", "TPFN"],
}

EXCLUDE = set(filter(None, os.environ.get("P1_EXCLUDE", "").split(",")))
ONLY = set(filter(None, os.environ.get("P1_ONLY", "").split(",")))
TAG = os.environ.get("P1_TAG", "")


def main(ds):
    out = os.path.join(os.path.dirname(__file__), "..", "results", f"p1{TAG}_{ds}.jsonl")
    done = set()
    if os.path.exists(out):
        for l in open(out):
            r = json.loads(l); done.add((r["split"], r["red"], r["head"]))
    X, y = load(ds)
    C = len(np.unique(y))
    for si, (tr, te) in enumerate(splits(y)):
        for red, heads in CONFIG.items():
            todo = [h for h in heads if (si, red, h) not in done and h not in EXCLUDE and (not ONLY or h in ONLY)]
            if not todo:
                continue
            t0 = time.time()
            R = Reduction(red).fit(X[tr], y[tr])
            Ztr, Zte = R.transform(X[tr]), R.transform(X[te])
            t_red = time.time() - t0
            for h in todo:
                t0 = time.time()
                rec = {"ds": ds, "split": si, "red": red, "head": h, "d_in": int(Ztr.shape[1]), "t_red": t_red}
                if hasattr(R, "n_groups_total"):
                    rec["n_groups_total"] = R.n_groups_total
                    rec["mean_group_size"] = float(np.mean(R.group_sizes))
                try:
                    m = make_head(h, C, seed=si)
                    m.fit(Ztr, y[tr])
                    p = np.asarray(m.predict(Zte)).ravel().astype(int)
                    rec.update(acc=float(np.mean(p == y[te])), bacc=balanced_acc(y[te], p))
                    if isinstance(m, ThresholdRules):
                        ptr = m.predict(Ztr)
                        rec["train_bacc"] = balanced_acc(y[tr], ptr)
                        rec["rule_train_q"] = float(np.mean(m.diag["train_q_sel"]))
                        # mean held-out quality of the selected individual rules (optimism)
                        qs = []
                        for (c, ch, _), Rte in zip(m.models, m.rule_outputs(Zte)):
                            yb = (y[te] == c).astype(int)
                            for j in range(Rte.shape[1]):
                                pr = (Rte[:, j] > 0).astype(int)
                                qs.append(0.5 * (np.mean(pr[yb == 1] == 1) + np.mean(pr[yb == 0] == 0)))
                        rec["rule_test_q"] = float(np.mean(qs))
                        rec["n_rules"] = int(np.sum(m.diag["n_rules"]))
                        rec["n_feat_used"] = int(len({int(f) for _, ch, _ in m.models for _, S, _, _ in ch for f in S}))
                except Exception as e:
                    rec["error"] = f"{type(e).__name__}: {str(e)[:200]}"
                rec["t_head"] = time.time() - t0
                with open(out, "a") as f:
                    f.write(json.dumps(rec) + "\n")
        print(ds, "split", si, "done", flush=True)

if __name__ == "__main__":
    main(sys.argv[1])
