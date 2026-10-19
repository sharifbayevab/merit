"""P14: are MERIT's gains driven by interactions between modules?

For every split (P4 protocol) we fit MERIT-1 (K=20 k-means modules, out-of-fold L2 module scores, 2-D threshold
rules with midpoint thresholds) and measure on the TEST fold, for the 10 heaviest rules of each class task:
  ba_pair   balanced accuracy of the 2-D rule alone
  ba_single best balanced accuracy of a 1-D Fisher/midpoint rule on either of its two modules alone (fitted on train)
  synergy = ba_pair - ba_single
We also fit an additive control on the same module scores: class-balanced logistic regression (MODLR) and
record the test balanced accuracy of MERIT-1 and MODLR.
Output: ../results/p14/p14_<ds>.jsonl (one record per split) and, for split 0, the module-score matrices of the
top rule (for the case-study figure).
"""
import json, os, sys, warnings
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
warnings.filterwarnings("ignore")
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from common import balanced_acc
from methods import ThresholdRules
from p4_bench import get_data, get_splits, clean
from p9_ablation import module_scores

HERE = os.path.dirname(os.path.abspath(__file__))


def bacc_bin(yb, pred):
    return 0.5 * (np.mean(pred[yb == 1] == 1) + np.mean(pred[yb == 0] == 0))


def rule_1d(a_tr, yb_tr):
    """1-D Fisher rule on one score: orientation by class means, midpoint threshold."""
    m1, m0 = a_tr[yb_tr == 1].mean(), a_tr[yb_tr == 0].mean()
    sgn = 1.0 if m1 >= m0 else -1.0
    return sgn, 0.5 * (m1 + m0) * sgn


def main(name, K=20):
    out = os.path.join(HERE, "..", "results", "p14", f"p14_{name.replace('/', '_').replace(':', '_')}.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if os.path.exists(out): os.remove(out)
    X, y, g = get_data(name)
    for si, (tr, te) in enumerate(get_splits(y, g)):
        Xtr, Xte = clean(X[tr], X[te]); mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        Ztr, Zte = (Xtr - mu) / sd, (Xte - mu) / sd; ytr, yte = y[tr], y[te]
        lab = MiniBatchKMeans(K, random_state=si, n_init=3, batch_size=4096).fit(Ztr.T / np.sqrt(len(tr))).labels_
        oof = None
        for a, b in StratifiedKFold(3, shuffle=True, random_state=si).split(Ztr, ytr):
            s = module_scores(Ztr[a], ytr[a], Ztr[b], lab, K, 0.1)
            if oof is None: oof = np.zeros((len(ytr), s.shape[1]))
            oof[b] = s
        Ste = module_scores(Ztr, ytr, Zte, lab, K, 0.1)
        m_, s_ = oof.mean(0), oof.std(0) + 1e-9
        A, B = (oof - m_) / s_, (Ste - m_) / s_
        r = ThresholdRules(k=2, thr="mid", seed=si).fit(A, ytr)
        ba_merit = balanced_acc(yte, r.predict(B))
        modlr = LogisticRegressionCV(Cs=[1e-3, 1e-2, 1e-1, 1, 10], cv=3, max_iter=3000, class_weight="balanced").fit(A, ytr)
        ba_modlr = balanced_acc(yte, modlr.predict(B))
        syn, pairs, singles, case = [], [], [], None
        for c, chosen, alphas in r.models:
            yb_tr, yb_te = (ytr == c).astype(int), (yte == c).astype(int)
            if yb_te.sum() == 0 or yb_te.sum() == len(yb_te): continue
            order = np.argsort(-alphas)[:10]
            for idx in order:
                q, S, w, t = chosen[idx]
                bp = bacc_bin(yb_te, (B[:, S] @ w > t).astype(int))
                bs = []
                for f in S:
                    sgn, th = rule_1d(A[:, f], yb_tr)
                    bs.append(bacc_bin(yb_te, (sgn * B[:, f] > th).astype(int)))
                syn.append(bp - max(bs)); pairs.append(bp); singles.append(max(bs))
                if case is None and si == 0:
                    case = {"S": [int(v) for v in S], "w": [float(v) for v in w], "t": float(t), "cls": int(c),
                            "A": A[:, S].round(4).tolist(), "ytr": (ytr == c).astype(int).tolist(),
                            "B": B[:, S].round(4).tolist(), "yte": yb_te.tolist(), "ba_pair": float(bp), "ba_single": [float(v) for v in bs]}
        rec = {"ds": name, "split": si, "ba_merit1": ba_merit, "ba_modlr": ba_modlr,
               "synergy": float(np.mean(syn)) if syn else None, "ba_pair": float(np.mean(pairs)) if pairs else None,
               "ba_single": float(np.mean(singles)) if singles else None, "frac_pos_syn": float(np.mean(np.array(syn) > 0)) if syn else None}
        if case is not None: rec["case"] = case
        open(out, "a").write(json.dumps(rec) + "\n")
        print(name, si, round(ba_merit, 3), round(ba_modlr, 3), None if not syn else round(float(np.mean(syn)), 3), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
