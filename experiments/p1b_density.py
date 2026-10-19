"""P1b: signal density diagnostic + two hybrids of dense LR and sparse Mirzaev rules.
(1) density curve: LR on top-k F features, k in KS (inside each training fold);
(2) SEL: inner 3-fold CV picks ALL:LR or CGf:TR2m;
(3) STACK: inner-CV out-of-fold scores of both -> logistic combiner.
Binary and multi-class (combiner on concatenated per-class scores)."""
import json, os, sys, warnings
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
warnings.filterwarnings("ignore")
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.model_selection import StratifiedKFold
from common import load, splits, balanced_acc
from methods import Reduction, ThresholdRules

KS = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000]

def lr():
    return LogisticRegressionCV(Cs=[1e-3, 1e-2, 1e-1, 1, 10], cv=3, max_iter=5000)

def lr_scores(Xa, ya, Xb):
    mu, sd = Xa.mean(0), Xa.std(0) + 1e-9
    m = lr().fit((Xa - mu) / sd, ya)
    return m.predict_proba((Xb - mu) / sd), m

def rule_scores(Xa, ya, Xb):
    R = Reduction("CGf").fit(Xa, ya)
    m = ThresholdRules(k=2, thr="mid").fit(R.transform(Xa), ya)
    return m.decision(R.transform(Xb)), m

def main(ds):
    out = os.path.join(os.path.dirname(__file__), "..", "results", f"p1b_{ds}.jsonl")
    if os.path.exists(out): os.remove(out)
    X, y = load(ds); classes = np.unique(y)
    for si, (tr, te) in enumerate(splits(y)):
        rec = {"ds": ds, "split": si}
        Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        Ztr, Zte = (Xtr - mu) / sd, (Xte - mu) / sd
        F = np.nan_to_num(f_classif(Ztr, ytr)[0]); order = np.argsort(-F)
        for k in KS + [X.shape[1]]:
            if k > X.shape[1]: continue
            c = order[:k]; m = lr().fit(Ztr[:, c], ytr)
            rec[f"lr_top{k if k < X.shape[1] else 'all'}"] = balanced_acc(yte, m.predict(Zte[:, c]))
        # inner CV out-of-fold scores
        skf = StratifiedKFold(3, shuffle=True, random_state=si)
        oof_l = np.zeros((len(tr), len(classes))); oof_r = None
        for a, b in skf.split(Xtr, ytr):
            pl, _ = lr_scores(Xtr[a], ytr[a], Xtr[b]); oof_l[b] = pl
            pr, _ = rule_scores(Xtr[a], ytr[a], Xtr[b])
            if oof_r is None: oof_r = np.zeros((len(tr), pr.shape[1]))
            oof_r[b] = pr
        acc_l = balanced_acc(ytr, classes[oof_l.argmax(1)])
        pr_lab = (np.where(oof_r[:, 0] > 0, classes[1], classes[0]) if len(classes) == 2 else classes[oof_r.argmax(1)])
        acc_r = balanced_acc(ytr, pr_lab)
        Pl, _ = lr_scores(Xtr, ytr, Xte); Pr, _ = rule_scores(Xtr, ytr, Xte)
        pred_l = classes[Pl.argmax(1)]
        pred_r = np.where(Pr[:, 0] > 0, classes[1], classes[0]) if len(classes) == 2 else classes[Pr.argmax(1)]
        rec["LR_all"] = balanced_acc(yte, pred_l); rec["rules"] = balanced_acc(yte, pred_r)
        rec["SEL"] = rec["rules"] if acc_r > acc_l else rec["LR_all"]; rec["sel_rules"] = bool(acc_r > acc_l)
        comb = LogisticRegression(C=1.0, max_iter=2000).fit(np.hstack([np.log(oof_l + 1e-6), oof_r]), ytr)
        rec["STACK"] = balanced_acc(yte, comb.predict(np.hstack([np.log(Pl + 1e-6), Pr])))
        open(out, "a").write(json.dumps(rec) + "\n")
        print(ds, si, {k: round(v, 3) for k, v in rec.items() if k in ("LR_all", "rules", "SEL", "STACK")}, flush=True)

if __name__ == "__main__":
    main(sys.argv[1])
