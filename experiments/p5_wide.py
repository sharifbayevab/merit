"""P5: TabPFN-Wide (Kolberg et al. 2025) as a strong HDLSS baseline, same 15 splits.
Cells: WIDE8k on F-top-min(d,8000) features (train-fold filter; = ALL when d <= 8000),
WIDE5k on F-top-5000, WIDE8k on CGf representatives (200)."""
import json, os, sys, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
from sklearn.feature_selection import f_classif
from common import load, splits, balanced_acc
from methods import Reduction

def main(ds):
    from tabpfnwide.classifier import TabPFNWideClassifier
    out = os.path.join(os.path.dirname(__file__), "..", "results", f"p5_{ds}.jsonl")
    done = {(json.loads(l)["split"], json.loads(l)["cell"]) for l in open(out)} if os.path.exists(out) else set()
    X, y = load(ds)
    for si, (tr, te) in enumerate(splits(y)):
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        Ztr, Zte = (X[tr] - mu) / sd, (X[te] - mu) / sd
        F = np.nan_to_num(f_classif(Ztr, y[tr])[0]); order = np.argsort(-F)
        cells = {"F8k:WIDE8k": ("wide-v2-8k", order[:8000]), "F5k:WIDE5k": ("wide-v2-5k", order[:5000])}
        for cell, (name, cols) in list(cells.items()) + [("CGf:WIDE8k", ("wide-v2-8k", None))]:
            if (si, cell) in done: continue
            t0 = time.time(); rec = {"ds": ds, "split": si, "cell": cell}
            try:
                if cols is None:
                    R = Reduction("CGf").fit(X[tr], y[tr]); A, B = R.transform(X[tr]), R.transform(X[te])
                else:
                    A, B = Ztr[:, cols], Zte[:, cols]
                m = TabPFNWideClassifier(model_name=name, device="cuda").fit(A, y[tr])
                p = np.asarray(m.predict(B)).astype(int)
                rec.update(bacc=balanced_acc(y[te], p), d_in=int(A.shape[1]))
            except Exception as e:
                rec["error"] = f"{type(e).__name__}: {str(e)[:200]}"
            rec["t"] = time.time() - t0
            open(out, "a").write(json.dumps(rec) + "\n")
        print(ds, si, flush=True)

if __name__ == "__main__":
    main(sys.argv[1])
