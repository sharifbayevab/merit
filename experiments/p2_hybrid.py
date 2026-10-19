"""P2: teacher-guided threshold rules (hybrid) vs plain rules, k-TSP and the teacher itself.
Same 15 splits and CGf/F500 reductions as P1. GPU teachers (TabPFN v2, TabICL v2) -> run on the Spark.
Usage: python p2_hybrid.py <dataset>  -> ../results/p2_<dataset>.jsonl (resumable)
"""
import json, os, sys, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
from common import load, splits, balanced_acc
from methods import Reduction, make_head, KTSP, TeacherRules, ThresholdRules

HEADS = ["KTSP", "T-PFN-TR2", "T-PFN-TR3", "T-PFN-TR3m", "T-ICL-TR3", "T-PFN-TR3-c8"]
REDS = ["CGf", "F500"]

def build(h, C, si):
    if h == "KTSP":
        return KTSP(seed=si)
    teacher = make_head("TPFN" if "PFN" in h else "TICL", C, seed=si)
    k = int(h.split("TR")[1][0])
    return TeacherRules(teacher, k=k, seed=si, thr="mid" if h.endswith("m") else "opt",
                        conf=0.8 if h.endswith("c8") else 0.0)

def main(ds):
    out = os.path.join(os.path.dirname(__file__), "..", "results", f"p2_{ds}.jsonl")
    done = set()
    if os.path.exists(out):
        for l in open(out):
            r = json.loads(l); done.add((r["split"], r["red"], r["head"]))
    X, y = load(ds); C = len(np.unique(y))
    for si, (tr, te) in enumerate(splits(y)):
        for red in REDS:
            todo = [h for h in HEADS if (si, red, h) not in done]
            if not todo:
                continue
            R = Reduction(red).fit(X[tr], y[tr]); Ztr, Zte = R.transform(X[tr]), R.transform(X[te])
            for h in todo:
                t0 = time.time(); rec = {"ds": ds, "split": si, "red": red, "head": h}
                try:
                    m = build(h, C, si).fit(Ztr, y[tr])
                    p = np.asarray(m.predict(Zte)).ravel().astype(int)
                    rec.update(acc=float(np.mean(p == y[te])), bacc=balanced_acc(y[te], p))
                    if isinstance(m, TeacherRules):
                        pt = np.asarray(m.teacher.predict(Zte)).ravel().astype(int)
                        rec["teacher_bacc"] = balanced_acc(y[te], pt)
                        rec["agree_teacher"] = float(np.mean(pt == p))
                        rec["n_aug"] = m.n_aug
                except Exception as e:
                    rec["error"] = f"{type(e).__name__}: {str(e)[:200]}"
                rec["t_head"] = time.time() - t0
                open(out, "a").write(json.dumps(rec) + "\n")
        print(ds, "split", si, "done", flush=True)

if __name__ == "__main__":
    main(sys.argv[1])
