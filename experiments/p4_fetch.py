"""P4: fetch all enabled TabBench-Bio classification datasets and save X/y as .npz (our protocol)."""
import json, os, sys, time, traceback
import numpy as np
from tabbench_bio import load_bio_as_dataset
# REG: dataset registry of a TabBench-Bio checkout (https://github.com/not-a-feature/TabBench-Bio); OUT: where the .npz files go
HERE = os.path.dirname(os.path.abspath(__file__))
REG = os.environ.get("TB_REGISTRY", "TabBench-Bio/src/tabbench_bio/bio/data/bio_datasets.json")
OUT = os.environ.get("TB_DIR", os.path.join(HERE, "..", "data_tb")); os.makedirs(OUT, exist_ok=True)
items = [i for i in json.load(open(REG)) if i.get("problem_type") in ("binary", "multiclass") and i.get("enabled", True)]
only = sys.argv[1:]
for it in items:
    if only and it["bio_id"] not in only: continue
    bid = it["bio_id"]; f = os.path.join(OUT, bid.replace("/", "_") + ".npz")
    if os.path.exists(f): continue
    t = time.time()
    try:
        ds = load_bio_as_dataset(bid, cache_dir=os.environ.get("TB_CACHE", os.path.expanduser("~/.cache/tabbench_bio")))
        X = np.asarray(ds.features, dtype="float32"); y = np.asarray(ds.targets).ravel().astype(str)
        g = np.asarray(ds.groups).astype(str) if getattr(ds, "groups", None) is not None else np.array([])
        np.savez_compressed(f, X=X, y=y, groups=g, feature_names=np.asarray(ds.feature_names, dtype=str))
        print("OK", bid, X.shape, len(set(y)), f"{time.time()-t:.0f}s", flush=True)
    except Exception as e:
        print("FAIL", bid, type(e).__name__, str(e)[:200], flush=True)
