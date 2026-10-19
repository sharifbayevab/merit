"""P10: GPU implementation of the module-rule hybrid and its partition ensembles.

All K module logistic regressions of one partition are fitted jointly as one block-structured model on the GPU
(padded tensor n x K x m_max, mask for padding). Objective per module is the sklearn one:
    sum_i CE(module logits_i, y_i) + ||w_k||^2 / (2C)          (binary: one logit, multi-class: C logits)
minimised with L-BFGS (strong Wolfe). Modules are independent, so the joint sum decouples exactly.
k-means on feature profiles is done in torch (k-means++ init, Lloyd iterations).
Rules on top: ThresholdRules (k=2, midpoint thresholds) on out-of-fold module scores, as in P7/P9.

Methods (cumulative where marked):
  gK20              single k-means partition, K=20 (validation against sklearn K20)
  gRAND20           single random partition, K=20
  gENS1..gENS12     vote over k-means partitions, K cycles 10/20/40, seeds added   (cumulative)
  gRENS6..gRENS24   vote over random partitions (support sets of estimate-calculation algorithms) (cumulative)
Usage: python p10_gpu_hybrid.py <dataset>
"""
import json, os, sys, time, warnings
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
warnings.filterwarnings("ignore")
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.cluster import MiniBatchKMeans
from common import balanced_acc
from methods import ThresholdRules
from p4_bench import get_data, get_splits, clean

HERE = os.path.dirname(os.path.abspath(__file__))
DEV = "cuda" if torch.cuda.is_available() else "cpu"
if DEV == "cuda":
    torch.cuda.set_per_process_memory_fraction(float(os.environ.get("MEMFRAC", "0.12")))
torch.backends.cuda.matmul.allow_tf32 = False


def kmeans_torch(P, K, seed, iters=40):
    g = torch.Generator(device=P.device).manual_seed(seed)
    d = P.shape[0]
    cent = P[torch.randint(d, (1,), generator=g, device=P.device)]
    for _ in range(1, K):                                     # k-means++
        dist = torch.cdist(P, cent).min(1).values ** 2
        cent = torch.cat([cent, P[torch.multinomial(dist / dist.sum(), 1, generator=g)]])
    for _ in range(iters):
        lab = torch.cdist(P, cent).argmin(1)
        new = torch.zeros_like(cent).index_add_(0, lab, P)
        cnt = torch.bincount(lab, minlength=K).clamp(min=1).unsqueeze(1)
        cent = torch.where(torch.bincount(lab, minlength=K).unsqueeze(1) > 0, new / cnt, cent)
    return torch.cdist(P, cent).argmin(1).cpu().numpy()


class Blocks:
    """Padded block layout of a partition: index tensor K x m_max and mask."""
    def __init__(self, lab, K):
        groups = [np.where(lab == k)[0] for k in range(K)]
        groups = [g for g in groups if len(g)]
        self.K = len(groups); m = max(len(g) for g in groups)
        idx = np.zeros((self.K, m), dtype=np.int64); mask = np.zeros((self.K, m), dtype=np.float32)
        for k, g in enumerate(groups):
            idx[k, :len(g)] = g; mask[k, :len(g)] = 1
        self.idx = torch.as_tensor(idx, device=DEV); self.mask = torch.as_tensor(mask, device=DEV)

    def gather(self, X):                                      # n x K x m_max (zeros on padding)
        return X[:, self.idx] * self.mask


def fit_modules(Xb, y, ncls, C=0.1, max_iter=300):
    """Joint L-BFGS fit of independent per-module logistic regressions (sklearn objective)."""
    n, K, m = Xb.shape
    out = 1 if ncls == 2 else ncls
    W = torch.zeros(K, m, out, device=DEV, requires_grad=True)
    b = torch.zeros(K, out, device=DEV, requires_grad=True)
    yt = torch.as_tensor(y, device=DEV)
    opt = torch.optim.LBFGS([W, b], lr=1, max_iter=max_iter, tolerance_grad=1e-6, tolerance_change=1e-9,
                            history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        z = torch.einsum("nkm,kmc->nkc", Xb, W) + b          # n x K x out
        if out == 1:
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                z[..., 0], yt.float().unsqueeze(1).expand(n, K), reduction="sum")
        else:
            loss = torch.nn.functional.cross_entropy(z.permute(0, 2, 1), yt.unsqueeze(1).expand(n, K), reduction="sum")
        loss = loss + (W ** 2).sum() / (2 * C)
        loss.backward()
        return loss

    opt.step(closure)
    return W.detach(), b.detach()


def scores(Xb, W, b):
    z = torch.einsum("nkm,kmc->nkc", Xb, W) + b
    return z.reshape(z.shape[0], -1).cpu().numpy()            # n x (K*out) — like stacked decision_function


def hybrid_decision(Ztr_t, ytr, Zte_t, lab, K, seed, ncls):
    B = Blocks(lab, K)
    Xtr, Xte = B.gather(Ztr_t), B.gather(Zte_t)
    oof = None
    for a, bb in StratifiedKFold(3, shuffle=True, random_state=seed).split(np.zeros(len(ytr)), ytr):
        W, b = fit_modules(Xtr[a], ytr[a], ncls)
        s = scores(Xtr[bb], W, b)
        if oof is None: oof = np.zeros((len(ytr), s.shape[1]))
        oof[bb] = s
    W, b = fit_modules(Xtr, ytr, ncls)
    Ste = scores(Xte, W, b)
    mu, sd = oof.mean(0), oof.std(0) + 1e-9
    r = ThresholdRules(k=2, thr="mid", seed=seed).fit((oof - mu) / sd, ytr)
    return r.classes, r.decision((Ste - mu) / sd)


def to_pred(cl, dec):
    return np.where(dec[:, 0] > 0, cl[1], cl[0]) if len(cl) == 2 else cl[np.argmax(dec, 1)]


def main(name):
    out = os.path.join(os.environ.get("P10_OUT", os.path.join(HERE, "..", "results", "p10")), f"p10_{name.replace('/', '_').replace(':', '_')}.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    done = {json.loads(l)["split"] for l in open(out) if '"gRENS24"' in l} if os.path.exists(out) else set()
    X, y, g = get_data(name); ncls = len(np.unique(y))
    for si, (tr, te) in enumerate(get_splits(y, g)):
        if si in done: continue
        t0 = time.time()
        Xtr, Xte = clean(X[tr], X[te]); mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        Ztr, Zte = (Xtr - mu) / sd, (Xte - mu) / sd
        Ztr_t = torch.as_tensor(Ztr, dtype=torch.float32, device=DEV); Zte_t = torch.as_tensor(Zte, dtype=torch.float32, device=DEV)
        P = Ztr_t.T / np.sqrt(len(tr))
        recs, ytr, yte = [], y[tr], y[te]

        def rec(method, cl, dec):
            recs.append({"ds": name, "split": si, "method": method, "bacc": balanced_acc(yte, np.asarray(to_pred(cl, dec)).astype(int))})

        Ks = [10, 20, 40]

        def km(K, seed):  # same partitions as P9 (sklearn MiniBatchKMeans on feature profiles)
            return MiniBatchKMeans(K, random_state=seed, n_init=3, batch_size=4096).fit(Ztr.T / np.sqrt(len(tr))).labels_

        cl, d = hybrid_decision(Ztr_t, ytr, Zte_t, km(20, si), 20, si, ncls); rec("gK20", cl, d)
        dec = 0
        for i in range(12):  # P9 ENS: K cycles 10/20/40, seed si*100 + i//3
            K = Ks[i % 3]; sd_ = si * 100 + i // 3
            cl, d = hybrid_decision(Ztr_t, ytr, Zte_t, km(K, sd_), K, sd_, ncls)
            dec = dec + d
            if i + 1 in (1, 2, 4, 6, 12): rec(f"gENS{i + 1}", cl, dec)
        # P9 RAND20: one random partition, K=20, seed si
        lab = np.random.default_rng(si).integers(0, 20, Ztr.shape[1])
        cl, d = hybrid_decision(Ztr_t, ytr, Zte_t, lab, 20, si, ncls); rec("gRAND20", cl, d)
        # random partitions: RENS vote (support sets), P9 seeds si*1000 + i
        dec = 0
        for i in range(24):
            K = Ks[i % 3]
            lab = np.random.default_rng(si * 1000 + i).integers(0, K, Ztr.shape[1])
            cl, d = hybrid_decision(Ztr_t, ytr, Zte_t, lab, K, si * 1000 + i, ncls)
            dec = dec + d
            if i + 1 in (6, 12, 24): rec(f"gRENS{i + 1}", cl, dec)
        with open(out, "a") as f:
            for r in recs:
                r["t_split"] = time.time() - t0; f.write(json.dumps(r) + "\n")
        print(name, si, f"{time.time() - t0:.1f}s", flush=True)
        del Ztr_t, Zte_t, P
        if DEV == "cuda": torch.cuda.empty_cache()


if __name__ == "__main__":
    main(sys.argv[1])
