"""P8: NeuroRules - end-to-end neural version of the module + threshold-rule hybrid.

Architecture (binary and multi-class, one-vs-rest rule heads):
  x (d) -> module layer: K modules from k-means on feature profiles (train fold); each module has its own
           sparse linear score s_k = <w_k, x_Mk> (L1 penalty)                          [Mirzaev: groups]
        -> rule layer: P pairs (i, j) of modules; soft threshold rule
           r_p = sigmoid((a_p s_i + b_p s_j - t_p) / tau)                               [Mirzaev: 2-D threshold rules]
        -> output: logit_c = sum_p alpha_{p,c} (2 r_p - 1) + bias_c                     [weighted voting]
  Ensembling: E implicit members with rank-1 multiplicative adapters on the module layer (TabM/BatchEnsemble);
  predictions averaged over members.
Training: AdamW, full batch, class-balanced cross-entropy, early stopping on an inner stratified 20% split,
then refit on the full training fold for the chosen number of epochs.
Initialisation of pairs: all pairs of the K modules (K=20 -> 190 pairs).
Usage: python p8_neurorules.py <dataset> [K] [E]
"""
import json, os, sys, time, warnings
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import StratifiedShuffleSplit
from common import balanced_acc
from p4_bench import get_data, get_splits, clean

HERE = os.path.dirname(os.path.abspath(__file__))
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
DEVICE = os.environ.get("NR_DEVICE", "cpu")
if DEVICE == "cuda":
    # hard per-process cap on CUDA memory (GB10 memory is unified with RAM and not visible to cgroup limits):
    # an allocation above the cap raises OOM inside this process instead of exhausting the machine
    torch.cuda.set_per_process_memory_fraction(float(os.environ.get("NR_MEMFRAC", "0.09")))


class NeuroRules(nn.Module):
    def __init__(self, lab, K, C, E=16, tau=0.5):
        super().__init__()
        d = len(lab)
        mask = torch.zeros(d, K); mask[torch.arange(d), torch.as_tensor(lab)] = 1.0
        self.register_buffer("mask", mask)                         # d x K block structure
        self.W = nn.Parameter(torch.randn(d, K) * 0.01)             # module weights (masked)
        self.R = nn.Parameter(torch.ones(E, d))                     # rank-1 adapters (TabM)
        self.S = nn.Parameter(torch.ones(E, K))
        self.bn = nn.BatchNorm1d(K, affine=False)
        ii, jj = torch.triu_indices(K, K, 1)
        self.register_buffer("ii", ii); self.register_buffer("jj", jj)
        P = len(ii)
        self.a = nn.Parameter(torch.randn(E, P) * 0.5)
        self.b = nn.Parameter(torch.randn(E, P) * 0.5)
        self.t = nn.Parameter(torch.zeros(E, P))
        self.alpha = nn.Parameter(torch.zeros(E, P, C))
        self.bias = nn.Parameter(torch.zeros(E, C))
        self.tau, self.E = tau, E
        self.register_buffer("rule_mask", torch.ones(P))

    def forward(self, x):                                           # x: n x d -> E x n x C
        Wm = self.W * self.mask
        s = torch.einsum("nd,ed,dk->enk", x, self.R, Wm) * self.S[:, None, :]
        s = torch.stack([self.bn(s[e]) for e in range(self.E)])     # standardise module scores
        z = self.a[:, None, :] * s[:, :, self.ii] + self.b[:, None, :] * s[:, :, self.jj] - self.t[:, None, :]
        r = torch.sigmoid(z / self.tau) * 2 - 1                     # soft rule outputs in (-1, 1)
        return torch.einsum("enp,epc->enc", r * self.rule_mask, self.alpha) + self.bias[:, None, :]

    def l1(self):
        return (self.W * self.mask).abs().sum()


def train(model, X, y, C, epochs, lr=3e-3, wd=1e-4, lam=1e-4, alpha_l1=1e-3, Xv=None, yv=None, eval_at=None):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    w = torch.as_tensor(len(y) / (C * np.bincount(y, minlength=C)), dtype=torch.float32, device=DEVICE)
    lossf = nn.CrossEntropyLoss(weight=w)
    Xt, yt = torch.as_tensor(X, dtype=torch.float32, device=DEVICE), torch.as_tensor(y, device=DEVICE)
    hist = []
    for ep in range(epochs):
        model.train(); opt.zero_grad()
        out = model(Xt)
        loss = sum(lossf(out[e], yt) for e in range(model.E)) / model.E + lam * model.l1() + alpha_l1 * model.alpha.abs().sum()
        loss.backward(); opt.step()
        if Xv is not None and (ep + 1) in (eval_at or set(range(10, epochs + 1, 10))):
            hist.append((ep + 1, balanced_acc(yv, predict(model, Xv))))
    return hist


@torch.no_grad()
def predict(model, X):
    model.eval()
    out = model(torch.as_tensor(X, dtype=torch.float32, device=DEVICE))
    return torch.softmax(out, -1).mean(0).argmax(-1).cpu().numpy()


ALPHA_L1 = float(os.environ.get('ALPHA_L1', '1e-3'))
PRUNE = int(os.environ.get('PRUNE', '0'))


def fit_predict(Ztr, ytr, Zte, K, E, seed, grid=(50, 100, 200)):
    """Epoch count chosen by inner 3-fold CV (early stopping on ~12 objects was too noisy)."""
    from sklearn.model_selection import StratifiedKFold
    C = int(ytr.max()) + 1
    lab = MiniBatchKMeans(K, random_state=seed, n_init=3, batch_size=4096).fit(Ztr.T / np.sqrt(len(Ztr))).labels_
    score = {g: [] for g in grid}
    for a, b in StratifiedKFold(3, shuffle=True, random_state=seed).split(Ztr, ytr):
        torch.manual_seed(seed); m = NeuroRules(lab, K, C, E).to(DEVICE)
        opt_hist = train(m, Ztr[a], ytr[a], C, max(grid), alpha_l1=ALPHA_L1, Xv=Ztr[b], yv=ytr[b], eval_at=set(grid))
        for ep, acc in opt_hist: score[ep].append(acc)
    best_ep = max(grid, key=lambda g: (np.mean(score[g]), -g))
    torch.manual_seed(seed)
    m = NeuroRules(lab, K, C, E).to(DEVICE)
    train(m, Ztr, ytr, C, best_ep, alpha_l1=ALPHA_L1)
    info = {}
    if PRUNE:
        info["bacc_full_pred"] = predict(m, Zte)
        imp = m.alpha.detach().abs().mean(0).sum(-1)            # rule importance averaged over members
        keep = torch.topk(imp, min(PRUNE, len(imp))).indices
        m.rule_mask.zero_(); m.rule_mask[keep] = 1.0
        train(m, Ztr, ytr, C, max(best_ep // 2, 25), alpha_l1=ALPHA_L1)   # fine-tune the pruned network
        mods = sorted(set(m.ii[keep].cpu().tolist()) | set(m.jj[keep].cpu().tolist()))
        Wm = (m.W.detach() * m.mask)[:, mods]
        thr = 1e-2 * Wm.abs().max()
        info.update(n_rules=int(len(keep)), n_modules=len(mods), n_feat=int((Wm.abs().max(1).values > thr).sum().item()))
    return predict(m, Zte), best_ep, info


def main(name, K=20, E=16):
    out = os.path.join(os.environ.get("P8_OUT", os.path.join(HERE, "..", "results", "p8")), f"p8_{name.replace('/', '_').replace(':', '_')}_K{K}_E{E}_a{ALPHA_L1:g}_p{PRUNE}.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    done = {json.loads(l)["split"] for l in open(out)} if os.path.exists(out) else set()
    X, y, g = get_data(name)
    for si, (tr, te) in enumerate(get_splits(y, g)):
        if si in done: continue
        Xtr, Xte = clean(X[tr], X[te]); mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        t0 = time.time()
        p, ep, info = fit_predict((Xtr - mu) / sd, y[tr], (Xte - mu) / sd, K, E, si)
        rec = {"ds": name, "split": si, "method": f"NR-K{K}-E{E}-a{ALPHA_L1:g}-p{PRUNE}", "bacc": balanced_acc(y[te], p), "epochs": ep, "alpha_l1": ALPHA_L1, "prune": PRUNE, "t": time.time() - t0}
        if "bacc_full_pred" in info:
            rec["bacc_unpruned"] = balanced_acc(y[te], info.pop("bacc_full_pred"))
        rec.update(info)
        open(out, "a").write(json.dumps(rec) + "\n")
        print(name, si, round(rec["bacc"], 3), ep, flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 20, int(sys.argv[3]) if len(sys.argv) > 3 else 16)
