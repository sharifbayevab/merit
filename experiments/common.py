"""Shared data protocol for the HDLSS recognition pilots.

Datasets: the eight scikit-feature HDLSS benchmarks used by GOTabPFN (2606.05441).
Protocol: 5-fold stratified CV x 3 repeats (seed 0) -> 15 splits per dataset.
Every fitted object (scaler, filter, grouping, head) sees only the training fold.
"""
import os
import numpy as np
import scipy.io as sio
from sklearn.model_selection import RepeatedStratifiedKFold

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
DATASETS = ["colon", "ALLAML", "Prostate_GE", "GLI_85", "SMK_CAN_187", "TOX_171", "lung", "arcene"]


def load(name):
    m = sio.loadmat(os.path.join(DATA_DIR, f"{name}.mat"))
    X = np.asarray(m["X"], dtype=np.float64)
    y_raw = np.ravel(m["Y"])
    _, y = np.unique(y_raw, return_inverse=True)
    return X, y.astype(int)


def splits(y, n_splits=5, n_repeats=3, seed=0):
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    return list(rskf.split(np.zeros(len(y)), y))


def balanced_acc(y, p):
    classes = np.unique(y)
    return float(np.mean([np.mean(p[y == c] == c) for c in classes]))
