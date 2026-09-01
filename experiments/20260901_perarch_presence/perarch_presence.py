"""Measure fully separate per architecture presence models against one shared
model, under the eighteen feature set.

The per architecture threshold split was declined on August 31 at minus 0.545
held out, but that split shared one fitted model and moved only the boundary.
This asks the stronger question the user's plan poses: fit an entire presence
model per detected architecture, so every weight can specialise, and route at
runtime by the lattice balance detector whose per pair outputs were recorded
in experiments/20260831_architecture_dispatch. Fitting splits the 216 pair
training harvest roughly in half and leaves about two dozen absents per
architecture, so the shrinkage risk is known going in; the measurement says
whether the specialisation pays for it.

Routing uses the DETECTED architecture on both suites, never the generator's
label, because the blind set carries no label.

    .venv/bin/python experiments/20260901_perarch_presence/perarch_presence.py \
        /tmp/rr_train.json /tmp/rr_holdout.json
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

REPO = Path(__file__).resolve().parent.parent.parent
DISPATCH = REPO / "experiments" / "20260831_architecture_dispatch"


def rerank_block(rr):
    rr = rr or {}
    return [float(rr.get("score", 0.0)), float(rr.get("margin", 0.0)),
            1.0 if rr.get("agree", True) else 0.0]


def feats_of(r):
    return list(r["v1_features"]) + rerank_block(r.get("rerank"))


def credit(e):
    return 1.0 if e <= 1 else 0.8 if e <= 2 else 0.6 if e <= 3 else 0.4 if e <= 5 else 0.0


def fit_logistic(X, y, l2=1.0):
    n, d = X.shape

    def nll(w):
        m = X @ w[:d] + w[d]
        wgt = np.where(y == 1, 1.0, (y == 1).sum() / max((y == 0).sum(), 1))
        return (wgt * (np.logaddexp(0, m) - y * m)).sum() + l2 * (w[:d] @ w[:d])

    return minimize(nll, np.zeros(d + 1), method="L-BFGS-B").x


def rej_f1(y, found):
    tp = int(np.sum((y == 0) & (found == 0)))
    fp = int(np.sum((y == 1) & (found == 0)))
    fn = int(np.sum((y == 0) & (found == 1)))
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    return 2 * p * r / max(p + r, 1e-9)


def auc_of(scores, labels):
    scores = np.asarray(scores, float)
    labels = np.asarray(labels, int)
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    pos = np.where(labels == 1)[0]
    neg = len(labels) - len(pos)
    if len(pos) == 0 or neg == 0:
        return 0.5
    return float((ranks[pos].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * neg))


def proxy(recs, probs, found):
    loc = {"A_nominal": [], "B_degraded": []}
    scores, labels = [], []
    y = np.array([1 if r["present"] else 0 for r in recs])
    for r, f, p in zip(recs, found, probs):
        if r["present"] and r["set"] in loc:
            e = r["err_classical"]
            loc[r["set"]].append(credit(e) if f and e >= 0 else 0.0)
        scores.append(max(p, 1.0 - p))
        labels.append(int(f == (1 if r["present"] else 0)))
    lp = 40 * (0.45 * np.mean(loc["A_nominal"]) + 0.55 * np.mean(loc["B_degraded"]))
    f1 = rej_f1(y, np.asarray(found, int))
    auc = auc_of(scores, labels)
    return lp + 15 * f1 + 10 * auc, lp, f1, auc


def cv_probs(Z, y, seed=0):
    rng = np.random.default_rng(seed)
    folds = np.zeros(len(y), int)
    for cls in (0, 1):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        for k, i in enumerate(idx):
            folds[i] = k % 5
    out = np.zeros(len(y))
    for k in range(5):
        if (folds != k).sum() == 0 or len(set(y[folds != k])) < 2:
            continue
        w = fit_logistic(Z[folds != k], y[folds != k])
        out[folds == k] = 1 / (1 + np.exp(-(Z[folds == k] @ w[:-1] + w[-1])))
    return out


def choose_threshold(recs, probs):
    grid = np.arange(0.02, 0.99, 0.01)
    return max(((proxy(recs, probs, (probs >= t).astype(int))[0], t) for t in grid))[1]


def main(train_path, hold_path):
    train = json.load(open(train_path))
    hold = json.load(open(hold_path))
    arch = {}
    for name in ("harvest_p2train.csv", "harvest_p2holdout2.csv"):
        for row in csv.DictReader(open(DISPATCH / name)):
            arch[(name, row["pair_id"])] = row["arch"]
    a_tr = [arch[("harvest_p2train.csv", r["pair_id"])] for r in train]
    a_ho = [arch[("harvest_p2holdout2.csv", r["pair_id"])] for r in hold]

    Xtr = np.array([feats_of(r) for r in train], float)
    ytr = np.array([1 if r["present"] else 0 for r in train])
    Xho = np.array([feats_of(r) for r in hold], float)

    def standardise(X, mu, sd):
        return (X - mu) / sd

    print("arm 1: one shared eighteen feature model")
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Ztr = standardise(Xtr, mu, sd)
    thr = choose_threshold(train, cv_probs(Ztr, ytr))
    w = fit_logistic(Ztr, ytr)
    pho = 1 / (1 + np.exp(-(standardise(Xho, mu, sd) @ w[:-1] + w[-1])))
    tot = proxy(hold, pho, (pho >= thr).astype(int))
    print(f"  threshold {thr:.2f}  holdout total {tot[0]:6.2f}  loc {tot[1]:5.2f}"
          f"  F1 {tot[2]:.3f}  auc {tot[3]:.3f}")
    shared = tot

    print("arm 2: shared model, per architecture thresholds")
    thr_by = {}
    cvp = cv_probs(Ztr, ytr)
    for a in ("dram", "finfet"):
        idx = [i for i, ai in enumerate(a_tr) if ai == a]
        thr_by[a] = choose_threshold([train[i] for i in idx], cvp[idx])
        print(f"  {a}: threshold {thr_by[a]:.2f} on {len(idx)} pairs")
    found = np.array([1 if pho[i] >= thr_by[a_ho[i]] else 0 for i in range(len(hold))])
    tot = proxy(hold, pho, found)
    print(f"  holdout total {tot[0]:6.2f}  loc {tot[1]:5.2f}  F1 {tot[2]:.3f}  auc {tot[3]:.3f}"
          f"   delta vs shared {tot[0] - shared[0]:+.2f}")

    print("arm 3: fully separate models per detected architecture")
    pho3 = np.zeros(len(hold))
    for a in ("dram", "finfet"):
        idx = [i for i, ai in enumerate(a_tr) if ai == a]
        Xa, ya = Xtr[idx], ytr[idx]
        mua, sda = Xa.mean(0), Xa.std(0) + 1e-9
        Za = standardise(Xa, mua, sda)
        thr_a = choose_threshold([train[i] for i in idx], cv_probs(Za, ya))
        wa = fit_logistic(Za, ya)
        hidx = [i for i, ai in enumerate(a_ho) if ai == a]
        pa = 1 / (1 + np.exp(-(standardise(Xho[hidx], mua, sda) @ wa[:-1] + wa[-1])))
        pho3[hidx] = pa
        thr_by[a] = thr_a
        print(f"  {a}: {len(idx)} train pairs ({int(ya.sum())} present), threshold {thr_a:.2f},"
              f" routes {len(hidx)} holdout pairs")
    found = np.array([1 if pho3[i] >= thr_by[a_ho[i]] else 0 for i in range(len(hold))])
    tot = proxy(hold, pho3, found)
    print(f"  holdout total {tot[0]:6.2f}  loc {tot[1]:5.2f}  F1 {tot[2]:.3f}  auc {tot[3]:.3f}"
          f"   delta vs shared {tot[0] - shared[0]:+.2f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
