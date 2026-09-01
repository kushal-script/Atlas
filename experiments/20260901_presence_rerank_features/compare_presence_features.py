"""Judge whether the re rank combiner's three diagnostics earn their place in
the presence model.

The combiner override was declined this morning with a held out delta of
exactly zero, but the override and the diagnostics are separate hypotheses:
disagreement between two independent evidence functions over which site is
true is close to a direct measurement of ambiguity, and ambiguity is what a
rejection is. This script fits the fifteen feature model and the eighteen
feature model with the same class weighted logistic and the same threshold
protocol on the p2train harvest, then judges both on the p2holdout2 harvest,
so the only difference between the arms is the three combiner features.

Pose credit is a mean over pairs whose localization already earned credit, so
it moves only through composition and is left out of the proxy; the proxy is
localization (40) plus rejection F1 (15) plus decision AUC (10), identical in
both arms.

    .venv/bin/python experiments/20260901_presence_rerank_features/compare_presence_features.py \
        /tmp/rr_train.json /tmp/rr_holdout.json
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))


def rerank_block(rr):
    rr = rr or {}
    return [float(rr.get("score", 0.0)), float(rr.get("margin", 0.0)),
            1.0 if rr.get("agree", True) else 0.0]


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


def proxy(recs, probs, thr):
    loc = {"A_nominal": [], "B_degraded": []}
    scores, labels = [], []
    y = np.array([1 if r["present"] else 0 for r in recs])
    found = (probs >= thr).astype(int)
    for r, f, p in zip(recs, found, probs):
        if r["present"] and r["set"] in loc:
            e = r["err_classical"]
            loc[r["set"]].append(credit(e) if f and e >= 0 else 0.0)
        scores.append(max(p, 1.0 - p))
        labels.append(int(f == (1 if r["present"] else 0)))
    lp = 40 * (0.45 * np.mean(loc["A_nominal"]) + 0.55 * np.mean(loc["B_degraded"]))
    f1 = rej_f1(y, found)
    auc = auc_of(scores, labels)
    return lp + 15 * f1 + 10 * auc, lp, f1, auc


def run_arm(name, feats_of, train, hold):
    Xtr = np.array([feats_of(r) for r in train], float)
    ytr = np.array([1 if r["present"] else 0 for r in train])
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Ztr = (Xtr - mu) / sd

    rng = np.random.default_rng(0)
    folds = np.zeros(len(ytr), int)
    for cls in (0, 1):
        idx = np.where(ytr == cls)[0]
        rng.shuffle(idx)
        for k, i in enumerate(idx):
            folds[i] = k % 5
    cvp = np.zeros(len(ytr))
    for k in range(5):
        w = fit_logistic(Ztr[folds != k], ytr[folds != k])
        cvp[folds == k] = 1 / (1 + np.exp(-(Ztr[folds == k] @ w[:-1] + w[-1])))

    grid = np.arange(0.02, 0.99, 0.01)
    thr = max(((proxy(train, cvp, t)[0], t) for t in grid))[1]

    w = fit_logistic(Ztr, ytr)
    tr_tot = proxy(train, 1 / (1 + np.exp(-(Ztr @ w[:-1] + w[-1]))), thr)

    Xho = np.array([feats_of(r) for r in hold], float)
    pho = 1 / (1 + np.exp(-(((Xho - mu) / sd) @ w[:-1] + w[-1])))
    ho_tot = proxy(hold, pho, thr)

    print(f"{name}: threshold {thr:.2f}")
    print(f"  train   total {tr_tot[0]:6.2f}  loc {tr_tot[1]:5.2f}  F1 {tr_tot[2]:.3f}  auc {tr_tot[3]:.3f}")
    print(f"  holdout total {ho_tot[0]:6.2f}  loc {ho_tot[1]:5.2f}  F1 {ho_tot[2]:.3f}  auc {ho_tot[3]:.3f}")
    return ho_tot, w, mu, sd, thr


def main(train_path, hold_path):
    train = json.load(open(train_path))
    hold = json.load(open(hold_path))
    v1 = run_arm("v1 fifteen features", lambda r: list(r["v1_features"]), train, hold)
    v2 = run_arm("v2 plus rerank block",
                 lambda r: list(r["v1_features"]) + rerank_block(r.get("rerank")),
                 train, hold)
    d = v2[0][0] - v1[0][0]
    print(f"held out delta v2 minus v1: {d:+.2f} proxy points "
          f"(F1 {v2[0][2] - v1[0][2]:+.3f}, auc {v2[0][3] - v1[0][3]:+.3f})")
    names = ["rr_score", "rr_margin", "rr_agree"]
    for n, wt in zip(names, v2[1][-4:-1]):
        print(f"  weight {n}: {wt:+.3f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
