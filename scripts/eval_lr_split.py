"""Honest LR evaluation: 70/30 split (Phase 2, Day 6, T3).

The production LR was reported on the full 200 pairs. To check generalization to
a blind official set, retrain on a 70% split and report F1 on the held-out 30%
and on the full set. No re-localization (uses an existing localize dump).

Usage:
    uv run python scripts/eval_lr_split.py --dump data/phase2_mixed/dump1.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

FEATURE_KEYS = ["peak_score", "num_candidates_wide", "uniqueness",
                "stage2_identifiability", "margin_strength",
                "peak_contrast", "peak_contrast_ratio", "geo_consistency",
                "search_noise_sigma", "inverted_contrast"]


def _f1(pred, gold):
    tp = int(np.sum((pred == 1) & (gold == 1)))
    fp = int(np.sum((pred == 1) & (gold == 0)))
    fn = int(np.sum((pred == 0) & (gold == 1)))
    prec = tp / (tp + fp) if (tp + fp) else 1.0
    rec = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return f1, prec, rec, tp, fp, fn


def _logreg(feats, present, l2=1e-2, iters=2000, lr=0.1):
    mu = feats.mean(0, keepdims=True)
    sd = feats.std(0, keepdims=True) + 1e-6
    X = (feats - mu) / sd
    X = np.hstack([np.ones((X.shape[0], 1)), X])
    y = present.astype(float)
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(X @ w)))
        grad = X.T @ (p - y) + l2 * w
        grad[0] -= l2 * w[0]
        w -= lr * grad
    return w, mu, sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--frac", type=float, default=0.30)
    args = ap.parse_args()

    rows = json.load(open(args.dump))
    idx = np.arange(len(rows))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(idx)
    n_test = int(len(idx) * args.frac)
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]

    def subset(idxs):
        X = np.array([[float(rows[i]["features"][k]) for k in FEATURE_KEYS] for i in idxs])
        y = np.array([int(rows[i]["present"]) for i in idxs])
        return X, y

    Xtr, ytr = subset(train_idx)
    Xte, yte = subset(test_idx)
    Xall, yall = subset(idx)

    w, mu, sd = _logreg(Xtr, ytr)
    def prob(X):
        Xs = (X - mu) / sd
        Xs = np.hstack([np.ones((Xs.shape[0], 1)), Xs])
        return 1.0 / (1.0 + np.exp(-(Xs @ w)))

    f1_test, p, r, tp, fp, fn = _f1((prob(Xte) >= 0.5).astype(int), yte)
    f1_full, pf, rf, tpf, fpf, fnf = _f1((prob(Xall) >= 0.5).astype(int), yall)
    print(f"split seed={args.seed} frac={args.frac}")
    print(f"  train={len(ytr)} (present={int(ytr.sum())}) test={len(yte)} (present={int(yte.sum())})")
    print(f"  HELD-OUT F1={f1_test:.4f} P={p:.3f} R={r:.3f} (tp={tp} fp={fp} fn={fn})")
    print(f"  FULL      F1={f1_full:.4f} P={pf:.3f} R={rf:.3f} (tp={tpf} fp={fpf} fn={fnf})")
    print("  decision: keep REJECTION_THRESHOLD=0.5" if f1_test >= 0.90
          else "  decision: held-out < 0.90 -> consider lowering threshold")


if __name__ == "__main__":
    main()
