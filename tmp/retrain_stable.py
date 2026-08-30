"""Stable retrain recipe (Phase 2 Day 8): standardized numpy logistic regression
with lr=0.01, iters=20000, l2=1e-2 (NOT the day5 default which diverges).

Reads a localize dump, fits the LR, writes bias/W/mu/sd to --out json, prints F1.
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


def _logreg(feats, present, l2=1e-2, iters=20000, lr=0.01):
    mu = feats.mean(0, keepdims=True)
    sd = feats.std(0, keepdims=True) + 1e-6
    X = (feats - mu) / sd
    X = np.hstack([np.ones((X.shape[0], 1)), X])
    y = present.astype(float)
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        z = X @ w
        z = np.clip(z, -50, 50)
        p = 1.0 / (1.0 + np.exp(-z))
        grad = X.T @ (p - y) + l2 * w
        grad[0] -= l2 * w[0]
        w -= lr * grad
    return w, mu, sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--l2", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = json.load(open(args.dump))
    X = np.array([[float(r["features"][k]) for k in FEATURE_KEYS] for r in rows])
    y = np.array([int(r["present"]) for r in rows])

    w, mu, sd = _logreg(X, y, l2=args.l2, iters=args.iters, lr=args.lr)
    Xs = (X - mu) / sd
    Xs = np.hstack([np.ones((Xs.shape[0], 1)), Xs])
    prob = 1.0 / (1.0 + np.exp(-(Xs @ w)))

    f1_05, p, r, tp, fp, fn = _f1((prob >= 0.5).astype(int), y)
    best = None
    for thr in np.linspace(0.0, 1.0, 1001):
        f1, _, _, _, _, _ = _f1((prob >= thr).astype(int), y)
        if best is None or f1 > best["f1"]:
            best = {"thr": float(thr), "f1": f1}

    print(f"pairs={len(y)} present={int(y.sum())} absent={int((y == 0).sum())}")
    print(f"[stable LR @0.5] F1={f1_05:.4f} P={p:.3f} R={r:.3f} (tp={tp} fp={fp} fn={fn})")
    print(f"[stable LR best thr] F1={best['f1']:.4f} @ thr={best['thr']:.3f}")
    print("bias =", float(w[0]))
    for k, v in zip(FEATURE_KEYS, w[1:]):
        print(f"  {k} = {float(v):.16f}")
    print("mu =", [float(v) for v in mu[0]])
    print("sd =", [float(v) for v in sd[0]])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "bias": float(w[0]),
        "weights": [float(v) for v in w[1:]],
        "mu": [float(v) for v in mu[0]],
        "sd": [float(v) for v in sd[0]],
        "f1_at_0.5": f1_05,
        "best_f1": best["f1"],
        "best_thr": best["thr"],
        "feature_keys": FEATURE_KEYS,
    }, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()