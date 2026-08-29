"""Retrain the presence logistic regression on the current (post-T3) features.

Reads a localize dump (scripts/localize_dump.py) so no re-localization is needed,
fits the same standardized numpy LR as scripts/tune_rejection.py, and prints the
new weights/mu/sd plus the F1 at 0.5 and the best-threshold F1. Use the printed
constants to update _REJECTION_* in src/drift_sense/api.py.

Usage:
    uv run python scripts/retrain_day5.py --dump data/phase2_mixed/dump1.json
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
        z = X @ w
        p = 1.0 / (1.0 + np.exp(-z))
        grad = X.T @ (p - y) + l2 * w
        grad[0] -= l2 * w[0]
        w -= lr * grad
    return w, mu, sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None,
                    help="optional json to write new constants")
    args = ap.parse_args()

    rows = json.load(open(args.dump))
    X = np.array([[float(r["features"][k]) for k in FEATURE_KEYS] for r in rows])
    y = np.array([int(r["present"]) for r in rows])

    w, mu, sd = _logreg(X, y)
    Xs = (X - mu) / sd
    Xs = np.hstack([np.ones((Xs.shape[0], 1)), Xs])
    prob = 1.0 / (1.0 + np.exp(-(Xs @ w)))

    f1_05, p, r, tp, fp, fn = _f1((prob >= 0.5).astype(int), y)
    # best-threshold sweep
    best = None
    for thr in np.linspace(0.0, 1.0, 201):
        f1, _, _, _, _, _ = _f1((prob >= thr).astype(int), y)
        if best is None or f1 > best["f1"]:
            best = {"thr": float(thr), "f1": f1}

    print(f"pairs={len(y)} present={int(y.sum())} absent={int((y == 0).sum())}")
    print(f"[retrained LR @0.5] F1={f1_05:.4f} P={p:.3f} R={r:.3f} (tp={tp} fp={fp} fn={fn})")
    print(f"[retrained LR best thr] F1={best['f1']:.4f} @ thr={best['thr']:.3f}")
    print("bias =", round(float(w[0]), 16))
    for k, v in zip(FEATURE_KEYS, w[1:]):
        print(f"  {k} = {float(v):.16f}")
    print("mu =", [round(float(v), 15) for v in mu[0]])
    print("sd =", [round(float(v), 15) for v in sd[0]])

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "bias": float(w[0]),
            "weights": [float(v) for v in w[1:]],
            "mu": [float(v) for v in mu[0]],
            "sd": [float(v) for v in sd[0]],
            "f1_at_0.5": f1_05, "best_f1": best["f1"], "best_thr": best["thr"],
        }, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
