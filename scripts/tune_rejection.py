"""Rejection threshold tuning (Phase 2, Day 3, T3).

Runs the localizer over a mixed Phase 2 dataset (present + Set C absent), computes
the combined presence score for every pair, and sweeps a single decision threshold
to maximize F1 on `found` against the ground-truth `present` label. If no single
threshold reaches the F1 >= 0.90 target, it escalates to a tiny numpy logistic
regression on presence_features (no torch) and reports that instead.

Usage:
    python scripts/tune_rejection.py --manifest data/phase2_mixed/manifest.csv
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from drift_sense.api import match_pair, presence_score, presence_features  # noqa: E402


FEATURE_KEYS = ["peak_score", "num_candidates_wide", "uniqueness",
                "stage2_identifiability", "margin_strength",
                "peak_contrast", "peak_contrast_ratio", "geo_consistency",
                "search_noise_sigma", "inverted_contrast"]


def _load_pairs(manifest):
    base = manifest.parent
    pairs = []
    with open(manifest, newline="") as fh:
        for row in csv.DictReader(fh):
            ref = (base / row["reference"]).resolve()
            search = (base / row["search"]).resolve()
            present = int(row["present"])
            pairs.append((ref, search, present))
    return pairs


def _scores(pairs):
    """Localize every pair once; return (scores, present, feature_matrix)."""
    scores, present, feats = [], [], []
    for ref, search, pres in pairs:
        r = match_pair(np.asarray(__import__("cv2").imread(str(ref), 0)),
                       np.asarray(__import__("cv2").imread(str(search), 0)))
        diag = r["diagnostics"]
        scores.append(presence_score(diag))
        f = presence_features(diag)
        feats.append([float(f[k]) for k in FEATURE_KEYS])
        present.append(pres)
    return np.array(scores), np.array(present), np.array(feats)


def _f1(pred, gold):
    tp = int(np.sum((pred == 1) & (gold == 1)))
    fp = int(np.sum((pred == 1) & (gold == 0)))
    fn = int(np.sum((pred == 0) & (gold == 1)))
    prec = tp / (tp + fp) if (tp + fp) else 1.0
    rec = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return f1, prec, rec, tp, fp, fn


def _sweep(scores, present, n=201):
    best = None
    for thr in np.linspace(0.0, 1.0, n):
        pred = (scores >= thr).astype(int)
        f1, p, r, tp, fp, fn = _f1(pred, present)
        if best is None or f1 > best["f1"]:
            best = {"threshold": float(thr), "f1": f1, "precision": p,
                    "recall": r, "tp": tp, "fp": fp, "fn": fn}
    return best


def _logreg(feats, present, l2=1e-2, iters=2000, lr=0.1):
    """Tiny numpy logistic regression with standardization + L2."""
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
        grad[0] -= l2 * w[0]   # no regularization on bias
        w -= lr * grad
    return w, mu, sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--target-f1", type=float, default=0.90)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    pairs = _load_pairs(args.manifest)
    scores, present, feats = _scores(pairs)
    n = len(present)
    n_absent = int(np.sum(present == 0))
    n_present = n - n_absent

    single = _sweep(scores, present)
    result = {
        "n_pairs": n, "n_present": n_present, "n_absent": n_absent,
        "single_threshold": single,
        "target_f1": args.target_f1,
    }
    print(f"pairs={n} present={n_present} absent={n_absent}")
    print(f"[single threshold] best F1={single['f1']:.4f} @ thr={single['threshold']:.3f}")
    print(f"  precision={single['precision']:.3f} recall={single['recall']:.3f} "
          f"(tp={single['tp']} fp={single['fp']} fn={single['fn']})")

    if single["f1"] < args.target_f1:
        w, mu, sd = _logreg(feats, present)
        X = (feats - mu) / sd
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        prob = 1.0 / (1.0 + np.exp(-(X @ w)))
        pred = (prob >= 0.5).astype(int)
        f1, p, r, tp, fp, fn = _f1(pred, present)
        result["logistic_regression"] = {
            "f1": f1, "precision": p, "recall": r,
            "tp": tp, "fp": fp, "fn": fn, "weights": w.tolist(),
            "mu": mu.tolist()[0], "sd": sd.tolist()[0],
        }
        print(f"[logistic regression] F1={f1:.4f} precision={p:.3f} recall={r:.3f} "
              f"(tp={tp} fp={fp} fn={fn})")
        print(f"  weights (bias + {FEATURE_KEYS}): {np.round(w, 3).tolist()}")
    else:
        print("single threshold meets target; no logistic regression needed")

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        with open(args.out / "tune_rejection.json", "w") as fh:
            json.dump(result, fh, indent=2)
        print(f"wrote {args.out / 'tune_rejection.json'}")

    return result


if __name__ == "__main__":
    main()
