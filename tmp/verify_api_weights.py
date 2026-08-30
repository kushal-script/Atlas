"""Verify api.py's _REJECTION_* constants give F1 >= target on the dump."""

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from drift_sense.api import (_REJECTION_BIAS, _REJECTION_W, _REJECTION_MU,
                             _REJECTION_SD, _PRESENCE_FEATURE_ORDER,
                             presence_probability_from_features)

FEATURE_KEYS = _PRESENCE_FEATURE_ORDER


def _f1(pred, gold):
    tp = int(np.sum((pred == 1) & (gold == 1)))
    fp = int(np.sum((pred == 1) & (gold == 0)))
    fn = int(np.sum((pred == 0) & (gold == 1)))
    prec = tp / (tp + fp) if (tp + fp) else 1.0
    rec = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return f1, prec, rec, tp, fp, fn


def _auroc_present(prob, y):
    """AUROC for 'present' class. y in {0,1}, prob is P(present)."""
    y = np.asarray(y).astype(int)
    p = np.asarray(prob)
    pos = p[y == 1]
    neg = p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    n_pos = len(pos)
    n_neg = len(neg)
    # Mann-Whitney U statistic / (n_pos*n_neg)
    # Efficient: sort once and count ranks
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    # average ranks for ties
    i = 0
    while i < len(p):
        j = i
        while j + 1 < len(p) and p[order[j + 1]] == p[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1  # 1-indexed average rank
        ranks[order[i:j + 1]] = avg
        i = j + 1
    sum_ranks_pos = ranks[y == 1].sum()
    u = sum_ranks_pos - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def main():
    if len(sys.argv) < 2:
        print("usage: verify_api_weights.py <dump.json>")
        sys.exit(2)
    rows_in = json.load(open(sys.argv[1]))
    feats = np.array([[float(r["features"][k]) for k in FEATURE_KEYS]
                      for r in rows_in])
    y = np.array([int(r["present"]) for r in rows_in])
    # Use the constants directly to ensure no other code path is involved
    z = (feats - _REJECTION_MU) / _REJECTION_SD
    logit = _REJECTION_BIAS + z @ _REJECTION_W
    prob = 1.0 / (1.0 + np.exp(-logit))
    f1_05, p, r, tp, fp, fn = _f1((prob >= 0.5).astype(int), y)
    auc = _auroc_present(prob, y)
    print(f"api.py weights @ thr=0.5: F1={f1_05:.4f} P={p:.3f} R={r:.3f} "
          f"(tp={tp} fp={fp} fn={fn})  AUROC={auc:.4f}")
    best = None
    for thr in np.linspace(0.0, 1.0, 1001):
        f1, _, _, _, _, _ = _f1((prob >= thr).astype(int), y)
        if best is None or f1 > best["f1"]:
            best = {"thr": float(thr), "f1": f1}
    print(f"best F1={best['f1']:.4f} @ thr={best['thr']:.3f}")
    print(f"FEATURE_KEYS order: {FEATURE_KEYS}")


if __name__ == "__main__":
    main()