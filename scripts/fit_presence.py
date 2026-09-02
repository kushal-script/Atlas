"""Fit the presence decision from recorded localizer diagnostics.

A single peak threshold cannot separate a degraded present pair from a clean
absent one: an impostor reference from the same architecture lets the scale
search rescale its lattice onto the search lattice and correlate respectably.
The decision therefore combines the evidence the localizer already produces,
peak strength in the context of the measured noise, how degenerate the
correlation surface was, and whether the deviation field singled a site out.

Fits a small logistic model on a generated training suite, reports stratified
cross validated reject class F1 against single feature baselines, and writes
the standardization constants and weights as json for register.py to embed.
Coefficient fitting on this repository's own generated data is threshold
tuning in the sense the addendum explicitly allows; no organiser data is read.

    .venv/bin/python scripts/fit_presence.py --records experiments/<stamp>_p2_features/records.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from drift_sense.presence import (ALL_FEATURE_NAMES, EXTENDED_FEATURES, FEATURES,
                                  RAW_CONFIRM_FEATURES, RERANK_FEATURES,
                                  features_for_model_record, features_from_record)

def rej_f1(y, pred_found):
    tp = int(np.sum((y == 0) & (pred_found == 0)))
    fp = int(np.sum((y == 1) & (pred_found == 0)))
    fn = int(np.sum((y == 0) & (pred_found == 1)))
    p = tp / max(tp + fp, 1); rc = tp / max(tp + fn, 1)
    return 2 * p * rc / max(p + rc, 1e-9), p, rc, tp, fp, fn


def fit_logistic(X, y, l2=1.0):
    n, d = X.shape
    def nll(w):
        m = X @ w[:d] + w[d]
        wgt = np.where(y == 1, 1.0, (y == 1).sum() / max((y == 0).sum(), 1))
        ll = wgt * (np.logaddexp(0, m) - y * m)
        return ll.sum() + l2 * (w[:d] @ w[:d])
    res = minimize(nll, np.zeros(d + 1), method="L-BFGS-B")
    return res.x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("models/presence_model.json"))
    ap.add_argument("--features", choices=("v1", "v2", "v3", "v4", "all"), default="v1",
                    help="v1 the fifteen diagnostics, v2 adds the rerank combiner block, "
                         "v3 the ambiguity block on v2, v4 the raw confirmation block "
                         "on v2, all every named feature")
    args = ap.parse_args()
    recs = json.load(open(args.records))
    feats = {"v1": FEATURES, "v2": RERANK_FEATURES, "v3": EXTENDED_FEATURES,
             "v4": RAW_CONFIRM_FEATURES, "all": ALL_FEATURE_NAMES}[args.features]
    pseudo = {"features": list(feats)}
    X = np.array([features_for_model_record(pseudo, r) for r in recs], float)
    y = np.array([1 if r["truth_found"] else 0 for r in recs])
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Xs = (X - mu) / sd

    rng = np.random.default_rng(0)
    folds = np.zeros(len(y), int)
    for cls in (0, 1):
        idx = np.where(y == cls)[0]; rng.shuffle(idx)
        for k, i in enumerate(idx): folds[i] = k % 5

    print("=== stratified 5 fold cross validation ===")
    probs = np.zeros(len(y))
    for k in range(5):
        tr, te = folds != k, folds == k
        w = fit_logistic(Xs[tr], y[tr])
        probs[te] = 1 / (1 + np.exp(-(Xs[te] @ w[:-1] + w[-1])))

    def loc_credit(e):
        return 1.0 if e <= 1 else 0.8 if e <= 2 else 0.6 if e <= 3 else 0.4 if e <= 5 else 0.0
    def sc_credit(zp):
        return 1.0 if zp <= 1 else 0.6 if zp <= 2 else 0.3 if zp <= 5 else 0.0
    def rc_credit(rd):
        return 1.0 if rd <= 0.25 else 0.6 if rd <= 0.5 else 0.3 if rd <= 1.0 else 0.0

    def total_score(found):
        credits = {"A_nominal": [], "B_degraded": []}
        pose = []
        for r, f in zip(recs, found):
            if not r["truth_found"]:
                continue
            c = loc_credit(r["err"]) if (f and r["err"] is not None) else 0.0
            credits[r["set"]].append(c)
            if f and c > 0:
                pose.append((sc_credit(r["zerr"]), rc_credit(r["rerr"])))
        loc = 40 * (0.45 * np.mean(credits["A_nominal"]) + 0.55 * np.mean(credits["B_degraded"]))
        pp = (10 * np.mean([p_ for p_, _ in pose]) + 10 * np.mean([q for _, q in pose])) if pose else 0.0
        f1v = rej_f1(y, np.asarray(found, int))[0]
        return loc + pp + 15 * f1v, loc, pp, 15 * f1v

    grid = np.arange(0.02, 0.99, 0.01)
    totals = [(total_score((probs >= t).astype(int))[0], t) for t in grid]
    t_best = max(totals)[1]
    tot, loc, pp, rj = total_score((probs >= t_best).astype(int))
    f1, p, rc, tp, fp, fn = rej_f1(y, (probs >= t_best).astype(int))
    f1best = max(rej_f1(y, (probs >= t).astype(int))[0] for t in grid)
    print(f"  total optimal threshold p>={t_best:.2f}: est core {tot:.1f} "
          f"(loc {loc:.1f} pose {pp:.1f} rej {rj:.1f})")
    print(f"  at that point: reject F1 {f1:.3f} precision {p:.2f} recall {rc:.2f} "
          f"(tp {tp} fp {fp} fn {fn}); best pure F1 on the sweep was {f1best:.3f}")
    best = (f1, t_best)

    peak = X[:, 0]
    b = max(((rej_f1(y, (peak >= t).astype(int))[0], t) for t in np.unique(np.round(peak, 3))))
    f1b, pb, rb, *_ = rej_f1(y, (peak >= b[1]).astype(int))
    print(f"  peak only baseline: reject F1 {f1b:.3f}  precision {pb:.2f} recall {rb:.2f} at peak>={b[1]:.3f}")

    w = fit_logistic(Xs, y)
    model = {"features": list(feats), "mu": mu.tolist(), "sd": sd.tolist(),
             "weights": w[:-1].tolist(), "bias": float(w[-1]),
             "prob_threshold": float(best[1]),
             "cv_reject_f1": float(f1),
             "trained_on": str(args.records), "n_pairs": len(y)}
    args.out.parent.mkdir(exist_ok=True)
    json.dump(model, open(args.out, "w"), indent=2)
    pf = 1 / (1 + np.exp(-(Xs @ w[:-1] + w[-1])))
    f1f, pfc, rfc, *_ = rej_f1(y, (pf >= best[1]).astype(int))
    print(f"  full fit train reject F1 {f1f:.3f} (precision {pfc:.2f} recall {rfc:.2f})")
    print(f"  wrote {args.out}")
    print("  weights by feature:")
    for name, wi in zip(feats, w[:-1]):
        print(f"    {name:10s} {wi:+.3f}")


if __name__ == "__main__":
    main()
