"""Fit the v2 presence model with richer features.

Generates training data, runs the localizer to collect v2 diagnostics,
and trains a regularised logistic model on the extended feature set.

The v2 features add spatial, residual, and response-surface diagnostics
that the localizer already computes but v1 did not use.  The extra dimensions
let the model separate degraded present pairs from absent impostors that
the v1 model confuses.

    .venv/bin/python scripts/fit_presence_v2.py --num 400 --seed 5001 --out models/presence_model.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from drift_sense.generator import generate_pair
from drift_sense.localize import MatchConfig, load_gray, locate, optical_config
from drift_sense.presence import FEATURES, features_from_diag


def rej_f1(y, pred_found):
    tp = int(np.sum((y == 0) & (pred_found == 0)))
    fp = int(np.sum((y == 1) & (pred_found == 0)))
    fn = int(np.sum((y == 0) & (pred_found == 1)))
    p = tp / max(tp + fp, 1); rc = tp / max(tp + fn, 1)
    return 2 * p * rc / max(p + rc, 1e-9), p, rc, tp, fp, fn


def loc_credit(e):
    return 1.0 if e <= 1 else 0.8 if e <= 2 else 0.6 if e <= 3 else 0.4 if e <= 5 else 0.0


def sc_credit(zp):
    return 1.0 if zp <= 1 else 0.6 if zp <= 2 else 0.3 if zp <= 5 else 0.0


def rc_credit(rd):
    return 1.0 if rd <= 0.25 else 0.6 if rd <= 0.5 else 0.3 if rd <= 1.0 else 0.0


def fit_logistic(X, y, l2=1.0):
    n, d = X.shape
    def nll(w):
        m = X @ w[:d] + w[d]
        wgt = np.where(y == 1, 1.0, max((y == 1).sum() / max((y == 0).sum(), 1), 0.1))
        ll = wgt * (np.logaddexp(0, m) - y * m)
        return ll.sum() + l2 * (w[:d] @ w[:d])
    res = minimize(nll, np.zeros(d + 1), method="L-BFGS-B",
                   options={"maxiter": 2000, "ftol": 1e-12})
    return res.x


def generate_and_featurize(num_pairs, seed, style="mixed"):
    """Generate pairs and collect v2 diagnostics from the localizer."""
    records = []
    cfg = MatchConfig()
    cfg_opt = optical_config()
    master_rng = np.random.default_rng(seed)

    for i in range(num_pairs):
        pair_seed = int(master_rng.integers(0, 2**31))
        degrade = 0
        absent = False
        modality = "sem"

        # Composition: 35% nominal, 35% degraded, 20% absent, 10% optical
        r = master_rng.random()
        if r < 0.35:
            degrade = 0; absent = False; modality = "sem"
        elif r < 0.70:
            degrade = master_rng.integers(1, 5); absent = False; modality = "sem"
        elif r < 0.90:
            degrade = master_rng.integers(0, 3); absent = True; modality = "sem"
        else:
            degrade = 0; absent = False; modality = "optical"

        style_pick = master_rng.choice(["dram", "finfet"]) if style == "mixed" else style

        try:
            pair = generate_pair(pair_seed, style_pick, modality=modality,
                                 absent=absent, degrade=degrade)
        except Exception:
            continue

        ref_img = pair["reference"]
        search_img = pair["search"]
        meta = pair["meta"]
        gt = meta["ground_truth"]

        ref, ref_rgb = load_gray(ref_img) if isinstance(ref_img, np.ndarray) else (ref_img, False)
        search, search_rgb = load_gray(search_img) if isinstance(search_img, np.ndarray) else (search_img, False)

        # Actually load from saved files to test the full pipeline
        import tempfile, json as _json, os
        from PIL import Image
        with tempfile.TemporaryDirectory() as td:
            ref_path = Path(td) / "ref.png"
            search_path = Path(td) / "search.png"
            Image.fromarray(pair["reference"]).save(ref_path)
            Image.fromarray(pair["search"]).save(search_path)
            ref_arr, ref_rgb = load_gray(ref_path)
            search_arr, search_rgb = load_gray(search_path)
            active_cfg = cfg_opt if (ref_rgb or search_rgb) else cfg
            try:
                x, y, diag, resp = locate(ref_arr, search_arr, active_cfg)
            except Exception:
                continue

        err = float(np.hypot(x - gt["x"], y - gt["y"]))
        truth_found = 0 if absent else 1
        feats = features_from_diag(diag)

        record = {name: float(val) for name, val in zip(FEATURES, feats)}
        record["truth_found"] = truth_found
        record["err"] = err
        record["set"] = "A_nominal" if degrade == 0 and not absent else "B_degraded"
        record["zerr"] = abs(float(diag["scale"]) * active_cfg.zoom - meta["zoom"]) / meta["zoom"] * 100
        record["rerr"] = abs(float(diag["theta_deg"]) - meta["relative_rotation_deg"])
        records.append(record)

        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{num_pairs}] pairs collected, "
                  f"{sum(1 for r in records if r['truth_found'])} present, "
                  f"{sum(1 for r in records if not r['truth_found'])} absent")

    return records


def train_model(records):
    """Train a regularised logistic model on v2 features."""
    X = np.array([[r[f] for f in FEATURES] for r in records], float)
    y = np.array([1 if r["truth_found"] else 0 for r in records])

    mu = X.mean(0)
    sd = X.std(0) + 1e-9
    Xs = (X - mu) / sd

    # Stratified 5-fold CV
    rng = np.random.default_rng(0)
    folds = np.zeros(len(y), int)
    for cls in (0, 1):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        for k, i in enumerate(idx):
            folds[i] = k % 5

    print("=== stratified 5 fold cross validation ===")
    probs = np.zeros(len(y))
    for k in range(5):
        tr, te = folds != k, folds == k
        w = fit_logistic(Xs[tr], y[tr])
        probs[te] = 1 / (1 + np.exp(-(Xs[te] @ w[:-1] + w[-1])))

    # Threshold sweep
    grid = np.arange(0.02, 0.99, 0.01)
    best_f1 = 0
    best_t = 0.5
    for t in grid:
        f1, _, _, _, _, _ = rej_f1(y, (probs >= t).astype(int))
        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    f1, p, rc, tp, fp, fn = rej_f1(y, (probs >= best_t).astype(int))
    print(f"  best reject F1 {f1:.3f} at threshold {best_t:.2f} "
          f"(precision {p:.2f}, recall {rc:.2f}, tp={tp} fp={fp} fn={fn})")

    # Total score sweep
    def total_score(found):
        credits = {"A_nominal": [], "B_degraded": []}
        pose_s, pose_r = [], []
        for r, f in zip(records, found):
            if not r["truth_found"]:
                continue
            c = loc_credit(r["err"]) if (f and r["err"] is not None) else 0.0
            credits[r["set"]].append(c)
            if f and c > 0:
                pose_s.append(sc_credit(r["zerr"]))
                pose_r.append(rc_credit(r["rerr"]))
        loc = 40 * (0.45 * np.mean(credits["A_nominal"]) + 0.55 * np.mean(credits["B_degraded"]))
        pp = (10 * np.mean(pose_s) + 10 * np.mean(pose_r)) if pose_s else 0.0
        f1v = rej_f1(y, np.asarray(found, int))[0]
        return loc + pp + 15 * f1v, loc, pp, 15 * f1v

    totals = [(total_score((probs >= t).astype(int))[0], t) for t in grid]
    t_best = max(totals)[1]
    tot, loc, pp, rj = total_score((probs >= t_best).astype(int))
    f1_at_total, _, _, _, _, _ = rej_f1(y, (probs >= t_best).astype(int))
    print(f"  total optimal threshold p>={t_best:.2f}: est core {tot:.1f} "
          f"(loc {loc:.1f} pose {pp:.1f} rej {rj:.1f})")
    print(f"  at that point: reject F1 {f1_at_total:.3f}")

    # Full fit
    w = fit_logistic(Xs, y)

    model = {
        "features": list(FEATURES),
        "mu": mu.tolist(),
        "sd": sd.tolist(),
        "weights": w[:-1].tolist(),
        "bias": float(w[-1]),
        "prob_threshold": float(t_best),
        "cv_reject_f1": float(f1),
        "n_pairs": len(records),
        "feature_version": "v2",
    }

    # Feature importance
    print("\n  weights by feature:")
    for name, wi in zip(FEATURES, w[:-1]):
        print(f"    {name:25s} {wi:+.4f}")

    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=400,
                    help="number of pairs to generate")
    ap.add_argument("--seed", type=int, default=5001)
    ap.add_argument("--out", type=Path, default=REPO / "models" / "presence_model.json")
    ap.add_argument("--style", default="mixed", choices=["dram", "finfet", "mixed"])
    args = ap.parse_args()

    print(f"Generating {args.num} pairs with seed {args.seed}...")
    t0 = time.perf_counter()
    records = generate_and_featurize(args.num, args.seed, args.style)
    elapsed = time.perf_counter() - t0
    print(f"Collected {len(records)} records in {elapsed:.1f}s")

    n_present = sum(1 for r in records if r["truth_found"])
    n_absent = sum(1 for r in records if not r["truth_found"])
    print(f"  {n_present} present, {n_absent} absent")

    model = train_model(records)
    args.out.parent.mkdir(exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(model, fh, indent=2)
    print(f"\nWrote {args.out} ({len(FEATURES)} features, {model['n_pairs']} pairs)")


if __name__ == "__main__":
    main()
