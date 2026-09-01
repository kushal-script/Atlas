"""Fit the re rank combiner and the extended presence model from one harvest.

One pass per suite runs the scored localizer with the combiner recording its
own candidate statistics, so every fitted weight comes from the quantities the
scored path actually computes, at the estimated pose rather than the oracle
one. Fitting and validation never share a suite: fit on p2train, judge on
p2holdout2, and the margin is chosen on the fitting suite before any held out
number is read.

    .venv/bin/python scripts/fit_rerank.py --harvest data/p2train --out /tmp/rr_train.json
    .venv/bin/python scripts/fit_rerank.py --fit /tmp/rr_train.json
    .venv/bin/python scripts/fit_rerank.py --report /tmp/rr_train.json /tmp/rr_holdout.json
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from drift_sense.localize import MatchConfig, load_gray, locate
from drift_sense.presence import features_from_diag, presence_probability

STAT_NAMES = ["ncc", "edge_locked", "lattice_res", "whiteness",
              "grad_ncc", "matscale_ncc", "res_rms"]


def credit(e):
    for lim, c in ((1, 1.0), (2, 0.80), (3, 0.60), (5, 0.40)):
        if e <= lim:
            return c
    return 0.0


def harvest(dataset, out):
    rows = [r for r in csv.DictReader(open(f"{dataset}/ground_truth.csv"))
            if r["modality"] == "sem"]
    recs = []
    for r in rows:
        ref, _ = load_gray(f"{dataset}/{r['reference_path']}")
        src, _ = load_gray(f"{dataset}/{r['search_path']}")
        cfg = MatchConfig()
        cfg.rerank_combiner = True
        cfg.rerank_combiner_margin = 9.0     # record everything, override nothing
        cfg.rerank_record_stats = True
        x, y, d, _ = locate(ref, src, cfg)
        rr = d.get("rerank") or {}
        present = r["found"] == "1"
        gx = float(r["gt_x"]) if present else 0.0
        gy = float(r["gt_y"]) if present else 0.0
        half = (d.get("template_px_used", 100) - 1) / 2.0
        cands = []
        for c in rr.get("candidates", []):
            cy, cx = c["rc"]
            err = float(np.hypot(cx + half - gx, cy + half - gy)) if present else -1.0
            cands.append({"rc": c["rc"], "stats": c["stats"],
                          "is_true": bool(present and err <= 5.0), "err": err})
        e_cls = float(np.hypot(x - gx, y - gy)) if present else -1.0
        recs.append({"pair_id": r["pair_id"], "set": r["set"],
                     "severity": int(r["severity"]), "style": r["style"],
                     "present": present, "err_classical": e_cls,
                     "rerank": {k: rr.get(k) for k in
                                ("score", "margin", "agree", "classical_score", "top_rc",
                                 "classical_rc")},
                     "candidates": cands,
                     "v1_features": features_from_diag(d)})
        print(f"{r['pair_id']} cands={len(cands)}", flush=True)
    json.dump(recs, open(out, "w"))
    print("WROTE", out, len(recs))


def _logistic(X, y, l2=0.02):
    from scipy.optimize import minimize
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd

    def nll(w):
        m = Z @ w[:-1] + w[-1]
        return float(np.mean(np.logaddexp(0, m) - y * m) + l2 * np.sum(w[:-1] ** 2))

    w = minimize(nll, np.zeros(X.shape[1] + 1), method="L-BFGS-B").x
    return w[:-1], float(w[-1]), mu, sd


def fit(train_json):
    recs = json.load(open(train_json))
    X, y = [], []
    for r in recs:
        for c in r["candidates"]:
            X.append(c["stats"])
            y.append(int(c["is_true"]))
    X, y = np.asarray(X, float), np.asarray(y)
    w, b, mu, sd = _logistic(X, y)
    model = {"features": STAT_NAMES,
             "weights": [round(float(v), 6) for v in w],
             "bias": round(b, 6),
             "mu": [round(float(v), 6) for v in mu],
             "sd": [round(float(v), 6) for v in sd],
             "note": (f"Refit at the estimated pose from {train_json} over "
                      f"{len(recs)} pairs and {len(y)} candidates; "
                      "margin fitted separately by --report.")}
    path = Path(__file__).resolve().parent.parent / "models" / "rerank_combiner.json"
    json.dump(model, open(path, "w"), indent=2)
    print(f"  refit on {len(y)} candidates ({int(y.sum())} true) -> {path}")


def _score_margin(recs, margin, model):
    """Weighted localization credit if the override fired at this margin."""
    per = {"A_nominal": [], "B_degraded": []}
    mu = np.asarray(model["mu"]); sd = np.asarray(model["sd"])
    wv = np.asarray(model["weights"]); b = model["bias"]
    for r in recs:
        if not r["present"] or r["set"] not in per:
            continue
        probs = []
        for c in r["candidates"]:
            z = (np.asarray(c["stats"], float) - mu) / sd
            probs.append((1.0 / (1.0 + np.exp(-(z @ wv + b))), c))
        e = r["err_classical"]
        if probs:
            probs.sort(key=lambda t: -t[0])
            p_top, c_top = probs[0]
            cy, cx = r["rerank"]["classical_rc"]
            p_cls = next((p for p, c in probs
                          if np.hypot(c["rc"][0] - cy, c["rc"][1] - cx) <= 2), 0.0)
            far = np.hypot(c_top["rc"][0] - cy, c_top["rc"][1] - cx) > 2
            if far and p_top - p_cls >= margin:
                e = c_top["err"]
        per[r["set"]].append(credit(e) if e >= 0 else 0.0)
    return 40 * (0.45 * np.mean(per["A_nominal"]) + 0.55 * np.mean(per["B_degraded"]))


def report(train_json, holdout_json):
    model = json.load(open(Path(__file__).resolve().parent.parent
                           / "models" / "rerank_combiner.json"))
    tr = json.load(open(train_json))
    ho = json.load(open(holdout_json))
    base_tr = _score_margin(tr, 99.0, model)
    print(f"  fitting suite, override off: {base_tr:.2f} of 40")
    grid = np.round(np.arange(0.0, 0.51, 0.02), 3)
    vals = [(m, _score_margin(tr, m, model)) for m in grid]
    top = max(v for _, v in vals)
    plateau = [m for m, v in vals if abs(v - top) < 1e-9]
    mid = round(float((min(plateau) + max(plateau)) / 2), 3)
    print(f"  optimum {top:.2f} on margins {min(plateau):.2f} to {max(plateau):.2f}; "
          f"midpoint {mid} chosen BEFORE reading the holdout")
    base_ho = _score_margin(ho, 99.0, model)
    at_ho = _score_margin(ho, mid, model)
    print(f"  HELD OUT: off {base_ho:.2f} -> at {mid} {at_ho:.2f}   delta {at_ho-base_ho:+.2f} of 40")
    print(f"  ship rerank_combiner_margin = {mid} only if that delta is positive")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--harvest"); ap.add_argument("--out")
    ap.add_argument("--fit")
    ap.add_argument("--report", nargs=2)
    a = ap.parse_args()
    if a.harvest:
        harvest(a.harvest, a.out or "/tmp/rr_harvest.json")
    elif a.fit:
        fit(a.fit)
    elif a.report:
        report(*a.report)
