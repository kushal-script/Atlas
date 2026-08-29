"""Phase 2 Day 5 analysis: T1 (calibration AUC), T2 (threshold sweep), T4 (scorecard).

Reads the one-pass localize dump (scripts/localize_dump.py) so the threshold
sweep and AUC analysis run instantly without re-localizing. Writes the final
predictions.csv (equal to register.py with score=presence_probability).

Usage:
    uv run python scripts/analyze_day5.py --dump1 data/phase2_mixed/dump1.json \
        --dump2 data/phase2_mixed/dump2.json \
        --predictions data/phase2_mixed/predictions.csv --json experiments/score_day5.json
"""

import argparse
import csv
import json
import sys
from bisect import bisect_left, bisect_right
from pathlib import Path

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from drift_sense.api import presence_probability_from_features


def _read(path):
    return json.load(open(path))


def _dist(r):
    if r["gt_x"] is None or r["gt_y"] is None:
        return None
    return float(np.hypot(r["x"] - r["gt_x"], r["y"] - r["gt_y"]))


def _loc_credit(dist):
    if dist <= 1.0:
        return 1.0
    if dist <= 2.0:
        return 0.8
    if dist <= 3.0:
        return 0.6
    if dist <= 5.0:
        return 0.4
    return 0.0


def _auc(scores, labels):
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return float("nan")
    neg_sorted = sorted(neg)
    total = 0.0
    for p in pos:
        lt = bisect_left(neg_sorted, p)
        le = bisect_right(neg_sorted, p)
        total += lt + 0.5 * (le - lt)
    return total / (len(pos) * len(neg))


def _found(r, thr):
    return 1 if r["prob"] >= thr else 0


def _correctness(r, thr):
    """correct iff (present and dist<=5 and found) or (absent and not found)."""
    f = _found(r, thr)
    if r["present"]:
        d = _dist(r)
        if d is None:
            return 0
        return 1 if (f == 1 and d <= 5.0) else 0
    return 1 if f == 0 else 0


def _rejection(rows, thr):
    tp = fp = fn = 0
    for r in rows:
        f = _found(r, thr)
        if r["present"] and f == 1:
            tp += 1
        elif (not r["present"]) and f == 1:
            fp += 1
        elif r["present"] and f == 0:
            fn += 1
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"f1": f1, "prec": prec, "rec": rec, "tp": tp, "fp": fp, "fn": fn}


def _localization(rows, thr):
    credits, pass5 = [], []
    for r in rows:
        if not r["present"]:
            continue
        f = _found(r, thr)
        if f == 1 and _dist(r) is not None:
            d = _dist(r)
            credits.append(_loc_credit(d))
            pass5.append(1 if d <= 5.0 else 0)
        else:
            credits.append(0.0)
            pass5.append(0)
    A = float(np.mean(pass5)) if pass5 else 0.0
    B = float(np.mean(credits)) if credits else 0.0
    return 40.0 * (0.45 * A + 0.55 * B), A, B


def _pose(rows, thr):
    sc, rc = [], []
    for r in rows:
        if not r["present"]:
            continue
        f = _found(r, thr)
        d = _dist(r)
        if f == 1 and d is not None and d <= 5.0:
            sc.append(max(0.0, 1.0 - abs(r["scale"] - 10.0) / 1.0))
            rc.append(max(0.0, 1.0 - abs(r["theta"]) / 5.0))
    return 20.0 * (0.5 * (np.mean(sc) if sc else 0.0) + 0.5 * (np.mean(rc) if rc else 0.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump1", required=True, type=Path)
    ap.add_argument("--dump2", type=Path, default=None)
    ap.add_argument("--thr", type=float, default=None, help="override chosen threshold")
    ap.add_argument("--predictions", type=Path, default=None)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    d1 = _read(args.dump1)
    d2 = _read(args.dump2) if args.dump2 else None

    # Recompute P(present) with the CURRENT trained model (dump may predate a
    # retrain); features are unchanged by retraining, only the weights are.
    for r in d1:
        r["prob"] = presence_probability_from_features(r["features"])
    if d2:
        for r in d2:
            r["prob"] = presence_probability_from_features(r["features"])

    sweep = [0.30, 0.35, 0.40, 0.45, 0.50, 0.52, 0.55]
    sweep_rows = []
    for thr in sweep:
        r1 = _rejection(d1, thr)
        r2 = _rejection(d2, thr) if d2 else None
        sweep_rows.append({
            "thr": thr,
            "f1_run1": round(r1["f1"], 4),
            "prec1": round(r1["prec"], 4), "rec1": round(r1["rec"], 4),
            "tp_fp_fn_1": [r1["tp"], r1["fp"], r1["fn"]],
            "f1_run2": (round(r2["f1"], 4) if r2 else None),
            "min_f1": (round(min(r1["f1"], r2["f1"]), 4) if r2 else round(r1["f1"], 4)),
        })

    if args.thr is not None:
        chosen = args.thr
    else:
        best = None
        for s in sweep_rows:
            score = s["min_f1"]
            if best is None or score > best[1] or (abs(score - best[1]) < 1e-9 and s["thr"] < best[0]):
                best = (s["thr"], score)
        chosen = best[0]

    # T1 calibration AUC (at chosen threshold's found decisions)
    corr1 = [_correctness(r, chosen) for r in d1]
    auc_before = _auc([r["diag_score"] for r in d1], corr1)
    auc_after = _auc([r["prob"] for r in d1], corr1)
    # Alternative scores ranked by the same localization-aware correctness:
    auc_geo = _auc([float(r["features"]["geo_consistency"]) for r in d1], corr1)
    # Rejection-correctness (found == present) -- what presence_probability is trained for
    rej_corr = [1 if (_found(r, chosen) == (1 if r["present"] else 0)) else 0 for r in d1]
    auc_prob_rej = _auc([r["prob"] for r in d1], rej_corr)
    auc_geo_rej = _auc([float(r["features"]["geo_consistency"]) for r in d1], rej_corr)
    # Combined score: presence x alignment
    combined = [r["prob"] * float(r["features"]["geo_consistency"]) for r in d1]
    auc_combined = _auc(combined, corr1)

    loc, A, B = _localization(d1, chosen)
    pose = _pose(d1, chosen)
    rej = _rejection(d1, chosen)

    result = {
        "chosen_threshold": chosen,
        "sweep": sweep_rows,
        "t1_calibration_auc": {
            "before_raw_peak_vs_loc_correct": round(auc_before, 4),
            "after_presence_prob_vs_loc_correct": round(auc_after, 4),
            "geo_consistency_vs_loc_correct": round(auc_geo, 4),
            "presence_prob_vs_rejection_correct": round(auc_prob_rej, 4),
            "geo_vs_rejection_correct": round(auc_geo_rej, 4),
            "combined_prob_x_geo_vs_loc_correct": round(auc_combined, 4),
        },
        "t4_scorecard": {
            "localization": round(loc, 2), "loc_A_pass5": round(A, 4), "loc_B_tiered": round(B, 4),
            "pose": round(pose, 2),
            "rejection_f1": round(rej["f1"], 4),
            "rejection_precision": round(rej["prec"], 4),
            "rejection_recall": round(rej["rec"], 4),
            "rejection_tp_fp_fn": [rej["tp"], rej["fp"], rej["fn"]],
            "calibration_auc": round(auc_combined, 4),
            "n_present": sum(1 for r in d1 if r["present"]),
            "n_absent": sum(1 for r in d1 if not r["present"]),
        },
    }

    if args.predictions:
        rows_out = []
        for r in d1:
            f = _found(r, chosen)
            x = y = theta = scale = 0.0
            if f == 1:
                x, y, theta, scale = r["x"], r["y"], r["theta"], r["scale"]
            conf = r["prob"] * float(r["features"]["geo_consistency"])
            rows_out.append({
                "pair_id": r["pid"], "x": f"{x:.3f}", "y": f"{y:.3f}",
                "theta": f"{theta:.4f}", "scale": f"{scale:.4f}",
                "found": int(f), "score": f"{conf:.4f}",
            })
        args.predictions.parent.mkdir(parents=True, exist_ok=True)
        with open(args.predictions, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["pair_id", "x", "y", "theta", "scale", "found", "score"])
            w.writeheader()
            w.writerows(rows_out)
        result["predictions_written"] = str(args.predictions)

    print(json.dumps(result, indent=2))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
