"""Score a predictions csv against a generated suite's ground truth.

Applies the Phase 2 scoring exactly as disclosed: tiered localization credit
weighted 0.45 nominal to 0.55 degraded, pose credit only where localization
earned credit, reject class F1 on the found flag, and AUC of the score column
against per pair correctness.

    .venv/bin/python scripts/score_predictions.py --pred pred.csv --truth data/p2holdout2/ground_truth.csv
"""

import argparse
import csv
import json

import numpy as np


def _pct(v):
    """Quartiles and tails, because a credit tier hides where inside it a value fell."""
    if not v:
        return {}
    a = np.asarray(v, float)
    return {"n": int(a.size), "median": float(np.median(a)),
            "p25": float(np.percentile(a, 25)), "p75": float(np.percentile(a, 75)),
            "p90": float(np.percentile(a, 90)), "max": float(a.max())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--truth", required=True)
    args = ap.parse_args()
    truth = {r["pair_id"]: r for r in csv.DictReader(open(args.truth))
             if r["modality"] == "sem"}
    preds = {r["pair_id"]: r for r in csv.DictReader(open(args.pred))}
    missing = set(truth) - set(preds)
    if missing:
        print(f"MISSING ROWS: {sorted(missing)[:5]} ...")

    loc = {"A_nominal": [], "B_degraded": []}
    pose_s, pose_r = [], []
    tp = fp = fn = tn = 0
    scores, labels = [], []
    by_sev = {}
    raw_scale, raw_rot, raw_err = [], [], []
    for pid, t in truth.items():
        p = preds.get(pid)
        t_found = int(t["found"]) == 1
        p_found = p is not None and p["found"] == "1"
        ok = False
        if t_found:
            err = (np.hypot(float(p["x"]) - float(t["gt_x"]),
                            float(p["y"]) - float(t["gt_y"])) if p_found else 1e9)
            c = 1.0 if err <= 1 else 0.8 if err <= 2 else 0.6 if err <= 3 else 0.4 if err <= 5 else 0.0
            loc[t["set"]].append(c)
            if c > 0:
                zerr = abs(float(p["scale"]) - float(t["gt_zoom"])) / float(t["gt_zoom"]) * 100
                rerr = abs(float(p["theta"]) - float(t["gt_rotation_deg"]))
                pose_s.append(1.0 if zerr <= 1 else 0.6 if zerr <= 2 else 0.3 if zerr <= 5 else 0.0)
                pose_r.append(1.0 if rerr <= 0.25 else 0.6 if rerr <= 0.5 else 0.3 if rerr <= 1.0 else 0.0)
                raw_scale.append(zerr); raw_rot.append(rerr)
            if p_found and err < 1e8:
                raw_err.append(err)
            key = f"{t['set']}/sev{t['severity']}"
            by_sev.setdefault(key, []).append(c)
            ok = p_found and err <= 5
        else:
            ok = not p_found
        scores.append(float(p["score"]) if p else 0.0)
        labels.append(1 if ok else 0)

    tp = sum(1 for pid, t in truth.items()
             if int(t["found"]) == 0 and preds.get(pid, {}).get("found") == "0")
    fp = sum(1 for pid, t in truth.items()
             if int(t["found"]) == 1 and preds.get(pid, {}).get("found") == "0")
    fn = sum(1 for pid, t in truth.items()
             if int(t["found"]) == 0 and preds.get(pid, {}).get("found") == "1")
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)

    credit_A = float(np.mean(loc["A_nominal"])) if loc["A_nominal"] else 0.0
    credit_B = float(np.mean(loc["B_degraded"])) if loc["B_degraded"] else 0.0
    loc_pts = 40 * (0.45 * credit_A + 0.55 * credit_B)
    pose_pts = (10 * float(np.mean(pose_s)) + 10 * float(np.mean(pose_r))) if pose_s else 0.0

    order = np.argsort(scores)
    ranks = np.empty(len(scores)); ranks[order] = np.arange(1, len(scores) + 1)
    pos = [i for i, l in enumerate(labels) if l]; neg = [i for i, l in enumerate(labels) if not l]
    auc = ((sum(ranks[i] for i in pos) - len(pos) * (len(pos) + 1) / 2)
           / max(len(pos) * len(neg), 1)) if pos and neg else 1.0

    sc = np.asarray(scores, float); lb = np.asarray(labels, float)
    brier = float(np.mean((sc - lb) ** 2)) if len(sc) else 0.0
    base = float(lb.mean()) if len(lb) else 0.0
    brier_ref = float(np.mean((base - lb) ** 2)) if len(lb) else 0.0
    rel = res = 0.0
    if len(sc):
        edges = np.linspace(0.0, 1.0 + 1e-9, 11)
        idx = np.clip(np.digitize(sc, edges) - 1, 0, 9)
        for b in range(10):
            m = idx == b
            if not m.any():
                continue
            w = m.sum() / len(sc)
            rel += w * (sc[m].mean() - lb[m].mean()) ** 2
            res += w * (lb[m].mean() - base) ** 2
    cdf = {f"within_{q}px": float(np.mean(np.asarray(raw_err) <= q)) if raw_err else 0.0
           for q in (0.5, 1.0, 2.0, 5.0)}

    rep = {"pairs": len(truth),
           "localization": {"credit_A": credit_A, "credit_B": credit_B, "points": loc_pts},
           "pose": {"scale_credit": float(np.mean(pose_s)) if pose_s else 0,
                    "rotation_credit": float(np.mean(pose_r)) if pose_r else 0,
                    "points": pose_pts},
           "rejection": {"f1": f1, "precision": prec, "recall": rec,
                         "tp": tp, "fp": fp, "fn": fn, "points": 15 * f1},
           "calibration": {"auc": float(auc), "points": 10 * float(auc),
                           "brier": brier, "brier_vs_base_rate": brier_ref,
                           "brier_reliability": rel, "brier_resolution": res},
           "per_severity_credit": {k: {"n": len(v), "credit": float(np.mean(v))}
                                   for k, v in sorted(by_sev.items())},
           "error_distribution": {
               "localization_px": _pct(raw_err),
               "scale_error_pct": _pct(raw_scale),
               "rotation_error_deg": _pct(raw_rot),
               "localization_cdf": cdf},
           "estimated_core": loc_pts + pose_pts + 15 * f1 + 10 * float(auc)}
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
