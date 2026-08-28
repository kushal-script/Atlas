"""Tune the Phase 2 presence rule and report the estimated blind score.

Runs the localizer over a generated Phase 2 suite, sweeps the found threshold,
and scores every component the addendum scores: tiered localization credit with
the degraded set weighted 0.55 against 0.45, pose credit only where the
location earned credit, rejection F1 over the grayscale pairs, and the AUC of
the score column against per pair correctness. The organiser sample pairs play
no part here; this runs only on this repository's own generator output.

    .venv/bin/python scripts/tune_phase2.py --dataset data/p2train --name p2_tune
    .venv/bin/python scripts/tune_phase2.py --dataset data/p2holdout --name p2_holdout --threshold 0.55
"""

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from drift_sense.localize import MatchConfig, load_gray, locate

REPO = Path(__file__).resolve().parent.parent


def loc_credit(err):
    if err <= 1: return 1.0
    if err <= 2: return 0.8
    if err <= 3: return 0.6
    if err <= 5: return 0.4
    return 0.0


def scale_credit(rel_pct):
    if rel_pct <= 1: return 1.0
    if rel_pct <= 2: return 0.6
    if rel_pct <= 5: return 0.3
    return 0.0


def rot_credit(err_deg):
    if err_deg <= 0.25: return 1.0
    if err_deg <= 0.5: return 0.6
    if err_deg <= 1.0: return 0.3
    return 0.0


def f1_at(recs, thr):
    """F1 of the reject class: the addendum's statement that a team which
    never rejects scores zero is only true of the reject class, since the
    present class F1 of an unconditional argmax is far from zero."""
    tp = sum(1 for r in recs if not r["truth_found"] and r["peak"] < thr)
    fp = sum(1 for r in recs if r["truth_found"] and r["peak"] < thr)
    fn = sum(1 for r in recs if not r["truth_found"] and r["peak"] >= thr)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return 2 * prec * rec / max(prec + rec, 1e-9), prec, rec, tp, fp, fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--name", default="p2_tune")
    ap.add_argument("--threshold", type=float, default=None,
                    help="evaluate at this fixed threshold instead of sweeping")
    args = ap.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = REPO / "experiments" / f"{stamp}_{args.name}"
    out_dir.mkdir(parents=True)

    rows = list(csv.DictReader(open(args.dataset / "ground_truth.csv")))
    gray = [r for r in rows if r["modality"] == "sem"]
    cfg = MatchConfig()
    recs = []
    for r in gray:
        ref, _ = load_gray(args.dataset / r["reference_path"])
        search, _ = load_gray(args.dataset / r["search_path"])
        t0 = time.perf_counter()
        x, y, d, _ = locate(ref, search, cfg)
        rt = time.perf_counter() - t0
        truth_found = int(r["found"]) == 1
        err = float(np.hypot(x - float(r["gt_x"]), y - float(r["gt_y"]))) if truth_found else None
        zerr = (abs(d["scale"] * cfg.zoom - float(r["gt_zoom"])) / float(r["gt_zoom"]) * 100
                if truth_found else None)
        rerr = (abs(cfg.theta_report_sign * d["theta_deg"] - float(r["gt_rotation_deg"]))
                if truth_found else None)
        s2 = d.get("stage2") or {}
        recs.append({"pair_id": r["pair_id"], "set": r["set"], "severity": int(r["severity"]),
                     "truth_found": truth_found, "err": err, "zerr": zerr, "rerr": rerr,
                     "peak": float(d["score"]), "prom": float(d.get("peak_prominence", 0)),
                     "wide": int(d.get("num_candidates_wide", 1)),
                     "strict": int(d.get("num_candidates", 1)),
                     "noise": float(d.get("search_noise_sigma", 0.0)),
                     "nominal": float(d.get("nominal_score", 0.0)),
                     "wide_score": float(d.get("wide_score", 0.0)),
                     "pose_source": d.get("pose_source", ""),
                     "over_p99": float(d.get("peak_over_p99", 0.0)),
                     "resp_median": float(d.get("resp_median", 0.0)),
                     "z": (float(s2["z"]) if s2.get("z") is not None else None),
                     "margin": (float(s2["margin"]) if s2.get("margin") is not None else None),
                     "mad": (float(s2["mad"]) if s2.get("mad") is not None else None),
                     "s2_used": bool(s2.get("used")),
                     "quad_disp": float(d.get("quad_disp", -1.0)),
                     "quad_agree": int(d.get("quad_agree", -1)),
                     "runtime": rt})
        print(f"{r['pair_id']} {r['set']:10s} peak {d['score']:.3f} "
              f"err {err if err is None else round(err, 1)}", flush=True)

    json.dump(recs, open(out_dir / "records.json", "w"), indent=1)

    if args.threshold is None:
        cands = sorted({round(r["peak"], 3) for r in recs})
        scored = [(f1_at(recs, t)[0], f1_at(recs, t)[1], t) for t in cands]
        best_f1 = max(s[0] for s in scored)
        # among thresholds within a hair of the best F1, take the most
        # precise, because a false grab silently corrupts a measurement while
        # a false reject costs one cheap rescan
        thr = max((s for s in scored if s[0] >= best_f1 - 1e-9), key=lambda s: (s[1], s[2]))[2]
    else:
        thr = args.threshold
    f1, prec, rec, tp, fp, fn = f1_at(recs, thr)

    A = [r for r in recs if r["set"] == "A_nominal"]
    B = [r for r in recs if r["set"] == "B_degraded"]
    present = [r for r in recs if r["truth_found"]]

    def set_loc(rs):
        return float(np.mean([loc_credit(r["err"]) for r in rs])) if rs else 0.0

    credit_A, credit_B = set_loc(A), set_loc(B)
    loc_pts = 40 * (0.45 * credit_A + 0.55 * credit_B)

    pose_scored = [r for r in present if loc_credit(r["err"]) > 0]
    sc = float(np.mean([scale_credit(r["zerr"]) for r in pose_scored])) if pose_scored else 0.0
    rc = float(np.mean([rot_credit(r["rerr"]) for r in pose_scored])) if pose_scored else 0.0
    pose_pts = 10 * sc + 10 * rc

    rej_pts = 15 * f1

    # AUC of the decision confidence against per pair correctness at thr
    def decision_ok(r):
        pred = r["peak"] >= thr
        if not r["truth_found"]:
            return not pred
        return pred and r["err"] is not None and r["err"] <= 5
    w = 0.08
    scores, labels = [], []
    for r in recs:
        p = 1.0 / (1.0 + np.exp(-(r["peak"] - thr) / w))
        conf = max(p, 1 - p)
        uniq = 1.0 / (1.0 + np.log1p(max(r["wide"], 1) - 1))
        scores.append(conf * (0.6 + 0.25 * uniq + 0.15 * min(max(r["prom"], 0) / 20, 1)))
        labels.append(1 if decision_ok(r) else 0)
    order = np.argsort(scores)
    ranks = np.empty(len(scores)); ranks[order] = np.arange(1, len(scores) + 1)
    pos = [i for i, l in enumerate(labels) if l]
    neg = [i for i, l in enumerate(labels) if not l]
    auc = ((sum(ranks[i] for i in pos) - len(pos) * (len(pos) + 1) / 2)
           / max(len(pos) * len(neg), 1)) if pos and neg else 1.0

    report = {
        "dataset": str(args.dataset), "pairs_gray": len(recs),
        "found_threshold": thr,
        "rejection": {"f1": f1, "precision": prec, "recall": rec,
                      "tp": tp, "fp": fp, "fn": fn},
        "localization": {"credit_A": credit_A, "credit_B": credit_B,
                         "points_of_40": loc_pts},
        "pose": {"scale_credit": sc, "rotation_credit": rc, "points_of_20": pose_pts},
        "calibration_auc": float(auc), "calibration_points_of_10": 10 * float(auc),
        "runtime_median_s": float(np.median([r["runtime"] for r in recs])),
        "runtime_max_s": float(max(r["runtime"] for r in recs)),
        "estimated_core_score": loc_pts + pose_pts + rej_pts + 10 * float(auc),
    }
    json.dump(report, open(out_dir / "report.json", "w"), indent=2)
    print(json.dumps(report, indent=2))
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
