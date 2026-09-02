"""Judge candidate presence models offline under the authoritative scoring.

Reads the fresh extended records, rebuilds each pair's decision from a fitted
model exactly as register.py would, applies the raw override at the shipped
0.05 floor to the effective error, and scores localization gated on the found
flag, pose gated on localization credit, rejection under both F1 readings,
and calibration from the shipped score construction, so a candidate's per
suite estimated core is what register.py would earn through the real scorer.

    .venv/bin/python experiments/20260902_presence_dev_refit/eval_models.py \
        --model models/presence_model.json --suites <name>=<records>,<gt> ...
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "src"))

from drift_sense.presence import features_for_model_record, presence_probability


def loc_credit(e):
    if e is None:
        return 0.0
    return 1.0 if e <= 1 else 0.8 if e <= 2 else 0.6 if e <= 3 else 0.4 if e <= 5 else 0.0


def sc_credit(z):
    return 1.0 if z <= 1 else 0.6 if z <= 2 else 0.3 if z <= 5 else 0.0


def rot_credit(r):
    return 1.0 if r <= 0.25 else 0.6 if r <= 0.5 else 0.3 if r <= 1.0 else 0.0


def load_suite(rec_path, gt_path):
    gt = {r["pair_id"]: r for r in csv.DictReader(open(REPO / gt_path))}
    rows = []
    for rec in json.load(open(REPO / rec_path)):
        g = gt[rec["pair_id"]]
        if g.get("modality", "sem") != "sem":
            continue
        err = rec["err"]
        rc = rec.get("raw_confirm") or {}
        if (rc.get("x") is not None and not rc.get("agree")
                and (rc.get("peak") or 0) >= 0.25 and (rc.get("margin") or 0) >= 0.05):
            err = float(np.hypot(float(rc["x"]) - float(g["gt_x"]),
                                 float(rc["y"]) - float(g["gt_y"])))
        rec = dict(rec, err_eff=err)
        rows.append(rec)
    return rows


def probs(model, rows):
    return np.array([presence_probability(model, features_for_model_record(model, r))
                     for r in rows])


def score_row(p, found, rec, mode):
    s = max(p, 1.0 - p)
    if found:
        qa = max(int(rec.get("quad_agree", -1)), 0)
        s *= 0.5 + 0.5 * min(qa / 4.0, 1.0)
        if mode in ("quad_raw", "quad_raw_rr"):
            rc = rec.get("raw_confirm") or {}
            agree_eff = bool(rc.get("agree")) or (
                rc.get("x") is not None and (rc.get("peak") or 0) >= 0.25
                and (rc.get("margin") or 0) >= 0.05)
            s *= 0.7 + 0.3 * (1.0 if agree_eff else 0.0)
        if mode in ("quad_rr", "quad_raw_rr"):
            rr = rec.get("rerank") or {}
            s *= 0.85 + 0.15 * (1.0 if rr.get("agree", True) else 0.0)
    return s


def evaluate(model, rows, thr, mode="quad"):
    ps = probs(model, rows)
    loc = {"A_nominal": [], "B_degraded": []}
    pose_s, pose_r = [], []
    tp_f = fp_f = fn_f = 0
    tp_r = fp_r = fn_r = 0
    scores, labels = [], []
    for p, r in zip(ps, rows):
        found = p >= thr
        t_found = bool(r["truth_found"])
        if t_found:
            c = loc_credit(r["err_eff"]) if found else 0.0
            if r["set"] in loc:
                loc[r["set"]].append(c)
            if found and c > 0 and r["zerr"] is not None:
                pose_s.append(sc_credit(r["zerr"]))
                pose_r.append(rot_credit(r["rerr"]))
            ok = found and r["err_eff"] is not None and r["err_eff"] <= 5
        else:
            ok = not found
        tp_f += t_found and found
        fp_f += (not t_found) and found
        fn_f += t_found and not found
        tp_r += (not t_found) and not found
        fp_r += t_found and not found
        fn_r += (not t_found) and found
        scores.append(score_row(p, found, r, mode))
        labels.append(1 if ok else 0)

    def f1(tp, fp, fn):
        pr = tp / max(tp + fp, 1)
        rc_ = tp / max(tp + fn, 1)
        return 2 * pr * rc_ / max(pr + rc_, 1e-9)

    order = np.argsort(scores)
    ranks = np.empty(len(scores)); ranks[order] = np.arange(1, len(scores) + 1)
    pos = [i for i, l in enumerate(labels) if l]
    neg = [i for i, l in enumerate(labels) if not l]
    auc = ((sum(ranks[i] for i in pos) - len(pos) * (len(pos) + 1) / 2)
           / max(len(pos) * len(neg), 1)) if pos and neg else 1.0

    la = float(np.mean(loc["A_nominal"])) if loc["A_nominal"] else 0.0
    lb = float(np.mean(loc["B_degraded"])) if loc["B_degraded"] else 0.0
    loc_pts = 40 * (0.45 * la + 0.55 * lb)
    pose_pts = (10 * float(np.mean(pose_s)) + 10 * float(np.mean(pose_r))) if pose_s else 0.0
    f1f, f1r = f1(tp_f, fp_f, fn_f), f1(tp_r, fp_r, fn_r)
    return {"loc": loc_pts, "pose": pose_pts, "f1_found": f1f, "f1_reject": f1r,
            "auc": float(auc),
            "core_found": loc_pts + pose_pts + 15 * f1f + 10 * float(auc),
            "core_reject": loc_pts + pose_pts + 15 * f1r + 10 * float(auc)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", nargs="+", required=True)
    ap.add_argument("--suites", nargs="+", required=True,
                    help="name=records.json,ground_truth.csv")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--sweep_pool", nargs="*", default=[],
                    help="suite names whose pairs pool the threshold sweep")
    ap.add_argument("--score_mode", default="quad",
                    choices=("quad", "quad_raw", "quad_rr", "quad_raw_rr"))
    args = ap.parse_args()

    suites = {}
    for spec in args.suites:
        name, paths = spec.split("=")
        rec_p, gt_p = paths.split(",")
        suites[name] = load_suite(rec_p, gt_p)

    for mp in args.model:
        model = json.load(open(REPO / mp))
        thr = args.threshold if args.threshold is not None else model.get("prob_threshold", 0.35)
        if args.sweep_pool:
            pool = [r for n in args.sweep_pool for r in suites[n]]
            grid = np.round(np.arange(0.05, 0.951, 0.025), 3)
            vals = []
            for t in grid:
                e = evaluate(model, pool, t, args.score_mode)
                vals.append((t, 0.5 * (e["core_found"] + e["core_reject"])))
            top = max(v for _, v in vals)
            plateau = [t for t, v in vals if v >= top - 0.25]
            thr = round(float((min(plateau) + max(plateau)) / 2), 3)
            print(f"{mp} pooled sweep: top {top:.2f}, plateau {min(plateau)} to "
                  f"{max(plateau)}, midpoint threshold {thr}")
        print(f"model {mp} ({len(model['weights'])} features) at threshold {thr}, "
              f"score mode {args.score_mode}")
        for name, rows in suites.items():
            e = evaluate(model, rows, thr, args.score_mode)
            print(f"  {name:12s} core {e['core_reject']:.2f}/{e['core_found']:.2f} "
                  f"(rej/found)  loc {e['loc']:.2f}  pose {e['pose']:.2f}  "
                  f"F1 {e['f1_reject']:.3f}/{e['f1_found']:.3f}  auc {e['auc']:.3f}")


if __name__ == "__main__":
    main()
