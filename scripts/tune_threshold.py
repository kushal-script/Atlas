"""Sweep the presence threshold against the score the addendum actually awards.

The threshold trades two things that are not symmetric. Rejecting a present
pair forfeits its localization and pose credit, but measured on held out data
the pairs the model rejects would almost all have mislocalized anyway, so that
forfeit is close to nothing. Rejecting it also counts as a false positive for
the reject class, which is scored on F1 across every grayscale pair and is
worth fifteen points. A threshold tuned on localization alone therefore sits in
the wrong place, and one tuned on F1 alone ignores the pairs that would have
scored.

This runs the localizer once per pair, records the presence probability beside
the errors the scorer needs, and then sweeps the threshold over the recorded
values, so the expensive part happens once rather than once per threshold.

    .venv/bin/python scripts/tune_threshold.py --dataset data/p2holdout --out local/threshold.csv
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from scipy.ndimage import grey_dilation, grey_erosion

from drift_sense.localize import MatchConfig, load_gray, locate, optical_config
from drift_sense.presence import features_from_diag, presence_probability

RESCUE_PEAK_BELOW = 0.62
RESCUE_MARGIN = 0.02
RESCUE_START_BEFORE = 0.5
MODEL = Path(__file__).resolve().parent.parent / "models" / "presence_model.json"


def credit(e):
    return 1.0 if e <= 1 else 0.8 if e <= 2 else 0.6 if e <= 3 else 0.4 if e <= 5 else 0.0


def scale_credit(rel):
    return 1.0 if rel <= 0.01 else 0.6 if rel <= 0.02 else 0.3 if rel <= 0.05 else 0.0


def rot_credit(d):
    return 1.0 if d <= 0.25 else 0.6 if d <= 0.5 else 0.3 if d <= 1.0 else 0.0


def harvest(dataset, model):
    cfg, cfg_opt = MatchConfig(), optical_config()
    rows = []
    for r in csv.DictReader(open(dataset / "ground_truth.csv")):
        if r["modality"] != "sem":
            continue
        t0 = time.perf_counter()
        ref, ref_rgb = load_gray(dataset / r["reference_path"])
        search, search_rgb = load_gray(dataset / r["search_path"])
        active = cfg_opt if (ref_rgb or search_rgb) else cfg
        x, y, diag, _ = locate(ref, search, active, t_start=t0)
        if (float(diag["score"]) < RESCUE_PEAK_BELOW
                and int(diag.get("num_candidates_wide", 1)) > 1
                and not (ref_rgb or search_rgb)):
            for op in (grey_erosion, grey_dilation):
                if time.perf_counter() - t0 > RESCUE_START_BEFORE * active.time_budget_s:
                    break
                x2, y2, d2, _ = locate(op(ref, size=(3, 3)).astype(ref.dtype), search,
                                       active, t_start=t0)
                if float(d2["score"]) > float(diag["score"]) + RESCUE_MARGIN:
                    x, y, diag = x2, y2, d2
        p = presence_probability(model, features_from_diag(diag))
        err = float(np.hypot(x - float(r["gt_x"]), y - float(r["gt_y"])))
        s_rel = abs(float(diag["scale"]) * active.zoom - float(r["gt_zoom"])) / float(r["gt_zoom"])
        r_abs = abs(active.theta_report_sign * float(diag["theta_deg"])
                    - float(r["gt_rotation_deg"]))
        rows.append({"pair_id": r["pair_id"], "set": r["set"], "truth_found": int(r["found"]),
                     "p_present": f"{p:.6f}", "err_px": f"{err:.3f}",
                     "scale_rel": f"{s_rel:.5f}", "rot_abs_deg": f"{r_abs:.4f}",
                     "quad_agree": int(max(diag.get("quad_agree", 0), 0)),
                     "seconds": f"{time.perf_counter() - t0:.2f}"})
        print(f"{r['pair_id']} p={p:.3f} err={err:.2f}", flush=True)
    return rows


def score_at(rows, thr):
    a, b, rej_tp, rej_fp, rej_fn, pose_s, pose_r, pose_n = [], [], 0, 0, 0, [], [], 0
    ys, ss = [], []
    for r in rows:
        present = r["truth_found"] == 1
        found = float(r["p_present"]) >= thr
        if not found and not present:
            rej_tp += 1
        elif not found and present:
            rej_fp += 1
        elif found and not present:
            rej_fn += 1
        if present:
            c = credit(float(r["err_px"])) if found else 0.0
            (a if r["set"] == "A_nominal" else b).append(c)
            if c > 0:
                pose_s.append(scale_credit(float(r["scale_rel"])))
                pose_r.append(rot_credit(float(r["rot_abs_deg"])))
                pose_n += 1
        p = float(r["p_present"])
        conf = max(p, 1.0 - p)
        if found:
            conf *= 0.5 + 0.5 * min(int(r["quad_agree"]) / 4.0, 1.0)
        correct = (found and present and float(r["err_px"]) <= 5) or (not found and not present)
        ys.append(1 if correct else 0); ss.append(conf)
    loc = (0.45 * (np.mean(a) if a else 0) + 0.55 * (np.mean(b) if b else 0)) * 40
    pose = ((np.mean(pose_s) if pose_s else 0) * 10 + (np.mean(pose_r) if pose_r else 0) * 10)
    f1 = 2 * rej_tp / (2 * rej_tp + rej_fp + rej_fn) if rej_tp else 0.0
    ys, ss = np.array(ys), np.array(ss)
    if 0 < ys.sum() < len(ys):
        order = np.argsort(ss)
        ranks = np.empty(len(ss)); ranks[order] = np.arange(1, len(ss) + 1)
        pos, neg = ys.sum(), len(ys) - ys.sum()
        auc = (ranks[ys == 1].sum() - pos * (pos + 1) / 2) / (pos * neg)
    else:
        auc = 0.5
    return {"threshold": thr, "loc": loc, "pose": pose, "f1": f1, "rej": f1 * 15,
            "auc": auc, "cal": auc * 10, "core": loc + pose + f1 * 15 + auc * 10,
            "tp": rej_tp, "fp": rej_fp, "fn": rej_fn,
            "credit_A": float(np.mean(a)) if a else 0.0,
            "credit_B": float(np.mean(b)) if b else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--harvest", type=Path, default=None,
                    help="reuse a previous harvest instead of running the localizer again")
    ap.add_argument("--pool", type=Path, nargs="*", default=[],
                    help="additional harvests to pool into the sweep. One suite of forty odd "
                         "degraded pairs carries enough spread that a threshold chosen on it "
                         "is chosen partly on its draw; pooling several disjoint seeds picks "
                         "an operating point that does not depend on which one was measured")
    args = ap.parse_args()
    model = json.loads(MODEL.read_text())

    if args.harvest and args.harvest.exists():
        rows = list(csv.DictReader(open(args.harvest)))
        rows = [dict(r, truth_found=int(r["truth_found"]), quad_agree=int(r["quad_agree"]))
                for r in rows]
    else:
        rows = harvest(args.dataset, model)
        target = args.harvest or args.out.with_suffix(".harvest.csv")
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    for extra in args.pool:
        add = list(csv.DictReader(open(extra)))
        rows += [dict(r, truth_found=int(r["truth_found"]), quad_agree=int(r["quad_agree"]))
                 for r in add]
        print(f"  pooled in {extra} ({len(add)} pairs)")
    if args.pool:
        n_abs = sum(1 for r in rows if r["truth_found"] == 0)
        print(f"  sweeping over {len(rows)} pairs, {n_abs} of them absent")

    shipped = float(model["prob_threshold"])
    grid = sorted(set(list(np.round(np.arange(0.05, 0.96, 0.025), 4)) + [shipped]))
    out = [score_at(rows, t) for t in grid]
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
    best = max(out, key=lambda d: d["core"])
    cur = min(out, key=lambda d: abs(d["threshold"] - shipped))
    print(f"\n  {'thr':>6}{'loc':>8}{'pose':>7}{'rej':>7}{'cal':>7}{'CORE':>8}"
          f"{'F1':>7}{'tp':>4}{'fp':>4}{'fn':>4}")
    for d in out:
        if d["threshold"] % 0.1 < 1e-6 or d is best or d is cur:
            tag = "  <- shipped" if d is cur else ("  <- best" if d is best else "")
            print(f"  {d['threshold']:>6.3f}{d['loc']:>8.2f}{d['pose']:>7.2f}{d['rej']:>7.2f}"
                  f"{d['cal']:>7.2f}{d['core']:>8.2f}{d['f1']:>7.3f}"
                  f"{d['tp']:>4}{d['fp']:>4}{d['fn']:>4}{tag}")
    print(f"\n  shipped threshold {shipped:.3f} scores {cur['core']:.2f}")
    print(f"  best threshold    {best['threshold']:.3f} scores {best['core']:.2f}"
          f"   gain {best['core'] - cur['core']:+.2f} points")


if __name__ == "__main__":
    main()
