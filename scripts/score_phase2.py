"""Phase 2 self-scorer (localization / pose / rejection / calibration).

Reads register.py predictions (pair_id,x,y,theta,scale,found,score) and the
mixed-set manifest (pair_id,reference,search,present,gt_x,gt_y) and reports the
four self-check metrics from the Phase 2 rubric:

  * Localization (40): tiered credit per present pair
      dist <=1 -> 1.0, <=2 -> 0.8, <=3 -> 0.6, <=5 -> 0.4, else 0.0
      A = fraction of present pairs with dist <= 5 (pass@5)
      B = mean tiered credit over present pairs
      score = 0.45*A + 0.55*B, scaled to 40.
  * Pose (20): scale credit (10) + rotation credit (10), only for present pairs
      that were found AND localized within 5 px. Linear credit with tolerance
      1.0x magnification and 5 deg; scaled to 20.
  * Rejection (15): F1 of `found` vs `present` over all pairs. Never rejecting
      (all found=1) scores 0.
  * Calibration (10): AUC of `score` vs whether the prediction was correct
      (found == present). Higher means the continuous score ranks correct
      predictions above incorrect ones.

Usage:
    python scripts/score_phase2.py --pred predictions.csv --manifest data/phase2_mixed/manifest.csv
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


def _read_rows(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


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
    cnt = sum(1 for p in pos for n in neg if p > n)
    return cnt / (len(pos) * len(neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, type=Path)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None,
                    help="optional JSON summary path")
    args = ap.parse_args()

    preds = {r["pair_id"]: r for r in _read_rows(args.pred)}
    man = {r["pair_id"]: r for r in _read_rows(args.manifest)}

    loc_credits, pass5, pose_scale, pose_rot, rej_labels = [], [], [], [], []
    cal_scores, cal_correct = [], []
    n_present = n_absent = 0
    for pid, m in man.items():
        p = preds[pid]
        present = int(m["present"])
        found = int(p["found"])
        rej_labels.append(found)
        cal_scores.append(float(p["score"]))
        cal_correct.append(1 if found == present else 0)
        if present:
            n_present += 1
            gx, gy = float(m["gt_x"]), float(m["gt_y"])
            if found:
                px, py = float(p["x"]), float(p["y"])
                dist = np.hypot(px - gx, py - gy)
                c = _loc_credit(dist)
                loc_credits.append(c)
                pass5.append(1 if dist <= 5.0 else 0)
                if dist <= 5.0:
                    se = abs(float(p["scale"]) - 10.0)
                    re = abs(float(p["theta"]))
                    pose_scale.append(max(0.0, 1.0 - se / 1.0))
                    pose_rot.append(max(0.0, 1.0 - re / 5.0))
            else:
                loc_credits.append(0.0)
                pass5.append(0)
        else:
            n_absent += 1

    A = float(np.mean(pass5)) if pass5 else 0.0
    B = float(np.mean(loc_credits)) if loc_credits else 0.0
    loc = 40.0 * (0.45 * A + 0.55 * B)

    pose = 20.0 * (0.5 * (np.mean(pose_scale) if pose_scale else 0.0)
                   + 0.5 * (np.mean(pose_rot) if pose_rot else 0.0))

    tp = sum(1 for a, b in zip(rej_labels, [int(m["present"]) for m in man.values()]) if a == 1 and b == 1)
    fp = sum(1 for a, b in zip(rej_labels, [int(m["present"]) for m in man.values()]) if a == 1 and b == 0)
    fn = sum(1 for a, b in zip(rej_labels, [int(m["present"]) for m in man.values()]) if a == 0 and b == 1)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    rej_f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    auc = _auc(cal_scores, cal_correct)

    result = {
        "n_pairs": len(man), "n_present": n_present, "n_absent": n_absent,
        "localization": round(loc, 2),
        "loc_A_pass5": round(A, 4), "loc_B_tiered": round(B, 4),
        "pose": round(pose, 2),
        "rejection_f1": round(rej_f1, 4),
        "rejection_precision": round(prec, 4),
        "rejection_recall": round(rec, 4),
        "rejection_tp_fp_fn": [tp, fp, fn],
        "calibration_auc": round(auc, 4),
        "weighted_total": round(loc + pose + rej_f1 * 15.0 + auc * 10.0, 2),
    }
    print(json.dumps(result, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
