"""Sweep every candidate presence model over the pooled feature harvests and
choose the shipping model and threshold by the shipped protocol.

The harvests carry the full twenty one feature vector per pair, so each
candidate is rescored without running the localizer again. The threshold is
the midpoint of the plateau within a quarter core point of the pooled top,
never the argmax, because the argmax of one pooled draw fits the draw; the
same rule that shipped 0.45 for the fifteen feature model.

    .venv/bin/python experiments/20260901_presence_v3_refit/choose_model.py <model.json> ...
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "src"))
from drift_sense.presence import presence_probability

HARVESTS = ["local/thr21_b180.harvest.csv", "local/thr21_h2.harvest.csv",
            "local/thr21_h1.harvest.csv"]


def credit(e):
    return 1.0 if e <= 1 else 0.8 if e <= 2 else 0.6 if e <= 3 else 0.4 if e <= 5 else 0.0


def scale_credit(rel):
    return 1.0 if rel <= 0.01 else 0.6 if rel <= 0.02 else 0.3 if rel <= 0.05 else 0.0


def rot_credit(d):
    return 1.0 if d <= 0.25 else 0.6 if d <= 0.5 else 0.3 if d <= 1.0 else 0.0


def score_at(rows, thr):
    a, b, tp, fp, fn = [], [], 0, 0, 0
    pose_s, pose_r, ys, ss = [], [], [], []
    for r in rows:
        present = int(r["truth_found"]) == 1
        p = float(r["p_present"])
        found = p >= thr
        if not found and not present:
            tp += 1
        elif not found and present:
            fp += 1
        elif found and not present:
            fn += 1
        if present:
            c = credit(float(r["err_px"])) if found else 0.0
            (a if r["set"] == "A_nominal" else b).append(c)
            if c > 0:
                pose_s.append(scale_credit(float(r["scale_rel"])))
                pose_r.append(rot_credit(float(r["rot_abs_deg"])))
        conf = max(p, 1.0 - p)
        if found:
            conf *= 0.5 + 0.5 * min(int(r["quad_agree"]) / 4.0, 1.0)
        correct = (found and present and float(r["err_px"]) <= 5) or (not found and not present)
        ys.append(1 if correct else 0)
        ss.append(conf)
    loc = (0.45 * np.mean(a) + 0.55 * np.mean(b)) * 40
    pose = (np.mean(pose_s) * 10 + np.mean(pose_r) * 10) if pose_s else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    ys, ss = np.array(ys), np.array(ss)
    order = np.argsort(ss)
    ranks = np.empty(len(ss))
    ranks[order] = np.arange(1, len(ss) + 1)
    pos, neg = ys.sum(), len(ys) - ys.sum()
    auc = ((ranks[ys == 1].sum() - pos * (pos + 1) / 2) / (pos * neg)
           if 0 < pos < len(ys) else 0.5)
    return loc + pose + 15 * f1 + 10 * auc, loc, pose, f1, auc


def main(paths):
    rows = []
    for h in HARVESTS:
        rows += list(csv.DictReader(open(REPO / h)))
    print(f"pooled {len(rows)} pairs over {len(HARVESTS)} harvests")
    grid = np.round(np.arange(0.05, 0.96, 0.025), 4)
    for mp in paths:
        model = json.load(open(mp))
        names = model["features"]
        for r in rows:
            r["p_present"] = presence_probability(
                model, [float(r[f"feat_{n}"]) for n in names])
        swept = [(score_at(rows, t), t) for t in grid]
        top = max(s[0][0] for s in swept)
        plateau = [t for (s, t) in swept if s[0] >= top - 0.25]
        mid = plateau[len(plateau) // 2]
        core, loc, pose, f1, auc = score_at(rows, mid)
        print(f"{Path(mp).name:20s} top {top:6.2f}  plateau {min(plateau):.3f} to {max(plateau):.3f}"
              f"  midpoint {mid:.3f}  core there {core:6.2f}"
              f"  (loc {loc:.2f} pose {pose:.2f} F1 {f1:.3f} auc {auc:.3f})")


if __name__ == "__main__":
    main(sys.argv[1:])
