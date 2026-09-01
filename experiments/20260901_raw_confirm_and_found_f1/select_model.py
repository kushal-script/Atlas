"""Choose the shipping presence model and threshold under the corrected
scoring objective, over pooled feature harvests spanning both generators.

The organisers' addendum defines the rejection F1's false positive as a found
absent pair and its false negative as a rejected present pair, and their
reference scorer implements exactly that, so the graded F1 is computed on the
found class. This selector therefore optimises localization plus pose plus
fifteen times found class F1 plus ten times decision AUC, prints the reject
class reading beside it so the cost under the other interpretation is always
visible, and applies the plateau midpoint rule to the threshold.

    .venv/bin/python experiments/20260901_raw_confirm_and_found_f1/select_model.py <model.json> ...
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "src"))
from drift_sense.presence import presence_probability

HARVESTS = ["local/thr24_b180.harvest.csv", "local/thr24_h2.harvest.csv",
            "local/thr24_h1.harvest.csv", "local/thr24_amat.harvest.csv"]


def credit(e):
    return 1.0 if e <= 1 else 0.8 if e <= 2 else 0.6 if e <= 3 else 0.4 if e <= 5 else 0.0


def scale_credit(rel):
    return 1.0 if rel <= 0.01 else 0.6 if rel <= 0.02 else 0.3 if rel <= 0.05 else 0.0


def rot_credit(d):
    return 1.0 if d <= 0.25 else 0.6 if d <= 0.5 else 0.3 if d <= 1.0 else 0.0


def score_at(rows, thr):
    a, b = [], []
    tp = fp = fn = tn = 0
    pose_s, pose_r, ys, ss = [], [], [], []
    for r in rows:
        present = int(r["truth_found"]) == 1
        p = float(r["p_present"])
        found = p >= thr
        if present and found:
            tp += 1
        elif present:
            fn += 1
        elif found:
            fp += 1
        else:
            tn += 1
        if present:
            c = credit(float(r["err_px"])) if found else 0.0
            (a if r["set"] == "A_nominal" else b).append(c)
            if c > 0:
                pose_s.append(scale_credit(float(r["scale_rel"])))
                pose_r.append(rot_credit(float(r["rot_abs_deg"])))
        conf = max(p, 1.0 - p)
        if found:
            conf *= 0.5 + 0.5 * min(int(r["quad_agree"]) / 4.0, 1.0)
        ys.append(1 if ((found and present and float(r["err_px"]) <= 5)
                        or (not found and not present)) else 0)
        ss.append(conf)
    loc = (0.45 * np.mean(a) + 0.55 * np.mean(b)) * 40
    pose = (np.mean(pose_s) * 10 + np.mean(pose_r) * 10) if pose_s else 0.0
    f1_found = 2 * tp / max(2 * tp + fn + fp, 1)
    f1_rej = 2 * tn / max(2 * tn + fn + fp, 1)
    ys, ss = np.array(ys), np.array(ss)
    order = np.argsort(ss)
    ranks = np.empty(len(ss))
    ranks[order] = np.arange(1, len(ss) + 1)
    pos, neg = ys.sum(), len(ys) - ys.sum()
    auc = ((ranks[ys == 1].sum() - pos * (pos + 1) / 2) / (pos * neg)
           if 0 < pos < len(ys) else 0.5)
    base = loc + pose + 10 * auc
    return {"found": base + 15 * f1_found, "reject": base + 15 * f1_rej,
            "loc": loc, "pose": pose, "f1_found": f1_found, "f1_rej": f1_rej,
            "auc": auc, "fn": fn, "fp": fp}


def load_pool(model):
    pools = {}
    for h in HARVESTS:
        rows = list(csv.DictReader(open(REPO / h)))
        names = model["features"]
        for r in rows:
            r["p_present"] = presence_probability(
                model, [float(r[f"feat_{n}"]) for n in names])
        pools[Path(h).stem.replace(".harvest", "")] = rows
    return pools


def main(paths):
    grid = np.round(np.arange(0.05, 0.96, 0.025), 4)
    for mp in paths:
        model = json.load(open(mp))
        pools = load_pool(model)
        allrows = [r for rows in pools.values() for r in rows]
        swept = [(score_at(allrows, t), t) for t in grid]
        top = max(s["found"] for s, _ in swept)
        plateau = [t for s, t in swept if s["found"] >= top - 0.25]
        mid = plateau[len(plateau) // 2]
        s = score_at(allrows, mid)
        print(f"{Path(mp).name}: plateau {min(plateau):.3f} to {max(plateau):.3f} midpoint {mid:.3f}")
        print(f"  pooled at {mid:.3f}: found graded {s['found']:.2f}  reject graded {s['reject']:.2f}"
              f"  loc {s['loc']:.2f}  F1found {s['f1_found']:.3f}  F1rej {s['f1_rej']:.3f}"
              f"  auc {s['auc']:.3f}  (fn {s['fn']} fp {s['fp']})")
        for name, rows in pools.items():
            si = score_at(rows, mid)
            print(f"    {name:12s} found {si['found']:6.2f}  reject {si['reject']:6.2f}"
                  f"  F1f {si['f1_found']:.3f} F1r {si['f1_rej']:.3f}")


if __name__ == "__main__":
    main(sys.argv[1:])
