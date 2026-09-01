"""Score a predictions csv against a ground truth in the organiser layout.

Applies the disclosed tiers: localization 1/0.8/0.6/0.4 at 1/2/3/5 px, scale
1/0.6/0.3 at 1/2/5 percent, rotation 1/0.6/0.3 at 0.25/0.5/1.0 degrees, pose
scored only where localization earned credit, reject class F1 on the found
flag, and rank AUC of the score column against per pair decision correctness.

    .venv/bin/python experiments/20260901_organiser_sample_validation/score_against_gt.py \
        --pred pred.csv --truth <suite>/ground_truth.csv
"""

import argparse
import csv

import numpy as np


def credit(e):
    return 1.0 if e <= 1 else 0.8 if e <= 2 else 0.6 if e <= 3 else 0.4 if e <= 5 else 0.0


def sc_credit(rel):
    return 1.0 if rel <= 0.01 else 0.6 if rel <= 0.02 else 0.3 if rel <= 0.05 else 0.0


def rc_credit(d):
    return 1.0 if d <= 0.25 else 0.6 if d <= 0.5 else 0.3 if d <= 1.0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--truth", required=True)
    args = ap.parse_args()
    gt = {r["pair_id"]: r for r in csv.DictReader(open(args.truth))}
    preds = {r["pair_id"]: r for r in csv.DictReader(open(args.pred))}

    per_set, errs = {}, {}
    tn = fp = fn = tp = 0
    pose_s, pose_r, ys, ss = [], [], [], []
    by_sev = {}
    for pid in sorted(gt):
        g, p = gt[pid], preds[pid]
        present, found = g["present"] == "1", p["found"] == "1"
        optical = g.get("set") == "D"
        if present and found:
            tp += 1
        elif present:
            fp += 1
        elif found:
            fn += 1
        else:
            tn += 1
        correct = False
        if present:
            err = (float(np.hypot(float(p["x"]) - float(g["x"]),
                                  float(p["y"]) - float(g["y"]))) if found else 1e9)
            c = credit(err) if err < 1e9 else 0.0
            per_set.setdefault(g.get("set", "?"), []).append(c)
            errs.setdefault(g.get("set", "?"), []).append(min(err, 9999))
            if not optical:
                by_sev.setdefault(int(g.get("severity", 0)), []).append(c)
            if found and c > 0:
                zrel = abs(float(p["scale"]) - float(g["scale"])) / float(g["scale"])
                terr = abs(float(p["theta"]) - float(g["theta"]))
                pose_s.append(sc_credit(zrel))
                pose_r.append(rc_credit(terr))
            correct = found and err <= 5
        else:
            correct = not found
        if not optical:
            ys.append(1 if correct else 0)
            ss.append(float(p["score"]))
    for s in sorted(per_set):
        print(f"  Set {s}: mean credit {np.mean(per_set[s]):.3f} over {len(per_set[s])}, "
              f"median err {np.median(errs[s]):.2f} px")
    for sv in sorted(by_sev):
        print(f"    severity {sv}: credit {np.mean(by_sev[sv]):.3f} over {len(by_sev[sv])}")
    a = per_set.get("A", [])
    b = per_set.get("B", [])
    loc = 40 * (0.45 * np.mean(a) + 0.55 * np.mean(b)) if a and b else 0.0
    pose = (10 * np.mean(pose_s) + 10 * np.mean(pose_r)) if pose_s else 0.0
    f1 = 2 * tn / max(2 * tn + fp + fn, 1)
    ys, ss = np.asarray(ys), np.asarray(ss)
    order = np.argsort(ss)
    ranks = np.empty(len(ss))
    ranks[order] = np.arange(1, len(ss) + 1)
    pos, neg = ys.sum(), len(ys) - ys.sum()
    auc = float((ranks[ys == 1].sum() - pos * (pos + 1) / 2) / (pos * neg)) \
        if 0 < pos < len(ys) else 0.5
    core = loc + pose + 15 * f1 + 10 * auc
    print(f"  loc {loc:.2f}/40  pose {pose:.2f}/20  reject F1 {f1:.3f} ({15 * f1:.2f}/15)  "
          f"auc {auc:.3f} ({10 * auc:.2f}/10)")
    print(f"  rejection detail: correct rejects {tn}, false rejects {fp}, false grabs {fn}")
    print(f"  ESTIMATED CORE {core:.2f} of 85")


if __name__ == "__main__":
    main()
