"""Resweep the raw override margin floor under the exact shipped trigger.

The original sweep chose 0.02 on a 0.02 step grid without the peak floor the
shipped gate applies. An external audit reproduced a damage case firing at
margin 0.0202, so the floor is reswept at 0.001 steps with the shipped
trigger, not agree and peak >= 0.25 and margin >= floor, choosing the plateau
midpoint on the fitting pool alone and validating on the held out suites.

    .venv/bin/python experiments/20260902_raw_override_hardening/sweep.py
"""

import csv
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
MIN_PEAK = 0.25

FIT = [
    ("experiments/20260902_161842_off_p2train/records.json", "data/p2train/ground_truth.csv"),
    ("experiments/20260902_162728_off_amatgen_train_s/records.json", "data/amatgen_train_s/ground_truth.csv"),
    ("experiments/20260902_162947_off_amatgen_train_h/records.json", "data/amatgen_train_h/ground_truth.csv"),
]
JUDGE = [
    ("experiments/20260902_163208_off_p2holdout/records.json", "data/p2holdout/ground_truth.csv"),
    ("experiments/20260902_163840_off_p2holdout2/records.json", "data/p2holdout2/ground_truth.csv"),
    ("experiments/20260902_164539_off_amatgen_holdout/records.json", "data/amatgen_holdout/ground_truth.csv"),
]


def credit(e):
    return 1.0 if e <= 1 else 0.8 if e <= 2 else 0.6 if e <= 3 else 0.4 if e <= 5 else 0.0


def load(rec_path, gt_path):
    gt = {r["pair_id"]: r for r in csv.DictReader(open(REPO / gt_path))}
    out = []
    for rec in json.load(open(REPO / rec_path)):
        g = gt[rec["pair_id"]]
        if g.get("modality", "sem") != "sem" or g["found"] != "1":
            continue
        rc = rec.get("raw_confirm") or {}
        row = {"set": rec["set"],
               "err_cls": rec["err"] if rec["err"] is not None else 1e9,
               "raw_err": 1e9, "margin": -1.0, "agree": True, "peak": 0.0}
        if rc.get("x") is not None:
            row.update(raw_err=float(np.hypot(float(rc["x"]) - float(g["gt_x"]),
                                              float(rc["y"]) - float(g["gt_y"]))),
                       margin=float(rc.get("margin") or 0.0),
                       agree=bool(rc.get("agree")),
                       peak=float(rc.get("peak") or 0.0))
        out.append(row)
    return out


def score(rows, floor):
    per = {"A_nominal": [], "B_degraded": []}
    fired = rescued = damaged = 0
    for r in rows:
        e = r["err_cls"]
        if not r["agree"] and r["peak"] >= MIN_PEAK and r["margin"] >= floor:
            fired += 1
            rescued += credit(r["raw_err"]) > credit(e)
            damaged += credit(r["raw_err"]) < credit(e)
            e = r["raw_err"]
        if r["set"] in per:
            per[r["set"]].append(credit(e))
    loc = 40 * (0.45 * np.mean(per["A_nominal"]) + 0.55 * np.mean(per["B_degraded"]))
    return loc, fired, rescued, damaged


def main():
    fit = [r for spec in FIT for r in load(*spec)]
    off = score(fit, 9.9)[0]
    ship = score(fit, 0.02)
    print(f"fitting pool {len(fit)} present pairs, override off {off:.2f} of 40, "
          f"shipped floor 0.02 {ship[0]:.2f} (fired {ship[1]}, rescued {ship[2]}, damaged {ship[3]})")
    grid = np.round(np.arange(0.0, 0.2501, 0.001), 4)
    vals = [(m, score(fit, m)) for m in grid]
    top = max(v[0] for _, v in vals)
    plateau = [m for m, v in vals if v[0] >= top - 1e-9]
    mid = round(float((min(plateau) + max(plateau)) / 2), 4)
    locm, fired, resc, dam = score(fit, mid)
    print(f"optimum {top:.2f} on floors {min(plateau):.3f} to {max(plateau):.3f}; "
          f"midpoint {mid} scores {locm:.2f} (fired {fired}, rescued {resc}, damaged {dam}) "
          f"chosen on the fitting pool BEFORE any held out number")
    for rec_path, gt_path in JUDGE:
        rows = load(rec_path, gt_path)
        off_j = score(rows, 9.9)[0]
        old = score(rows, 0.02)
        new = score(rows, mid)
        print(f"HELD OUT {Path(rec_path).parent.name}: off {off_j:.2f}, shipped 0.02 "
              f"{old[0]:.2f} (r{old[2]} d{old[3]}), floor {mid} {new[0]:.2f} "
              f"(r{new[2]} d{new[3]}), delta {new[0] - old[0]:+.2f}")


if __name__ == "__main__":
    main()
