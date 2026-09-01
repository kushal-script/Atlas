"""Sweep the raw confirmation override offline from recorded diagnostics.

The override moves the answer to the raw full reference correlation's global
argmax when it disagrees with the pipeline's answer and its margin clears m.
Everything needed is in the records: the classical answer's error, the raw
argmax position, and the ground truth, so the sweep runs without touching the
localizer, on the fitting suites alone; the chosen margin is validated on the
held out suites afterwards, never chosen on them.

    .venv/bin/python experiments/20260901_raw_confirm_and_found_f1/sweep_raw_override.py \
        --fit rec_p2train.json rec_amat_s.json rec_amat_h.json --judge rec_holdout.json
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent


def credit(e):
    return 1.0 if e <= 1 else 0.8 if e <= 2 else 0.6 if e <= 3 else 0.4 if e <= 5 else 0.0


def load(records_path, gt_path):
    gt = {r["pair_id"]: r for r in csv.DictReader(open(gt_path))}
    out = []
    for rec in json.load(open(records_path)):
        g = gt[rec["pair_id"]]
        if g["modality"] != "sem" or g["found"] != "1":
            continue
        rc = rec.get("raw_confirm") or {}
        if rc.get("x") is None:
            continue
        raw_err = float(np.hypot(float(rc["x"]) - float(g["gt_x"]),
                                 float(rc["y"]) - float(g["gt_y"])))
        out.append({"set": rec["set"], "err_cls": rec["err"] if rec["err"] is not None else 1e9,
                    "raw_err": raw_err, "margin": float(rc.get("margin") or 0.0),
                    "agree": bool(rc.get("agree")), "peak": float(rc.get("peak") or 0.0)})
    return out


def score(rows, m):
    per = {"A_nominal": [], "B_degraded": []}
    fired = rescued = damaged = 0
    for r in rows:
        e = r["err_cls"]
        if not r["agree"] and r["margin"] >= m:
            fired += 1
            if credit(r["raw_err"]) > credit(e):
                rescued += 1
            elif credit(r["raw_err"]) < credit(e):
                damaged += 1
            e = r["raw_err"]
        if r["set"] in per:
            per[r["set"]].append(credit(e))
    loc = 40 * (0.45 * np.mean(per["A_nominal"]) + 0.55 * np.mean(per["B_degraded"]))
    return loc, fired, rescued, damaged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", nargs="+", required=True,
                    help="records.json,ground_truth.csv pairs joined by a comma")
    ap.add_argument("--judge", nargs="+", default=[])
    args = ap.parse_args()

    def load_many(specs):
        rows = []
        for spec in specs:
            rec, gt = spec.split(",")
            rows += load(rec, gt)
        return rows

    fit = load_many(args.fit)
    base = score(fit, 9.9)[0]
    print(f"fitting pool {len(fit)} present pairs, override off: {base:.2f} of 40")
    grid = np.round(np.arange(0.0, 0.61, 0.02), 3)
    vals = [(m, score(fit, m)) for m in grid]
    top = max(v[0] for _, v in vals)
    plateau = [m for m, v in vals if v[0] >= top - 1e-9]
    mid = round(float((min(plateau) + max(plateau)) / 2), 3)
    _, (locm, fired, resc, dam) = next(v for v in vals if v[0] == mid)
    print(f"optimum {top:.2f} on margins {min(plateau):.2f} to {max(plateau):.2f}; "
          f"midpoint {mid} (fired {fired}, rescued {resc}, damaged {dam}) chosen "
          f"BEFORE any held out number")
    for spec in args.judge:
        rec, gt = spec.split(",")
        rows = load(rec, gt)
        off = score(rows, 9.9)[0]
        on, fired, resc, dam = score(rows, mid)
        print(f"HELD OUT {Path(rec).parent.name}: off {off:.2f} -> at {mid} {on:.2f} "
              f"delta {on - off:+.2f} of 40 (fired {fired}, rescued {resc}, damaged {dam})")


if __name__ == "__main__":
    main()
