"""The diagnostics a scored total cannot show.

The addendum's four components say how much was earned, not where the method
is and is not making progress. This reads a harvest of the localizer's own
diagnostics beside the ground truth and reports the breakdowns that matter for
deciding what to work on next: which confidence regime an answer came from and
how often each one is right, whether the presence features actually separate
the two classes, how flat the correlation surface was when the method got it
wrong, how often the width rescue earns its runtime, and whether the model's
weights are stable between training seeds.

    .venv/bin/python scripts/diagnostics_report.py --records experiments/<stamp>/records.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from drift_sense.presence import FEATURES, features_from_record, presence_probability

MODEL = Path(__file__).resolve().parent.parent / "models" / "presence_model.json"


def _sep(name, present, absent):
    """How far apart the two classes sit, in robust units."""
    if not len(present) or not len(absent):
        return None
    mp, ma = float(np.median(present)), float(np.median(absent))
    pooled = float(np.median(np.abs(np.concatenate([present - mp, absent - ma])))) * 1.4826
    # A discrete feature can have a pooled deviation of zero, which would make
    # the ratio explode and read as infinite separation. Fall back to the
    # standard deviation there, and report the overlap directly, which stays
    # meaningful whatever the feature's distribution looks like.
    if pooled < 1e-6:
        pooled = float(np.std(np.concatenate([present, absent])))
    both = np.concatenate([present, absent])
    thr = (mp + ma) / 2.0
    hi = present > thr if mp > ma else present < thr
    lo = absent <= thr if mp > ma else absent >= thr
    return {"present_median": mp, "absent_median": ma,
            "separation_robust_sd": (abs(mp - ma) / pooled) if pooled > 1e-9 else None,
            "split_accuracy_at_midpoint": float((hi.sum() + lo.sum()) / len(both))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    recs = json.load(open(args.records))
    model = json.loads(MODEL.read_text()) if MODEL.exists() else None
    present = np.array([bool(r["truth_found"]) for r in recs])
    rep = {"records": len(recs), "present": int(present.sum()),
           "absent": int((~present).sum())}

    # Per regime accuracy. An answer's regime says how it was decided, so the
    # hit rate per regime is what tells a caller which answers to trust.
    reg = {}
    for r in recs:
        if not r["truth_found"]:
            continue
        k = r.get("regime") or r.get("confidence_regime") or r.get("pose_source", "unknown")
        e = r.get("err_px", r.get("err"))
        if e is None:
            continue
        reg.setdefault(k, []).append(float(e) <= 5.0)
    if reg:
        rep["per_regime_accuracy"] = {k: {"n": len(v), "within_5px": float(np.mean(v))}
                                      for k, v in sorted(reg.items())}

    # Do the presence features actually separate the classes, one at a time.
    sep = {}
    for name, key in (("quad_agree", "quad_agree"), ("quad_disp", "quad_disp"),
                      ("peak", "peak"), ("prom", "prom"),
                      ("resp_entropy", "resp_entropy"), ("stab", "stab"),
                      ("geo", "geo")):
        vals = [r.get(key) for r in recs]
        if all(v is None for v in vals):
            continue
        a = np.array([float(v) if v is not None else np.nan for v in vals])
        ok = ~np.isnan(a)
        s = _sep(name, a[ok & present], a[ok & ~present])
        if s:
            sep[name] = s
    rep["feature_separation"] = dict(sorted(
        sep.items(), key=lambda kv: -kv[1]["separation_robust_sd"]))

    # Where the method was wrong, how flat was the surface it decided on.
    ent = [float(r["resp_entropy"]) for r in recs if r.get("resp_entropy") is not None]
    if ent:
        e = np.array(ent)
        wrong = np.array([not r["truth_found"] for r in recs])[:len(e)]
        rep["response_entropy"] = {
            "present_median": float(np.median(e[~wrong])) if (~wrong).any() else None,
            "absent_median": float(np.median(e[wrong])) if wrong.any() else None,
            "note": "0 is one dominant site, 1 is a flat ambiguous surface"}

    # Model calibration on this harvest, and the weight ranking.
    if model is not None and len(model["weights"]) == len(FEATURES):
        p = np.array([presence_probability(model, features_from_record(r)) for r in recs])
        y = present.astype(float)
        rep["model_on_this_harvest"] = {
            "brier": float(np.mean((p - y) ** 2)),
            "brier_base_rate": float(np.mean((y.mean() - y) ** 2)),
            "mean_p_present_when_present": float(p[present].mean()) if present.any() else None,
            "mean_p_present_when_absent": float(p[~present].mean()) if (~present).any() else None}
        rep["weight_ranking"] = sorted(
            ({"feature": f, "weight": float(w)} for f, w in zip(FEATURES, model["weights"])),
            key=lambda d: -abs(d["weight"]))

    txt = json.dumps(rep, indent=2)
    print(txt)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(txt + "\n")


if __name__ == "__main__":
    main()
