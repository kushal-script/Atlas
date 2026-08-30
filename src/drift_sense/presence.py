"""Feature construction for the presence decision.

register.py builds features from a live locate() diagnostics dict, and the
fitting script builds them from recorded rows. Both constructions live here,
side by side, because a silent divergence between them would ship a model that
was fitted on one feature space and evaluated on another; a unit test asserts
they agree.

margin_over_mad is the absolute residual evidence: how far the winning site's
deviation field score stands above the candidate median, in robust units. The
z statistic asks whether one site stands out among its peers, which an
impostor site can do by chance; this asks whether the deviation field matched
at all, which is the question presence actually turns on.
"""

import numpy as np

FEATURES = ("peak", "prom_l", "wide_l", "strict_l", "noise", "nominal",
            "gap", "over_p99", "z_fill", "z_missing", "mom", "pose_wide",
            "quad_disp", "quad_agree", "quad_missing")


def _assemble(peak, prom, wide, strict, noise, nominal, over_p99,
              z, margin, mad, pose_source, quad_disp=-1.0, quad_agree=-1):
    mom = 0.0
    if margin is not None and mad is not None:
        mom = float(margin) / max(1.4826 * float(mad), 1e-6)
    return [
        float(peak),
        float(np.log1p(max(float(prom), 0.0))),
        float(np.log1p(int(wide))),
        float(np.log1p(int(strict))),
        float(noise),
        float(nominal),
        float(peak) - float(nominal),
        float(over_p99),
        float(z) if z is not None else 0.0,
        0.0 if z is not None else 1.0,
        float(np.clip(mom, -20.0, 20.0)),
        # The localizer reports this as "wide_grid" or "nominal". An earlier
        # revision compared against "wide", which never matches, so the feature
        # was constant zero through the fit that produced the shipped weights
        # and carries a weight of exactly zero there: correcting the comparison
        # changes no shipped decision, and the feature stays inert until a
        # refit gives it a weight. Both the fit and the inference path build
        # features through this one function, so they cannot disagree.
        1.0 if str(pose_source).startswith("wide") else 0.0,
        float(np.clip(quad_disp, 0.0, 17.0)) if quad_disp >= 0 else 0.0,
        float(quad_agree) if quad_agree >= 0 else 0.0,
        1.0 if quad_disp < 0 else 0.0,
    ]


def features_from_diag(diag):
    s2 = diag.get("stage2") or {}
    return _assemble(diag["score"], diag.get("peak_prominence", 0.0),
                     diag.get("num_candidates_wide", 1),
                     diag.get("num_candidates", 1),
                     diag.get("search_noise_sigma", 0.0),
                     diag.get("nominal_score", 0.0),
                     diag.get("peak_over_p99", 0.0),
                     s2.get("z"), s2.get("margin"), s2.get("mad"),
                     diag.get("pose_source", ""),
                     diag.get("quad_disp", -1.0), diag.get("quad_agree", -1))


def features_from_record(rec):
    return _assemble(rec["peak"], rec["prom"], rec["wide"],
                     rec.get("strict", 1), rec.get("noise", 0.0),
                     rec.get("nominal", 0.0), rec.get("over_p99", 0.0),
                     rec.get("z"), rec.get("margin"), rec.get("mad"),
                     rec.get("pose_source", ""),
                     rec.get("quad_disp", -1.0), rec.get("quad_agree", -1))


def presence_probability(model, feats):
    # A model fitted against a different feature list would silently misalign
    # every weight with the wrong quantity and still return a plausible looking
    # probability, so the mismatch is refused rather than tolerated.
    if len(model["weights"]) != len(feats):
        raise ValueError(
            f"presence model has {len(model['weights'])} weights for "
            f"{len(feats)} features; refit it with scripts/fit_presence.py")
    x = (np.asarray(feats, float) - np.asarray(model["mu"])) / np.asarray(model["sd"])
    m = float(x @ np.asarray(model["weights"]) + model["bias"])
    return 1.0 / (1.0 + np.exp(-m))
