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


RERANK_FEATURES = FEATURES + ("rr_score", "rr_margin", "rr_agree")


def _fin(v, default):
    """A recorded value, or the block's neutral default when it is not finite.

    A degenerate image can push nan through a diagnostic, and nan times any
    weight is nan, which turns the whole probability into an accident of how
    comparisons treat it. The neutral default makes the conservative outcome
    deliberate instead."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return default
    return v if np.isfinite(v) else default


def _rerank_block(rr):
    """The re ranker's contributions: its probability for the chosen site, its
    margin over the runner up, and whether it agreed with the correlation's
    choice. Neutral defaults carry no signal when it did not run."""
    rr = rr or {}
    return [
        _fin(rr.get("score", 0.0), 0.0),
        _fin(rr.get("margin", 0.0), 0.0),
        1.0 if rr.get("agree", True) else 0.0,
    ]


def features_from_diag_v2(diag):
    return features_from_diag(diag) + _rerank_block(diag.get("rerank"))


def features_from_record_v2(rec):
    return features_from_record(rec) + _rerank_block(rec.get("rerank"))


EXTENDED_FEATURES = RERANK_FEATURES + ("lattice_balance", "period_ratio", "peak_curv")


def _extended_block(d):
    """Ambiguity statistics: the reference's spectral balance, the period aware
    second peak ratio, and the peak curvature."""
    return [
        _fin(d.get("lattice_balance", 0.0), 0.0),
        _fin(d.get("period_ratio", 1.0), 1.0),
        _fin(d.get("peak_curv", 0.0), 0.0),
    ]


def features_from_diag_v3(diag):
    return features_from_diag_v2(diag) + _extended_block(diag)


def features_from_record_v3(rec):
    return features_from_record_v2(rec) + _extended_block(rec)


RAW_CONFIRM_FEATURES = RERANK_FEATURES + ("raw_peak", "raw_margin", "raw_agree")


def _raw_block(d):
    """The raw full reference confirmation's peak, its margin over the best peak
    outside the site, and whether its argmax agreed with the answer. Neutral
    defaults carry no signal when it did not run."""
    rc = d.get("raw_confirm") or {}
    return [
        _fin(rc.get("peak", 0.0), 0.0),
        _fin(rc.get("margin", 0.0), 0.0),
        1.0 if rc.get("agree", True) else 0.0,
    ]


DEV_FEATURES = ("dev_best", "dev_excess", "dev_missing")


def _dev_block(d):
    """The deviation field's absolute evidence at the answer: the winner's
    normalized deviation correlation, its gap over the candidate field median,
    and an indicator for the stage not having run. The z statistic asks
    whether one site stands out among its peers, which a decoy site can do by
    being the only zone with the right geometry; these ask whether the site
    unique content matched at all, in absolute units a decoy cannot fake."""
    s2 = d.get("stage2") or {}
    best = d.get("dev_best", s2.get("best_residual_score"))
    med = d.get("dev_median", s2.get("residual_median"))
    have = np.isfinite(_fin(best, np.nan))
    excess = (float(best) - float(med)) if (have and med is not None
                                            and np.isfinite(_fin(med, np.nan))) else 0.0
    return [
        _fin(best, 0.0) if have else 0.0,
        _fin(excess, 0.0),
        0.0 if have else 1.0,
    ]


ALL_FEATURE_NAMES = tuple(list(FEATURES)
                          + ["rr_score", "rr_margin", "rr_agree",
                             "lattice_balance", "period_ratio", "peak_curv",
                             "raw_peak", "raw_margin", "raw_agree",
                             "dev_best", "dev_excess", "dev_missing"])


def _all_named(diag_or_rec, base):
    vals = (base + _rerank_block(diag_or_rec.get("rerank"))
            + _extended_block(diag_or_rec) + _raw_block(diag_or_rec)
            + _dev_block(diag_or_rec))
    return dict(zip(ALL_FEATURE_NAMES, vals))


def features_all_from_diag(diag):
    return [_all_named(diag, features_from_diag(diag))[n] for n in ALL_FEATURE_NAMES]


def features_for_model(model, diag):
    """The feature vector the given model was fitted on, selected by name.

    Every fitted model names its features; the vector is assembled from the
    named quantities so tiers of equal length cannot be confused, and a name
    this construction does not know is refused rather than guessed at. A
    model file without a feature list is the original fifteen feature one."""
    if "features" not in model:
        return features_from_diag(diag)
    table = _all_named(diag, features_from_diag(diag))
    return [table[n] for n in model["features"]]


def features_for_model_record(model, rec):
    if "features" not in model:
        return features_from_record(rec)
    table = _all_named(rec, features_from_record(rec))
    return [table[n] for n in model["features"]]


def presence_probability(model, feats):
    if len(model["weights"]) != len(feats):
        raise ValueError(
            f"presence model has {len(model['weights'])} weights for "
            f"{len(feats)} features; refit it with scripts/fit_presence.py")
    x = (np.asarray(feats, float) - np.asarray(model["mu"])) / np.asarray(model["sd"])
    m = float(x @ np.asarray(model["weights"]) + model["bias"])
    return 1.0 / (1.0 + np.exp(-m))
