"""SEM image formation model.

Signal chain per capture: secondary electron emission from the specimen
(material yield times secant of local surface tilt, plus detector side
asymmetry), beam point spread blur, point sampling on the scan grid with
per line drift, jitter and vibration offsets, dielectric charging, Poisson
shot noise scaled by electron dose, Gaussian detector read noise, and an
auto brightness contrast tone map quantized to 8 bit. The physical model and
every parameter range are justified in docs/citations.md.
"""

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

from ..params import BASE_SE_YIELD, MATERIAL_NITRIDE, MATERIAL_STI


def build_se_canvas(mat, hgt, pixel_nm, rng, cp):
    lut = np.zeros(max(BASE_SE_YIELD) + 1, dtype=np.float32)
    yields = {}
    for m, y in BASE_SE_YIELD.items():
        jittered = y * (1.0 + rng.uniform(-cp.yield_jitter, cp.yield_jitter))
        lut[m] = jittered
        yields[m] = float(jittered)
    yield_map = lut[mat]

    hs = gaussian_filter(hgt, sigma=cp.sidewall_sigma_nm / pixel_nm)
    gy, gx = np.gradient(hs, pixel_nm)
    slope = np.hypot(gx, gy)
    max_slope = np.tan(np.deg2rad(cp.max_tilt_deg))
    np.clip(slope, 0.0, max_slope, out=slope)
    sec = np.sqrt(1.0 + slope * slope).astype(np.float32)

    asym = rng.uniform(*cp.detector_asymmetry)
    az = rng.uniform(0.0, 2.0 * np.pi)
    directional = (gx * np.cos(az) + gy * np.sin(az)) / (1.0 + slope)
    factor = (1.0 + asym * directional).astype(np.float32)

    se = yield_map * sec * factor
    norm = float(np.percentile(se, 99.5))
    se /= norm
    info = {"yields": yields, "detector_asymmetry": float(asym),
            "detector_azimuth_rad": float(az), "norm": norm}
    return se.astype(np.float32), info


def _scan_offsets(rng, n_rows, cap):
    total_x = rng.uniform(*cap.drift_total_px) * rng.choice([-1.0, 1.0])
    total_y = rng.uniform(*cap.drift_total_px) * rng.choice([-1.0, 1.0]) * 0.5
    t = np.linspace(0.0, 1.0, n_rows, dtype=np.float32)
    curve = rng.uniform(-0.3, 0.3)
    profile = t + curve * t * (1.0 - t)
    drift_x = total_x * profile
    drift_y = total_y * profile

    sigma_j = rng.uniform(*cap.jitter_sigma_px)
    rho = 0.5
    white = rng.normal(0.0, sigma_j * np.sqrt(1 - rho * rho), n_rows)
    jitter = np.empty(n_rows, dtype=np.float32)
    acc = 0.0
    for i in range(n_rows):
        acc = rho * acc + white[i]
        jitter[i] = acc

    amp = rng.uniform(*cap.vib_amp_px)
    period = rng.uniform(30.0, 200.0)
    phase = rng.uniform(0.0, 2.0 * np.pi)
    vib = amp * np.sin(2.0 * np.pi * np.arange(n_rows) / period + phase)

    dx = (drift_x + jitter + vib).astype(np.float32)
    dy = drift_y.astype(np.float32)
    meta = {"drift_total_px": [float(total_x), float(total_y)],
            "jitter_sigma_px": float(sigma_j), "vib_amp_px": float(amp),
            "vib_period_rows": float(period)}
    return dx, dy, meta


def _sample_capture_settings(rng, cap):
    sigma = rng.uniform(*cap.psf_sigma_nm)
    ratio = rng.uniform(*cap.astig_ratio)
    if rng.random() < 0.5:
        sx, sy = sigma, sigma * ratio
    else:
        sx, sy = sigma * ratio, sigma
    return {
        "psf_sigma_x_nm": float(sx),
        "psf_sigma_y_nm": float(sy),
        "dose_e": float(rng.uniform(*cap.dose_e)),
        "read_noise_e": float(rng.uniform(*cap.read_noise_e)),
        "charging_amp": float(rng.uniform(*cap.charging_amp)),
        "charging_scale_nm": float(rng.uniform(*cap.charging_scale_nm)),
    }


def _sample_canvas(canvas, jv, iu, order=1):
    return map_coordinates(canvas, [jv, iu], order=order, mode="nearest")


def render_capture(se_canvas, mat_canvas, canvas_pixel_nm, pose, cap, rng):
    n = cap.out_px
    ss = cap.supersample
    m = n * ss
    q = pose["pixel_nm"] / ss
    theta = pose["theta_rad"]
    cu, cv = pose["center_u_nm"], pose["center_v_nm"]
    settings = _sample_capture_settings(rng, cap)
    dx, dy, scan_meta = _scan_offsets(rng, n, cap)

    cos_t, sin_t = np.cos(theta), np.sin(theta)
    half = (m - 1) / 2.0
    fine = np.empty((m, m), dtype=np.float32)
    cf = np.arange(m, dtype=np.float64)
    duf = (cf - half) * q
    chunk = 512
    for r0 in range(0, m, chunk):
        r1 = min(r0 + chunk, m)
        dvf = (np.arange(r0, r1, dtype=np.float64) - half) * q
        u = cu + duf[None, :] * cos_t - dvf[:, None] * sin_t
        v = cv + duf[None, :] * sin_t + dvf[:, None] * cos_t
        iu = u / canvas_pixel_nm - 0.5
        jv = v / canvas_pixel_nm - 0.5
        fine[r0:r1] = _sample_canvas(se_canvas, jv, iu)

    gaussian_filter(fine, sigma=(settings["psf_sigma_y_nm"] / q,
                                 settings["psf_sigma_x_nm"] / q), output=fine)

    off = (ss - 1) / 2.0
    rows = np.arange(n, dtype=np.float64)
    cols = np.arange(n, dtype=np.float64)
    rf = (rows[:, None] + dy[:, None]) * ss + off
    cf_out = (cols[None, :] + dx[:, None]) * ss + off
    rf, cf_out = np.broadcast_arrays(rf, cf_out)
    img = map_coordinates(fine, [rf.ravel(), cf_out.ravel()], order=1,
                          mode="nearest").reshape(n, n).astype(np.float32)
    del fine

    p = pose["pixel_nm"]
    du = (cols - (n - 1) / 2.0) * p
    dv = (rows - (n - 1) / 2.0) * p
    u = cu + du[None, :] * cos_t - dv[:, None] * sin_t
    v = cv + du[None, :] * sin_t + dv[:, None] * cos_t
    iu = u / canvas_pixel_nm - 0.5
    jv = v / canvas_pixel_nm - 0.5
    mat_s = _sample_canvas(mat_canvas.astype(np.float32), jv, iu, order=0)
    dielectric = (mat_s == MATERIAL_STI) | (mat_s == MATERIAL_NITRIDE)
    oxide = gaussian_filter(dielectric.astype(np.float32), sigma=30.0 / p)
    field = rng.standard_normal((n, n)).astype(np.float32)
    field = gaussian_filter(field, sigma=settings["charging_scale_nm"] / p)
    std = field.std()
    if std > 1e-8:
        field /= std
    img *= 1.0 + settings["charging_amp"] * field * oxide

    dose = settings["dose_e"]
    electrons = rng.poisson(np.clip(img, 0.0, None) * dose).astype(np.float32)
    electrons += rng.normal(0.0, settings["read_noise_e"], electrons.shape).astype(np.float32)
    img = electrons / dose

    lo = np.percentile(img, 1.0)
    hi = np.percentile(img, 99.7)
    span = max(hi - lo, 1e-6)
    lo -= rng.uniform(0.0, 0.06) * span
    hi += rng.uniform(0.0, 0.06) * span
    img8 = np.clip((img - lo) / (hi - lo) * 255.0, 0.0, 255.0).astype(np.uint8)

    meta = {"pose": {k: float(val) for k, val in pose.items()},
            "settings": settings, "scan": scan_meta}
    return img8, dx, dy, meta


def capture_to_specimen(r, c, pose, dx, dy, out_px):
    n = out_px
    dxr = np.interp(r, np.arange(n), dx)
    dyr = np.interp(r, np.arange(n), dy)
    c_eff = c + dxr
    r_eff = r + dyr
    p = pose["pixel_nm"]
    du = (c_eff - (n - 1) / 2.0) * p
    dv = (r_eff - (n - 1) / 2.0) * p
    cos_t, sin_t = np.cos(pose["theta_rad"]), np.sin(pose["theta_rad"])
    u = pose["center_u_nm"] + du * cos_t - dv * sin_t
    v = pose["center_v_nm"] + du * sin_t + dv * cos_t
    return u, v


def specimen_to_capture(u, v, pose, dx, dy, out_px):
    n = out_px
    p = pose["pixel_nm"]
    cos_t, sin_t = np.cos(pose["theta_rad"]), np.sin(pose["theta_rad"])
    ru = u - pose["center_u_nm"]
    rv = v - pose["center_v_nm"]
    du = ru * cos_t + rv * sin_t
    dv = -ru * sin_t + rv * cos_t
    c = du / p + (n - 1) / 2.0
    r = dv / p + (n - 1) / 2.0
    for _ in range(3):
        dxr = np.interp(r, np.arange(n), dx)
        dyr = np.interp(r, np.arange(n), dy)
        c = du / p + (n - 1) / 2.0 - dxr
        r = dv / p + (n - 1) / 2.0 - dyr
    return c, r
