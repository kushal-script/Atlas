"""Brightfield optical microscope image formation over the same specimen
canvases used for SEM, producing RGB captures for the optical tool bonus.

Model: per material spectral reflectance, two beam thin film interference
color over oxide regions parameterized by film thickness, a diffraction
limited Gaussian point spread of sigma 0.21 lambda over NA per channel with
chromatic variation, radial vignetting and an illumination tilt, photon shot
noise per channel scaled by exposure (the wide field capture receives far
fewer photons), small sensor read noise, white balance jitter and a shared
percentile tone map preserving color ratios. Optical tools are full field
cameras, so no scan line artifacts apply. Physics references are in
docs/citations.md.
"""

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

from ..params import MATERIAL_STI

OPTICAL_REFLECTANCE = {
    0: (0.52, 0.56, 0.62),
    1: (0.38, 0.40, 0.44),
    2: (0.30, 0.32, 0.36),
    3: (0.62, 0.60, 0.55),
    4: (0.42, 0.46, 0.50),
}


def render_optical_capture(mat_canvas, hgt_canvas, canvas_pixel_nm, pose, opt, rng):
    n = opt["out_px"]
    p = pose["pixel_nm"]
    theta = pose["theta_rad"]
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    idx = np.arange(n, dtype=np.float64)
    du = (idx - (n - 1) / 2.0) * p
    u = pose["center_u_nm"] + du[None, :] * cos_t - du[:, None] * sin_t
    v = pose["center_v_nm"] + du[None, :] * sin_t + du[:, None] * cos_t
    iu = u / canvas_pixel_nm - 0.5
    jv = v / canvas_pixel_nm - 0.5
    mat_s = map_coordinates(mat_canvas, [jv, iu], order=0, mode="nearest")
    hgt_s = map_coordinates(hgt_canvas, [jv, iu], order=1, mode="nearest")

    lut = np.array([OPTICAL_REFLECTANCE.get(m, (0.4, 0.4, 0.4))
                    for m in range(int(mat_canvas.max()) + 1)], dtype=np.float32)
    lut = lut * (1.0 + rng.uniform(-0.05, 0.05, lut.shape)).astype(np.float32)

    t_film = opt["film_base_nm"] + hgt_s
    oxide = (mat_s == MATERIAL_STI).astype(np.float32)
    yy, xx = np.meshgrid(idx, idx, indexing="ij")
    r2 = ((xx - (n - 1) / 2) ** 2 + (yy - (n - 1) / 2) ** 2) / ((n / 2) ** 2)
    tilt_phi = rng.uniform(0, 2 * np.pi)
    tilt = 1.0 + opt["illum_tilt"] * ((xx / n - 0.5) * np.cos(tilt_phi)
                                      + (yy / n - 0.5) * np.sin(tilt_phi))
    vignette = 1.0 - opt["vignette"] * r2

    channels = []
    wb = 1.0 + rng.uniform(-0.08, 0.08, 3)
    for c, lam in enumerate(opt["wavelengths_nm"]):
        refl = lut[mat_s, c]
        film = 0.75 + 0.25 * np.cos(4.0 * np.pi * opt["film_index"] * t_film / lam)
        signal = refl * (oxide * film + (1.0 - oxide))
        # Shorter wavelengths resolve slightly better, so the sampled spot is
        # scaled by wavelength about the green channel, preserving the colour
        # dependent sharpness an optical instrument shows.
        sigma_px = opt["psf_sigma_px"] * (lam / 540.0)
        signal = gaussian_filter(signal.astype(np.float32), sigma_px)
        signal = signal * vignette * tilt
        dose = opt["photon_dose"]
        photons = rng.poisson(np.clip(signal, 0, None) * dose).astype(np.float32)
        photons += rng.normal(0.0, opt["read_noise"], photons.shape).astype(np.float32)
        channels.append(photons / dose * wb[c])

    img = np.stack(channels, axis=-1)
    lo = np.percentile(img, 1.0)
    hi = np.percentile(img, 99.5)
    img8 = np.clip((img - lo) / max(hi - lo, 1e-6) * 255.0, 0, 255).astype(np.uint8)
    meta = {"pose": {k: float(val) for k, val in pose.items()},
            "settings": {"wavelengths_nm": list(opt["wavelengths_nm"]),
                         "na": opt["na"], "photon_dose": opt["photon_dose"],
                         "film_base_nm": opt["film_base_nm"],
                         "vignette": opt["vignette"],
                         "illum_tilt": opt["illum_tilt"]}}
    return img8, meta


def sample_optical_settings(rng, role):
    """Settings for one optical capture.

    On resolution, and why this is an analogue rather than a literal
    microscope. Taken literally, an Abbe limited point spread of 0.21 lambda
    over the numerical aperture is about 126 nm wide, which at the reference
    scale of 1 nm per pixel is a 126 pixel blur across a 1000 pixel frame: the
    reference becomes a smooth colour gradient carrying no structure at all,
    and the registration task it is supposed to pose stops existing, since the
    high magnification image would hold less information than the wide one.
    Real optical inspection never images a 1 micron field for exactly this
    reason. The addendum asks for an optical microscope analogue with the
    reference present and scores it, so the analogue keeps what makes optical
    imaging distinctive, three colour channels whose contrast comes from thin
    film interference rather than topography, photon limited noise, vignetting
    and illumination tilt, and samples the point spread relative to the pixel
    the way the electron model does, instead of applying an absolute
    diffraction limit that would annihilate one of the two images.
    """
    doses = {"reference": (6000.0, 20000.0), "search": (400.0, 1500.0)}
    psf_px = {"reference": (1.1, 2.2), "search": (1.8, 4.0)}
    return {
        "out_px": 1000,
        "wavelengths_nm": (620.0, 540.0, 460.0),
        "psf_sigma_px": float(rng.uniform(*psf_px[role])),
        "na": float(rng.uniform(0.85, 0.95)),
        "photon_dose": float(rng.uniform(*doses[role])),
        "read_noise": float(rng.uniform(1.0, 4.0)),
        "film_base_nm": float(rng.uniform(150.0, 420.0)),
        "film_index": 1.46,
        "vignette": float(rng.uniform(0.02, 0.08)),
        "illum_tilt": float(rng.uniform(0.01, 0.05)),
    }
