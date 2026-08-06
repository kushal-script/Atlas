"""Rasterization primitives for painting layout features onto canvas arrays.

Canvas arrays are indexed [row, col] where row maps to v (nm, downward) and
col maps to u (nm, rightward). Pixel centers sit at (index + 0.5) * pixel_nm.
Later paints overwrite earlier ones, mirroring deposition order.
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d


def corr_noise_1d(rng, n, sigma_nm, corr_nm, pixel_nm):
    if sigma_nm <= 0 or n <= 0:
        return np.zeros(n, dtype=np.float32)
    raw = rng.standard_normal(n).astype(np.float32)
    smooth = gaussian_filter1d(raw, sigma=max(corr_nm / pixel_nm, 0.5), mode="wrap")
    std = smooth.std()
    if std < 1e-8:
        return np.zeros(n, dtype=np.float32)
    return (smooth * (sigma_nm / std)).astype(np.float32)


def _clip_index(x, n):
    return int(np.clip(x, 0, n))


def paint_rect(mat, hgt, pixel_nm, u0, v0, u1, v1, material, height):
    nrows, ncols = mat.shape
    j0 = _clip_index(np.floor(v0 / pixel_nm), nrows)
    j1 = _clip_index(np.ceil(v1 / pixel_nm), nrows)
    i0 = _clip_index(np.floor(u0 / pixel_nm), ncols)
    i1 = _clip_index(np.ceil(u1 / pixel_nm), ncols)
    if j1 <= j0 or i1 <= i0:
        return None
    mat[j0:j1, i0:i1] = material
    hgt[j0:j1, i0:i1] = height
    return (slice(j0, j1), slice(i0, i1))


def paint_vstripe(mat, hgt, pixel_nm, u_center, width, v0, v1, material, height,
                  edge_left=None, edge_right=None):
    """Vertical stripe running along v, with optional per row edge deviation arrays."""
    nrows, ncols = mat.shape
    j0 = _clip_index(np.floor(v0 / pixel_nm), nrows)
    j1 = _clip_index(np.ceil(v1 / pixel_nm), nrows)
    if j1 <= j0:
        return None
    half = width / 2.0
    rows = np.arange(j0, j1)
    el = edge_left[rows] if edge_left is not None else np.zeros(rows.size, dtype=np.float32)
    er = edge_right[rows] if edge_right is not None else np.zeros(rows.size, dtype=np.float32)
    left = u_center - half + el
    right = u_center + half + er
    i0 = _clip_index(np.floor(left.min() / pixel_nm) - 1, ncols)
    i1 = _clip_index(np.ceil(right.max() / pixel_nm) + 1, ncols)
    if i1 <= i0:
        return None
    ucols = (np.arange(i0, i1, dtype=np.float32) + 0.5) * pixel_nm
    mask = (ucols[None, :] >= left[:, None]) & (ucols[None, :] < right[:, None])
    sub = (slice(j0, j1), slice(i0, i1))
    mat[sub][mask] = material
    if np.isscalar(height):
        hgt[sub][mask] = height
    else:
        hgt[sub][mask] = height[mask]
    return sub, mask


def paint_hstripe(mat, hgt, pixel_nm, v_center, width, u0, u1, material, height,
                  edge_top=None, edge_bot=None):
    nrows, ncols = mat.shape
    i0 = _clip_index(np.floor(u0 / pixel_nm), ncols)
    i1 = _clip_index(np.ceil(u1 / pixel_nm), ncols)
    if i1 <= i0:
        return None
    half = width / 2.0
    cols = np.arange(i0, i1)
    et = edge_top[cols] if edge_top is not None else np.zeros(cols.size, dtype=np.float32)
    eb = edge_bot[cols] if edge_bot is not None else np.zeros(cols.size, dtype=np.float32)
    top = v_center - half + et
    bot = v_center + half + eb
    j0 = _clip_index(np.floor(top.min() / pixel_nm) - 1, nrows)
    j1 = _clip_index(np.ceil(bot.max() / pixel_nm) + 1, nrows)
    if j1 <= j0:
        return None
    vrows = (np.arange(j0, j1, dtype=np.float32) + 0.5) * pixel_nm
    mask = (vrows[:, None] >= top[None, :]) & (vrows[:, None] < bot[None, :])
    sub = (slice(j0, j1), slice(i0, i1))
    mat[sub][mask] = material
    if np.isscalar(height):
        hgt[sub][mask] = height
    else:
        hgt[sub][mask] = height[mask]
    return sub, mask


def paint_dots(mat, hgt, pixel_nm, centers_u, centers_v, radii, material, height):
    nrows, ncols = mat.shape
    radii = np.broadcast_to(np.atleast_1d(radii), centers_u.shape)
    for cu, cv, r in zip(centers_u, centers_v, radii):
        j0 = _clip_index(np.floor((cv - r) / pixel_nm), nrows)
        j1 = _clip_index(np.ceil((cv + r) / pixel_nm) + 1, nrows)
        i0 = _clip_index(np.floor((cu - r) / pixel_nm), ncols)
        i1 = _clip_index(np.ceil((cu + r) / pixel_nm) + 1, ncols)
        if j1 <= j0 or i1 <= i0:
            continue
        vv = (np.arange(j0, j1, dtype=np.float32) + 0.5) * pixel_nm - cv
        uu = (np.arange(i0, i1, dtype=np.float32) + 0.5) * pixel_nm - cu
        mask = (vv[:, None] ** 2 + uu[None, :] ** 2) <= r * r
        sub = (slice(j0, j1), slice(i0, i1))
        mat[sub][mask] = material
        hgt[sub][mask] = height
