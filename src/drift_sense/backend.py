"""Compute backends for the two operations that dominate localization runtime.

Profiling one 1000x1000 pair puts 52 percent of the time in building the
reference blur bank and 45 percent in normalized cross correlation, so those
are the only two operations worth abstracting. Everything else stays plain
numpy.

The default backend is numpy and OpenCV, which needs no framework and is the
submitted inference path. A torch backend is available for machines with an
accelerator; it is selected explicitly and never automatically, so a missing
or mismatched CUDA install can never change what the submitted path does.

Correctness requirement: every backend must return the same answer. The torch
correlation reproduces the OpenCV TM_CCOEFF_NORMED definition exactly rather
than approximating it, and `scripts/verify_backends.py` asserts agreement on
real pairs.
"""

import numpy as np
import cv2

_TORCH = None


def _torch():
    global _TORCH
    if _TORCH is None:
        import torch
        import torch.nn.functional as F
        _TORCH = (torch, F)
    return _TORCH


def available_devices():
    devices = ["cpu"]
    try:
        torch, _ = _torch()
    except ImportError:
        return devices
    if torch.cuda.is_available():
        devices.append("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        devices.append("mps")
    return devices


def resolve_device(name):
    """Map a requested device onto one that exists, without silent downgrades."""
    if name in (None, "cpu"):
        return "cpu"
    have = available_devices()
    if name == "auto":
        for pick in ("cuda", "mps"):
            if pick in have:
                return pick
        return "cpu"
    if name not in have:
        raise SystemExit(f"device {name!r} is not available; found {have}")
    return name


def _reflect(t, radius, dim):
    """Edge repeating reflection, matching scipy and cv2.BORDER_REFLECT.

    Torch's own reflect padding drops the edge sample, which is
    cv2.BORDER_REFLECT_101 and blurs the border differently.
    """
    torch, _ = _torch()
    lo = t.narrow(dim, 0, radius).flip(dim)
    hi = t.narrow(dim, t.shape[dim] - radius, radius).flip(dim)
    return torch.cat([lo, t, hi], dim=dim)


def gaussian(img, sigma, device="cpu"):
    """Separable Gaussian blur.

    The kernel radius follows scipy's truncate of 4 standard deviations so
    that switching backends cannot change the amount of blur, only the
    hardware that applies it.
    """
    sigma = float(sigma)
    if sigma <= 0:
        return np.asarray(img, dtype=np.float32)
    radius = int(4.0 * sigma + 0.5)
    ksize = 2 * radius + 1
    if device == "cpu":
        src = np.asarray(img, dtype=np.float32)
        return cv2.GaussianBlur(src, (ksize, ksize), sigma, borderType=cv2.BORDER_REFLECT)
    torch, F = _torch()
    k = cv2.getGaussianKernel(ksize, sigma).astype(np.float32).ravel()
    t = torch.as_tensor(np.asarray(img, dtype=np.float32), device=device)[None, None]
    kt = torch.as_tensor(k, device=device)
    t = _reflect(t, radius, -1)
    t = F.conv2d(t, kt.view(1, 1, 1, -1))
    t = _reflect(t, radius, -2)
    t = F.conv2d(t, kt.view(1, 1, -1, 1))
    return t[0, 0].cpu().numpy()


def match_ccoeff_normed(search, tmpl, device="cpu"):
    """Zero mean normalized cross correlation, valid positions only.

    Reproduces cv2.TM_CCOEFF_NORMED: the numerator correlates the mean removed
    template against the search image, and the denominator is the product of
    the template's norm and the per window standard deviation of the search
    image, the latter obtained from running sums rather than recomputed.
    """
    if device == "cpu":
        return cv2.matchTemplate(np.asarray(search, dtype=np.float32),
                                 np.asarray(tmpl, dtype=np.float32),
                                 cv2.TM_CCOEFF_NORMED)
    torch, F = _torch()
    s = torch.as_tensor(np.asarray(search, dtype=np.float32), device=device)[None, None]
    t = torch.as_tensor(np.asarray(tmpl, dtype=np.float32), device=device)[None, None]
    n = float(t.numel())
    t0 = t - t.mean()
    ones = torch.ones_like(t)
    num = F.conv2d(s, t0)
    s1 = F.conv2d(s, ones)
    s2 = F.conv2d(s * s, ones)
    var = (s2 - s1 * s1 / n).clamp_min(0.0)
    den = torch.sqrt(var * (t0 * t0).sum())
    out = torch.where(den > 1e-12, num / den, torch.zeros_like(num))
    return out[0, 0].cpu().numpy()


def match_ccoeff_normed_batch(search, tmpls, device="cpu", want_min=False):
    """Correlate many templates against one search image, returning extrema.

    Templates are grouped by shape and batched so the search image crosses
    the bus once per group rather than once per template.
    """
    if device == "cpu":
        out = []
        for t in tmpls:
            r = match_ccoeff_normed(search, t, "cpu")
            out.append((float(r.min()), float(r.max())) if want_min else float(r.max()))
        return out

    torch, F = _torch()
    s = torch.as_tensor(np.asarray(search, dtype=np.float32), device=device)[None, None]
    s_sq = s * s
    results = [None] * len(tmpls)
    groups = {}
    for i, t in enumerate(tmpls):
        groups.setdefault(np.shape(t), []).append(i)

    for shape, idx in groups.items():
        stack = np.stack([np.asarray(tmpls[i], dtype=np.float32) for i in idx])
        t = torch.as_tensor(stack, device=device)[:, None]
        n = float(shape[0] * shape[1])
        t0 = t - t.mean(dim=(2, 3), keepdim=True)
        ones = torch.ones((1, 1) + tuple(shape), device=device, dtype=t.dtype)
        num = F.conv2d(s, t0)
        s1 = F.conv2d(s, ones)
        s2 = F.conv2d(s_sq, ones)
        var = (s2 - s1 * s1 / n).clamp_min(0.0)
        norm = (t0 * t0).sum(dim=(1, 2, 3)).view(1, -1, 1, 1)
        den = torch.sqrt(var * norm)
        r = torch.where(den > 1e-12, num / den, torch.zeros_like(num))
        mx = r.amax(dim=(2, 3))[0].cpu().numpy()
        mn = r.amin(dim=(2, 3))[0].cpu().numpy() if want_min else None
        for k, i in enumerate(idx):
            results[i] = (float(mn[k]), float(mx[k])) if want_min else float(mx[k])
    return results


_POOL = {"ex": None}


def _pool():
    if _POOL["ex"] is None:
        import os
        from concurrent.futures import ThreadPoolExecutor
        _POOL["ex"] = ThreadPoolExecutor(max_workers=min(4, os.cpu_count() or 1))
    return _POOL["ex"]


def map_parallel(fn, items):
    """Order preserving parallel map over independent per template work.

    The heavy calls, OpenCV transforms and scipy resampling, release the GIL,
    so a small thread pool uses the reference machine's four cores; each item
    is computed by the same function on the same input as the serial loop, so
    the returned list is identical element for element."""
    items = list(items)
    if len(items) < 3:
        return [fn(i) for i in items]
    return list(_pool().map(fn, items))


class _CpuCorrelator:
    """OpenCV correlation, which already transforms large templates."""

    def __init__(self, search):
        self.search = np.asarray(search, dtype=np.float32)

    def peaks(self, tmpls, want_min=False):
        def one(t):
            r = cv2.matchTemplate(self.search, np.asarray(t, dtype=np.float32),
                                  cv2.TM_CCOEFF_NORMED)
            return (float(r.min()), float(r.max())) if want_min else float(r.max())
        if min(self.search.shape[:2]) <= 512:
            return map_parallel(one, tmpls)
        return [one(t) for t in tmpls]

    def full(self, tmpl):
        return cv2.matchTemplate(self.search, np.asarray(tmpl, dtype=np.float32),
                                 cv2.TM_CCOEFF_NORMED)


class _TorchCorrelator:
    """Accelerated correlation against one fixed search image, with the search
    side transforms and window sums computed once and held on the device.
    """

    CHUNK = 32

    def __init__(self, search, device):
        torch, _ = _torch()
        self.torch = torch
        self.device = device
        s = torch.as_tensor(np.asarray(search, dtype=np.float32), device=device)
        self.h, self.w = int(s.shape[0]), int(s.shape[1])
        self.fs = torch.fft.rfft2(s, s=(self.h, self.w))
        self.fs2 = torch.fft.rfft2(s * s, s=(self.h, self.w))
        self._sums = {}

    def _valid(self, spec, kernels):
        torch = self.torch
        th, tw = int(kernels.shape[-2]), int(kernels.shape[-1])
        fk = torch.fft.rfft2(torch.flip(kernels, (-2, -1)), s=(self.h, self.w))
        out = torch.fft.irfft2(spec * fk, s=(self.h, self.w))
        return out[..., th - 1:self.h, tw - 1:self.w]

    def _win_sums(self, th, tw):
        if (th, tw) not in self._sums:
            torch = self.torch
            ones = torch.ones((1, th, tw), device=self.device)
            s1 = self._valid(self.fs, ones)
            s2 = self._valid(self.fs2, ones)
            self._sums[(th, tw)] = (s1, s2)
        return self._sums[(th, tw)]

    def _scores(self, stack):
        torch = self.torch
        t = torch.as_tensor(stack, device=self.device)
        th, tw = int(t.shape[-2]), int(t.shape[-1])
        n = float(th * tw)
        t0 = t - t.mean(dim=(1, 2), keepdim=True)
        s1, s2 = self._win_sums(th, tw)
        var = (s2 - s1 * s1 / n).clamp_min(0.0)
        num = self._valid(self.fs, t0)
        norm = (t0 * t0).sum(dim=(1, 2)).view(-1, 1, 1)
        den = torch.sqrt(var * norm)
        return torch.where(den > 1e-12, num / den, torch.zeros_like(num))

    def peaks(self, tmpls, want_min=False):
        results = [None] * len(tmpls)
        groups = {}
        for i, t in enumerate(tmpls):
            groups.setdefault(np.shape(t), []).append(i)
        for shape, idx in groups.items():
            for c in range(0, len(idx), self.CHUNK):
                part = idx[c:c + self.CHUNK]
                stack = np.stack([np.asarray(tmpls[i], dtype=np.float32) for i in part])
                r = self._scores(stack)
                mx = r.amax(dim=(1, 2)).cpu().numpy()
                mn = r.amin(dim=(1, 2)).cpu().numpy() if want_min else None
                for k, i in enumerate(part):
                    results[i] = (float(mn[k]), float(mx[k])) if want_min else float(mx[k])
        return results

    def full(self, tmpl):
        stack = np.asarray(tmpl, dtype=np.float32)[None]
        return self._scores(stack)[0].cpu().numpy()


class _FftCpuCorrelator:
    """The OpenCV correlation with the search side work computed once: the search
    spectrum and integral images are cached, so each call costs one template
    transform, one spectrum product and one inverse transform. The surface
    equals TM_CCOEFF_NORMED to float rounding."""

    def __init__(self, search):
        s = np.asarray(search, dtype=np.float32)
        self.search = s
        self.h, self.w = s.shape
        self.fs = cv2.dft(s, flags=cv2.DFT_COMPLEX_OUTPUT)
        self.ii, self.ii2 = cv2.integral2(s)
        self._wsums = {}

    def _window_var(self, th, tw):
        key = (th, tw)
        if key not in self._wsums:
            H, W = self.h - th + 1, self.w - tw + 1
            ii, ii2 = self.ii, self.ii2
            s1 = (ii[th:th + H, tw:tw + W] - ii[:H, tw:tw + W]
                  - ii[th:th + H, :W] + ii[:H, :W])
            s2 = (ii2[th:th + H, tw:tw + W] - ii2[:H, tw:tw + W]
                  - ii2[th:th + H, :W] + ii2[:H, :W])
            n = float(th * tw)
            var = np.maximum(s2 - s1 * s1 / n, 0.0)
            self._wsums[key] = var.astype(np.float64)
        return self._wsums[key]

    def _surface(self, tmpl):
        t = np.asarray(tmpl, dtype=np.float32)
        th, tw = t.shape
        if th > self.h or tw > self.w:
            raise ValueError("template larger than search")
        t0 = t - float(t.mean())
        norm = float((t0 * t0).sum())
        pad = np.zeros((self.h, self.w), np.float32)
        pad[:th, :tw] = t0
        ft = cv2.dft(pad, flags=cv2.DFT_COMPLEX_OUTPUT)
        prod = cv2.mulSpectrums(self.fs, ft, 0, conjB=True)
        cross = cv2.dft(prod, flags=cv2.DFT_INVERSE | cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
        H, W = self.h - th + 1, self.w - tw + 1
        num = cross[:H, :W].astype(np.float64)
        den = np.sqrt(self._window_var(th, tw) * norm)
        out = np.where(den > 1e-9, num / np.maximum(den, 1e-12), 0.0)
        return out.astype(np.float32)

    def peaks(self, tmpls, want_min=False):
        for t in tmpls:
            th, tw = np.asarray(t).shape
            self._window_var(th, tw)
        def one(t):
            r = self._surface(t)
            return (float(r.min()), float(r.max())) if want_min else float(r.max())
        return map_parallel(one, tmpls)

    def full(self, tmpl):
        return self._surface(tmpl)


_CPU_CHOICE = {"cls": None}


def _pick_cpu_correlator(search):
    """The faster of the two CPU correlators, decided once per process by a warm
    timed call of each path against the first search image, since the winner
    is platform dependent. The paths agree to float rounding, so the choice
    affects speed alone."""
    if _CPU_CHOICE["cls"] is None:
        import time as _time
        s = np.asarray(search, dtype=np.float32)
        if min(s.shape[:2]) < 256:
            return _CpuCorrelator
        t = np.ascontiguousarray(s[: s.shape[0] // 8, : s.shape[1] // 8])
        best = {}
        fft = _FftCpuCorrelator(s)
        for name, call in (("direct", lambda: cv2.matchTemplate(
                s, t, cv2.TM_CCOEFF_NORMED)), ("fft", lambda: fft.full(t))):
            call()
            t0 = _time.perf_counter()
            call()
            best[name] = _time.perf_counter() - t0
        _CPU_CHOICE["cls"] = (_CpuCorrelator if best["direct"] <= best["fft"]
                              else _FftCpuCorrelator)
    return _CPU_CHOICE["cls"]


def make_correlator(search, device="cpu"):
    if device == "cpu":
        return _pick_cpu_correlator(search)(search)
    if device == "cpu_direct":
        return _CpuCorrelator(search)
    if device == "cpu_fft":
        return _FftCpuCorrelator(search)
    return _TorchCorrelator(search, device)
