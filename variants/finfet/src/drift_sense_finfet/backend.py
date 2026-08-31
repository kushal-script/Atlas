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

    The pose prescreen evaluates on the order of two hundred hypotheses and
    keeps only the peak of each, so correlating them one at a time on an
    accelerator spends nearly all its time moving a fresh copy of the search
    image across the bus. Templates are therefore grouped by shape and each
    group is issued as a single batched correlation, with the reduction done
    on the device so only the scalars come back.
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


class _CpuCorrelator:
    """OpenCV correlation, which already transforms large templates."""

    def __init__(self, search):
        self.search = np.asarray(search, dtype=np.float32)

    def peaks(self, tmpls, want_min=False):
        out = []
        for t in tmpls:
            r = cv2.matchTemplate(self.search, np.asarray(t, dtype=np.float32),
                                  cv2.TM_CCOEFF_NORMED)
            out.append((float(r.min()), float(r.max())) if want_min else float(r.max()))
        return out

    def full(self, tmpl):
        return cv2.matchTemplate(self.search, np.asarray(tmpl, dtype=np.float32),
                                 cv2.TM_CCOEFF_NORMED)


class _TorchCorrelator:
    """Accelerated correlation against one fixed search image.

    Two things make the difference between a port that wins and one that
    loses. First, the correlation is evaluated through the Fourier transform
    rather than as a direct convolution: at a 90 by 90 template over a 1000 by
    1000 image the direct form is roughly twenty times slower, which is why
    OpenCV transforms large templates too. Second, the search image, its
    square, and the per window sums for each template size are computed once
    and held on the device, because the search image does not change while two
    hundred pose hypotheses are scored against it.
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


def make_correlator(search, device="cpu"):
    return _CpuCorrelator(search) if device == "cpu" else _TorchCorrelator(search, device)
