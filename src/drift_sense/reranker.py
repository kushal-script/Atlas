"""Optional learned re-ranker for the stage two decision.

A small convolutional network scores each degenerate candidate from a four
channel stack (search window, template, window deviation field, template
deviation field) plus two scalars (lattice correlation and dense residual
score). A learnable null logit competes in the softmax so the network can
abstain, in which case the classical tie break applies. Inference here is
pure numpy from an npz weight file, so the default installation needs no
deep learning framework; training lives in scripts/train_reranker.py.

Architecture, fixed and mirrored exactly by the trainer:
conv 4 to 16, kernel 5, stride 2, relu
conv 16 to 32, kernel 3, stride 2, relu
conv 32 to 32, kernel 3, stride 2, relu
global average pool, concat the two scalars
fc 34 to 32, relu, fc 32 to 1, plus a learnable null logit
"""

from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

_CACHE = {}


def load_weights(path):
    key = str(path)
    if key not in _CACHE:
        _CACHE[key] = dict(np.load(path))
    return _CACHE[key]


def _conv(x, w, b, stride):
    v = sliding_window_view(x, (w.shape[2], w.shape[3]), axis=(2, 3))
    v = v[:, :, ::stride, ::stride]
    out = np.einsum("nchwkl,ockl->nohw", v, w, optimize=True)
    return out + b[None, :, None, None]


def forward(weights, x, scalars):
    h = np.maximum(_conv(x, weights["c1_w"], weights["c1_b"], 2), 0.0)
    h = np.maximum(_conv(h, weights["c2_w"], weights["c2_b"], 2), 0.0)
    h = np.maximum(_conv(h, weights["c3_w"], weights["c3_b"], 2), 0.0)
    feat = h.mean(axis=(2, 3))
    feat = np.concatenate([feat, scalars], axis=1)
    h = np.maximum(feat @ weights["f1_w"].T + weights["f1_b"], 0.0)
    return (h @ weights["f2_w"].T + weights["f2_b"]).ravel()


def build_pool(resp, r2, wide, pool_size):
    ncc_vals = resp[wide[:, 0], wide[:, 1]]
    r2_vals = r2[wide[:, 0], wide[:, 1]]
    half = pool_size // 2
    picked = list(np.argsort(-ncc_vals)[:half])
    for i in np.argsort(-r2_vals):
        if len(picked) >= pool_size:
            break
        if i not in picked:
            picked.append(int(i))
    return np.array(picked, dtype=np.int64)


def _norm(c):
    return (c - c.mean()) / (c.std() + 1e-6)


def build_stacks(search, tmpl, med, rt0, resp, r2, wide, pool_idx):
    t = tmpl.shape[0]
    tmpl_n = _norm(tmpl)
    rt0_n = _norm(rt0)
    xs, ss = [], []
    for i in pool_idx:
        py, px = int(wide[i, 0]), int(wide[i, 1])
        win = search[py:py + t, px:px + t]
        xs.append(np.stack([_norm(win), tmpl_n, _norm(win - med), rt0_n]))
        ss.append([float(resp[py, px]), float(r2[py, px])])
    return (np.asarray(xs, dtype=np.float32),
            np.asarray(ss, dtype=np.float32))


def rerank(search, tmpl, med, rt0, resp, r2, wide, cfg):
    weights = load_weights(cfg.reranker_path)
    pool_idx = build_pool(resp, r2, wide, cfg.reranker_pool)
    x, s = build_stacks(search, tmpl, med, rt0, resp, r2, wide, pool_idx)
    logits = forward(weights, x, s)
    all_logits = np.append(logits, float(np.ravel(weights["null_b"])[0]))
    all_logits -= all_logits.max()
    p = np.exp(all_logits)
    p /= p.sum()
    win = int(np.argmax(p))
    if win == len(logits) or p[win] < cfg.reranker_prob:
        return None, float(p[:-1].max()) if len(logits) else 0.0
    return int(pool_idx[win]), float(p[win])
