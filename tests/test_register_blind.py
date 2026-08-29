"""Blind-CSV submission-safety test (Phase 2, Day 6, T0).

The official entry point is:

    python register.py --input pairs.csv --output predictions.csv

where pairs.csv contains ONLY pair_id, reference_path, search_path (no ground
truth). This test proves register.py (1) runs end-to-end on a blind manifest,
(2) emits a valid found/score decision, and (3) IGNORES any present/gt_* columns
that happen to be present (leakage guard): adding those columns with fabricated
values must not change the output.

It does NOT require the deep-learning framework and never touches the network.
"""

import csv
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import cv2

REPO = Path(__file__).resolve().parents[1]


def _write_pair(d, pid):
    rng = np.random.default_rng(abs(hash(pid)) % (2**32))
    ref = (rng.random((1000, 1000)) * 255).astype(np.uint8)
    # Search contains the reference rolled by a known shift; a real match exists.
    dx, dy = 120, 80
    search = np.roll(ref, shift=(dy, dx), axis=(0, 1)).astype(np.uint8)
    rp = d / f"{pid}_ref.png"
    sp = d / f"{pid}_search.png"
    cv2.imwrite(str(rp), ref)
    cv2.imwrite(str(sp), search)
    return rp, sp


def _run(blind_csv, out_csv):
    return subprocess.run(
        [sys.executable, str(REPO / "register.py"),
         "--input", str(blind_csv), "--output", str(out_csv)],
        capture_output=True, text=True,
    )


def _read_found_score(out_csv):
    rows = list(csv.DictReader(open(out_csv)))
    return [(r["found"], r["score"]) for r in rows]


def test_register_runs_on_blind_csv():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        pids = [f"pair_{i:03d}" for i in range(3)]
        pairs = [_write_pair(td, p) for p in pids]

        blind = td / "blind.csv"
        with open(blind, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["pair_id", "reference_path", "search_path"])
            for pid, (rp, sp) in zip(pids, pairs):
                w.writerow([pid, str(rp), str(sp)])

        out = td / "preds.csv"
        r = _run(blind, out)
        assert r.returncode == 0, r.stderr

        rows = list(csv.DictReader(open(out)))
        assert len(rows) == 3
        assert list(rows[0].keys()) == ["pair_id", "x", "y", "theta", "scale", "found", "score"]
        for row in rows:
            assert row["found"] in ("0", "1")
            float(row["score"])  # score is a parseable float in [0, 1]


def test_register_ignores_present_and_gt_columns():
    """Adding fabricated present/gt_* columns must not change the decision."""
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        pids = [f"pair_{i:03d}" for i in range(3)]
        pairs = [_write_pair(td, p) for p in pids]

        blind = td / "blind.csv"
        with open(blind, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["pair_id", "reference_path", "search_path"])
            for pid, (rp, sp) in zip(pids, pairs):
                w.writerow([pid, str(rp), str(sp)])

        # Same images, but with LYING ground-truth columns.
        leaking = td / "leaking.csv"
        with open(leaking, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["pair_id", "reference_path", "search_path", "present", "gt_x", "gt_y"])
            for pid, (rp, sp) in zip(pids, pairs):
                w.writerow([pid, str(rp), str(sp), 0, 17.0, 23.0])

        out_blind = td / "preds_blind.csv"
        out_leak = td / "preds_leak.csv"
        rb = _run(blind, out_blind)
        rl = _run(leaking, out_leak)
        assert rb.returncode == 0, rb.stderr
        assert rl.returncode == 0, rl.stderr

        blind_dec = _read_found_score(out_blind)
        leak_dec = _read_found_score(out_leak)
        assert blind_dec == leak_dec, (
            "register.py leaked ground-truth columns into the decision: "
            f"{blind_dec} != {leak_dec}"
        )
