"""The pair loop without SIGALRM, which is how Windows runs it.

The hard alarm is Unix only; on a platform without it the loop must still
produce one well formed row per pair from the shared wall clock budget and
the conservative exception paths alone. Forcing the flag off and driving
main() over a degenerate pair and a missing file exercises exactly the
branches a Windows evaluator would hit.
"""

import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import register


def test_loop_without_alarm_writes_conservative_rows(tmp_path, monkeypatch):
    blank = np.zeros((1000, 1000), dtype=np.uint8)
    cv2.imwrite(str(tmp_path / "blank.png"), blank)
    with open(tmp_path / "pairs.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pair_id", "reference_path", "search_path"])
        w.writerow(["blank_pair", "blank.png", "blank.png"])
        w.writerow(["missing_pair", "nope.png", "nope.png"])
    out = tmp_path / "pred.csv"
    monkeypatch.setattr(register, "_HAS_ALARM", False)
    monkeypatch.setattr(sys, "argv", ["register.py", "--input", str(tmp_path / "pairs.csv"),
                                      "--output", str(out)])
    register.main()
    rows = {r["pair_id"]: r for r in csv.DictReader(open(out))}
    assert set(rows) == {"blank_pair", "missing_pair"}
    for r in rows.values():
        assert r["found"] == "0"
        assert (r["x"], r["y"], r["theta"], r["scale"]) == ("0", "0", "0", "0")
    assert rows["blank_pair"]["score"] == "0.50000"
    assert rows["missing_pair"]["score"] == "0.50000"
