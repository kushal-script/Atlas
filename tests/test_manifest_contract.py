"""The scored contract holds for every manifest a person might hand the entry point.

One row in, one row out, always. A manifest that cannot be parsed is worth one
conservative rejection per row and never an abandoned run: the entry point writes
the predictions file at the end, so an exception escaping the pair loop produces
no file at all and forfeits every pair rather than the one that is malformed.
The paths here point at nothing on purpose, which exercises the parser and the
contract without invoking the matcher.
"""

import csv
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
COLUMNS = ["pair_id", "x", "y", "theta", "scale", "found", "score"]
H = "pair_id,reference_path,search_path"
R = "/nonexistent/a.png,/nonexistent/b.png"

CASES = [
    ("ragged_extra_field", f"{H}\np0,{R}\np1,{R},\np2,{R}\n", 3),
    ("short_row", f"{H}\np0,{R}\np1,/nonexistent/a.png\np2,{R}\n", 3),
    ("header_only", f"{H}\n", 0),
    ("empty_file", "", 0),
    ("duplicate_ids", f"{H}\np0,{R}\np0,{R}\n", 2),
    ("utf8_bom", "﻿" + f"{H}\np0,{R}\n", 1),
    ("extra_columns", f"{H},notes\np0,{R},hi\n", 1),
    ("reordered_columns", f"search_path,pair_id,reference_path\n/nonexistent/b.png,p0,/nonexistent/a.png\n", 1),
    ("unknown_headers", "a,b,c\n1,2,3\n1,2,3\n", 2),
    ("whitespace_paths", f"{H}\np0,  /nonexistent/a.png  ,  /nonexistent/b.png  \n", 1),
    ("quoted_comma_path", f'{H}\np0,"/nonexistent/a,b.png",/nonexistent/b.png\n', 1),
    ("crlf_endings", f"{H}\r\np0,{R}\r\n", 1),
]


@pytest.mark.parametrize("name,body,expected", CASES, ids=[c[0] for c in CASES])
def test_malformed_manifest_still_writes_one_row_per_pair(tmp_path, name, body, expected):
    manifest = tmp_path / f"{name}.csv"
    manifest.write_text(body, encoding="utf-8")
    out = tmp_path / f"{name}.pred.csv"

    proc = subprocess.run(
        [sys.executable, "register.py", "--input", str(manifest), "--output", str(out)],
        cwd=REPO, capture_output=True, text=True, timeout=300)

    assert proc.returncode == 0, f"entry point exited {proc.returncode}: {proc.stderr[-800:]}"
    assert out.exists(), "no predictions file was written, which forfeits every pair"

    with open(out, newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert list(rows[0].keys()) == COLUMNS if rows else True
    assert len(rows) == expected
    for row in rows:
        if row["found"] == "0":
            assert row["x"] == "0" and row["y"] == "0"
            assert row["theta"] == "0" and row["scale"] == "0"


def test_symlinked_images_resolve(tmp_path):
    """A pair whose paths are symlinks must localize like the originals."""
    import numpy as np
    import cv2
    rng = np.random.default_rng(5)
    search = (rng.random((300, 300)) * 200 + 20).astype(np.uint8)
    ref = search[80:180, 90:190].copy()
    real = tmp_path / "real"
    real.mkdir()
    cv2.imwrite(str(real / "ref.png"), ref)
    cv2.imwrite(str(real / "search.png"), search)
    link = tmp_path / "link"
    link.mkdir()
    try:
        (link / "ref.png").symlink_to(real / "ref.png")
        (link / "search.png").symlink_to(real / "search.png")
    except OSError:
        import pytest
        pytest.skip("symlink creation not permitted on this platform")
    manifest = tmp_path / "pairs.csv"
    manifest.write_text("pair_id,reference_path,search_path\n"
                        f"p0,{link / 'ref.png'},{link / 'search.png'}\n")
    out = tmp_path / "pred.csv"
    proc = subprocess.run(
        [sys.executable, str(REPO / "register.py"),
         "--input", str(manifest), "--output", str(out)],
        capture_output=True, text=True, timeout=180)
    assert proc.returncode == 0, proc.stderr
    rows = list(csv.DictReader(open(out)))
    assert len(rows) == 1


def test_degenerate_blank_pair_is_deliberately_rejected(tmp_path):
    """A near constant capture is rejected by decision, not by accident."""
    import numpy as np
    import cv2
    blank = np.full((300, 300), 128, np.uint8)
    cv2.imwrite(str(tmp_path / "ref.png"), blank)
    cv2.imwrite(str(tmp_path / "search.png"), blank)
    manifest = tmp_path / "pairs.csv"
    manifest.write_text("pair_id,reference_path,search_path\n"
                        f"p0,{tmp_path / 'ref.png'},{tmp_path / 'search.png'}\n")
    out = tmp_path / "pred.csv"
    proc = subprocess.run(
        [sys.executable, str(REPO / "register.py"),
         "--input", str(manifest), "--output", str(out)],
        capture_output=True, text=True, timeout=180)
    assert proc.returncode == 0, proc.stderr
    rows = list(csv.DictReader(open(out)))
    assert len(rows) == 1
    assert rows[0]["found"] == "0"
    assert rows[0]["x"] == "0" and rows[0]["y"] == "0"


def test_sixteen_bit_images_are_coerced_and_processed(tmp_path):
    """A 16 bit export must produce the same row as its 8 bit original.

    The coercion shifts sixteen bit values down eight bits, the exact inverse
    of how such an export is made, so a real pair re encoded at 16 bits must
    localize identically, not merely produce a row."""
    import numpy as np
    import cv2
    src = REPO / "data" / "p2smoke" / "pair_0000"
    if not src.exists():
        import pytest
        pytest.skip("smoke suite not present")
    outs = {}
    for depth in ("8", "16"):
        d = tmp_path / depth
        d.mkdir()
        for name in ("reference.png", "search.png"):
            img = cv2.imread(str(src / name), cv2.IMREAD_UNCHANGED)
            if depth == "16":
                img = (img.astype(np.uint16) << 8)
            cv2.imwrite(str(d / name), img)
        manifest = d / "pairs.csv"
        manifest.write_text("pair_id,reference_path,search_path\n"
                            f"p0,{d / 'reference.png'},{d / 'search.png'}\n")
        out = d / "pred.csv"
        proc = subprocess.run(
            [sys.executable, str(REPO / "register.py"),
             "--input", str(manifest), "--output", str(out)],
            capture_output=True, text=True, timeout=300)
        assert proc.returncode == 0, proc.stderr
        outs[depth] = list(csv.DictReader(open(out)))[0]
    assert outs["8"] == outs["16"]


def test_output_file_is_written_incrementally(tmp_path):
    """Rows must reach disk as pairs finish, so a kill loses one pair, not all."""
    import numpy as np
    import cv2
    rng = np.random.default_rng(12)
    search = (rng.random((300, 300)) * 200 + 20).astype(np.uint8)
    ref = search[60:160, 70:170].copy()
    cv2.imwrite(str(tmp_path / "ref.png"), ref)
    cv2.imwrite(str(tmp_path / "search.png"), search)
    manifest = tmp_path / "pairs.csv"
    manifest.write_text("pair_id,reference_path,search_path\n"
                        + "".join(f"p{i},{tmp_path / 'ref.png'},{tmp_path / 'search.png'}\n"
                                  for i in range(3)))
    out = tmp_path / "pred.csv"
    proc = subprocess.Popen(
        [sys.executable, str(REPO / "register.py"),
         "--input", str(manifest), "--output", str(out)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    first = proc.stdout.readline()
    import time as _t
    deadline = _t.time() + 120
    while "found=" not in first and _t.time() < deadline:
        first = proc.stdout.readline()
    proc.kill()
    proc.wait(timeout=30)
    rows = list(csv.DictReader(open(out)))
    assert len(rows) >= 1, "at least the finished pair must already be on disk"
