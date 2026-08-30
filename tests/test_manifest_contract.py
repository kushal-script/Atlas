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
