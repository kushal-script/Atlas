"""Pair discovery must work for every layout an evaluator might hand us.

These are regression tests for a defect that broke the graded path: the role
word was stripped as a bare substring, so "reference" became "erence" and the
two halves of a pair could never agree on a token. A directory holding exactly
reference.png and search.png, which is the most obvious way to supply a single
pair, failed with "no pairs found".
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import localize as cli


def _write(path, seed=0):
    rng = np.random.default_rng(seed)
    Image.fromarray(rng.integers(0, 255, (16, 16), dtype=np.uint8)).save(path)


@pytest.mark.parametrize("ref_name,search_name", [
    ("reference.png", "search.png"),
    ("pair_0000_reference.png", "pair_0000_search.png"),
    ("reference_001.png", "search_001.png"),
    ("pair_0000_ref.png", "pair_0000_search.png"),
    ("case7-reference.png", "case7-wide.png"),
])
def test_flat_pairs_are_discovered(tmp_path, ref_name, search_name):
    _write(tmp_path / ref_name, 1)
    _write(tmp_path / search_name, 2)
    pairs = cli.discover_flat(tmp_path)
    assert len(pairs) == 1, f"{ref_name} and {search_name} did not pair"
    _, ref, search = pairs[0]
    assert ref.name == ref_name and search.name == search_name


def test_single_pair_directory_is_discovered(tmp_path):
    """--batch on a directory holding one pair must not report nothing found."""
    _write(tmp_path / "reference.png", 1)
    _write(tmp_path / "search.png", 2)
    pairs = cli.discover_nested(tmp_path)
    assert len(pairs) == 1
    assert pairs[0][1].name == "reference.png"
    assert pairs[0][2].name == "search.png"


def test_nested_pair_folders_still_work(tmp_path):
    for i in range(3):
        d = tmp_path / f"pair_{i:04d}"
        d.mkdir()
        _write(d / "reference.png", i)
        _write(d / "search.png", i + 10)
    pairs = cli.discover_nested(tmp_path)
    assert [p[0] for p in pairs] == ["pair_0000", "pair_0001", "pair_0002"]


def test_role_token_keeps_the_whole_word():
    assert cli._role_and_token("pair_0000_reference") == ("reference", "pair_0000")
    assert cli._role_and_token("pair_0000_search") == ("search", "pair_0000")
    assert cli._role_and_token("reference") == ("reference", "")
    assert cli._role_and_token("search") == ("search", "")
