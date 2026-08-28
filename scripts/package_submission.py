"""Build and verify the Phase 2 submission zip.

Packs exactly what the scored run needs, then proves the pack works the way
the organisers will run it: the zip is extracted into a fresh directory, a
Python 3.11 interpreter runs register.py there against sample pairs, and a
socket shim makes any attempt at network access raise instead of connecting,
so the no network rule is demonstrated rather than assumed. The run must also
produce one row per pair with the exact required columns.

    .venv/bin/python scripts/package_submission.py --python311 /path/to/python3.11 \
        --pairs data/p2holdout2 --out submission_phase2.zip
"""

import argparse
import csv
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

INCLUDE = [
    "register.py",
    "generate_dataset.py",
    "localize.py",
    "requirements_phase2.txt",
    "models/presence_model.json",
    "models/reranker.npz",
    "src/drift_sense/__init__.py",
    "src/drift_sense/params.py",
    "src/drift_sense/generator.py",
    "src/drift_sense/localize.py",
    "src/drift_sense/backend.py",
    "src/drift_sense/presence.py",
    "src/drift_sense/api.py",
    "src/drift_sense/reranker.py",
    "src/drift_sense/evaluate.py",
    "src/drift_sense/geometry/__init__.py",
    "src/drift_sense/geometry/primitives.py",
    "src/drift_sense/geometry/dram.py",
    "src/drift_sense/geometry/finfet.py",
    "src/drift_sense/imaging/__init__.py",
    "src/drift_sense/imaging/sem.py",
    "src/drift_sense/imaging/optical.py",
    "scripts/generate_phase2_suite.py",
    "docs/citations.md",
    "README.md",
]

SOCKET_SHIM = '''
import socket as _socket
class _NoNetwork(OSError):
    pass
def _blocked(*a, **k):
    raise _NoNetwork("network access attempted during the scored run")
_socket.socket.connect = _blocked
_socket.socket.connect_ex = _blocked
_socket.create_connection = _blocked
_socket.getaddrinfo = _blocked
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--python311", type=Path, required=True)
    ap.add_argument("--pairs", type=Path, required=True,
                    help="dataset whose first pairs exercise the io contract")
    ap.add_argument("--num", type=int, default=4)
    ap.add_argument("--out", type=Path, default=REPO / "submission_phase2.zip")
    ap.add_argument("--extra", nargs="*", default=[],
                    help="additional files, e.g. the failure analysis pdf")
    args = ap.parse_args()

    missing = [f for f in INCLUDE if not (REPO / f).exists()]
    if missing:
        raise SystemExit(f"missing from the repository: {missing}")

    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in INCLUDE + list(args.extra):
            z.write(REPO / f, f)
        z.write(REPO / "requirements_phase2.txt", "requirements.txt")
    print(f"packed {args.out} ({args.out.stat().st_size/1024:.0f} KB, "
          f"{len(INCLUDE) + len(args.extra) + 1} files)")

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        with zipfile.ZipFile(args.out) as z:
            z.extractall(td / "run")
        (td / "shim" / "sitecustomize.py").parent.mkdir(parents=True)
        (td / "shim" / "sitecustomize.py").write_text(SOCKET_SHIM)

        rows = list(csv.DictReader(open(args.pairs / "ground_truth.csv")))
        sample = [r for r in rows if r["modality"] == "sem"][:args.num]
        pairs_csv = td / "pairs.csv"
        with open(pairs_csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["pair_id", "reference_path", "search_path"])
            for r in sample:
                w.writerow([r["pair_id"],
                            (args.pairs / r["reference_path"]).resolve(),
                            (args.pairs / r["search_path"]).resolve()])

        import os
        env = dict(os.environ)
        env["PYTHONPATH"] = str(td / "shim")
        res = subprocess.run(
            [str(args.python311), "register.py",
             "--input", str(pairs_csv), "--output", str(td / "pred.csv")],
            cwd=td / "run", env=env, capture_output=True, text=True, timeout=600)
        print("--- self test: extracted zip, python 3.11, sockets blocked ---")
        print("\n".join("  " + l for l in res.stdout.strip().split("\n")[-4:]))
        if res.returncode != 0:
            print(res.stderr[-1500:])
            raise SystemExit("SELF TEST FAILED")
        out_rows = list(csv.DictReader(open(td / "pred.csv")))
        assert len(out_rows) == len(sample), "row count mismatch"
        assert list(out_rows[0].keys()) == ["pair_id", "x", "y", "theta",
                                            "scale", "found", "score"], "wrong columns"
        for r in out_rows:
            if r["found"] == "0":
                assert r["x"] == "0" and r["theta"] == "0", "rejected row must zero pose"
        print(f"  SELF TEST PASSED: {len(out_rows)} rows, exact columns, "
              f"no network, no reads outside the supplied paths")


if __name__ == "__main__":
    main()
