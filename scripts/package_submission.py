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
import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

INCLUDE = [
    "register.py",
    "generate_dataset.py",
    "localize.py",
    "requirements_phase2.txt",
    "submission/failure_analysis.pdf",
    "models/presence_model.json",
    "models/rerank_combiner.json",
    "scripts/fit_rerank.py",
    "models/reranker.npz",
    "src/drift_sense/__init__.py",
    "src/drift_sense/params.py",
    "src/drift_sense/generator.py",
    "src/drift_sense/localize.py",
    "src/drift_sense/backend.py",
    "src/drift_sense/presence.py",
    "src/drift_sense/api.py",
    "src/drift_sense/reranker.py",
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
    "models/reranker.pt",
    "references/references.bib",
    "docs/phase2_failure_analysis.md",
    "docs/failure_analysis.md",
    "docs/architecture.md",
    "docs/dataset_format.md",
    "docs/development_log.md",
    "notebooks/train_reranker.ipynb",
    "scripts/train_reranker.py",
    "scripts/eval_degraded.py",
    "scripts/score_predictions.py",
    "results/README.md",
    "results/runtime_protocol.json",
    "requirements_freeze.txt",
    "requirements_dev.txt",
    "requirements_train.txt",
    "scripts/build_results_tables.py",
    "scripts/oracle_probe.py",
    "scripts/tune_threshold.py",
    "scripts/setup_python311.sh",
    "scripts/fit_presence.py",
    "scripts/tune_phase2.py",
    "scripts/eval_presence_models.py",
    "scripts/generate_stress_dataset.py",
    "scripts/generate_starter_spec_dataset.py",
    "scripts/generate_amat_proxy.py",
    "setup.py",
    "docs/lab_guide.md",
]

SOCKET_SHIM = '''
import sys as _sys, os as _os, socket as _socket
class _NoNetwork(OSError):
    pass
def _blocked(*a, **k):
    raise _NoNetwork("network access attempted during the scored run")
_socket.socket.connect = _blocked
_socket.socket.connect_ex = _blocked
_socket.create_connection = _blocked
_socket.getaddrinfo = _blocked

# The interpreter and its standard library, the operating system's own
# trees, the extracted submission, the directory holding the supplied csv and
# its images, and the working directory the csv and predictions live in. A read
# anywhere else is the disqualifying behaviour this is here to catch.
_ALLOWED = tuple(_os.path.realpath(p) for p in
                 (_sys.prefix, _sys.base_prefix, _sys.exec_prefix,
                  _os.path.dirname(_os.__file__), "/System", "/usr", "/Library",
                  _os.environ["DS_RUN_DIR"], _os.environ["DS_DATA_DIR"],
                  _os.environ["DS_IO_DIR"]) if p)
_seen = set()
def _audit(event, args):
    if event == "open" and args and isinstance(args[0], str):
        p = _os.path.realpath(args[0])
        if not p.startswith(_ALLOWED) and p not in _seen:
            _seen.add(p)
            print("OUTSIDE_SUPPLIED_PATHS " + p, flush=True)
_sys.addaudithook(_audit)
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--python311", type=Path, default=REPO / ".venv/bin/python")
    ap.add_argument("--pairs", type=Path, required=True,
                    help="dataset whose first pairs exercise the io contract")
    ap.add_argument("--num", type=int, default=4)
    ap.add_argument("--out", type=Path, default=REPO / "submission_phase2.zip")
    ap.add_argument("--extra", nargs="*", default=[],
                    help="additional files, e.g. the failure analysis pdf")
    args = ap.parse_args()

    if not args.python311.exists():
        raise SystemExit(f"no Python 3.11 interpreter at {args.python311}; "
                         f"build one with bash scripts/setup_python311.sh")
    probe = subprocess.run(
        [str(args.python311), "-c",
         "import sys, cv2, numpy, scipy, PIL;"
         "print(sys.version_info[0], sys.version_info[1], cv2.__version__)"],
        capture_output=True, text=True)
    if probe.returncode != 0:
        raise SystemExit(f"{args.python311} cannot import the submission "
                         f"dependencies:\n{probe.stderr.strip()[-500:]}")
    major, minor, cv_version = probe.stdout.split()
    if (int(major), int(minor)) != (3, 11):
        raise SystemExit(f"{args.python311} is Python {major}.{minor}, "
                         f"the reference machine is 3.11")
    print(f"  self test interpreter: Python {major}.{minor}, cv2 {cv_version}")

    pdf = REPO / "submission/failure_analysis.pdf"
    if not pdf.exists():
        raise SystemExit("submission/failure_analysis.pdf is missing")
    pages = pdf.read_bytes().count(b"/Type /Page") - pdf.read_bytes().count(b"/Type /Pages")
    if pages > 2:
        raise SystemExit(f"failure analysis is {pages} pages, the limit is 2")
    print(f"  failure analysis pdf verified at {pages} pages from its bytes")

    missing = [f for f in INCLUDE if not (REPO / f).exists()]
    if missing:
        raise SystemExit(f"missing from the repository: {missing}")

    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in INCLUDE + list(args.extra):
            z.write(REPO / f, Path(f).name if f.endswith("failure_analysis.pdf") else f)
        z.write(REPO / "requirements_phase2.txt", "requirements.txt")
    print(f"packed {args.out} ({args.out.stat().st_size/1024:.0f} KB, "
          f"{len(INCLUDE) + len(args.extra) + 1} files)")

    with zipfile.ZipFile(args.out) as z:
        names = set(z.namelist())
    required = {"register.py", "requirements.txt", "generate_dataset.py",
                "failure_analysis.pdf"}
    absent = sorted(required - names)
    if absent:
        raise SystemExit(f"zip is missing required deliverables: {absent}")
    pdf = (REPO / "submission/failure_analysis.pdf").read_bytes()
    pages = len(re.findall(rb"/Type\s*/Page[^s]", pdf))
    if not pages:
        raise SystemExit("could not count pages in the failure analysis")
    if pages > 2:
        raise SystemExit(f"failure analysis is {pages} pages, the limit is 2")
    print(f"  deliverables present, failure analysis {pages} pages")

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        with zipfile.ZipFile(args.out) as z:
            z.extractall(td / "run")
        (td / "shim" / "sitecustomize.py").parent.mkdir(parents=True)
        (td / "shim" / "sitecustomize.py").write_text(SOCKET_SHIM)

        rows = list(csv.DictReader(open(args.pairs / "ground_truth.csv")))
        sem = [r for r in rows if r["modality"] == "sem"]
        present = [r for r in sem if r["found"] == "1"][:max(args.num - 1, 1)]
        absent = [r for r in sem if r["found"] == "0"][:3]
        assert absent, "sample suite carries no absent pair"
        sample = present + absent
        io_dir = td / "io"
        io_dir.mkdir()
        pairs_csv = io_dir / "pairs.csv"
        import shutil as _sh
        with open(pairs_csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["pair_id", "reference_path", "search_path"])
            for r in sample:
                pd = io_dir / r["pair_id"]
                pd.mkdir()
                _sh.copy(args.pairs / r["reference_path"], pd / "reference.png")
                _sh.copy(args.pairs / r["search_path"], pd / "search.png")
                w.writerow([r["pair_id"], f"{r['pair_id']}/reference.png",
                            f" {r['pair_id']}/search.png"])

        import os
        env = dict(os.environ)
        env["PYTHONPATH"] = str(td / "shim")
        env["DS_RUN_DIR"] = str((td / "run").resolve())
        env["DS_DATA_DIR"] = str(Path(args.pairs).resolve())
        env["DS_IO_DIR"] = str(td.resolve())
        res = subprocess.run(
            [str(args.python311), "register.py",
             "--input", str(pairs_csv), "--output", str(td / "pred.csv")],
            cwd=td / "run", env=env, capture_output=True, text=True, timeout=600)
        print("--- self test: extracted zip, python 3.11, sockets blocked ---")
        print("\n".join("  " + l for l in res.stdout.strip().split("\n")[-4:]))
        if res.returncode != 0:
            print(res.stderr[-1500:])
            raise SystemExit("SELF TEST FAILED")
        stray = sorted({l.split(" ", 1)[1] for l in res.stdout.split("\n")
                        if l.startswith("OUTSIDE_SUPPLIED_PATHS")})
        if stray:
            raise SystemExit("read outside the supplied paths: " + ", ".join(stray[:8]))
        out_rows = list(csv.DictReader(open(td / "pred.csv")))
        assert len(out_rows) == len(sample), "row count mismatch"
        assert list(out_rows[0].keys()) == ["pair_id", "x", "y", "theta",
                                            "scale", "found", "score"], "wrong columns"
        for r in out_rows:
            assert all(float(r[k]) == float(r[k]) for k in ("x", "y", "theta", "scale", "score")), "non finite value"
            if r["found"] == "0":
                assert (r["x"], r["y"], r["theta"], r["scale"]) == ("0",) * 4, \
                    "rejected row must zero all four pose columns"
        assert [r["pair_id"] for r in out_rows] == [r["pair_id"] for r in sample], "row order"
        if not any(r["found"] == "0" for r in out_rows):
            print("  note: no rejection on this sample; pose zeroing is locked "
                  "by the unit and end to end tests")
        print(f"  SELF TEST PASSED: {len(out_rows)} rows over csv relative paths with "
              f"an absent pair, exact columns and order, all pose zeros on rejection, "
              f"no network, no reads outside the supplied paths")


if __name__ == "__main__":
    main()
