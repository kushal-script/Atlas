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
import tempfile
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# The addendum names four things the zip must carry: the entry point, a pip
# freeze under the name requirements.txt, a documented generator, and a failure
# analysis of at most two pages. All four are listed here rather than left to a
# command line flag, because a required deliverable that ships only when the
# packager is invoked with the right argument is a deliverable that will one day
# not ship; REQUIRED below fails the build loudly instead.
INCLUDE = [
    "register.py",
    "generate_dataset.py",
    "localize.py",
    "requirements_phase2.txt",
    "submission/failure_analysis.pdf",
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
    # Everything the shipped README points at. A readme that references a file
    # the zip does not carry is a broken reference for whoever extracts it,
    # and these are all small: the whole set roughly doubles a 190 KB archive.
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
]

# Two of the disqualifying behaviours are checked here rather than asserted.
# The socket shim turns any network attempt into an exception. The audit hook
# records every file the run opens, so reading outside the supplied paths is
# demonstrated absent rather than promised: the interpreter, its standard
# library and the extracted submission are the only trees a scored run has any
# business touching, and anything else is reported by path.
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
            # The failure analysis is named by the addendum, so it lands at the
            # top of the zip under that name rather than under its repository
            # directory.
            z.write(REPO / f, Path(f).name if f.endswith("failure_analysis.pdf") else f)
        z.write(REPO / "requirements_phase2.txt", "requirements.txt")
    print(f"packed {args.out} ({args.out.stat().st_size/1024:.0f} KB, "
          f"{len(INCLUDE) + len(args.extra) + 1} files)")

    # Every deliverable the addendum names by name, checked against the zip that
    # was actually written rather than against the list that was meant to build
    # it, and the page limit checked rather than assumed.
    with zipfile.ZipFile(args.out) as z:
        names = set(z.namelist())
    required = {"register.py", "requirements.txt", "generate_dataset.py",
                "failure_analysis.pdf"}
    absent = sorted(required - names)
    if absent:
        raise SystemExit(f"zip is missing required deliverables: {absent}")
    pages = int(subprocess.run(["pdfinfo", str(REPO / "submission/failure_analysis.pdf")],
                               capture_output=True, text=True
                               ).stdout.split("Pages:")[1].split("\n")[0].strip())
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
        # The two trees a scored run may read: the extracted submission it runs
        # from, and the directory holding the pairs csv and the images it names.
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
            if r["found"] == "0":
                assert r["x"] == "0" and r["theta"] == "0", "rejected row must zero pose"
        print(f"  SELF TEST PASSED: {len(out_rows)} rows, exact columns, "
              f"no network, no reads outside the supplied paths")


if __name__ == "__main__":
    main()
