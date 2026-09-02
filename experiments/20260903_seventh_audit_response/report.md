# Seventh external audit: nothing blocking by its own assessment, one stale artifact fixed, one false finding, the rest already dispositioned

The seventh audit's own bottom line is that nothing blocks submission, all
findings are verification steps rather than code changes, and its do not
fix list independently endorses four earlier dispositions: the SIGALRM
absence on Windows with the shared wall clock budget as the protection,
the reranker weights staying in the archive as a deliverable, the RGB
spread tolerance of two, and the 15 second budget. Dispositions of what
remained:

configs/matcher_default.json was stale, the audit's one genuine catch.
Nothing reads the file, but it recorded time_budget_s 9.0 and predated the
raw override, pose arbiter and streak fields, so a reader comparing it to
the README would find a contradiction. It is regenerated from the live
MatchConfig, all 57 fields, and now documents exactly what ships. The file
is not in the archive, so the archive is unchanged.

The Unix only interpreter path in generate_dataset.py is false at head:
the delegation uses sys.executable, not a hardcoded venv path, and runs on
any platform the entry point runs on.

The missing Windows path test was the audit's one worthwhile suggestion
and now exists: forcing the alarm flag off and driving main() over a
degenerate pair and a missing file asserts the loop writes one well formed
conservative row per pair from the shared budget and the exception paths
alone, the exact branches a Windows evaluator would hit. 69 tests pass.

Everything else restates settled ground: the 0.35 threshold the audit
itself confirms, the always entered wide grid whose 9.9 gate the sixth
audit already read correctly, the deliberate 0.0 score on timeout rows,
the per process correlator calibration whose two answers agree to float
rounding, the seed spawn tree independence, and the corner tests whose
real generator coverage lives in the pose robustness experiments.
