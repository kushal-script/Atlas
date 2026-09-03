# Eighth audit: a version comparison that would zero the very scores it claims to protect, refuted on the organisers' own ground truth

The eighth review compared this repository against an old flat copy it
calls Version 1 and recommended submitting that instead, on four claims.
Each was tested rather than argued.

The critical claim, that theta_report_sign of minus one flips the reported
rotation and will zero rotation scores, is backwards. The constant converts
the localizer's internal grid angle, the rotation applied to the template,
into the reported convention, counter clockwise positive in the search
frame; the sixth audit traced that chain end to end and the September 1
organiser convention experiment settled it against their labels. Today's
fresh run on the organisers' shared twenty pairs makes it arithmetic:
fourteen present pairs carry rotations up to five degrees, the shipped
sign matches every one within 0.28 degrees, rotation credit 0.975, and the
flip the review recommends would score 0.144. Following the review would
have destroyed the scores it claimed to protect.

The OpenCV claim, that the evaluator environment may not carry it, is
refuted by the evaluators themselves: the organisers' own generator,
baseline and scorer import cv2, and the shipped requirements pin
opencv-python 5.0.0.93, which the archive's socket blocked self test
installs and runs.

The template size, 90 px scale adaptive against the old fixed 100, is the
August 12 ablation's measured choice, the change that lifted low zoom
coverage from 60 to 90 percent; it is recorded in the ledger, not drift.

Unvalidated is the one word the record refutes most simply: 0.988 mean
credit and 0.38 px median error on the organisers' sample this repository
never fitted on, against their baseline's 0.800, an eight suite battery,
69 tests, and seven prior audit rounds. Version 1 as described, no test
suite, no timeout, no presence decision and no seven column output, is the
Phase 1 era build; submitting it would forfeit rejection, calibration and
pose wholesale. Nothing changes.
