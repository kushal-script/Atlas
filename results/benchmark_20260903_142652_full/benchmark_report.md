# Drift-Sense Benchmark — full (120 pairs, seed 999)

Generated at 2026-09-03T14:49:00.919185+00:00 on `macOS-26.5.1-arm64-arm-64bit`
CPU: Apple M3 | Memory: 16 GiB | Python 3.11.14 | numpy 2.4.6 scipy 1.17.1 cv2 5.0.0

Suite: `/Users/kushalsathyanarayan/Desktop/semicon_india_hackathon/PS2/results/benchmark_20260903_142652_full/suite` composition `{'A_nominal': 42, 'B_degraded': 42, 'C_absent': 24, 'D_optical': 12}`

## Runtime

| median | mean | p90 | max | over 5s | budget_gated | requirement |
|---|---|---|---|---|---|---|
| 2.67s | 3.94s | 7.81s | 8.32s | 28.3% | 0 | median 5.0s, hard 20s |

Median: **PASS**, Hard timeout: **PASS**

## Score (SEM pairs, organiser scheme)

| Loc A (nominal) | Loc B (degraded) | Loc pts /40 | Pose /20 | Reject F1 | AUC | Est. core /85 |
|---|---|---|---|---|---|---|
| 0.833 | 0.624 | 28.72 | 18.08 | 0.651 | 0.963 | 66.20 |

### Per-severity credit

| tier | n | credit |
|---|---|---|
| A_nominal/sev0 | 42 | 0.833 |
| B_degraded/sev1 | 11 | 0.818 |
| B_degraded/sev2 | 7 | 0.971 |
| B_degraded/sev3 | 11 | 0.727 |
| B_degraded/sev4 | 13 | 0.185 |

Predictions: `/Users/kushalsathyanarayan/Desktop/semicon_india_hackathon/PS2/results/benchmark_20260903_142652_full/predictions.csv` | Full report: `benchmark_report.json`

## How to compare across devices

Commit `benchmark_report.json` from each device and diff `runtime.median_s_per_pair`, `runtime.p90_s`, `runtime.max_s`, `estimated_core`, `rejection.f1`, `calibration.auc`. Headline full-suite numbers should sit in the 76.6–81.7 band for `full` (120) and scale similarly for `quick` (40).
