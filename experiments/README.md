# The experiment record

Every measured claim in the top level readme traces to a folder here. Two
naming forms coexist: a folder named `YYYYMMDD_HHMMSS_name` is one recorded
run, its `records.json`, predictions and `report.json` written by the harness
that ran it; a folder named `YYYYMMDD_name` is a campaign, and its
`report.md` states what was asked, what was measured and what shipped or was
declined. A run folder holds `config.json` with the dataset path and the full
matcher configuration, `results.csv` or `records.json` with one row per pair,
`metrics.json` or `report.json` with the aggregates, and often `plots/`, so
every number stays traceable to the exact configuration that produced it. Reading the campaign reports in order reconstructs every decision
in the repository; the single run folders are their raw material.

The campaigns that decide the shipped configuration, in reading order:

| Folder | What it settled |
| --- | --- |
| `20260830_threshold_and_oracle` | the presence threshold protocol and the optical always found rule |
| `20260830_degrade_attribution` | the severity loss split factor by factor: 65 percent is dose and blur, information never collected |
| `20260830_optical_margin` | the Set D bonus gate measured at 132 plus 72 pairs with a bootstrap on the 0.40 gate |
| `20260830_bank_factorial` | the blur bank extension full factorial, negative, three earlier claims retracted |
| `20260901_organiser_convention` | the reported rotation sign settled against the organisers' own labels |
| `20260901_raw_confirm_and_found_f1` | the found class F1 reading and the raw full reference confirmation override |
| `20260901_presence_v3_refit` | the eighteen feature presence model chosen over fifteen and twenty one |
| `20260901_stress_and_decoys` | the held out battery, the decoy weakness priced, five post freeze surprise seeds |
| `20260901_organiser_sample_validation` | 0.988 credit on the organisers' shared twenty, never fitted on |
| `20260901_alien_distribution` | six unseen appearance families, the polarity fix, 1.000 after it |
| `20260902_raw_override_hardening` | the override margin floor moved to 0.05 on fresh records after an external audit |
| `20260902_sev4_oracle_revisited` | the severity four information limit re established under four independent statistics |
| `20260902_presence_dev_refit` | the decoy discriminating signal found, five deployments measured and declined |
| `20260902_final_adversarial_audit` | five dimension adversarial audit, six findings shipped or closed by measurement |
| `20260903_reference_machine_runtime` | the pose surfaces pooled for the four core reference machine, byte identical math |

The audit response folders, `20260902_sixth_audit_response` through
`20260903_eighth_audit_response`, disposition every external review finding
by finding, each one either shipped with a regression test or refuted with
recorded evidence.
