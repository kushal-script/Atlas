# Experiments

Every evaluation or training run gets one folder named `YYYYMMDD_HHMMSS_<name>`, created by `scripts/evaluate.py`. A run folder contains:

```
config.json     dataset path and full matcher configuration of the run
results.csv     one row per pair: truth, prediction, error, score, runtime
metrics.json    aggregate metrics overall and split by style and placement
plots/          error distribution, scatter diagnostics, success and failure montages
```

Runs are committed to the repository so results referenced in the presentation stay reproducible and traceable to the exact configuration that produced them.
