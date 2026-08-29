import csv
import sys
import time
import numpy as np
import cv2

sys.path.insert(0, "src")

from drift_sense.api import match_pair, presence_features

MANIFEST = "data/phase2_mixed/manifest.csv"
ROOT = "data/phase2_mixed"

rows = []
with open(MANIFEST) as f:
    for r in csv.DictReader(f):
        rows.append(r)

print(f"loaded {len(rows)} pairs", flush=True)

present = []
absent = []
records = []
t0 = time.time()
for i, r in enumerate(rows):
    ref = cv2.imread(f"{ROOT}/{r['reference']}", 0)
    search = cv2.imread(f"{ROOT}/{r['search']}", 0)
    d = match_pair(ref, search)["diagnostics"]
    p = int(r["present"])
    geo = float(np.clip(d.get("geo_consistency", 0.0), 0.0, 1.0))
    score = float(d["score"])
    wide = int(d.get("num_candidates_wide", d["num_candidates"]))
    peak_contrast = float(d.get("peak_contrast", 0.0))
    records.append((p, geo, score, wide, peak_contrast))
    if p == 1:
        present.append(geo)
    else:
        absent.append(geo)
    if (i + 1) % 10 == 0 or (i + 1) == len(rows):
        print(f"  {i+1}/{len(rows)}  elapsed {time.time()-t0:.0f}s", flush=True)


def stats(name, vals):
    a = np.array(vals, dtype=float)
    print(f"\n=== {name} (n={len(a)}) ===")
    print(f"  min    : {np.min(a):.4f}")
    print(f"  p10    : {np.percentile(a,10):.4f}")
    print(f"  median : {np.percentile(a,50):.4f}")
    print(f"  p90    : {np.percentile(a,90):.4f}")
    print(f"  max    : {np.max(a):.4f}")


stats("PRESENT", present)
stats("ABSENT", absent)

n_pres_under = int(np.sum(np.array(present) < 0.4))
n_abs_over = int(np.sum(np.array(absent) >= 0.4))
print(f"\nPRESENT with geo_consistency < 0.4 : {n_pres_under} / {len(present)}")
print(f"ABSENT  with geo_consistency >= 0.4: {n_abs_over} / {len(absent)}")

y = np.array([p for p, *_ in records])
geo = np.array([g for _, g, *_ in records])

best_f1 = -1.0
best_t = 0.0
best_conf = None
for t in np.arange(0.0, 1.0001, 0.02):
    pred = (geo >= t).astype(int)
    tp = int(np.sum((pred == 1) & (y == 1)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    if f1 > best_f1:
        best_f1 = f1
        best_t = float(t)
        best_conf = (tp, fp, fn, tn)

print("\n=== GEO-ONLY THRESHOLD SWEEP ===")
print(f"best F1        : {best_f1:.4f}")
print(f"threshold      : {best_t:.2f}")
tp, fp, fn, tn = best_conf
print(f"tp={tp} fp={fp} fn={fn} tn={tn}")
