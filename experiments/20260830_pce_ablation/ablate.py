import json, sys, numpy as np
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from drift_sense.presence import FEATURES, features_from_record
from fit_presence import fit_logistic

recs = json.load(open("experiments/20260830_190246_p2_pce/records.json"))
Xf = np.array([features_from_record(r) for r in recs], float)
y = np.array([1 if r["truth_found"] else 0 for r in recs])
rng = np.random.default_rng(0)
folds = np.zeros(len(y), int)
for cls in (0, 1):
    idx = np.where(y == cls)[0]; rng.shuffle(idx)
    for k, i in enumerate(idx): folds[i] = k % 5

def cv_f1(cols, label):
    X = Xf[:, cols]
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Xs = (X - mu) / sd
    probs = np.zeros(len(y))
    for k in range(5):
        tr, te = folds != k, folds == k
        w = fit_logistic(Xs[tr], y[tr])
        probs[te] = 1 / (1 + np.exp(-(Xs[te] @ w[:-1] + w[-1])))
    best = 0.0
    for t in np.arange(0.05, 0.96, 0.005):
        pred = probs < t                      # predicted absent
        tp = int(((y == 0) & pred).sum()); fp = int(((y == 1) & pred).sum())
        fn = int(((y == 0) & ~pred).sum())
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        best = max(best, f1)
    print(f"  {label:<38} cv reject F1 {best:.4f}")
    return best

i_pce = FEATURES.index("pce"); i_pw = FEATURES.index("pose_wide")
allc = list(range(len(FEATURES)))
a = cv_f1(allc, "16 features, pce + live pose_wide")
b = cv_f1([c for c in allc if c != i_pce], "15 features, live pose_wide, NO pce")
c = cv_f1([c for c in allc if c != i_pw], "15 features, pce, NO pose_wide")
d = cv_f1([c for c in allc if c not in (i_pce, i_pw)], "14 features, neither")
print(f"\n  pce alone contributes   {a - b:+.4f}")
print(f"  pose_wide alone         {a - c:+.4f}")
print(f"  both together vs neither{a - d:+.4f}")
