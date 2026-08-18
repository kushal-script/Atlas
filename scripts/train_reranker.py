"""Train the stage two re-ranker and export numpy weights.

Loads harvested candidate pools, trains the small CNN with a softmax over
each pair's candidates plus a learnable null class, applies dihedral
augmentation, calibrates the abstention threshold on the validation split,
verifies that the exported numpy forward pass matches torch, writes the
weights to models/reranker.npz and records the run with plots under
experiments/.

Usage:
    python scripts/train_reranker.py --data data/reranker_harvest --epochs 40
"""

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

REPO = Path(__file__).resolve().parents[1]


class Reranker(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(4, 16, 5, stride=2)
        self.c2 = nn.Conv2d(16, 32, 3, stride=2)
        self.c3 = nn.Conv2d(32, 32, 3, stride=2)
        self.f1 = nn.Linear(34, 32)
        self.f2 = nn.Linear(32, 1)
        self.null_b = nn.Parameter(torch.zeros(1))

    def forward(self, x, s):
        h = torch.relu(self.c1(x))
        h = torch.relu(self.c2(h))
        h = torch.relu(self.c3(h))
        feat = h.mean(dim=(2, 3))
        feat = torch.cat([feat, s], dim=1)
        h = torch.relu(self.f1(feat))
        return self.f2(h).squeeze(-1)


def load_pairs(data_dir):
    pairs = []
    for f in sorted(Path(data_dir).glob("*.npz")):
        d = np.load(f)
        pairs.append({"x": d["x"].astype(np.float32), "s": d["s"].astype(np.float32),
                      "label": int(d["label"]), "name": f.stem,
                      "placement": str(d["placement"])})
    return pairs


def augment(x, rng):
    k = int(rng.integers(4))
    x = torch.rot90(x, k, dims=(-2, -1))
    if rng.integers(2):
        x = torch.flip(x, dims=(-1,))
    return x


def pair_loss(model, pair, device, rng=None):
    x = torch.from_numpy(pair["x"]).to(device)
    s = torch.from_numpy(pair["s"]).to(device)
    if rng is not None:
        x = augment(x, rng)
    logits = model(x, s)
    all_logits = torch.cat([logits, model.null_b])
    target = pair["label"] if pair["label"] >= 0 else len(logits)
    return all_logits, torch.tensor(target, device=device)


def evaluate_split(model, pairs, device):
    model.eval()
    correct = 0
    records = []
    with torch.no_grad():
        for p in pairs:
            all_logits, target = pair_loss(model, p, device)
            prob = torch.softmax(all_logits, dim=0)
            pred = int(torch.argmax(prob))
            correct += int(pred == int(target))
            records.append({"pred": pred, "target": int(target),
                            "n": len(all_logits) - 1,
                            "prob": float(prob[pred])})
    return correct / max(len(pairs), 1), records


def calibrate_tau(records):
    table = {}
    best = 0.5
    for tau in np.arange(0.30, 0.91, 0.05):
        fired = [r for r in records
                 if r["pred"] < r["n"] and r["prob"] >= tau]
        right = [r for r in fired if r["pred"] == r["target"]]
        prec = len(right) / len(fired) if fired else 1.0
        table[round(float(tau), 2)] = {"fired": len(fired), "correct": len(right),
                                       "precision": round(prec, 4)}
        if prec >= 0.97 and len(fired) > 0:
            best = round(float(tau), 2)
            break
    return best, table


def export_weights(model, path):
    sd = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    np.savez(path,
             c1_w=sd["c1.weight"], c1_b=sd["c1.bias"],
             c2_w=sd["c2.weight"], c2_b=sd["c2.bias"],
             c3_w=sd["c3.weight"], c3_b=sd["c3.bias"],
             f1_w=sd["f1.weight"], f1_b=sd["f1.bias"],
             f2_w=sd["f2.weight"], f2_b=sd["f2.bias"],
             null_b=sd["null_b"])


def verify_parity(model, pairs, weight_path, device):
    from drift_sense.reranker import forward, load_weights
    weights = load_weights(weight_path)
    p = pairs[0]
    with torch.no_grad():
        torch_logits = model(torch.from_numpy(p["x"]).to(device),
                             torch.from_numpy(p["s"]).to(device)).cpu().numpy()
    np_logits = forward(weights, p["x"], p["s"])
    diff = float(np.abs(torch_logits - np_logits).max())
    assert diff < 1e-3, f"numpy and torch forward disagree by {diff}"
    return diff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = "cpu"
    pairs = load_pairs(args.data)
    val = [p for i, p in enumerate(pairs) if i % 7 == 0]
    train = [p for i, p in enumerate(pairs) if i % 7 != 0]
    print(f"{len(train)} train pairs, {len(val)} val pairs")

    model = Reranker().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=30, gamma=0.1)
    ce = nn.CrossEntropyLoss()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = REPO / "experiments" / f"{stamp}_reranker_training"
    run_dir.mkdir(parents=True)
    history = []
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        order = rng.permutation(len(train))
        total = 0.0
        opt.zero_grad()
        for j, idx in enumerate(order):
            all_logits, target = pair_loss(model, train[idx], device, rng)
            loss = ce(all_logits[None], target[None])
            loss.backward()
            total += float(loss)
            if (j + 1) % 8 == 0 or j == len(order) - 1:
                opt.step()
                opt.zero_grad()
        sched.step()
        val_acc, _ = evaluate_split(model, val, device)
        train_loss = total / len(train)
        history.append({"epoch": epoch, "train_loss": round(train_loss, 4),
                        "val_acc": round(val_acc, 4)})
        print(f"epoch {epoch:3d} loss {train_loss:.4f} val_acc {val_acc:.3f}")

    val_acc, records = evaluate_split(model, val, device)
    tau, table = calibrate_tau(records)
    weight_path = REPO / "models" / "reranker.npz"
    torch_path = REPO / "models" / "reranker.pt"
    weight_path.parent.mkdir(exist_ok=True)
    export_weights(model, weight_path)
    torch.save(model.state_dict(), torch_path)
    print(f"wrote {weight_path} and {torch_path}")
    parity = verify_parity(model, val, weight_path, device)

    with open(run_dir / "training_log.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(history[0].keys()))
        w.writeheader()
        w.writerows(history)
    with open(run_dir / "config.json", "w") as fh:
        json.dump({"data": str(args.data), "epochs": args.epochs, "lr": args.lr,
                   "seed": args.seed, "train_pairs": len(train),
                   "val_pairs": len(val), "final_val_acc": val_acc,
                   "tau": tau, "calibration": table,
                   "numpy_torch_max_diff": parity,
                   "runtime_s": time.time() - t0}, fh, indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax1 = plt.subplots(figsize=(6.4, 4.2), dpi=150)
    ax1.plot([h["epoch"] for h in history], [h["train_loss"] for h in history],
             color="#2a78d6", linewidth=2, label="train loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("train loss")
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.grid(True, color="#ececea", linewidth=0.8)
    ax1.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(run_dir / "loss_curve.png")
    fig2, ax2 = plt.subplots(figsize=(6.4, 4.2), dpi=150)
    ax2.plot([h["epoch"] for h in history], [h["val_acc"] for h in history],
             color="#2a78d6", linewidth=2)
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("validation pair accuracy")
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.grid(True, color="#ececea", linewidth=0.8)
    ax2.set_axisbelow(True)
    fig2.tight_layout()
    fig2.savefig(run_dir / "val_accuracy.png")

    print(f"val accuracy {val_acc:.3f}, calibrated tau {tau}, "
          f"numpy parity diff {parity:.2e}, weights at {weight_path}, "
          f"run recorded in {run_dir}")


if __name__ == "__main__":
    main()
