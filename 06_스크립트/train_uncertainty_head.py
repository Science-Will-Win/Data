#!/usr/bin/env python3
"""Uncertainty Head Training - Surrogate Feature MLP"""
import json, os, argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

LABELS_FILE = "trajectory_output_phase2_v2/uncertainty_labels.json"
TRAJ_FILE = "trajectory_output/deduped_r0_eval1.jsonl"
OUTPUT_DIR = "uncertainty_head_output"
DEVICE = "cuda:0"

class UncertaintyHead(nn.Module):
    def __init__(self, input_dim=12, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1), nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

def extract_features(trajectory, task_type=""):
    if not trajectory or len(trajectory) == 0:
        return [0.0] * 12
    steps = len(trajectory)
    full_text, tool_calls, error_count, think_count, solution_found = "", {}, 0, 0, False
    for step in trajectory:
        content = str(step.get("content", "") if isinstance(step, dict) else step)
        full_text += content + " "
        if "execute" in content.lower() or "tool" in content.lower():
            tn = content[:50]
            tool_calls[tn] = tool_calls.get(tn, 0) + 1
        for ew in ["error", "exception", "traceback", "failed", "not found"]:
            if ew in content.lower(): error_count += 1
        if "<think>" in content: think_count += 1
        if "<solution>" in content: solution_found = True
    return [
        min(steps / 30.0, 1.0), min(error_count / 10.0, 1.0),
        min(sum(1 for v in tool_calls.values() if v >= 4) / 3.0, 1.0),
        min(len(tool_calls) / 10.0, 1.0), min(len(full_text) / 50000.0, 1.0),
        min(think_count / 10.0, 1.0), 1.0 if solution_found else 0.0,
        min(steps / 5.0, 1.0) if steps > 15 else 0.0,
        1.0 if error_count > 3 else 0.0,
        1.0 if "gwas" in task_type.lower() else 0.0,
        1.0 if "rare" in task_type.lower() else 0.0,
        1.0 if "patient" in task_type.lower() else 0.0,
    ]

class UncertaintyDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx): return self.features[idx], self.labels[idx]

def load_data():
    with open(LABELS_FILE) as f: labels = json.load(f)
    traj_map = {}
    with open(TRAJ_FILE) as f:
        for line in f:
            d = json.loads(line); traj_map[d["task_id"]] = d
    features, targets, task_ids = [], [], []
    for item in labels:
        tid = item["task_id"]
        td = traj_map.get(tid, {})
        feat = extract_features(td.get("trajectory", []),
                                item.get("task_type", td.get("task_type", "")))
        features.append(feat)
        if item["should_refuse"]:
            target = 0.9
        else:
            diff = item.get("difficulty", "medium")
            target = {"easy": 0.1, "medium": 0.4}.get(diff, 0.6)
        targets.append(target)
        task_ids.append(tid)
    print(f"Data: {len(features)} | refuse=True: {sum(1 for l in labels if l['should_refuse'])} | False: {sum(1 for l in labels if not l['should_refuse'])}")
    return features, targets, task_ids

def train(args):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    features, targets, task_ids = load_data()
    X_tr, X_v, y_tr, y_v, _, _ = train_test_split(features, targets, task_ids, test_size=0.2, random_state=42)
    print(f"Train: {len(X_tr)} | Val: {len(X_v)}")
    tr_loader = DataLoader(UncertaintyDataset(X_tr, y_tr), batch_size=args.batch_size, shuffle=True)
    va_loader = DataLoader(UncertaintyDataset(X_v, y_v), batch_size=args.batch_size, shuffle=False)
    model = UncertaintyHead(input_dim=12, hidden_dim=args.hidden_dim).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    def cal_loss(p, t):
        return nn.MSELoss()(p, t) + 0.5*((1-p)*t).mean() + 0.3*(p*(1-t)).mean()
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    best_vl, best_ep = float("inf"), 0
    for ep in range(args.epochs):
        model.train(); tl = 0
        for bx, by in tr_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            loss = cal_loss(model(bx), by)
            opt.zero_grad(); loss.backward(); opt.step(); tl += loss.item()
        tl /= len(tr_loader); sched.step()
        model.eval(); vl = 0; ap, at = [], []
        with torch.no_grad():
            for bx, by in va_loader:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                p = model(bx); vl += cal_loss(p, by).item()
                ap.extend(p.cpu().tolist()); at.extend(by.cpu().tolist())
        vl /= len(va_loader)
        bp = [1 if p > 0.5 else 0 for p in ap]
        bt = [1 if t > 0.5 else 0 for t in at]
        acc = accuracy_score(bt, bp); f1 = f1_score(bt, bp, zero_division=0)
        print(f"  Epoch {ep+1:>3}/{args.epochs} | train={tl:.4f} | val={vl:.4f} | acc={acc:.3f} | f1={f1:.3f}")
        if vl < best_vl:
            best_vl = vl; best_ep = ep + 1
            torch.save({"model_state_dict": model.state_dict(), "epoch": ep,
                         "val_loss": vl, "accuracy": acc, "f1": f1},
                        os.path.join(OUTPUT_DIR, "best_model.pt"))
    print(f"\nBest: epoch {best_ep} | val_loss={best_vl:.4f}")
    ckpt = torch.load(os.path.join(OUTPUT_DIR, "best_model.pt"))
    model.load_state_dict(ckpt["model_state_dict"]); model.eval()
    ap, at = [], []
    with torch.no_grad():
        for bx, by in va_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            ap.extend(model(bx).cpu().tolist()); at.extend(by.cpu().tolist())
    bp = [1 if p > 0.5 else 0 for p in ap]
    bt = [1 if t > 0.5 else 0 for t in at]
    print(classification_report(bt, bp, target_names=["certain", "uncertain"], zero_division=0))
    json.dump({"best_epoch": best_ep, "val_loss": best_vl, "accuracy": ckpt["accuracy"],
               "f1": ckpt["f1"], "total_data": len(features), "train_size": len(X_tr),
               "val_size": len(X_v)},
              open(os.path.join(OUTPUT_DIR, "training_results.json"), "w"), indent=2)
    print(f"Saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden_dim", type=int, default=64)
    train(p.parse_args())
