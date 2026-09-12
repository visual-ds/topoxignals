"""
Node classification on High School Dynamic Contact Network (SocioPatterns 2012).

Architecture (DTDG):
    per-snapshot GAT → GRU over snapshot sequence → MLP classifier

Usage::

    python -m xplain_pipeline.models.gnn --data_dir /path/to/data
"""

import argparse
import json
import math
import os
import pickle
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data
from torch_geometric.nn import GATConv

SCHOOL_START_HOUR = 7
SCHOOL_END_HOUR = 19
NODE_IN = 5  # [gender, tod_sin, tod_cos, dow_sin, dow_cos]


# ── Data loading ─────────────────────────────────────────────────────────────

def load_raw(data_dir: str):
    contacts = pd.read_csv(
        os.path.join(data_dir, "highschool_2012.csv"),
        sep="\t", header=None, names=["t", "i", "j", "Ci", "Cj"],
    )
    metadata = pd.read_csv(
        os.path.join(data_dir, "highschool_2012_metadata.txt"),
        sep="\t", header=None, names=["id", "class", "gender"],
    )
    return contacts, metadata


def get_school_days(contacts: pd.DataFrame):
    tz = timezone.utc
    seen = sorted({datetime.fromtimestamp(int(t), tz=tz).date() for t in contacts["t"].unique()})
    days = []
    for date in seen:
        if date.weekday() >= 5:
            continue
        t_start = int(datetime(date.year, date.month, date.day, SCHOOL_START_HOUR, 0, 0, tzinfo=tz).timestamp())
        t_end = int(datetime(date.year, date.month, date.day, SCHOOL_END_HOUR, 0, 0, tzinfo=tz).timestamp())
        days.append(dict(date=date, weekday=date.weekday(), t_start=t_start, t_end=t_end))
    return days


# ── Feature encoding ─────────────────────────────────────────────────────────

GENDER_ENC = {"M": 0.0, "F": 1.0}


def circular(value: float, period: float):
    a = 2 * math.pi * value / period
    return math.sin(a), math.cos(a)


# ── Snapshot construction ────────────────────────────────────────────────────

def build_snapshot(win_contacts, meta, class_le, snap_mid_sec, t_start, t_end, weekday):
    if win_contacts.empty:
        return None

    students = sorted(set(win_contacts["i"]) | set(win_contacts["j"]))
    node_map = {s: idx for idx, s in enumerate(students)}

    day_dur = t_end - t_start
    tod_sin, tod_cos = circular((snap_mid_sec - t_start) / day_dur, 1.0)
    dow_sin, dow_cos = circular(weekday, 5.0)

    node_feats, node_labels, valid_flags = [], [], []
    for s in students:
        if s in meta.index:
            row = meta.loc[s]
            g_enc = GENDER_ENC.get(str(row["gender"]), 0.5)
            label = int(class_le.transform([row["class"]])[0])
            valid = True
        else:
            g_enc, label, valid = 0.5, -1, False
        node_feats.append([g_enc, tod_sin, tod_cos, dow_sin, dow_cos])
        node_labels.append(label)
        valid_flags.append(valid)

    edge_cnt = {}
    for row in win_contacts.itertuples(index=False):
        u, v = node_map[row.i], node_map[row.j]
        key = (min(u, v), max(u, v))
        edge_cnt[key] = edge_cnt.get(key, 0) + 1

    src, dst, wts = [], [], []
    for (u, v), cnt in edge_cnt.items():
        src += [u, v]; dst += [v, u]; wts += [cnt, cnt]

    # Retained for exported snapshots; SnapshotGAT currently uses topology only.
    edge_attr = torch.tensor(wts, dtype=torch.float).unsqueeze(1)
    edge_attr = edge_attr / (edge_attr.max() + 1e-8)

    return Data(
        x=torch.tensor(node_feats, dtype=torch.float),
        edge_index=torch.tensor([src, dst], dtype=torch.long),
        edge_attr=edge_attr,
        y=torch.tensor(node_labels, dtype=torch.long),
        num_nodes=len(students),
        student_ids=torch.tensor(students, dtype=torch.long),
        valid_mask=torch.tensor(valid_flags, dtype=torch.bool),
    )


def build_day(contacts, metadata, class_le, day_info, window_sec=3600, overlap_min=30):
    meta = metadata.set_index("id")
    step = window_sec - overlap_min * 60
    assert step > 0
    t_start, t_end, weekday = day_info["t_start"], day_info["t_end"], day_info["weekday"]
    day_contacts = contacts[(contacts["t"] >= t_start) & (contacts["t"] <= t_end)]

    snapshots = []
    for ws in range(t_start, t_end - window_sec + 1, step):
        we = ws + window_sec
        win = day_contacts[(day_contacts["t"] >= ws) & (day_contacts["t"] <= we)]
        snap = build_snapshot(win, meta, class_le, (ws + we) / 2.0, t_start, t_end, weekday)
        if snap is not None:
            snapshots.append(snap)
    return snapshots


def build_or_load_dataset(contacts, metadata, class_le, day_infos, window_sec, overlap_min, cache_path):
    if os.path.exists(cache_path):
        print(f"  Loading processed data from cache: {cache_path}")
        with open(cache_path, "rb") as f:
            all_days = pickle.load(f)
        print(f"  Loaded {len(all_days)} days from cache.")
        return all_days

    print("  Building dataset from raw files...")
    all_days = []
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    for d in day_infos:
        snaps = build_day(contacts, metadata, class_le, d, window_sec=window_sec, overlap_min=overlap_min)
        all_days.append(snaps)
        print(f"    {d['date']} ({day_names[d['weekday']]})  ->  {len(snaps)} snapshots")

    with open(cache_path, "wb") as f:
        pickle.dump(all_days, f)
    print(f"  Dataset cached to: {cache_path}")
    return all_days


# ── Node mask sampling ───────────────────────────────────────────────────────

def sample_masks(all_days: list, test_ratio: float, seed: int = None):
    rng = np.random.default_rng(seed)
    masks = []
    for day_snaps in all_days:
        sids = set()
        labels = {}
        for snap in day_snaps:
            for sid, lbl, valid in zip(
                snap.student_ids.tolist(), snap.y.tolist(), snap.valid_mask.tolist()
            ):
                if valid and lbl >= 0:
                    sids.add(sid)
                    labels[sid] = lbl

        sids = sorted(sids)
        n = len(sids)
        idx = np.arange(n)
        rng.shuffle(idx)
        n_test = max(1, int(n * test_ratio))
        test_idx = set(idx[:n_test].tolist())
        train_mask = torch.tensor([i not in test_idx for i in range(n)], dtype=torch.bool)
        test_mask = ~train_mask
        masks.append((train_mask, test_mask, sids))
    return masks


# ── Model ────────────────────────────────────────────────────────────────────

class SnapshotGAT(nn.Module):
    def __init__(self, in_ch, hidden_ch, out_ch, heads=4, dropout=0.3):
        super().__init__()
        self.conv1 = GATConv(in_ch, hidden_ch, heads=heads, dropout=dropout, concat=True)
        self.conv2 = GATConv(hidden_ch * heads, out_ch, heads=1, dropout=dropout, concat=False)
        self.norm1 = nn.LayerNorm(hidden_ch * heads)
        self.norm2 = nn.LayerNorm(out_ch)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        h = F.elu(self.norm1(self.conv1(x, edge_index)))
        h = self.drop(h)
        h = F.elu(self.norm2(self.conv2(h, edge_index)))
        return h


class DayGNN(nn.Module):
    """per-snapshot GAT → GRU → MLP classifier."""

    def __init__(self, node_in, gat_hidden, gat_out, gru_hidden, num_classes, heads=4, dropout=0.3):
        super().__init__()
        self.gat = SnapshotGAT(node_in, gat_hidden, gat_out, heads, dropout)
        self.gru = nn.GRU(gat_out, gru_hidden, num_layers=2, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(gru_hidden, gru_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gru_hidden // 2, num_classes),
        )

    def forward(self, snapshots: list, train_mask=None, test_mask=None, sids_order=None):
        device = snapshots[0].x.device

        snap_embs = []
        for snap in snapshots:
            h = self.gat(snap.x, snap.edge_index)
            snap_embs.append({int(sid): h[i] for i, sid in enumerate(snap.student_ids.tolist())})

        label_map = {}
        for snap in snapshots:
            for sid, lbl, valid in zip(
                snap.student_ids.tolist(), snap.y.tolist(), snap.valid_mask.tolist()
            ):
                if valid and lbl >= 0 and sid not in label_map:
                    label_map[sid] = lbl

        students = sids_order if sids_order is not None else sorted(label_map.keys())
        sid_to_row = {sid: i for i, sid in enumerate(students)}
        N, T = len(students), len(snapshots)
        emb_dim = next(iter(snap_embs[0].values())).shape[0]

        seq = torch.zeros(N, T, emb_dim, device=device)
        for t, emb_dict in enumerate(snap_embs):
            for sid, emb in emb_dict.items():
                if sid in sid_to_row:
                    seq[sid_to_row[sid], t] = emb

        _, h_n = self.gru(seq)
        h_final = h_n[-1]
        logits = self.head(h_final)
        labels = torch.tensor([label_map[sid] for sid in students], dtype=torch.long, device=device)

        return logits, labels, train_mask, test_mask


# ── Training / evaluation ────────────────────────────────────────────────────

def train_epoch(model, all_days, masks, optimizer, device):
    model.train()
    day_order = np.random.permutation(len(all_days))
    total_loss = 0.0

    for i in day_order:
        train_mask, _, sids = masks[i]
        snaps = [s.to(device) for s in all_days[i]]
        train_mask_d = train_mask.to(device)

        logits, labels, tr_mask, _ = model(snaps, train_mask=train_mask_d, sids_order=sids)
        loss = F.cross_entropy(logits[tr_mask], labels[tr_mask])
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(all_days)


@torch.no_grad()
def evaluate(model, all_days, masks, device):
    model.eval()
    tr_preds, tr_truths = [], []
    te_preds, te_truths = [], []
    tr_loss_sum = te_loss_sum = 0.0

    for i, day_snaps in enumerate(all_days):
        train_mask, test_mask, sids = masks[i]
        snaps = [s.to(device) for s in day_snaps]
        train_mask_d = train_mask.to(device)
        test_mask_d = test_mask.to(device)

        logits, labels, _, _ = model(snaps, sids_order=sids)

        if train_mask_d.any():
            tr_loss_sum += F.cross_entropy(logits[train_mask_d], labels[train_mask_d]).item()
            tr_preds.extend(logits[train_mask_d].argmax(1).cpu().tolist())
            tr_truths.extend(labels[train_mask_d].cpu().tolist())

        if test_mask_d.any():
            te_loss_sum += F.cross_entropy(logits[test_mask_d], labels[test_mask_d]).item()
            te_preds.extend(logits[test_mask_d].argmax(1).cpu().tolist())
            te_truths.extend(labels[test_mask_d].cpu().tolist())

    n = len(all_days)
    tr_acc = accuracy_score(tr_truths, tr_preds) if tr_truths else 0.0
    te_acc = accuracy_score(te_truths, te_preds) if te_truths else 0.0
    return tr_acc, te_acc, tr_loss_sum / n, te_loss_sum / n, te_truths, te_preds


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_curves(history: dict, out_path: str):
    epochs = history["epochs"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, history["train_loss"], label="Train loss", linewidth=1.5)
    ax1.plot(epochs, history["test_loss"], label="Test loss", linewidth=1.5, linestyle="--")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Cross-entropy loss")
    ax1.set_title("Loss"); ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(epochs, history["train_acc"], label="Train acc", linewidth=1.5)
    ax2.plot(epochs, history["test_acc"], label="Test acc", linewidth=1.5, linestyle="--")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
    ax2.set_ylim(0, 1); ax2.set_title("Accuracy")
    ax2.legend(); ax2.grid(alpha=0.3)

    best_ep = epochs[int(np.argmax(history["test_acc"]))]
    best_acc = max(history["test_acc"])
    ax2.axvline(best_ep, color="grey", linestyle=":", alpha=0.7)
    ax2.annotate(f"best={best_acc:.3f}", xy=(best_ep, best_acc),
                 xytext=(best_ep + 2, best_acc - 0.05), fontsize=8, color="grey")

    fig.suptitle("DayGNN — Node Masking Training", fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Training curves saved: {out_path}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Dynamic GNN node classification — transductive node masking")
    p.add_argument("--data_dir", type=str, default=".")
    p.add_argument("--out_dir", type=str, default="results")
    p.add_argument("--overlap", type=int, default=30)
    p.add_argument("--test_ratio", type=float, default=0.3)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--gat_hidden", type=int, default=32)
    p.add_argument("--gat_out", type=int, default=64)
    p.add_argument("--gru_hidden", type=int, default=64)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_every", type=int, default=10)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"\n{'=' * 62}")
    print(f"  Dynamic GNN | Node Masking | High School Contact Network")
    print(f"  Device: {device}  Overlap: {args.overlap}min  Test: {args.test_ratio:.0%}")
    print(f"{'=' * 62}\n")

    contacts, metadata = load_raw(args.data_dir)
    class_le = LabelEncoder()
    class_le.fit(metadata["class"].values)
    num_classes = len(class_le.classes_)
    day_infos = get_school_days(contacts)

    cache_name = f"dataset_overlap{args.overlap}_{os.path.basename(os.path.abspath(args.data_dir))}.pkl"
    cache_path = os.path.join(args.out_dir, cache_name)
    all_days = build_or_load_dataset(
        contacts, metadata, class_le, day_infos,
        window_sec=3600, overlap_min=args.overlap, cache_path=cache_path,
    )

    init_masks = sample_masks(all_days, args.test_ratio, seed=args.seed)

    model = DayGNN(
        node_in=NODE_IN, gat_hidden=args.gat_hidden, gat_out=args.gat_out,
        gru_hidden=args.gru_hidden, num_classes=num_classes,
        heads=args.heads, dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    history = dict(epochs=[], train_loss=[], test_loss=[], train_acc=[], test_acc=[])
    best_test_acc, best_epoch, best_report = 0.0, 0, ""

    weights_best = os.path.join(args.out_dir, "weights_best.pt")
    weights_final = os.path.join(args.out_dir, "weights_final.pt")
    plot_path = os.path.join(args.out_dir, "training_curves.png")

    for epoch in range(1, args.epochs + 1):
        masks = sample_masks(all_days, args.test_ratio, seed=args.seed + epoch)
        train_loss = train_epoch(model, all_days, masks, optimizer, device)
        scheduler.step()

        if epoch % args.log_every == 0 or epoch == 1 or epoch == args.epochs:
            tr_acc, te_acc, tr_loss_eval, te_loss, te_truths, te_preds = evaluate(model, all_days, masks, device)

            history["epochs"].append(epoch)
            history["train_loss"].append(tr_loss_eval)
            history["test_loss"].append(te_loss)
            history["train_acc"].append(tr_acc)
            history["test_acc"].append(te_acc)

            print(f"{epoch:>6}  {tr_loss_eval:>10.4f}  {te_loss:>9.4f}  {tr_acc:>9.4f}  {te_acc:>8.4f}")

            if te_acc > best_test_acc:
                best_test_acc, best_epoch = te_acc, epoch
                best_report = classification_report(te_truths, te_preds, target_names=class_le.classes_, zero_division=0)
                torch.save(
                    {"epoch": epoch, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
                     "test_acc": te_acc, "args": vars(args), "class_names": list(class_le.classes_)},
                    weights_best,
                )

    torch.save(
        {"epoch": args.epochs, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
         "test_acc": history["test_acc"][-1] if history["test_acc"] else 0.0,
         "args": vars(args), "class_names": list(class_le.classes_)},
        weights_final,
    )

    plot_curves(history, plot_path)

    te_f1 = f1_score(te_truths, te_preds, average="macro", zero_division=0)
    results = dict(
        best_epoch=best_epoch, best_test_acc=round(best_test_acc, 6),
        final_test_acc=round(history["test_acc"][-1] if history["test_acc"] else 0.0, 6),
        macro_f1=round(te_f1, 6), num_classes=num_classes,
        class_names=list(class_le.classes_), args=vars(args),
    )
    results_path = os.path.join(args.out_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n  Best test accuracy: {best_test_acc:.4f} (epoch {best_epoch})")
    print(f"  Macro F1: {te_f1:.4f}")
    print(f"\n{best_report}")


if __name__ == "__main__":
    main()
