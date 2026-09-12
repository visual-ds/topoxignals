"""
Experimental edge-scoring for the High School Contact Network.

Two methods:

1. **GAT Attention Weights** (``--method attention``)
   Reads local message-passing coefficients from GATConv layers.

2. **Gradient surrogate** (``--method gradient``)
   Injects gates into an experimental surrogate before backpropagation.

Neither method is a causal explanation of the temporal model's predictions.
They produce edge-scoring heuristics for exploratory analysis.

Usage::

    python -m xplain_pipeline.models.explain \\
        --model_path results/weights_best.pt \\
        --dataset_cache results/dataset_overlap30_mydata.pkl \\
        --method attention
"""

import argparse
import json
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

from xplain_pipeline.models.gnn import (
    DayGNN,
    SnapshotGAT,
    load_raw,
    get_school_days,
    build_or_load_dataset,
    NODE_IN,
)


# ── Method 1: GAT Attention Weights ─────────────────────────────────────────

@torch.no_grad()
def attention_edge_importance(gat: SnapshotGAT, snap, device: torch.device) -> torch.Tensor:
    """
    Return local GAT-attention scores aligned to the input edges.

    GATConv may append self-loops internally. Its returned edge index is used
    to project each layer's coefficients back onto ``snap.edge_index`` instead
    of relying on implicit ordering.
    """
    gat.eval()
    snap = snap.to(device)
    x, ei = snap.x, snap.edge_index

    _, (attention_edges_1, alpha1) = gat.conv1(x, ei, return_attention_weights=True)
    h1 = F.elu(gat.norm1(gat.conv1(x, ei)))
    _, (attention_edges_2, alpha2) = gat.conv2(h1, ei, return_attention_weights=True)

    def align_scores(attention_edges: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        by_edge: dict[tuple[int, int], list[torch.Tensor]] = {}
        for index, (source, target) in enumerate(attention_edges.t().tolist()):
            by_edge.setdefault((source, target), []).append(alpha[index].mean())

        aligned = []
        for source, target in ei.t().tolist():
            values = by_edge.get((source, target))
            if not values:
                raise ValueError("GAT attention output does not contain an input edge.")
            aligned.append(torch.stack(values).mean())
        return torch.stack(aligned)

    return ((align_scores(attention_edges_1, alpha1) + align_scores(attention_edges_2, alpha2)) / 2.0).detach()


# ── Method 2: Gradient surrogate ────────────────────────────────────────────

def gradient_edge_importance(
    full_model: DayGNN, all_snaps: list, target_idx: int, device: torch.device
) -> torch.Tensor:
    """
    Score injected destination-node gates with gradients.

    This is not gradient × input for the original GAT edges: its values are a
    surrogate heuristic and must not be interpreted as causal attribution.
    """
    snap = all_snaps[target_idx].to(device)
    x, ei, y = snap.x, snap.edge_index, snap.y
    E = ei.shape[1]

    all_sids = sorted(set().union(*[set(s.student_ids.tolist()) for s in all_snaps]))
    sid_to_row = {sid: i for i, sid in enumerate(all_sids)}
    N = len(all_sids)
    T = len(all_snaps)
    emb_dim = full_model.gru.input_size

    with torch.no_grad():
        context_seq = torch.zeros(N, T, emb_dim, device=device)
        for i, s in enumerate(all_snaps):
            if i == target_idx:
                continue
            sd = s.to(device)
            h = full_model.gat(sd.x, sd.edge_index)
            for li, sid in enumerate(sd.student_ids.tolist()):
                row = sid_to_row.get(sid)
                if row is not None:
                    context_seq[row, i] = h[li]

    edge_weights = torch.ones(E, device=device, requires_grad=True)

    src_nodes = ei[0]
    dst_nodes = ei[1]

    h1_raw = full_model.gat.conv1(x, ei)
    h1 = F.elu(full_model.gat.norm1(h1_raw))

    N_snap = x.shape[0]
    node_weight_sum = torch.zeros(N_snap, device=device).index_add(0, dst_nodes, edge_weights)
    node_count = torch.zeros(N_snap, device=device).index_add(0, dst_nodes, torch.ones(E, device=device))
    node_weights = (node_weight_sum / node_count.clamp(min=1)).clamp(min=1e-6)

    h1_weighted = h1 * node_weights.unsqueeze(-1)
    h2 = F.elu(full_model.gat.norm2(full_model.gat.conv2(h1_weighted, ei)))

    seq = context_seq.clone()
    target_sids = all_snaps[target_idx].student_ids.tolist()
    for local_idx, sid in enumerate(target_sids):
        row = sid_to_row.get(sid)
        if row is not None:
            seq[row, target_idx] = h2[local_idx]

    _, h_n = full_model.gru(seq)
    logits = full_model.head(h_n[-1])

    snap_sids = all_snaps[target_idx].student_ids.tolist()
    vm = all_snaps[target_idx].valid_mask & (all_snaps[target_idx].y >= 0)
    rows = torch.tensor([sid_to_row[s] for s in snap_sids], device=device)
    loss = F.cross_entropy(logits[rows][vm.to(device)], y[vm.to(device)])
    loss.backward()

    return edge_weights.grad.abs().detach()


# ── Edge-removal diagnostics ────────────────────────────────────────────────

@torch.no_grad()
def compute_fidelity(
    full_model: DayGNN, all_snaps: list, target_idx: int,
    edge_mask: torch.Tensor, tau: float, device: torch.device,
) -> dict:
    snap = all_snaps[target_idx].to(device)
    vm = snap.valid_mask & (snap.y >= 0)
    if not vm.any():
        return dict(fidelity_plus=None, fidelity_minus=None, sparsity=None,
                    baseline_acc=None, n_important=0, n_edges=snap.edge_index.shape[1], tau=tau)

    E_snap = snap.edge_index.shape[1]
    edge_mask = edge_mask[:E_snap]
    important = edge_mask >= tau
    n_imp = important.sum().item()

    all_sids = sorted(set().union(*[set(s.student_ids.tolist()) for s in all_snaps]))
    sid_to_row = {sid: i for i, sid in enumerate(all_sids)}
    N = len(all_sids)
    T = len(all_snaps)
    emb_dim = full_model.gru.input_size

    context_seq = torch.zeros(N, T, emb_dim, device=device)
    for i, s in enumerate(all_snaps):
        if i == target_idx:
            continue
        sd = s.to(device)
        h = full_model.gat(sd.x, sd.edge_index)
        for li, sid in enumerate(sd.student_ids.tolist()):
            row = sid_to_row.get(sid)
            if row is not None:
                context_seq[row, i] = h[li]

    snap_sids = snap.student_ids.tolist()
    rows = torch.tensor([sid_to_row[s] for s in snap_sids], device=device)

    def acc_on(keep: torch.Tensor) -> float:
        seq = context_seq.clone()
        ei_sub = snap.edge_index[:, keep]
        h = full_model.gat(snap.x, ei_sub)
        for local_idx, sid in enumerate(snap_sids):
            row = sid_to_row.get(sid)
            if row is not None:
                seq[row, target_idx] = h[local_idx]
        _, h_n = full_model.gru(seq)
        logits = full_model.head(h_n[-1])
        preds = logits[rows][vm].argmax(1).cpu().numpy()
        truth = snap.y[vm].cpu().numpy()
        return float(accuracy_score(truth, preds))

    all_keep = torch.ones(E_snap, dtype=torch.bool, device=device)
    base_acc = acc_on(all_keep)
    fid_minus = acc_on(important) if important.any() else 0.0
    comp_acc = acc_on(~important) if (~important).any() else base_acc
    fid_plus = base_acc - comp_acc
    sparsity = 1.0 - n_imp / E_snap if E_snap > 0 else 0.0

    return dict(
        fidelity_plus=round(float(fid_plus), 6),
        fidelity_minus=round(float(fid_minus), 6),
        sparsity=round(float(sparsity), 6),
        baseline_acc=round(float(base_acc), 6),
        n_important=n_imp, n_edges=E_snap, tau=tau,
    )


# ── Per-day pipeline ────────────────────────────────────────────────────────

def explain_day(day_snaps, full_model, metadata, day_label, out_dir, args, device):
    meta = metadata.set_index("id")
    long_rows = []
    fid_recs = []

    for snap_idx, snap in enumerate(day_snaps):
        print(f"\n  Snapshot {snap_idx:>2}/{len(day_snaps) - 1}  "
              f"nodes={snap.num_nodes}  edges={snap.edge_index.shape[1]}")

        snap_d = snap.to(device)

        if args.method == "attention":
            edge_mask = attention_edge_importance(full_model.gat, snap_d, device)
        else:
            full_model.gru.train()
            full_model.gat.eval()
            full_model.head.eval()
            for p in full_model.parameters():
                p.requires_grad_(False)
            edge_mask = gradient_edge_importance(full_model, day_snaps, snap_idx, device)

        if args.tau_pct is not None:
            tau = float(torch.quantile(edge_mask, args.tau_pct))
        else:
            tau = args.tau

        full_model.gat.eval()
        full_model.head.eval()
        full_model.gru.train()
        for p in full_model.parameters():
            p.requires_grad_(False)

        fid = compute_fidelity(full_model, day_snaps, snap_idx, edge_mask, tau, device)
        fid.update(dict(day=day_label, snapshot=snap_idx, method=args.method))
        fid_recs.append(fid)

        src_l = snap.edge_index[0].cpu().numpy()
        dst_l = snap.edge_index[1].cpu().numpy()
        mask_np = edge_mask.cpu().numpy()
        sids = snap.student_ids.tolist()

        pair_imp: dict = {}
        for s, d, imp in zip(src_l, dst_l, mask_np):
            key = (min(sids[s], sids[d]), max(sids[s], sids[d]))
            pair_imp[key] = max(pair_imp.get(key, 0.0), float(imp))

        for (sid_a, sid_b), imp in pair_imp.items():
            cls_a = meta.loc[sid_a, "class"] if sid_a in meta.index else "?"
            cls_b = meta.loc[sid_b, "class"] if sid_b in meta.index else "?"
            gen_a = meta.loc[sid_a, "gender"] if sid_a in meta.index else "?"
            gen_b = meta.loc[sid_b, "gender"] if sid_b in meta.index else "?"
            long_rows.append(dict(
                snapshot=snap_idx, src_student=sid_a, dst_student=sid_b,
                src_class=cls_a, dst_class=cls_b, src_gender=gen_a, dst_gender=gen_b,
                same_class=(cls_a == cls_b and cls_a != "?"),
                importance=round(imp, 8),
            ))

    long_df = (
        pd.DataFrame(long_rows)
        .sort_values(["snapshot", "importance"], ascending=[True, False])
        .reset_index(drop=True)
    )

    wide_df = long_df.pivot_table(
        index=["src_student", "dst_student", "src_class", "dst_class", "src_gender", "dst_gender", "same_class"],
        columns="snapshot", values="importance", aggfunc="first",
    )
    wide_df.columns = [f"snapshot_{c}" for c in wide_df.columns]
    wide_df = wide_df.reset_index()

    return long_df, wide_df, fid_recs


# ── Edge-removal diagnostic plot ─────────────────────────────────────────────

def plot_fidelity(fid_m1, fid_m2, out_path: str, method: str):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    metrics = ["fidelity_minus", "fidelity_plus", "sparsity"]
    titles = ["Accuracy using selected edges", "Accuracy drop after removal", "Sparsity"]
    colors = [("steelblue", "cornflowerblue"), ("tomato", "salmon"), ("seagreen", "mediumseagreen")]

    for ax, m, title, (c1, c2) in zip(axes, metrics, titles, colors):
        r1 = [(r["snapshot"], r[m]) for r in fid_m1 if r.get(m) is not None]
        r2 = [(r["snapshot"], r[m]) for r in fid_m2 if r.get(m) is not None]
        if r1:
            xs, ys = zip(*r1)
            ax.plot(xs, ys, "o-", color=c1, lw=1.8, ms=4, label="Monday 1")
        if r2:
            xs, ys = zip(*r2)
            ax.plot(xs, ys, "s--", color=c2, lw=1.8, ms=4, label="Monday 2")
        ax.set_title(title); ax.set_xlabel("Snapshot index"); ax.set_ylabel(title)
        ax.set_ylim(-0.05, 1.05); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.suptitle(f"Edge Importance ({method})", fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, default="results/weights_best.pt")
    p.add_argument("--dataset_cache", type=str, default=None)
    p.add_argument("--data_dir", type=str, default=".")
    p.add_argument("--out_dir", type=str, default="explain_out")
    p.add_argument("--arch_from_ckpt", action="store_true")
    p.add_argument("--gat_hidden", type=int, default=32)
    p.add_argument("--gat_out", type=int, default=64)
    p.add_argument("--gru_hidden", type=int, default=64)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--method", type=str, default="attention", choices=["attention", "gradient"])
    p.add_argument("--tau", type=float, default=0.5)
    p.add_argument("--tau_pct", type=float, default=None)
    p.add_argument("--top_k", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    ckpt = torch.load(args.model_path, map_location=device)
    overlap = 30
    if args.arch_from_ckpt and "args" in ckpt:
        ca = ckpt["args"]
        args.gat_hidden = ca.get("gat_hidden", args.gat_hidden)
        args.gat_out = ca.get("gat_out", args.gat_out)
        args.gru_hidden = ca.get("gru_hidden", args.gru_hidden)
        args.heads = ca.get("heads", args.heads)
        args.dropout = ca.get("dropout", args.dropout)
        overlap = ca.get("overlap", overlap)

    class_names = ckpt.get("class_names", None)
    num_classes = len(class_names) if class_names else None

    if args.dataset_cache and os.path.exists(args.dataset_cache):
        with open(args.dataset_cache, "rb") as f:
            all_days = pickle.load(f)
        if num_classes is None:
            lbls = set()
            for day in all_days:
                for snap in day:
                    lbls.update(snap.y[snap.valid_mask].tolist())
            lbls.discard(-1)
            num_classes = len(lbls)
        contacts, metadata = load_raw(args.data_dir)
        class_le = LabelEncoder()
        class_le.fit(metadata["class"].values)
        day_infos = get_school_days(contacts)
    else:
        contacts, metadata = load_raw(args.data_dir)
        class_le = LabelEncoder()
        class_le.fit(metadata["class"].values)
        num_classes = len(class_le.classes_)
        day_infos = get_school_days(contacts)
        cache_path = os.path.join(args.out_dir, f"dataset_overlap{overlap}.pkl")
        all_days = build_or_load_dataset(
            contacts, metadata, class_le, day_infos,
            window_sec=3600, overlap_min=overlap, cache_path=cache_path,
        )

    mon_idx = [i for i, d in enumerate(day_infos) if d["weekday"] == 0]
    assert mon_idx, "No Mondays found."
    targets = [("monday1", mon_idx[0]), ("monday2", mon_idx[-1])]

    full_model = DayGNN(
        node_in=NODE_IN, gat_hidden=args.gat_hidden, gat_out=args.gat_out,
        gru_hidden=args.gru_hidden, num_classes=num_classes,
        heads=args.heads, dropout=args.dropout,
    )
    full_model.load_state_dict(ckpt["state_dict"])
    full_model.to(device)
    full_model.gat.eval()
    full_model.head.eval()
    full_model.gru.train()
    for p in full_model.parameters():
        p.requires_grad_(False)

    all_fidelity = {}
    for day_label, day_idx in targets:
        long_df, wide_df, fid_recs = explain_day(
            day_snaps=all_days[day_idx], full_model=full_model,
            metadata=metadata, day_label=day_label,
            out_dir=args.out_dir, args=args, device=device,
        )
        all_fidelity[day_label] = fid_recs

        long_path = os.path.join(args.out_dir, f"edge_importance_{day_label}_{args.method}.csv")
        wide_path = os.path.join(args.out_dir, f"edge_importance_{day_label}_{args.method}_wide.csv")
        long_df.to_csv(long_path, index=False)
        wide_df.to_csv(wide_path, index=False)

    fid_path = os.path.join(args.out_dir, f"fidelity_report_{args.method}.json")
    with open(fid_path, "w") as f:
        json.dump(all_fidelity, f, indent=2, default=str)

    plot_fidelity(
        all_fidelity["monday1"], all_fidelity["monday2"],
        os.path.join(args.out_dir, f"fidelity_plot_{args.method}.png"),
        method=args.method,
    )


if __name__ == "__main__":
    main()
