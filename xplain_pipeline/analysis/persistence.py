"""
Compute persistence-diagram time series for snapshot graphs and their
explanation subgraphs.

For a day with T snapshots, produces five series:

- ``dist_G_E``   (T)   — d(G_t, E_t)
- ``dist_G_seq`` (T-1) — d(G_t, G_{t+1})
- ``dist_E_seq`` (T-1) — d(E_t, E_{t+1})
- ``jacc_G_seq`` (T-1) — Jaccard(G_t, G_{t+1})
- ``jacc_E_seq`` (T-1) — Jaccard(E_t, E_{t+1})

The bundled HSNet results can be regenerated with::

    python -m xplain_pipeline.analysis.persistence \
      --snapshots data/hsnet/latest/monday1_dfs.pkl \
      --importance data/hsnet/latest/edge_importance_monday1_attention.csv \
      --output-dir data/hsnet/latest/results
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from persim import bottleneck, wasserstein
from tqdm import tqdm

from temporal_tda import (
    associate_graph_with_metric_space,
    compute_persistence_diagram,
)


# ── Edge-set helpers ─────────────────────────────────────────────────────────

def _edge_set(
    edges_df: pd.DataFrame,
    src_col: str = "src_student",
    dst_col: str = "dst_student",
) -> set:
    """Undirected edge set as frozenset pairs ``{(min, max), …}``."""
    return {
        (min(int(r[src_col]), int(r[dst_col])), max(int(r[src_col]), int(r[dst_col])))
        for _, r in edges_df.iterrows()
    }


def _build_explanation_edges(snap_imp: pd.DataFrame, tau: float) -> pd.DataFrame:
    """Filter importance rows ≥ *tau* and return as a directed edge DataFrame."""
    top = snap_imp[snap_imp["importance"] >= tau].copy()
    if top.empty:
        return pd.DataFrame(columns=["src_student", "dst_student", "edge_weight"])

    fwd = top[["src_student", "dst_student", "importance"]].rename(columns={"importance": "edge_weight"})
    rev = top[["dst_student", "src_student", "importance"]].rename(
        columns={"dst_student": "src_student", "src_student": "dst_student", "importance": "edge_weight"}
    )
    return pd.concat([fwd, rev], ignore_index=True)


# ── Single-graph persistence diagram ─────────────────────────────────────────

def _edges_to_pd(
    edges_df: pd.DataFrame,
    node_list: list,
    src_col: str,
    dst_col: str,
    weight_col: str,
    homology_dim: int,
    distance_func: str,
    k_eigen: int,
) -> np.ndarray:
    """edges → metric space → persistence diagram (``persim``-ready)."""
    edge_input = edges_df[[src_col, dst_col]].copy()
    edge_input.columns = ["source", "target"]
    if weight_col and weight_col in edges_df.columns:
        edge_input["weight"] = edges_df[weight_col].values
    else:
        edge_input["weight"] = 1.0

    dist_mat = associate_graph_with_metric_space(
        edges=edge_input,
        source_col="source",
        target_col="target",
        distance_func=distance_func,
        k_eigen=k_eigen,
        node_list=node_list,
    )

    return compute_persistence_diagram(dist_mat, homology_dims=homology_dim)


# ── Distance helpers ─────────────────────────────────────────────────────────

def _diagram_distance(dgm_a: np.ndarray, dgm_b: np.ndarray, metric: str) -> float:
    if dgm_a.shape[0] == 0 and dgm_b.shape[0] == 0:
        return 0.0
    fn = bottleneck if metric == "bottleneck" else wasserstein
    return float(fn(dgm_a, dgm_b))


def _jaccard(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 1.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union > 0 else 0.0


# ── Main analysis ────────────────────────────────────────────────────────────

def run_analysis(
    day_dfs: list,
    importance_csv: str,
    tau: float = 0.5,
    tau_pct: float | None = None,
    metric: str = "bottleneck",
    homology_dim: int = 0,
    distance_func: str = "shortest_path",
    k_eigen: int = 4,
    out_path: str | None = None,
) -> dict:
    """
    Compute five time series for one day.

    Parameters
    ----------
    day_dfs        : output of ``snapshots_to_dataframes()``
    importance_csv : path to edge_importance CSV
    tau            : fixed importance threshold
    tau_pct        : if set, overrides *tau* with a percentile per snapshot
    metric         : ``"bottleneck"`` or ``"wasserstein"``
    homology_dim   : 0 (components) or 1 (loops)
    distance_func  : ``"shortest_path"`` or ``"commute_time"``
    k_eigen        : eigenvectors for commute_time
    out_path       : save path (``.json`` → JSON, else pickle)
    """
    assert metric in ("bottleneck", "wasserstein")

    imp_all = pd.read_csv(importance_csv)
    T = len(day_dfs)

    pd_G, pd_E = [], []
    edges_G, edges_E = [], []

    print(f"Computing PDs for {T} snapshots (metric={metric}, H{homology_dim}, tau_pct={tau_pct})...")

    for entry in tqdm(day_dfs):
        snap_idx = entry["snapshot"]
        e_df = entry["edges"]
        n_df = entry["nodes"]
        node_list = n_df["student_id"].tolist()

        diag_g = _edges_to_pd(
            e_df, node_list, "src_student", "dst_student", "edge_weight",
            homology_dim, distance_func, k_eigen,
        )
        pd_G.append(diag_g)
        edges_G.append(_edge_set(e_df))

        snap_imp = imp_all[imp_all["snapshot"] == snap_idx]
        if snap_imp.empty or snap_imp["importance"].isna().all():
            pd_E.append(np.zeros((0, 2)))
            edges_E.append(set())
        else:
            tau_used = float(snap_imp["importance"].quantile(tau_pct)) if tau_pct is not None else tau
            expl_dir = _build_explanation_edges(snap_imp, tau_used)

            diag_e = _edges_to_pd(
                expl_dir, node_list, "src_student", "dst_student", "edge_weight",
                homology_dim, distance_func, k_eigen,
            )
            pd_E.append(diag_e)
            edges_E.append(_edge_set(snap_imp[snap_imp["importance"] >= tau_used]))

    print("  Computing dist_G_E...")
    dist_G_E = [_diagram_distance(pd_G[t], pd_E[t], metric) for t in range(T)]

    print("  Computing dist_G_seq and dist_E_seq...")
    dist_G_seq, dist_E_seq = [], []
    for t in range(T - 1):
        dist_G_seq.append(_diagram_distance(pd_G[t], pd_G[t + 1], metric))
        dist_E_seq.append(_diagram_distance(pd_E[t], pd_E[t + 1], metric))

    print("  Computing Jaccard similarities...")
    jacc_G_seq, jacc_E_seq = [], []
    for t in range(T - 1):
        jacc_G_seq.append(_jaccard(edges_G[t], edges_G[t + 1]))
        jacc_E_seq.append(_jaccard(edges_E[t], edges_E[t + 1]))

    snap_indices = [e["snapshot"] for e in day_dfs]

    results = dict(
        dist_G_E=dist_G_E,
        dist_G_seq=dist_G_seq,
        dist_E_seq=dist_E_seq,
        jacc_G_seq=jacc_G_seq,
        jacc_E_seq=jacc_E_seq,
        snapshots=snap_indices,
        meta=dict(
            importance_csv=importance_csv, tau=tau, tau_pct=tau_pct,
            metric=metric, homology_dim=homology_dim,
            distance_func=distance_func, T=T,
        ),
    )

    if out_path is not None:
        if out_path.endswith(".json"):
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"  Saved JSON: {out_path}")
        else:
            with open(out_path, "wb") as f:
                pickle.dump(results, f)
            print(f"  Saved pickle: {out_path}")

    _print_summary(results)
    return results


def _print_summary(results: dict) -> None:
    meta = results["meta"]
    T = meta["T"]
    print(f"\n{'=' * 52}")
    print(f"  Time series summary  (H{meta['homology_dim']}, {meta['metric']}, tau={meta['tau']}, tau_pct={meta['tau_pct']})")
    print(f"{'=' * 52}")
    print(f"  {'Series':<14} {'len':>4}  {'min':>8}  {'max':>8}  {'mean':>8}")
    print(f"  {'─' * 48}")
    for key, label in [
        ("dist_G_E", "dist G-E"),
        ("dist_G_seq", "dist G seq"),
        ("dist_E_seq", "dist E seq"),
        ("jacc_G_seq", "Jacc G seq"),
        ("jacc_E_seq", "Jacc E seq"),
    ]:
        v = results[key]
        if v:
            print(f"  {label:<14} {len(v):>4}  {min(v):>8.4f}  {max(v):>8.4f}  {sum(v) / len(v):>8.4f}")


def load_results(path: str) -> dict:
    """Load previously saved results from JSON or pickle."""
    if path.endswith(".json"):
        with open(path) as f:
            return json.load(f)
    with open(path, "rb") as f:
        return pickle.load(f)


def load_snapshot_table(path: str | Path) -> list[dict]:
    """Load the serialized HSNet snapshot table used by the viewer."""
    with Path(path).open("rb") as file:
        snapshots = pickle.load(file)
    if not isinstance(snapshots, list):
        raise TypeError("Snapshot file must contain a list of snapshot mappings.")
    required = {"snapshot", "nodes", "edges"}
    if any(not isinstance(snapshot, dict) or not required <= snapshot.keys() for snapshot in snapshots):
        raise TypeError("Each snapshot must define snapshot, nodes, and edges.")
    return snapshots


def regenerate_hsnet_results(
    snapshots_path: str | Path,
    importance_path: str | Path,
    output_dir: str | Path,
    tau: float = 0.5,
) -> None:
    """Regenerate the four bundled HSNet time-series files."""
    snapshots = load_snapshot_table(snapshots_path)
    importance_path = Path(importance_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for prefix, metric in (("b", "bottleneck"), ("w", "wasserstein")):
        for homology_dim in (0, 1):
            run_analysis(
                snapshots,
                str(importance_path),
                tau=tau,
                metric=metric,
                homology_dim=homology_dim,
                distance_func="shortest_path",
                out_path=str(output_dir / f"{prefix}{homology_dim}_timeseries_monday1.json"),
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate HSNet persistence and Jaccard time series.")
    parser.add_argument("--snapshots", type=Path, required=True, help="Path to monday1_dfs.pkl")
    parser.add_argument("--importance", type=Path, required=True, help="Path to edge-importance CSV")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for b/w × H0/H1 JSON files")
    parser.add_argument("--tau", type=float, default=0.5, help="Explanation edge-importance threshold")
    args = parser.parse_args()
    regenerate_hsnet_results(args.snapshots, args.importance, args.output_dir, args.tau)


if __name__ == "__main__":
    main()
