"""Basic tests for the temporal_tda core pipeline."""

from temporal_tda import (
    associate_graph_with_metric_space,
    compute_persistence_diagram,
    compute_topological_distances,
)
import numpy as np
import pandas as pd


def test_pipeline():
    edges_g1 = [[0, 1], [1, 2], [2, 0]]          # Triangle
    edges_g2 = [[0, 1], [1, 2], [2, 3], [3, 0]]  # Square
    edges_g3 = pd.DataFrame({"s": [0, 1], "t": [1, 2]})  # Line

    print("Testing CT Metric Space...")
    d_mat = associate_graph_with_metric_space(edges_g1, distance_func="commute_time")
    print("G1 CT Distance matrix shape:", d_mat.shape)

    print("Testing SP Metric Space...")
    d_mat_sp = associate_graph_with_metric_space(edges_g2, distance_func="shortest_path")
    print("G2 SP Distance matrix shape:", d_mat_sp.shape)

    print("Testing Persistence Diagram...")
    diag = compute_persistence_diagram(d_mat_sp, homology_dims=[0, 1])
    print("G2 diagrams H0 shape:", diag[0].shape)

    seq = [edges_g1, edges_g2, edges_g3, []]  # include an empty graph
    print("Testing distances...")
    dist = compute_topological_distances(seq, fixed_num_nodes=5, persistence_distance_metric="bottleneck")

    print("Final pairwise distance matrix:")
    print(dist)


if __name__ == "__main__":
    test_pipeline()
