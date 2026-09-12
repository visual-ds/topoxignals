from __future__ import annotations

from typing import Union, List, Tuple, Optional, Dict, Any
import numpy as np
import pandas as pd
import networkx as nx
from scipy.sparse import csgraph # type: ignore
from scipy.sparse.linalg import eigsh # type: ignore
import gudhi as gd # type: ignore
from persim import bottleneck, wasserstein # type: ignore
from tqdm import tqdm # type: ignore

import kmapper as km
from kneed import KneeLocator
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN

def associate_graph_with_metric_space(
    edges: Union[np.ndarray, List, pd.DataFrame, nx.Graph],
    source_col: Optional[Union[str, int]] = 0,
    target_col: Optional[Union[str, int]] = 1,
    distance_func: str = 'shortest_path',
    k_eigen: int = 4,
    node_list: Optional[List[int]] = None
) -> np.ndarray:
    """
    Computes a dense distance matrix representing a graph's metric space.
    
    Parameters:
        edges: Edge list as a numpy array, list, pandas DataFrame, or an already constructed nx.Graph.
        source_col: Column index or name for the source nodes. Defaults to 0.
        target_col: Column index or name for the target nodes. Defaults to 1.
        distance_func: Distance metric to use ('shortest_path' or 'commute_time').
        k_eigen: Number of eigenvalues/vectors for commute_time distance computation.
        node_list: Optional list to enforce specific nodes and matrix ordering.
        
    Returns:
        np.ndarray: A fully dense distance matrix of the metric space.
    """
    if isinstance(edges, nx.Graph):
        G = edges.copy()
        # Ensure all edges have weight and dist properties
        for u, v, data in G.edges(data=True):
            if 'weight' not in data:
                data['weight'] = 1.0
            data['dist'] = 1.0 / data['weight']
            
        if node_list is not None:
            G.add_nodes_from(node_list)
            
    else:
        if isinstance(edges, pd.DataFrame):
            df = edges.copy()
            # If columns are passed as indices but df has string columns, use iloc to get names
            s_col = df.columns[source_col] if isinstance(source_col, int) else source_col
            t_col = df.columns[target_col] if isinstance(target_col, int) else target_col
            
            if 'weight' in df.columns:
                df = df[[s_col, t_col, 'weight']]
                df.columns = ['source', 'target', 'weight']
            else:
                df = df[[s_col, t_col]]
                df.columns = ['source', 'target']
                df['weight'] = 1.0
        elif isinstance(edges, (list, np.ndarray)):
            arr = np.array(edges)
            if arr.size == 0:
                df = pd.DataFrame(columns=['source', 'target', 'weight'])
            else:
                if arr.ndim > 1 and arr.shape[1] >= 3:
                    df = pd.DataFrame(arr[:, :3], columns=['source', 'target', 'weight'])
                    df['weight'] = pd.to_numeric(df['weight'])
                else:
                    # Assume 0 and 1 index for source and target
                    df = pd.DataFrame(arr[:, :2], columns=['source', 'target'])
                    df['weight'] = 1.0
        else:
            raise TypeError("Edges must be a numpy array, list, pandas DataFrame, or nx.Graph.")
    
        if df.empty:
            n = len(node_list) if node_list is not None else 0
            return np.zeros((n, n), dtype=np.float32)
    
        agg_df = df.groupby(['source', 'target'])['weight'].sum().reset_index()
    
        G = nx.Graph()
        if node_list is not None:
            G.add_nodes_from(node_list)
            
        for _, row in agg_df.iterrows():
            # Higher strength means a shorter traversal distance.
            G.add_edge(row['source'], row['target'], weight=row['weight'], dist=1.0/row['weight'])

    nodes = list(G.nodes())
    if node_list is not None:
        nodes = node_list
    else:
        nodes.sort()
        
    n = len(nodes)
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    
    if n == 0:
        return np.zeros((0, 0), dtype=np.float32)

    dist_matrix = np.zeros((n, n), dtype=np.float32)

    if distance_func == 'shortest_path':
        all_pairs = nx.all_pairs_dijkstra_path_length(G, weight='dist')
        dist_matrix = np.full((n, n), np.inf, dtype=np.float32)
        np.fill_diagonal(dist_matrix, 0.0)
        
        for src, dist_map in all_pairs:
            i = node_to_idx[src]
            for dst, dist in dist_map.items():
                j = node_to_idx[dst]
                dist_matrix[i, j] = dist
                
        finite_max = np.nanmax(dist_matrix[dist_matrix != np.inf]) if np.any(dist_matrix != np.inf) else 1.0
        dist_matrix[dist_matrix == np.inf] = (finite_max + 1.0) * 2.0

    elif distance_func == 'commute_time':
        A = nx.to_scipy_sparse_array(G, nodelist=nodes, weight='weight', format='csr')
        # Compute Unnormalized Laplacian
        L = csgraph.laplacian(A, normed=False)
        
        # Note: L is positive semi-definite. The smallest eigenvalue is 0.
        
        try:
            k = min(k_eigen + 1, n - 1)
            if n <= k + 1:
                evals, evecs = np.linalg.eigh(L.toarray())
            else:
                evals, evecs = eigsh(L, k=k, which='SA', tol=1e-5)
            
            # Start from index 1 to skip the 0 eigenvalue
            valid_idx = evals > 1e-10
            ev = evals[valid_idx][:k_eigen]
            evc = evecs[:, valid_idx][:, :k_eigen]
            
            # Compute CT formula: sqrt( sum_i (1/lambda_i) (phi_i(u) - phi_i(v))^2 )
            # Vectorized approach:
            # P_i = evc_i / sqrt(ev_i)
            # CT(u, v) = || P[u] - P[v] ||_2
            
            P = evc / np.sqrt(ev)
            # Compute pairwise euclidean distances of rows in P
            # dist_matrix[u, v] = sqrt(sum((P[u] - P[v])^2))
            
            # broadcasting to get pairwise differences
            diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
            dist_matrix = np.sqrt(np.sum(diff**2, axis=-1)).astype(np.float32)
            
        except Exception as e:
            print(f"Warning: CT computation failed ({e}), falling back to 0 distance matrix.")
            pass
            
    else:
        raise ValueError(f"Unknown distance function: {distance_func}. Use 'shortest_path' or 'commute_time'.")

    return dist_matrix


def compute_persistence_diagram(
    dist_matrix: np.ndarray,
    homology_dims: Union[int, List[int]] = 0,
    max_edge_length: Optional[float] = None
) -> Union[np.ndarray, Dict[int, np.ndarray]]:
    """
    Computes the persistence diagram from a distance matrix.
    
    Parameters:
        dist_matrix: Dense square distance matrix representing the metric space.
        homology_dims: The homology dimension(s) to compute. Default is 0.
        max_edge_length: Maximum edge length for the Vietoris-Rips complex.
        
    Returns:
        np.ndarray or dict: The persistence diagrams. Returns a single numpy array if
        homology_dims is an integer, or a dictionary of arrays if it is a list.
    """
    n = dist_matrix.shape[0]
    
    if isinstance(homology_dims, int):
        dims = [homology_dims]
        return_dict = False
    else:
        dims = homology_dims
        return_dict = True

    if n == 0:
        empty_res = {dim: np.zeros((0, 2), dtype=float) for dim in dims}
        return empty_res if return_dict else empty_res[dims[0]]

    if max_edge_length is None:
        max_edge_length = float(np.nanmax(dist_matrix)) * 1.1

    # Initialize RipsComplex and SimplexTree
    # gudhi requires a list of lists or 1D array for distance_matrix in newer versions, 
    # but 2D array typically works. If not, fallback to list.
    rips = gd.RipsComplex(distance_matrix=dist_matrix.tolist(), max_edge_length=max_edge_length)
    max_dim_to_compute = max(dims) + 1
    simplex_tree = rips.create_simplex_tree(max_dimension=max_dim_to_compute)
    
    simplex_tree.compute_persistence()
    
    res = {}
    for dim in dims:
        intervals = simplex_tree.persistence_intervals_in_dimension(dim)
        if len(intervals) == 0:
            res[dim] = np.zeros((0, 2), dtype=float)
        else:
            # Some gudhi versions return infinite death times; replace with max distance
            arr = np.array(intervals, dtype=float)
            arr[np.isinf(arr[:, 1]), 1] = max_edge_length
            res[dim] = arr
            
    return res if return_dict else res[dims[0]]


def compute_pairwise_diagram_distances(
    diagrams_list: List[Any],
    is_empty_graph_arr: List[bool],
    persistence_distance_metric: str = 'bottleneck'
) -> np.ndarray:
    """
    Computes a full pairwise distance matrix between persistence diagrams.
    """
    num_graphs = len(diagrams_list)
    distance_matrix = np.zeros((num_graphs, num_graphs), dtype=np.float32)
    total_pairs = (num_graphs * (num_graphs - 1)) // 2
    
    print("Computing pairwise diagram distances...")
    with tqdm(total=total_pairs, desc="Calculating pairwise distances") as pbar:
        for i in range(num_graphs):
            for j in range(i + 1, num_graphs):
                if is_empty_graph_arr[i] and is_empty_graph_arr[j]:
                    d = 0.0
                elif is_empty_graph_arr[i] or is_empty_graph_arr[j]:
                    d = 0.0
                else:
                    if persistence_distance_metric == 'bottleneck':
                        d = bottleneck(diagrams_list[i], diagrams_list[j])
                    else: 
                        d = wasserstein(diagrams_list[i], diagrams_list[j])
                
                distance_matrix[i, j] = d
                distance_matrix[j, i] = d
                pbar.update(1)

    return distance_matrix


def compute_topological_distances(
    sequence_of_edges: List[Union[np.ndarray, List, pd.DataFrame, nx.Graph]],
    fixed_num_nodes: Optional[int] = None,
    metadata: Optional[Dict[int, Dict[str, Any]]] = None,
    source_col: Optional[Union[str, int]] = 0,
    target_col: Optional[Union[str, int]] = 1,
    graph_distance_func: str = 'shortest_path',
    k_eigen: int = 4,
    persistence_distance_metric: str = 'bottleneck',
    homology_dim: int = 0,
    subsequence: Optional[str] = None
) -> np.ndarray:
    """
    Computes an n x n distance matrix between the persistence diagrams of a sequence of graphs.
    
    Parameters:
        sequence_of_edges: A sequence of graph edge lists.
        fixed_num_nodes: Optional. The maximum number of nodes expected across all graphs. If None, considers just the nodes present in the edges lists or in the nx graphs.
        metadata: Optional metadata for each node, including node labels and other features.
        source_col: Column index or name for the source nodes. Defaults to 0.
        target_col: Column index or name for the target nodes. Defaults to 1.
        graph_distance_func: Metric to compute graph distances ('shortest_path' or 'commute_time').
        k_eigen: Number of eigenvectors used if commute_time is selected.
        persistence_distance_metric: Metric for topological distance ('bottleneck' or 'wasserstein').
        homology_dim: Compute distances based on this homology dimension. Defaults to 0.
        
    Returns:
        np.ndarray: A fully dense square matrix of the pairwise distances between the graphs' diagrams,
                    or a 1D array if subsequence is 'n-1' or 'n'.
    """
    
    if persistence_distance_metric not in ['bottleneck', 'wasserstein']:
        raise ValueError("persistence_distance_metric must be 'bottleneck' or 'wasserstein'")
    if subsequence not in [None, 'n-1', 'n']:
        raise ValueError("subsequence must be None, 'n-1', or 'n'")
        
    num_graphs = len(sequence_of_edges)
    
    if num_graphs == 0:
        return np.zeros((0, 0), dtype=np.float32) if subsequence is None else np.zeros((0,), dtype=np.float32)

    if metadata is not None:
        node_list = list(metadata.keys())
    elif fixed_num_nodes is not None:
        # Pre-define the node list to enforce fixed_num_nodes
        node_list = list(range(fixed_num_nodes))
    else:
        node_list = None
    
    # 1. First pass: compute diagrams for all graphs 
    diagrams_list: List[Any] = []
    is_empty_graph_arr: List[bool] = []
    
    print("Computing diagrams...")
    for i in tqdm(range(num_graphs), desc="Processing graphs"):
        edges = sequence_of_edges[i]
        
        # Check empty
        if isinstance(edges, pd.DataFrame) and edges.empty:
            diagrams_list.append(np.zeros((0, 2)))
            is_empty_graph_arr.append(True)
            continue
        elif isinstance(edges, (list, np.ndarray)) and len(edges) == 0:
            diagrams_list.append(np.zeros((0, 2)))
            is_empty_graph_arr.append(True)
            continue
            
        elif isinstance(edges, nx.Graph) and edges.number_of_edges() == 0:
            diagrams_list.append(np.zeros((0, 2)))
            is_empty_graph_arr.append(True)
            continue
            
        # Validation checks
        if isinstance(edges, pd.DataFrame):
            # Check unique nodes
            s_col = edges.columns[source_col] if isinstance(source_col, int) else source_col
            t_col = edges.columns[target_col] if isinstance(target_col, int) else target_col
            unique_nodes = pd.concat([edges[s_col], edges[t_col]]).unique()
        elif isinstance(edges, nx.Graph):
            unique_nodes = list(edges.nodes())
        elif isinstance(edges, np.ndarray):
            unique_nodes = np.unique(edges[:, :2])
        else: # list
            arr = np.array(edges)[:, :2]
            unique_nodes = np.unique(arr)
            
        if fixed_num_nodes is not None:
            if len(unique_nodes) > fixed_num_nodes:
                raise ValueError(f"Graph at index {i} has {len(unique_nodes)} nodes, which exceeds fixed_num_nodes ({fixed_num_nodes}).")
            
            # Actually checking if any node id is >= fixed_num_nodes
            # Because if we use list(range(fixed_num_nodes)), maximum ID can be fixed_num_nodes - 1
            max_node_id = len(unique_nodes) if len(unique_nodes) > 0 else -1
            if max_node_id >= fixed_num_nodes:
                raise ValueError(f"Graph at index {i} has node ID {max_node_id}, which exceeds max allowed ID ({fixed_num_nodes - 1}).")

        # Compute Metric Space
        dist_mat = associate_graph_with_metric_space(
            edges, 
            source_col=source_col, 
            target_col=target_col, 
            distance_func=graph_distance_func, 
            k_eigen=k_eigen,
            node_list=node_list
        )
        
        # Compute Diagram
        diag = compute_persistence_diagram(dist_mat, homology_dims=homology_dim)
        diagrams_list.append(diag)
        is_empty_graph_arr.append(False)
        
        
    if subsequence is None:
        return compute_pairwise_diagram_distances(diagrams_list, is_empty_graph_arr, persistence_distance_metric)
    
    # Subsequence distances
    n = num_graphs
    distances = np.zeros(n - 1, dtype=np.float32)
    
    print(f"Computing subsequence diagram distances (type: {subsequence})...")
    for i in tqdm(range(n - 1), desc="Calculating sequential distances"):
        if is_empty_graph_arr[i] or is_empty_graph_arr[i + 1]:
            d = 0.0
        else:
            if persistence_distance_metric == 'bottleneck':
                d = bottleneck(diagrams_list[i], diagrams_list[i + 1])
            else:
                d = wasserstein(diagrams_list[i], diagrams_list[i + 1])
        distances[i] = d

    if subsequence == 'n-1':
        return distances
    else: # 'n'
        averaged_distances = np.zeros(n, dtype=np.float32)
        if n == 1:
            return averaged_distances  # single graph = 0 dist
            
        # compute average of (i-1, i) and (i, i+1) for n timestamps
        averaged_distances[0] = distances[0]
        for i in range(1, n - 1):
            if distances[i - 1]==0 or distances[i]==0:
                averaged_distances[i] = distances[i - 1] + distances[i]
            else:
                averaged_distances[i] = (distances[i - 1] + distances[i]) / 2.0
        averaged_distances[-1] = distances[-1]
        
        return averaged_distances

def get_hiperparameters(betti_numbers,finer=1.,k=2):
    N = len(betti_numbers)
    X_data = np.array(betti_numbers).reshape(-1,1)

    dynamic_n_cubes = int(np.ceil(1+np.log2(N)) * finer)

    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(X_data)
    distances,_= nn.kneighbors(X_data)
    sorted_dist = np.sort(distances[:,1])
    kneedle = KneeLocator(range(len(sorted_dist)),sorted_dist,curve='convex',direction='increasing')
    dynamic_eps = sorted_dist[kneedle.elbow] if kneedle.elbow else 1.

    return X_data,dynamic_n_cubes,dynamic_eps


def get_mapper_bettin1D(betti_numbers,finer=1.,k=2,return_nx=False):
    X_data,dynamic_n_cubes,dynamic_eps = get_hiperparameters(betti_numbers,finer=finer,k=k)
    mapper = km.KeplerMapper(verbose=0)
    cover = km.Cover(n_cubes=dynamic_n_cubes, perc_overlap=0.3) # Overlap stays static
    clusterer = DBSCAN(eps=dynamic_eps, min_samples=2)

    graph = mapper.map(
        lens=X_data,
        X=X_data,
        clusterer=clusterer,
        cover=cover
    )
    # mapper.visualize(graph, path_html="mapper_1d_floats.html")
    if return_nx:
        g = nx.Graph()
        g.add_nodes_from([(k,{'sub_elements':v})for k,v in graph['nodes'].items()])
        g.add_edges_from([(u,v) for u in graph['links'] for v in graph['links'][u]])
        return g
    return graph
