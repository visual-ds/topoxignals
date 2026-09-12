import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d

from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling
from torch_geometric.nn import GINConv
from torch_geometric.explain import Explainer, PGExplainer
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, f1_score
from torch.optim import lr_scheduler
import torch.nn as nn

class GINNodeClassifier(torch.nn.Module):
    """
    A Graph Isomorphism Network (GIN) for extracting node embeddings.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            in_dims = in_channels if i == 0 else hidden_channels
            nn_seq = Sequential(
                Linear(in_dims, hidden_channels),
                BatchNorm1d(hidden_channels),
                ReLU(),
                Linear(hidden_channels, hidden_channels),
                BatchNorm1d(hidden_channels),
                ReLU()
            )
            self.convs.append(GINConv(nn_seq))
        
        # We output node embeddings of size out_channels (which can be same as hidden_channels)
        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)
        x = self.lin(x)
        return x  # Return node embeddings

class LinkPredictor(torch.nn.Module):
    """
    Predicts links given two node embeddings.
    """
    def __init__(self, in_channels, hidden_channels, out_channels=1):
        super().__init__()
        # We concatenate two node embeddings, so input is in_channels * 2
        self.lin1 = Linear(in_channels * 2, hidden_channels)
        self.lin2 = Linear(hidden_channels, out_channels)

    def forward(self, z, edge_index):
        # z: node embeddings (N, F)
        # edge_index: (2, E)
        row, col = edge_index
        z_out = torch.cat([z[row], z[col]], dim=-1)
        z_out = F.relu(self.lin1(z_out))
        return self.lin2(z_out).squeeze(-1) # Output raw logits (E,)

class GNNLinkPredictionModel(torch.nn.Module):
    def __init__(self, node_encoder, link_predictor):
        super().__init__()
        self.node_encoder = node_encoder
        self.link_predictor = link_predictor

    def forward(self, x, edge_index, edge_label_index=None):
        """
        x: Node features
        edge_index: Message passing edges
        edge_label_index: Edges to predict (defaults to message passing edges if None)
        """
        z = self.node_encoder(x, edge_index)
        
        if edge_label_index is None:
            edge_label_index = edge_index
            
        logits = self.link_predictor(z, edge_label_index)
        return logits

def prepare_data(graphs, labels, num_classes, test_ratio=0.2, time_attr='t'):
    """
    Converts NetworkX graphs to PyTorch Geometric Data objects for Link Prediction.
    Node features are generated as one-hot encodings of the node labels.
    Performs an in-snapshot temporal split:
    - 80% (oldest) edges are used for message passing (embedding nodes).
    - 20% (newest) edges are used as prediction targets, mixed with negative samples.
    """
    dataset = []
    
    # Generate node features directly from labels (one-hot)
    y_tensor = torch.tensor(labels, dtype=torch.int64)
    x = F.one_hot(y_tensor, num_classes=num_classes).to(torch.float)

    with tqdm(total=len(graphs), desc=f"Preparing data") as pbar:
        for G in graphs:
            # Sort edges temporally if the time attribute exists
            try:
                edges = sorted(G.edges(data=True), key=lambda e: float(e[2].get(time_attr, 0)) if (len(e)>2 and time_attr in e[2]) else 0)
                edges = [(u, v) for u, v, d in edges]
            except Exception:
                edges = list(G.edges())
                
            num_edges = len(edges)
            # Default to predicting the last 20% of edges in the snapshot
            num_test = max(1, int(num_edges * test_ratio)) if num_edges > 1 else 0
            num_mp = num_edges - num_test
            
            mp_edges = edges[:num_mp]
            test_edges = edges[num_mp:]
            
            def get_edge_index(edge_list, is_undir):
                if not edge_list:
                    return torch.empty((2, 0), dtype=torch.long)
                ei = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
                if is_undir:
                    ei = torch.cat([ei, ei[[1, 0]]], dim=1)
                return ei
                
            num_nodes = G.number_of_nodes()
            is_undir = not G.is_directed()
            
            # Message Passing Edges (80%)
            mp_ei = get_edge_index(mp_edges, is_undir)
            
            # Target Edges (20%) + Negative Samples
            pos_labels = get_edge_index(test_edges, is_undir)
            # We sample negatives equal to the number of test positive edges
            neg_samples = negative_sampling(
                get_edge_index(edges, is_undir), # Exclude ALL true edges from negative samples
                num_nodes=num_nodes, 
                num_neg_samples=pos_labels.size(1)
            ) if pos_labels.size(1) > 0 else torch.empty((2,0), dtype=torch.long)
            
            label_index = torch.cat([pos_labels, neg_samples], dim=1)
            labels_tensor = torch.cat([torch.ones(pos_labels.size(1)), torch.zeros(neg_samples.size(1))], dim=0)

            data = Data(x=x, edge_index=mp_ei, edge_label_index=label_index, edge_label=labels_tensor)
            dataset.append(data)

            pbar.update(1)
        
    return dataset

def evaluate_gnn(model, dataset):
    """
    Evaluate the GNN Link Prediction model and compute AUC, AP, Accuracy, Precision, Recall, and F1.
    Predictions are made over the edge_label_index provided by RandomLinkSplit.
    """
    model.eval()
    all_preds_prob = []
    all_preds_class = []
    all_labels = []
    
    with torch.no_grad():
        for data in dataset:
            # For evaluation, data is the specific test/val split data object
            logits = model(data.x, data.edge_index, data.edge_label_index)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds_class = (probs > 0.5).astype(int)
            labels = data.edge_label.cpu().numpy()
            
            all_preds_prob.extend(probs)
            all_preds_class.extend(preds_class)
            all_labels.extend(labels)
            
    if len(all_labels) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            
    auc = roc_auc_score(all_labels, all_preds_prob)
    ap = average_precision_score(all_labels, all_preds_prob)
    acc = accuracy_score(all_labels, all_preds_class)
    prec = precision_score(all_labels, all_preds_class, zero_division=0)
    rec = recall_score(all_labels, all_preds_class, zero_division=0)
    f1 = f1_score(all_labels, all_preds_class, zero_division=0)
    
    return auc, ap, acc, prec, rec, f1

def train_gnn(train_dataset, num_features, epochs=100, lr=0.01, hidden_dim=32, embed_dim=32):
    """
    Initialize and train the GNN model for Link Prediction over the sequence of train graphs.
    """
    node_encoder = GINNodeClassifier(in_channels=num_features, 
                                     hidden_channels=hidden_dim, 
                                     out_channels=embed_dim)
    link_predictor = LinkPredictor(in_channels=embed_dim, 
                                   hidden_channels=hidden_dim, 
                                   out_channels=1)
    
    model = GNNLinkPredictionModel(node_encoder, link_predictor)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    criterion = nn.BCEWithLogitsLoss()
    losses = []
    
    with tqdm(total=epochs, desc=f"training") as pbar:
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for train_data in train_dataset:
                optimizer.zero_grad()
                
                # Forward pass: predict on edge_label_index using edge_index for message passing
                logits = model(train_data.x, train_data.edge_index, train_data.edge_label_index)
                
                # Compute loss
                loss = criterion(logits, train_data.edge_label.float())
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()        
                
            avg_loss = total_loss/len(train_dataset) if len(train_dataset) > 0 else 0
            pbar.set_postfix({"loss": avg_loss})
            pbar.update(1)
            scheduler.step()
            losses.append(avg_loss)
            
    return model, losses

def map_pyg_mask_to_nx(nx_graph, pyg_edge_index, pyg_edge_mask):
    """ 
    Maps PyG bidirectional edge importance masks back to the original 
    NetworkX edge list by taking the max over bidirectional importance scores.
    """
    nx_edges = list(nx_graph.edges())
    mask_dict = {}

    for i in range(pyg_edge_index.size(1)):
        u = int(pyg_edge_index[0, i])
        v = int(pyg_edge_index[1, i])
        edge = tuple(sorted((u, v))) if not nx_graph.is_directed() else (u, v)
        
        if edge not in mask_dict:
            mask_dict[edge] = []
        mask_dict[edge].append(float(pyg_edge_mask[i]))

    nx_edge_mask = []
    for u, v in nx_edges:
        edge = tuple(sorted((u, v))) if not nx_graph.is_directed() else (u, v)
        if edge in mask_dict:
            nx_edge_mask.append(np.max(mask_dict[edge]))
        else:
            nx_edge_mask.append(0.0)

    return np.array(nx_edge_mask)

def get_aggregated_edge_masks(graphs, dataset, model, explain_epochs=30):
    """
    Runs the PGExplainer to generate a global message-passing edge mask per graph.
    PGExplainer learns a global explainer model, thus we must train it first 
    over all graphs/edges before fetching explanations.
    """
    model.eval()
    
    explainer = Explainer(
        model=model.node_encoder, # Explain the node classification part
        algorithm=PGExplainer(epochs=explain_epochs, lr=0.003),
        explanation_type='phenomenon',
        node_mask_type=None,
        edge_mask_type='object',
        model_config=dict(
            mode='multiclass_classification', # Treat node encoder outputs as targets
            task_level='node',
            return_type='raw',
        ),
    )

    # 1. Prepare full graphs data objects since PGExplainer needs them for training
    full_data_list = []
    with tqdm(total=len(dataset), desc="Preparing data to explain") as pbar:
        for g_idx, explain_data in enumerate(dataset):
                
            full_edges = list(graphs[g_idx].edges())
            if not full_edges:
                continue
                
            edge_index = torch.tensor(full_edges, dtype=torch.long).t().contiguous()
            if not graphs[g_idx].is_directed():
                edge_index = torch.cat([edge_index, edge_index[[1, 0]]], dim=1)
                
            # edge_index of explain_data is already 80% if we use the same Data object.
            # But we want to explain the full graph or the 80% message passing graph.
            # We'll use the existing 80% message passing edge_index.
            with torch.no_grad():
                node_embeddings = model.node_encoder(explain_data.x, explain_data.edge_index)
                # Use argmax of node embeddings as dummy target classes
                explain_data.target = node_embeddings.argmax(dim=-1).long()
            full_data_list.append(explain_data)
            pbar.update(1)

    # 2. Train PGExplainer on all graphs (using edge_label_index edges as prediction targets)
    if not full_data_list:
        return [np.array([]) for _ in graphs]
        
    print(f"Training PGExplainer for {explain_epochs} epochs...")
    for epoch in tqdm(range(explain_epochs), desc="Training Explainer"):
        for explain_data in full_data_list:
            # PGExplainer trains by accumulating gradients over individual nodes
            num_nodes_to_explain = explain_data.x.size(0)
            for node_idx in range(num_nodes_to_explain):
                explainer.algorithm.train(
                    epoch=epoch,
                    model=model.node_encoder,
                    x=explain_data.x,
                    edge_index=explain_data.edge_index,
                    target=explain_data.target,
                    index=node_idx,
                )

    all_graph_masks = []
    
    # 3. Get explanations now that PGExplainer is trained
    with tqdm(total=len(full_data_list), desc="Explaining") as pbar:
        # We mapped graphs idx directly to full_data_list except for empty graphs
        data_ptr = 0
        for g_idx in range(len(graphs)):
            full_edges = list(graphs[g_idx].edges())
            if not full_edges:
                all_graph_masks.append(np.array([]))
                continue
                
            explain_data = full_data_list[data_ptr]
            data_ptr += 1
            num_edges = explain_data.edge_index.size(1)
            edge_masks_for_graph = []

            # Explain each existing node and aggregate edge masks
            for node_idx in range(explain_data.x.size(0)):
                explanation = explainer(
                    x=explain_data.x, 
                    edge_index=explain_data.edge_index, 
                    target=explain_data.target,
                    index=int(node_idx),
                )
                edge_masks_for_graph.append(explanation.edge_mask.detach().cpu())

            # Aggregate masks across all explained edges by taking the max
            if len(edge_masks_for_graph) > 0:
                max_pyg_mask = torch.stack(edge_masks_for_graph).max(dim=0).values
            else:
                max_pyg_mask = torch.zeros(explain_data.edge_index.size(1))

            # Map back to NetworkX strict edge ordering
            nx_mask = map_pyg_mask_to_nx(graphs[g_idx], explain_data.edge_index, max_pyg_mask)
            all_graph_masks.append(nx_mask)
            pbar.update(1)

    return all_graph_masks

def pipeline(graphs, labels, epochs=100, explain_epochs=50, train_ratio=0.8):
    """
    Main pipeline function for Link Prediction:
    1. Prepares temporal sequence data
    2. Splits graphs into Train and Validation randomly (e.g. 80/20) at the graph sequence level
    3. Trains a link-prediction GNN model
    4. Evaluates classification metrics
    5. Runs edge-level explainer
    6. Aggregates masks globally for each graph.
    """
    num_classes = len(set(labels))
    print(f"Preparing dataset with {len(graphs)} graphs and {num_classes} classes for Link Prediction...")
    # dataset is a list of Data objects for each snapshot
    dataset = prepare_data(graphs, labels, num_classes=num_classes)
    
    # Shuffle and split the dataset at the graph snapshot level (e.g., random 80% graphs used for training)
    num_graphs = len(dataset)
    indices = np.random.permutation(num_graphs)
    split_idx = int(num_graphs * train_ratio)
    
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    # Extract the train and val datasets based on indices
    train_dataset = [dataset[i] for i in train_indices]
    val_dataset = [dataset[i] for i in val_indices]
    
    print(f"Split completed: {len(train_dataset)} Train Snapshots, {len(val_dataset)} Validation/Test Snapshots.")
    
    num_features = train_dataset[0].x.size(1) # should match num_classes
    
    print("\nTraining GNN model for Link Prediction...")
    model, _ = train_gnn(train_dataset, num_features=num_features, epochs=epochs)
    
    print("\nFinal Model Evaluation on Test Set...")
    val_auc, val_ap, val_acc, val_prec, val_rec, val_f1 = evaluate_gnn(model, val_dataset)
    print(f"Final Test Metrics -> AUC: {val_auc:.4f} | AP: {val_ap:.4f} | Accuracy: {val_acc:.4f} | Precision: {val_prec:.4f} | Recall: {val_rec:.4f} | F1: {val_f1:.4f}")
    
    print("\nRunning GNNExplainer on the entire dataset (explaining 80% message passing edges)...")
    # Provide the full dataset list so it can reconstruct the explanations
    masks = get_aggregated_edge_masks(graphs, dataset, model, 
                                      explain_epochs=explain_epochs)
    
    print("Pipeline complete!")
    return masks

if __name__ == "__main__":
    # --- Dummy Example Snippet ---
    print("Running Dummy Example...")
    num_nodes = 50
    
    # 10 mock graphs with moving edges but identical nodes
    graphs = [nx.erdos_renyi_graph(num_nodes, 0.1) for _ in range(10)]
    for G in graphs:
        for u, v, d in G.edges(data=True):
            d['t'] = np.random.rand()  # mock time attribute
    
    # Node labels 0 or 1
    labels = np.random.randint(0, 2, num_nodes).tolist()
    
    # Run the full pipeline
    resulting_masks = pipeline(graphs, labels, epochs=50, explain_epochs=10, train_ratio=0.8)
    
    print(f"\nGenerated {len(resulting_masks)} masks for {len(graphs)} graphs.")
    for i, mask in enumerate(resulting_masks[:3]): # only print first 3
        print(f"Graph {i+1} Output Mask shape: {mask.shape}, matches nx edges: {len(graphs[i].edges())}")
