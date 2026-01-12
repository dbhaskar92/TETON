import torch
import numpy as np
from scipy.sparse import coo_matrix

def sort_edges(edge_list):
    return sorted([sorted(edge) for edge in edge_list])

def build_complete_topological_data(node_features, edge_list, triangle_list, agg_func='mean'):
    """
    Constructs topological matrices AND lifts node features to edges/triangles.
    
    Args:
        node_features (torch.Tensor): Shape (num_nodes, feature_dim)
        edge_list (list of tuples): [(u, v), ...].
        triangle_list (list of tuples): [(u, v, w), ...].
        agg_func (str): 'mean', 'sum', or 'max'. How to combine node features.
        
    Returns:
        snapshot (dict): Dictionary containing:
            - 'features': {rank_0: x0, rank_1: x1, rank_2: x2}
            - 'incidences': {rank_1: B1, rank_2: B2}
            - 'adjacencies': {rank_0: A0, rank_1: A1, rank_2: A2}
    """


    node_features = torch.tensor(node_features)
    num_nodes = node_features.shape[0]
    
    # --- 1. Map Simplices to Indices ---
    # Sort tuples to ensure consistency (0,1) is same as (1,0)
    edge_list = [tuple(sorted(e)) for e in edge_list]
    edge_count = torch.unique(torch.tensor(edge_list), dim=0).shape[0]
    print(f"Number of unique edges: {edge_count}")
    print(f"Total number of edges provided: {len(edge_list)}")
    triangle_list = [tuple(sorted(t)) for t in triangle_list]
    
    edge_to_idx = {e: i for i, e in enumerate(edge_list)}
    
    # --- 2. Feature Lifting (Nodes -> Edges/Triangles) ---
    # We use PyTorch indexing for speed instead of loops
    
    # Convert lists to LongTensors for indexing
    # Shape: (num_edges, 2)
    if len(edge_list) > 0:
        edge_indices = torch.tensor(edge_list, dtype=torch.long)
        # Gather features: (num_edges, 2, feats)
        edge_feat_stack = node_features[edge_indices] 
    else:
        edge_feat_stack = torch.zeros((0, 2, node_features.shape[1]))

    # Shape: (num_triangles, 3)
    if len(triangle_list) > 0:
        tri_indices = torch.tensor(triangle_list, dtype=torch.long)
        # Gather features: (num_tris, 3, feats)
        tri_feat_stack = node_features[tri_indices]
    else:
        tri_feat_stack = torch.zeros((0, 3, node_features.shape[1]))

    # Apply Aggregation
    if agg_func == 'mean':
        x_1 = torch.mean(edge_feat_stack, dim=1)
        x_2 = torch.mean(tri_feat_stack, dim=1)
    elif agg_func == 'sum':
        x_1 = torch.sum(edge_feat_stack, dim=1)
        x_2 = torch.sum(tri_feat_stack, dim=1)
    elif agg_func == 'max':
        x_1, _ = torch.max(edge_feat_stack, dim=1)
        x_2, _ = torch.max(tri_feat_stack, dim=1)
    else:
        raise ValueError("agg_func must be 'mean', 'sum', or 'max'")

    # Pack features
    features = {
        0: node_features, # (N0, C)
        1: x_1,           # (N1, C)
        2: x_2            # (N2, C)
    }
    # --- 3. Build Matrices ---
    # Incidence 1: Nodes -> Edges
    rows_1, cols_1, data_1 = [], [], []
    for e_idx, (u, v) in enumerate(edge_list):
        rows_1.extend([u, v])
        cols_1.extend([e_idx, e_idx])
        data_1.extend([-1.0, 1.0])
    
    # Handle empty edge lists for safety
    if len(edge_list) > 0:
        B1 = coo_matrix((data_1, (rows_1, cols_1)), shape=(num_nodes, len(edge_list)))
    else:
        B1 = coo_matrix(([], ([], [])), shape=(num_nodes, 0))

    # Incidence 2: Edges -> Triangles
    rows_2, cols_2, data_2 = [], [], []
    for t_idx, (u, v, w) in enumerate(triangle_list):
        e1, e2, e3 = tuple(sorted((u, v))), tuple(sorted((v, w))), tuple(sorted((u, w)))
        if e1 in edge_to_idx and e2 in edge_to_idx and e3 in edge_to_idx:
            rows_2.extend([edge_to_idx[e1], edge_to_idx[e2], edge_to_idx[e3]])
            cols_2.extend([t_idx, t_idx, t_idx])
            data_2.extend([1.0, 1.0, -1.0])

    if len(triangle_list) > 0:
        B2 = coo_matrix((data_2, (rows_2, cols_2)), shape=(len(edge_list), len(triangle_list)))
    else:
        B2 = coo_matrix(([], ([], [])), shape=(len(edge_list), 0))

    # Adjacencies
    
    A0 = B1.T.dot(B1)
    A1 = B1.T.dot(B1) + B2.dot(B2.T)
    A2 = B2.T.dot(B2)

    def to_sparse_tensor(mat):
        if mat.nnz == 0:
            # Handle empty matrix case
            return torch.sparse_coo_tensor(
                torch.empty((2, 0), dtype=torch.long), 
                torch.empty(0, dtype=torch.float), 
                mat.shape
            )
        mat = mat.tocoo()
        indices = torch.LongTensor(np.vstack((mat.row, mat.col)))
        values = torch.FloatTensor(mat.data)
        return torch.sparse_coo_tensor(indices, values, mat.shape)

    incidences = {'rank_1': to_sparse_tensor(B1), 'rank_2': to_sparse_tensor(B2)}
    adjacencies = {'rank_0': to_sparse_tensor(A0), 'rank_1': to_sparse_tensor(A1), 'rank_2': to_sparse_tensor(A2)}
    
    return features, incidences, adjacencies


import torch

def construct_topological_snapshot(
    node_features,
    edge_list,
    triangle_list,
    agg_func="mean",
):
    """
    Constructs topological matrices and lifts node features to edges/triangles.
    """
    device = node_features.device
    N, F = node_features.shape
    E = len(edge_list)
    
    edge_list = sort_edges(edge_list)
    

    #T = len(triangle_list)
    #print(f"Number of nodes: {N}")
    #print(f"Number of edges: {E}")
    #print(f"NUmber of triangles: {T}")
    edge_to_idx = {tuple(sorted(e)): i for i, e in enumerate(edge_list)}
    
    # ------------------
    # Aggregation helper
    # ------------------
    def aggregate(x, dim=0):
        if agg_func == "mean":
            return x.mean(dim)
        elif agg_func == "sum":
            return x.sum(dim)
        elif agg_func == "max":
            return x.max(dim).values
        else:
            raise ValueError("agg_func must be 'mean', 'sum', or 'max'")

    # ------------------
    # Lift features
    # ------------------
    node_features = torch.tensor(node_features)
    x0 = node_features                               # (N, F)

    x1 = torch.stack([
        aggregate(node_features[list(e)], dim=0)
        for e in edge_list
    ], dim=0)                                        # (E, F)

    # ------------------
    # Incidence B1 (nodes → edges)
    # ------------------
    rows, cols, vals = [], [], []
    for e_idx, (u, v) in enumerate(edge_list):
        rows.append(u)
        cols.append(e_idx)
        vals.append(1.0)
        rows.append(v)
        cols.append(e_idx)
        vals.append(1.0)

    B1 = torch.sparse_coo_tensor(
        torch.tensor([rows, cols], device=device),
        torch.tensor(vals, device=device),
        size=(N, E)
    )

    # ------------------
    # Incidence B2 (edges → triangles)
    # ------------------

    rows, cols, vals = [], [], []
    t_idx = 0
    new_triangle_list = []
    for (u, v, w) in triangle_list:
        e1, e2, e3 = tuple(sorted((u, v))), tuple(sorted((v, w))), tuple(sorted((u, w)))
        if e1 in edge_to_idx and e2 in edge_to_idx and e3 in edge_to_idx:
            e_idx_uv = edge_to_idx[e1]
            e_idx_vw = edge_to_idx[e2]
            e_idx_uw = edge_to_idx[e3]
            rows.append(e_idx_uv)
            cols.append(t_idx)
            vals.append(1.0)
            rows.append(e_idx_vw)
            cols.append(t_idx)
            vals.append(1.0)
            rows.append(e_idx_uw)
            cols.append(t_idx)
            vals.append(1.0)
            t_idx = t_idx + 1
            new_triangle_list.append((u, v, w))
            

    x2 = torch.stack([
        aggregate(node_features[list(t)], dim=0)
        for t in new_triangle_list
    ], dim=0)                                        # (T, F)

    T = len(new_triangle_list)
    B2 = torch.sparse_coo_tensor(
        torch.tensor([rows, cols], device=device),
        torch.tensor(vals, device=device),
        size=(E, T)
    )

    # ------------------
    # Adjacencies
    # ------------------
    A0 = torch.sparse.mm(B1, B1.transpose(0, 1)).coalesce()
    A1 = (
        torch.sparse.mm(B1.transpose(0, 1), B1)
        + torch.sparse.mm(B2, B2.transpose(0, 1))
    ).coalesce()
    A2 = torch.sparse.mm(B2.transpose(0, 1), B2).coalesce()


    features = {0: x0, 1: x1, 2: x2}
    incidences = {'rank_1': B1, 'rank_2': B2}
    adjacencies = {'rank_0': A0, 'rank_1': A1, 'rank_2': A2}
    
    return features, incidences, adjacencies


            