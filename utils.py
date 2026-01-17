import torch
import numpy as np
from scipy.sparse import coo_matrix

def sort_edges(edge_list):
    return sorted([sorted(edge) for edge in edge_list])

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