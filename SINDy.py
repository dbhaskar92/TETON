from curses import meta
import numpy as np
from itertools import combinations
import math
from scipy.signal import savgol_filter
from utils import *
from tqdm import tqdm 
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("✓ GPU (CuPy) detected")
    _ = cp.zeros(1)
except ImportError:
    GPU_AVAILABLE = False
    cp = np
    print("⚠️ GPU not available (using NumPy)")
    
xp = cp if GPU_AVAILABLE else np
# ----- Inline ground-truth (V2 Structure) -----
GT_EDGES_INLINE = {
    frozenset((0,1)), frozenset((0,2)), frozenset((0,3)), frozenset((1,2)),
    frozenset((2,3)), frozenset((2,8)), frozenset((2,9)),
    frozenset((3,4)), frozenset((3,5)), frozenset((4,5)),
    frozenset((5,6)), frozenset((5,8)), frozenset((5,9)),
    frozenset((6,7)), frozenset((7,8)), frozenset ((6,8)), frozenset((7,9)), frozenset((8,9)),
}
GT_TRIANGLES_INLINE = {
    frozenset((0,1,2)),
    frozenset((3,4,5)), # Added for V2
    frozenset((7,8,9)), frozenset((6,7,8)), frozenset((2,8,9)),
}


def edge_F1(S2, gt_edges_set, q):
    nloc = S2.shape[0]
    vals = [S2[i, j] for i in range(nloc) for j in range(i+1, nloc) if S2[i, j] > 0]
    tau2 = float(np.quantile(vals, q))
    pred = {frozenset((i, j)) for i in range(nloc) for j in range(i+1, nloc) if S2[i, j] >= tau2}
    TP = len(pred & gt_edges_set)
    FP = len(pred - gt_edges_set)
    FN = len(gt_edges_set - pred)
    P  = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    R  = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    F1 = 2*P*R/(P+R) if (P+R) > 0 else 0.0
    out = {"q": q, "tau2": tau2, "P": P, "R": R, "F1": F1}
    return out

def triangle_F1(S3_scores, gt_tris_set, q):
    if not S3_scores:                       
        return {"q": float(q), "tau3": 0.0, "P": 0.0, "R": 0.0, "F1": 0.0,
                "TP": 0, "FP": 0, "FN": len(gt_tris_set)}
    vals = np.array(list(S3_scores.values())) if S3_scores else np.array([])
    tau3 = float(np.quantile(vals, q))
    pred = {t for t, s in S3_scores.items() if s >= tau3}
    TP = len(pred & gt_tris_set)
    FP = len(pred - gt_tris_set)
    FN = len(gt_tris_set - pred)
    P  = TP / (TP + FP) if (TP+FP)>0 else 0.0
    R  = TP / (TP + FN) if (TP+FN)>0 else 0.0
    F1 = 2*P*R/(P+R) if (P+R)>0 else 0.0
    out = {"q": float(q), "tau3": tau3, "P":P, "R":R, "F1":F1,
                "TP":TP, "FP":FP, "FN":FN}
    return out

def kkt_grad_row_norms(G, B, Xi):
    R = G @ Xi - B
    return (cp.asnumpy(cp.linalg.norm(R, axis=1)) if GPU_AVAILABLE
            else np.linalg.norm(R, axis=1))
    
def kkt_grad_row_norms_single_dim(G, Xi):
    R = G @ Xi
    return (cp.asnumpy(cp.linalg.norm(R, axis=1)) if GPU_AVAILABLE
            else np.linalg.norm(R, axis=1))
    
def degree_weights(maps, g, args=None):
    w = xp.ones(g, dtype=xp.float32)
    w[0] = 0.0   # constant row
    for (_,), r in maps.get(1, {}).items():
        w[r] *= xp.array(1.0, dtype=xp.float32)
    for (_,_), r in maps.get(2, {}).items():
        w[r] *= xp.array(args.pair_lambda_mult, dtype=xp.float32)
    return w

def build_soc_edges_from_maps(dmax, maps):
    edges = []
    if dmax >= 2:
        for (i,j), child in maps[2].items():
            edges += [(child, maps[1][(i,)]), (child, maps[1][(j,)])]
    if dmax >= 3:
        for (i,j,k), child in maps[3].items():
            for face in combinations((i,j,k), 2):
                edges.append((child, maps[2][tuple(sorted(face))]))
    if dmax >= 4:
        for (i,j,k,l), child in maps[4].items():
            for face in combinations((i,j,k,l), 3):
                edges.append((child, maps[3][tuple(sorted(face))]))
    return edges

def select_local(Xw, args):
    x0 = np.median(Xw, axis=1, keepdims=True)
    Xc = Xw - x0
    sigma = np.maximum(
        1e-8,
        1.4826 * np.median(
            np.abs(Xc - np.median(Xc, axis=1, keepdims=True)),
            axis=1, keepdims=True
        )
    )
    d_pc = np.sqrt(((Xc / sigma) ** 2).sum(axis=0)) / np.sqrt(Xc.shape[0])
    keep = np.where(d_pc <= args.r_target_pc)[0]
    if keep.size < args.k_min:
        keep = np.argsort(d_pc)[:args.k_min]
    elif keep.size > args.k_max:
        keep = np.argsort(d_pc)[:args.k_max]
    return x0, np.sort(keep), d_pc

def build_library_fast(XT, n, dmax, idx):
    T = XT.shape[0]
    parts = [xp.ones((T, 1), dtype=XT.dtype)]  # constant term
    if dmax >= 1:
        parts.append(XT)
    if dmax >= 2 and 'pairs' in idx:
        p = idx['pairs']
        parts.append(XT[:, p[:, 0]] * XT[:, p[:, 1]])
    if dmax >= 3 and 'triples' in idx:
        tr = idx['triples']
        parts.append(XT[:, tr[:,0]] * XT[:, tr[:,1]] * XT[:, tr[:,2]])
    if dmax >= 4 and 'quads' in idx:
        qd = idx['quads']
        parts.append(XT[:, qd[:,0]] * XT[:, qd[:,1]] * XT[:, qd[:,2]] * XT[:, qd[:,3]])
    return xp.concatenate(parts, axis=1)

def precompute_gram(Theta_std, Y_scaled):
    T = max(1, Theta_std.shape[0])
    G = (Theta_std.T @ Theta_std) / T
    eps_ridge = 5e-3
    G = G + eps_ridge * xp.eye(G.shape[0], dtype=G.dtype)
    B = (Theta_std.T @ Y_scaled)  / T
    return G, B

def precompute_gram_single_dim(Theta_std):
    T = max(1, Theta_std.shape[0])
    G = (Theta_std.T @ Theta_std) / T
    eps_ridge = 5e-3
    G = G + eps_ridge * xp.eye(G.shape[0], dtype=G.dtype)
    return G

def project_rows_dykstra(Z, edges, alpha, iters=25, tol=1e-7):
    Zc = Z.copy()
    if not edges:
        return Zc
    ci = xp.asarray([e[0] for e in edges], dtype=xp.int32)
    pi = xp.asarray([e[1] for e in edges], dtype=xp.int32)
    E, ncols = ci.shape[0], Zc.shape[1]
    inc_c = xp.zeros((E, ncols), dtype=Zc.dtype)
    inc_p = xp.zeros((E, ncols), dtype=Zc.dtype)
    for _ in range(iters):
        Zold = Zc.copy()
        c_tmp = Zc[ci] - inc_c
        p_tmp = Zc[pi] - inc_p
        C = xp.linalg.norm(c_tmp, axis=1)
        P = xp.linalg.norm(p_tmp, axis=1)
        feas = (C <= alpha * P + 1e-12)
        if (~feas).any():
            idx = xp.where(~feas)[0]
            Ct, Pt = C[idx], P[idx]
            cti, pti = c_tmp[idx], p_tmp[idx]
            denom = 1.0 + alpha * alpha
            t = (alpha * Ct + Pt) / denom
            sc = xp.where(Ct > 0, (alpha * t) / Ct, 0.0)
            sp = xp.where(Pt > 0, t / Pt, 0.0)
            c_tmp[idx] = cti * sc[:, None]
            p_tmp[idx] = pti * sp[:, None]
        inc_c = c_tmp - (Zc[ci] - inc_c)
        inc_p = p_tmp - (Zc[pi] - inc_p)
        Zc[ci] = c_tmp
        Zc[pi] = p_tmp
        if float(xp.linalg.norm(Zc - Zold)) < tol:
            break
    return Zc

def solve_admm_gl_soc(
    G, B, edges, lamb, rho_hier, w=None,
    max_iters=300, rho_admm=3.0, Xi0=None,
    log_every=25, adaptive_rho=True, overrelax=1.6,
    trace_every=1, return_trace=True, freeze_cholesky=False,
    early_proj_every=1, late_proj_every=1,
    early_proj_iters=15, late_proj_iters=50,
    early_proj_tol=1e-7,  late_proj_tol=1e-8,
    dtype_for_solves=None, args=None):

    g, k = G.shape[0], B.shape[1]
    if w is None:
        w = xp.ones(g, dtype=G.dtype)

    # Optional dtype casting for stability
    if dtype_for_solves is not None and dtype_for_solves != G.dtype:
        Gs = G.astype(dtype_for_solves, copy=True)
        Bs = B.astype(dtype_for_solves, copy=True)
        Is = xp.eye(g, dtype=dtype_for_solves)
    else:
        Gs, Bs = G, B
        Is = xp.eye(g, dtype=G.dtype)

    # Variables
    Xi = xp.zeros((g, k), dtype=Gs.dtype) if Xi0 is None else Xi0.astype(Gs.dtype, copy=True)
    Z  = Xi.copy()
    C  = Xi.copy()
    Uz = xp.zeros_like(Xi)
    Uc = xp.zeros_like(Xi)

    # Linear system matrix
    A = Gs + (2.0 * rho_admm) * Is
    try:
        Lc = xp.linalg.cholesky(A)
        chol = True
    except Exception:
        chol = False

    # Should we actually record a trace?
    record_trace = (trace_every is not None) and (trace_every > 0) and return_trace

    trace = None
    if record_trace:
        trace = {
            "obj": [], "ls": [], "gl": [], "r_pri": [], "r_dual": [],
            "rho": [], "lam": [], "nnz_rows": [], "nnz_params": [],
            "soc_violations_before": [], "soc_violations_after": [], "cond_A": []
        }

    def count_soc_violations(M):
        if not edges:
            return 0
        ci = xp.asarray([e[0] for e in edges], dtype=xp.int32)
        pi = xp.asarray([e[1] for e in edges], dtype=xp.int32)
        cn = xp.linalg.norm(M[ci], axis=1)
        pn = xp.linalg.norm(M[pi], axis=1)
        return int((cn > rho_hier * pn + 1e-12).sum())

    def condition_number(A_mat):
        try:
            s = xp.linalg.svd(A_mat, compute_uv=False)
            return float(s[0] / max(s[-1], 1e-16))
        except Exception:
            return float("nan")

    if record_trace:
        trace["cond_A"].append(condition_number(A))

    lam = lamb

    for it in range(max_iters):
        Z_prev = Z.copy()
        C_prev = C.copy()

        # ---- Xi-step ----
        RHS = Bs + rho_admm * (Z - Uz + C - Uc)
        if chol and not freeze_cholesky:
            Y = xp.linalg.solve(Lc, RHS)
            Xi = xp.linalg.solve(Lc.T, Y)
        else:
            Xi = xp.linalg.solve(A, RHS)

        # Over-relaxation
        Xi_hat = overrelax * Xi + (1.0 - overrelax) * Z_prev

        # ---- Z-step: group lasso (row-wise shrinkage) ----
        Wz = Xi_hat + Uz
        norms = xp.sqrt((Wz * Wz).sum(axis=1) + 1e-16)
        shrink = xp.maximum(0.0, 1.0 - (lam * w) / (rho_admm * norms))
        Z = (Wz.T * shrink).T

        # ---- C-step: hierarchical SOC projection ----
        two_thirds = (2 * max_iters) // 3
        if it < two_thirds:
            proj_every = early_proj_every
            proj_iters = early_proj_iters
            proj_tol   = early_proj_tol
        else:
            proj_every = late_proj_every
            proj_iters = late_proj_iters
            proj_tol   = late_proj_tol

        soc_before = None
        if proj_every and (it % proj_every) == 0:
            if record_trace:
                soc_before = count_soc_violations(Xi_hat + Uc)
            C = project_rows_dykstra(Xi_hat + Uc, edges, rho_hier,
                                     iters=proj_iters, tol=proj_tol)
        else:
            C = Xi_hat + Uc

        soc_after = count_soc_violations(C) if record_trace else None

        # ---- Dual updates ----
        Uz += (Xi_hat - Z)
        Uc += (Xi_hat - C)

        # ---- Residuals ----
        r_pri = float(xp.linalg.norm(Xi - Z) + xp.linalg.norm(Xi - C))
        r_dual = float(rho_admm * xp.linalg.norm((Z - Z_prev) + (C - C_prev)))

        # ---- Trace bookkeeping ----
        if record_trace and (it % trace_every) == 0:
            R = Gs @ Xi - Bs
            ls = 0.5 * float(xp.sum(R * R))
            Z_row_norms = xp.sqrt((Z * Z).sum(axis=1) + 1e-16)
            gl = float(lam * xp.sum(w * Z_row_norms))

            trace["ls"].append(ls)
            trace["gl"].append(gl)
            trace["obj"].append(ls + gl)
            trace["r_pri"].append(r_pri)
            trace["r_dual"].append(r_dual)
            trace["rho"].append(rho_admm)
            trace["lam"].append(float(lam))
            nnz_rows = int((xp.linalg.norm(Z, axis=1) > args.ROW_NORM_NNZ_THR).sum())
            nnz_params = int((xp.abs(Z) > args.PARAM_ABS_NNZ_THR).sum())
            trace["nnz_rows"].append(nnz_rows)
            trace["nnz_params"].append(nnz_params)
            trace["soc_violations_before"].append(int(soc_before) if soc_before is not None else None)
            trace["soc_violations_after"].append(int(soc_after) if soc_after is not None else None)

        # ---- Optional logging (for the "full run") ----
        if log_every and ((it % log_every) == 0 or it == max_iters - 1):
            # If we didn't just compute ls/gl, recompute quickly
            if not (record_trace and trace_every and (it % trace_every) == 0):
                R = Gs @ Xi - Bs
                ls = 0.5 * float(xp.sum(R * R))
                gl = float(lam * xp.sum(w * xp.sqrt((Z * Z).sum(axis=1) + 1e-16)))
            nnz_rows = int((xp.linalg.norm(Z, axis=1) > args.ROW_NORM_NNZ_THR).sum())
            print(
                f"    ADMM it={it:3d} | ls={ls:.3e} gl={gl:.3e} "
                f"| obj={ls+gl:.3e} | r_pri={r_pri:.2e} r_dual={r_dual:.2e} ρ={rho_admm:.2g} "
                f"| nnz_rows={nnz_rows}"
            )

        # ---- Adaptive ρ ----
        if adaptive_rho and it % 5 == 0 and it > 20:
            ratio = r_pri / max(r_dual, 1e-12)
            if ratio > 1.8:
                rho_admm *= 1.25
                Uz /= 1.25
                Uc /= 1.25
            elif ratio < 0.55:
                rho_admm /= 1.25
                Uz *= 1.25
                Uc *= 1.25
            rho_admm = float(min(max(rho_admm, 2.0), 10.0))
            A = Gs + (2.0 * rho_admm) * Is
            if not freeze_cholesky:
                try:
                    Lc = xp.linalg.cholesky(A)
                    chol = True
                except Exception:
                    chol = False
            if record_trace:
                trace["cond_A"].append(condition_number(A))

        # ---- Convergence test ----
        if r_pri < 1e-3 and r_dual < 1e-3:
            break

    if return_trace:
        return Xi, C, trace
    else:
        return Xi, C

def precompute_index_patterns(n, dmax, use_xp=True):
    out = {}
    if dmax >= 2 and n >= 2:
        iu0, iu1 = np.triu_indices(n, k=1)
        pairs = np.stack([iu0.astype(np.int32), iu1.astype(np.int32)], axis=1)
        out['pairs'] = cp.asarray(pairs) if (use_xp and GPU_AVAILABLE) else pairs
    if dmax >= 3 and n >= 3:
        triples = np.array(list(combinations(range(n), 3)), dtype=np.int32)
        out['triples'] = cp.asarray(triples) if (use_xp and GPU_AVAILABLE) else triples
    if dmax >= 4 and n >= 4:
        quads = np.array(list(combinations(range(n), 4)), dtype=np.int32)
        out['quads'] = cp.asarray(quads) if (use_xp and GPU_AVAILABLE) else quads
    return out

def build_maps_from_patterns(n, dmax, idx):
    maps = {1:{}, 2:{}, 3:{}, 4:{}}
    row = 0
    if dmax >= 1:
        for i in range(n):
            row += 1
            maps[1][(i,)] = row
    if dmax >= 2 and 'pairs' in idx:
        for a, b in (idx['pairs'].get() if GPU_AVAILABLE else idx['pairs']):
            row += 1
            maps[2][(int(a), int(b))] = row
    if dmax >= 3 and 'triples' in idx:
        for i, j, k in (idx['triples'].get() if GPU_AVAILABLE else idx['triples']):
            row += 1
            maps[3][(int(i), int(j), int(k))] = row
    if dmax >= 4 and 'quads' in idx:
        for i, j, k, l in (idx['quads'].get() if GPU_AVAILABLE else idx['quads']):
            row += 1
            maps[4][(int(i), int(j), int(k), int(l))] = row
    g_total = row + 1   # include constant row 0
    return maps, g_total

def readout_scores_multi_mode(Xi, n, dmax, maps):
    """
    Returns S2 (edges) and THREE variants of S3 (triangles):
      - S3_max:  max(w_i, w_j, w_k)  (Most lenient, best for recall)
      - S3_mean: mean(w_i, w_j, w_k) (Balanced)
      - S3_geom: geom(w_i, w_j, w_k) (Strict, requires symmetry)
    """
    A = cp.asnumpy(Xi) if GPU_AVAILABLE else Xi
    absA = np.abs(A)
    out = {}

    # ----- S2: edge scores (Unchanged, max is standard for edges) -----
    if dmax >= 1 and maps[1]:
        S2_dir = np.zeros((n, n))
        for j in range(n):
            row = maps[1][(j,)]
            for i in range(n):
                if i == j: continue
                S2_dir[i, j] = absA[row, i]
        out['S2'] = np.maximum(S2_dir, S2_dir.T)

    # ----- S3: Multi-mode triangle scores -----
    if dmax >= 2 and maps[2]:
        S3_max = {}
        S3_mean = {}
        S3_geom = {}

        pair_row = maps[2]
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    tri = frozenset((i, j, k))

                    # Get row indices for pairs (j,k), (i,k), (i,j)
                    r_jk = pair_row.get(tuple(sorted((j, k))))
                    r_ik = pair_row.get(tuple(sorted((i, k))))
                    r_ij = pair_row.get(tuple(sorted((i, j))))

                    if (r_jk is None) or (r_ik is None) or (r_ij is None):
                        continue

                    # Extract weights from the three equations
                    # w_i is the coeff of x_j*x_k in equation i
                    w_i = absA[r_jk, i]
                    w_j = absA[r_ik, j]
                    w_k = absA[r_ij, k]

                    weights = [w_i, w_j, w_k]

                    # 1. Max (Lenient)
                    s_max = max(weights)
                    if s_max > 1e-9: S3_max[tri] = s_max

                    # 2. Arithmetic Mean (Balanced)
                    s_mean = sum(weights) / 3.0
                    if s_mean > 1e-9: S3_mean[tri] = s_mean

                    # 3. Geometric Mean (Strict - Current)
                    s_geom = (w_i * w_j * w_k) ** (1.0/3.0)
                    if s_geom > 1e-9: S3_geom[tri] = s_geom

        out['S3_max'] = S3_max
        out['S3_mean'] = S3_mean
        out['S3_geom'] = S3_geom

    return out

def sweep_edge_thresholds(S2, q):
    nloc = S2.shape[0]
    vals = [S2[i, j] for i in range(nloc) for j in range(i+1, nloc) if S2[i, j] > 0]
    tau2 = float(np.quantile(vals, q))
    pred = set()
    for i in range(nloc):
        for j in range(i+1, nloc):
            if S2[i, j] >= tau2:
                pred.add(frozenset((i, j)))
    return pred

def sweep_triangle_thresholds(S3_scores, q):
    if not S3_scores:                          
        return set()
    vals = np.array(list(S3_scores.values())) if S3_scores else np.array([])
    tau3 = float(np.quantile(vals, q))
    pred = set()
    for t,s in S3_scores.items():
        if s >= tau3:
            pred.add(t)
    return pred

def SINDy_windowed(X, Y, n, idx_patterns, maps0, edges0, w_degree_base, args):
    
    X_gpu = cp.asarray(X) if GPU_AVAILABLE else X
    Y_gpu = cp.asarray(Y) if GPU_AVAILABLE else Y
    
    w_start = 0
    w_end   = args.win_len
    Xw_full = X[:, w_start:w_end]
    x0_full, keep_full, d_pc_full = select_local(Xw_full, args)
    metas = [{
        "w_start": w_start,
        "t_mid": (w_start + args.win_len / 2) * args.dt,
        "x0": x0_full.flatten(),
        "keep": keep_full
    }]
    meta = metas[0]
    w_start, w_end = meta["w_start"], meta["w_start"] + args.win_len

    if GPU_AVAILABLE:
        Xw = X_gpu[:, w_start:w_end]
        Yw = Y_gpu[:, w_start:w_end]
        x0 = cp.asarray(meta["x0"].reshape(-1, 1))
        Xw_c = Xw - x0
        Xw_use = Xw_c[:, meta["keep"]]
        Yw_use = Yw[:, meta["keep"]]
        if Xw_use.shape[1] > args.max_rows:
            idx = cp.linspace(0, Xw_use.shape[1] - 1, args.max_rows, dtype=cp.int32)
            Xw_use = Xw_use[:, idx]
            Yw_use = Yw_use[:, idx]
        XT = Xw_use.T
        YT = Yw_use.T
    else:
        Xw = X[:, w_start:w_end]
        Yw = Y[:, w_start:w_end]
        x0 = meta["x0"].reshape(-1, 1)
        Xw_c = Xw - x0
        Xw_use = Xw_c[:, meta["keep"]]
        Yw_use = Yw[:, meta["keep"]]
        if Xw_use.shape[1] > args.max_rows:
            idx = np.linspace(0, Xw_use.shape[1] - 1, args.max_rows, dtype=int)
            Xw_use = Xw_use[:, idx]
            Yw_use = Yw_use[:, idx]
        XT = Xw_use.T
        YT = Yw_use.T
    Theta = build_library_fast(XT, n, args.d_max, idx_patterns)

    Theta_mean = xp.mean(Theta, axis=0, keepdims=True)
    Theta_stdv = xp.std(Theta, axis=0, keepdims=True) + 1e-8
    Thetaz = (Theta - Theta_mean) / Theta_stdv

    Y_mean = xp.mean(YT, axis=0, keepdims=True)
    Y_stdv = xp.std(YT, axis=0, keepdims=True) + 1e-8
    Y_scaled = (YT - Y_mean) / Y_stdv

    Thetaz   = Thetaz.astype(xp.float32, copy=False)
    Y_scaled = Y_scaled.astype(xp.float32, copy=False)

    G, B = precompute_gram(Thetaz, Y_scaled)
    col_std = xp.std(Thetaz, axis=0).astype(xp.float32)
    w_rows = xp.ones(G.shape[0], dtype=xp.float32)
    bad = (col_std < 1e-6)
    if (cp.any(bad).item() if GPU_AVAILABLE else bool(np.any(bad))):
        w_rows = w_rows.copy()
        w_rows[bad] = xp.array(3.0, dtype=w_rows.dtype)
        
    # Stats needed for lambda
    T_eff = Thetaz.shape[0]
    G_eff = Thetaz.shape[1]
    
    if GPU_AVAILABLE:
        med = cp.median(Y_scaled, axis=0, keepdims=True)
        mad = cp.median(cp.abs(Y_scaled - med), axis=0, keepdims=True)
        res_std = float(1.4826 * cp.mean(mad))
    else:
        med = np.median(Y_scaled, axis=0, keepdims=True)
        mad = np.median(np.abs(Y_scaled - med), axis=0, keepdims=True)
        res_std = 1.4826 * float(np.mean(mad))

    
    lam = args.scale * (res_std / max(1, math.sqrt(T_eff))) * math.sqrt(2.0 * math.log(max(2, G_eff)))
    w = (w_degree_base.astype(xp.float32) * w_rows).astype(xp.float32)
    
    Xi_gpu, C_gpu = solve_admm_gl_soc(
        G, B, edges0, lamb=lam,
        rho_hier=args.rho_val,   # <--- DYNAMIC RHO
        w=w, max_iters=args.max_iters, rho_admm=args.admm_rho,
        Xi0=None, log_every=0, adaptive_rho=True, overrelax=args.admm_overrelax,
        trace_every=0, return_trace=False
    )

    Xi_gpu = project_rows_dykstra(C_gpu, edges0, args.rho_val, iters=200, tol=1e-9)

    A_main = cp.asnumpy(Xi_gpu) if GPU_AVAILABLE else Xi_gpu
    row_norms = np.linalg.norm(A_main, axis=1)
    w_host = cp.asnumpy(w) if GPU_AVAILABLE else w
    tau_vec = (float(lam) * w_host) / float(args.admm_rho)

    has_active_child = np.zeros(len(row_norms), dtype=bool)
    if 3 in maps0 and maps0[3]:
        for (i, j, k), r_tri in maps0[3].items():
            if row_norms[r_tri] > args.row_norm_nnz_thr:
                for (a, b) in [(i,j), (i,k), (j,k)]:
                    edge_key = tuple(sorted((a, b)))
                    r_edge = maps0[2].get(edge_key)
                    if r_edge is not None:
                        has_active_child[r_edge] = True
    if 2 in maps0 and maps0[2]:
        for (i, j), r_edge in maps0[2].items():
            if row_norms[r_edge] > args.row_norm_nnz_thr:
                has_active_child[maps0[1][(i,)]] = True
                has_active_child[maps0[1][(j,)]] = True

    row_is_lin = np.zeros_like(row_norms, dtype=bool)
    for r in maps0.get(1, {}).values():
        row_is_lin[r] = True
    row_is_pair = np.zeros_like(row_norms, dtype=bool)
    for r in maps0.get(2, {}).values():
        row_is_pair[r] = True

    grad = kkt_grad_row_norms(G, B, Xi_gpu)

    cand = np.zeros_like(row_norms, dtype=bool)
    cand[row_is_lin]  |= (row_norms[row_is_lin]  <= 0.8 * tau_vec[row_is_lin])
    cand[row_is_pair] |= (row_norms[row_is_pair] <= 0.7 * tau_vec[row_is_pair])

    # Disable pair protection for now to let strict SOC work
    pair_parent_active = np.zeros_like(row_norms, dtype=bool)

    safe = grad <= (float(lam) * w_host + 1e-9)
    mask = cand & safe & (~has_active_child) & (~pair_parent_active)

    if np.any(mask):
        if GPU_AVAILABLE:
            Xi_gpu[cp.asarray(mask), :] = 0.0
        else:
            Xi_gpu[mask, :] = 0.0
    
    A_after = cp.asnumpy(Xi_gpu) if GPU_AVAILABLE else Xi_gpu
    active_rows = np.where(np.linalg.norm(A_after, axis=1) > args.row_norm_nnz_thr)[0]
    if len(active_rows) > 0:
        if GPU_AVAILABLE:
            G_sub = G[active_rows][:, active_rows]
            B_sub = B[active_rows]
            G_sub = G_sub + 1e-5 * cp.eye(len(active_rows))
            Xi_sub = cp.linalg.solve(G_sub, B_sub)
            Xi_gpu.fill(0.0)
            Xi_gpu[active_rows] = Xi_sub
        else:
            G_sub = G[active_rows][:, active_rows]
            B_sub = B[active_rows]
            G_sub = G_sub + 1e-5 * np.eye(len(active_rows))
            Xi_sub = np.linalg.solve(G_sub, B_sub)
            Xi_gpu.fill(0.0)
            Xi_gpu[active_rows] = Xi_sub
            
    Xi_final = Xi_gpu
    scores = readout_scores_multi_mode(Xi_final, n, args.d_max, maps0)
    
    S2 = scores.get('S2', np.zeros((n, n)))
    edge_sweep = sweep_edge_thresholds(S2, args.q)
    S3_mode = scores.get(f'S3_{args.S3_mode}', {})
    tri_sweep = sweep_triangle_thresholds(S3_mode, args.q)
    
    return edge_sweep, tri_sweep
    
def SINDy_readout_predictions(data, args):
    T_raw, D = data.shape
    sol = data.reshape(T_raw, args.num_nodes, args.input_dim)
    
    # extract x, y, z -> each shape (N, T)
    x_raw = sol[:, :, 0].T
    y_raw = sol[:, :, 1].T
    z_raw = sol[:, :, 2].T
    n, T_raw = x_raw.shape
    half = (args.win_sg - 1) // 2
    
    # Smooth x, y, z with the SAME Savitzky–Golay settings
    X_smooth = savgol_filter(x_raw, args.win_sg, args.order, axis=1, mode='interp')[:, half:-half]
    Y_smooth = savgol_filter(y_raw, args.win_sg, args.order, axis=1, mode='interp')[:, half:-half]
    Z_smooth = savgol_filter(z_raw, args.win_sg, args.order, axis=1, mode='interp')[:, half:-half]  # not strictly needed, but nice to have
    
    # Total derivative of x (what you were using before)
    dXdt_raw = savgol_filter(
        x_raw, args.win_sg, args.order, deriv=1, delta=args.dt, axis=1, mode='interp'
    )[:, half:-half]

    # Per-node mean/std for X (same as before — we standardize x only)
    mu  = X_smooth.mean(axis=1, keepdims=True)
    sig = X_smooth.std(axis=1, keepdims=True) + 1e-8

    # Standardized state used for library
    X = (X_smooth - mu) / sig

    # ---- NEW: intrinsic Lorenz term and coupling residual ----
    # intrinsic: σ (y_i - x_i) evaluated on the smoothed trajectories
    dx_intrinsic = args.sigma * (Y_smooth - X_smooth)

    # target for SINDy: "coupling only"
    Y = (dXdt_raw - dx_intrinsic) / sig

    idx_patterns = precompute_index_patterns(n, args.d_max, use_xp=True)
    maps0, g_total = build_maps_from_patterns(n, args.d_max, idx_patterns)
    edges0 = build_soc_edges_from_maps(args.d_max, maps0)
    w_degree_base = degree_weights(maps0, g_total, args)
    
    edge_sweep, tri_sweep = SINDy_windowed(X, Y, n, idx_patterns, maps0, edges0, w_degree_base, args)

def SINDy_readout_predictions_single_dim(data, args):
    #chose 0, 3, 6, ... dimensions for single dim
    #new_data = data[:, ::args.input_dim]
    raw = data[:, np.ptp(data, axis=0) > 1e-12].T
    n, T_raw = raw.shape
    X_smooth  = savgol_filter(raw, args.win_sg, args.order, axis=1, mode='interp')[:, args.half:-args.half]
    dXdt_raw  = savgol_filter(raw, args.win_sg, args.order, deriv=1, delta=args.dt, axis=1, mode='interp')[:, args.half:-args.half]
    mu  = X_smooth.mean(axis=1, keepdims=True)
    sig = X_smooth.std(axis=1, keepdims=True) + 1e-8
    X = (X_smooth - mu) / sig
    Y = dXdt_raw / sig
    
    idx_patterns = precompute_index_patterns(n, args.d_max, use_xp=True)
    maps0, g_total = build_maps_from_patterns(n, args.d_max, idx_patterns)
    edges0 = build_soc_edges_from_maps(args.d_max, maps0)
    
    w_degree_base = degree_weights(maps0, g_total, args)
    #edge_sweep, tri_sweep = SINDy_windowed(X, Y, n, idx_patterns, maps0, edges0, w_degree_base, args)
    
    #divide the data into windows and get edge and triangle predictions for each window
    new_dataset = []
    for w_start in range(0, T_raw - args.win_len + 1, args.stride):
        w_end = w_start + args.win_len
        Xw = X[:, w_start:w_end]
        Yw = Y[:, w_start:w_end]
        edge_sweep, tri_sweep = SINDy_windowed(Xw, Yw, n, idx_patterns, maps0, edges0, w_degree_base, args)
        new_dataset.append(construct_topological_snapshot(Xw, [list(edge) for edge in edge_sweep], [list(tri) for tri in tri_sweep], args.feature_aggr))
    return new_dataset


def SINDy_EEG_window_analysis(sample_windows, args):
    """
    Process EEG windows from EEG_Dataset_Creation notebook and extract edge/triangle predictions.
    
    This function takes pre-extracted windows from the EEG dataset creation pipeline
    and applies SINDy analysis to discover topological structure (edges and triangles).
    
    Args:
        sample_windows: numpy array of shape (num_windows, window_length, num_channels)
                       - Pre-extracted windows from an EEG event segment
                       - Each window is (window_length, num_channels), e.g., (768, 31)
        args: argument namespace with SINDy parameters:
              - win_sg: Savitzky-Golay window size for smoothing
              - order: polynomial order for Savitzky-Golay filter
              - dt: time step between samples
              - d_max: maximum polynomial degree for library
              - half: (win_sg - 1) // 2 for edge trimming
              - Other ADMM/SINDy parameters (rho_val, admm_rho, etc.)
    
    Returns:
        results: list of dicts, one per window, each containing:
                 - 'edge_sweep': set of frozensets representing predicted edges
                 - 'tri_sweep': set of frozensets representing predicted triangles
                 - 'window_idx': index of the window
    """
    num_windows, window_length, num_channels = sample_windows.shape
    
    # Precompute patterns for the number of channels (nodes)
    n = num_channels
    idx_patterns = precompute_index_patterns(n, args.d_max, use_xp=True)
    maps0, g_total = build_maps_from_patterns(n, args.d_max, idx_patterns)
    edges0 = build_soc_edges_from_maps(args.d_max, maps0)
    w_degree_base = degree_weights(maps0, g_total, args)
    
    results = []
    
    for w_idx in range(num_windows):
        # Extract single window: (window_length, num_channels) -> transpose to (num_channels, window_length)
        window_data = sample_windows[w_idx].T  # Shape: (num_channels, window_length)
        
        # Apply Savitzky-Golay smoothing and compute derivatives
        X_smooth = savgol_filter(window_data, args.win_sg, args.order, axis=1, mode='interp')[:, args.half:-args.half]
        dXdt_raw = savgol_filter(window_data, args.win_sg, args.order, deriv=1, delta=args.dt, axis=1, mode='interp')[:, args.half:-args.half]
        
        # Skip if window becomes too small after trimming
        if X_smooth.shape[1] < 10:
            results.append({
                'edge_sweep': set(),
                'tri_sweep': set(),
                'window_idx': w_idx
            })
            continue
        
        # Standardize
        mu = X_smooth.mean(axis=1, keepdims=True)
        sig = X_smooth.std(axis=1, keepdims=True) + 1e-8
        X = (X_smooth - mu) / sig
        Y = dXdt_raw / sig
        
        # Run SINDy analysis on the window
        edge_sweep, tri_sweep = SINDy_windowed(X, Y, n, idx_patterns, maps0, edges0, w_degree_base, args)
        
        results.append({
            'edge_sweep': edge_sweep,
            'tri_sweep': tri_sweep,
            'window_idx': w_idx
        })
    
    return results


def SINDy_EEG_sample_analysis(sample_windows, args, aggregate=False):
    """
    Process an EEG sample (multiple windows from one event segment) and extract topology.
    
    This is a higher-level function that processes all windows from a sample and
    optionally aggregates the edge/triangle predictions across windows.
    
    Args:
        sample_windows: numpy array of shape (num_windows, window_length, num_channels)
        args: SINDy parameters namespace
        aggregate: if True, return aggregated edges/triangles across all windows
                   if False, return per-window predictions
    
    Returns:
        if aggregate=True:
            dict with:
            - 'edges': set of frozensets (edges appearing in at least one window)
            - 'triangles': set of frozensets (triangles appearing in at least one window)
            - 'edge_counts': dict mapping edge -> count of windows it appears in
            - 'triangle_counts': dict mapping triangle -> count of windows it appears in
        if aggregate=False:
            list of per-window results from SINDy_EEG_window_analysis
    """
    window_results = SINDy_EEG_window_analysis(sample_windows, args)
    
    if not aggregate:
        return window_results
    
    # Aggregate across windows
    all_edges = set()
    all_triangles = set()
    edge_counts = {}
    triangle_counts = {}
    
    for res in window_results:
        for edge in res['edge_sweep']:
            all_edges.add(edge)
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
        
        for tri in res['tri_sweep']:
            all_triangles.add(tri)
            triangle_counts[tri] = triangle_counts.get(tri, 0) + 1
    
    return {
        'edges': all_edges,
        'triangles': all_triangles,
        'edge_counts': edge_counts,
        'triangle_counts': triangle_counts
    }
        
    
        
 
