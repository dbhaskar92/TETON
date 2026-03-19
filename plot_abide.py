import pickle
import json
import numpy as np

def load_complexes(filepath):
    """Load pickle — frozensets are restored exactly."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def load_complexes_json(filepath):
    """Load JSON version — converts lists back to frozensets."""
    with open(filepath, 'r') as f:
        raw = json.load(f)
    complexes = []
    for c in raw:
        complexes.append({
            'window':    c['window'],
            't_start_s': c['t_start_s'],
            'edges':     {frozenset(e) for e in c['edges']},
            'triangles': {frozenset(t) for t in c['triangles']},
        })
    return complexes

def load_metadata(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

# ---- Load everything ----
complexes_asd  = load_complexes('/HOME/garciare/TETON/results_abide/complexes_asd.pkl')
complexes_ctrl = load_complexes('/HOME/garciare/TETON/results_abide/complexes_ctrl.pkl')
metadata       = load_metadata('/HOME/garciare/TETON/results_abide/metadata.json')

dmn_indices = metadata['dmn_indices']
dmn_names   = metadata['dmn_names']
n_nodes     = metadata['n_nodes']

print(f"Loaded ASD:     {len(complexes_asd)} windows")
print(f"Loaded Control: {len(complexes_ctrl)} windows")
print(f"Nodes: {n_nodes}")
print(f"Subjects: {metadata['subject_asd']} (ASD), "
      f"{metadata['subject_ctrl']} (Control)")
print(f"Sites:    {metadata['site_asd']} (ASD), "
      f"{metadata['site_ctrl']} (Control)")
print(f"Args used: {metadata['args']}")

# ============================================================
# Simplicial Complex Visualization on Glass Brain
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.gridspec as gridspec
from nilearn import plotting, image, datasets
from nilearn.plotting import find_parcellation_cut_coords
import nibabel as nib
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. Extract MNI coordinates for each DMN ROI
#    using the Harvard-Oxford atlas centroids
# ============================================================

def get_dmn_coords(ho_atlas, dmn_indices):
    """
    Compute MNI centroid for each DMN ROI directly from the atlas label image.
    More reliable than find_parcellation_cut_coords for bilateral atlases.
    """
    atlas_img  = ho_atlas.maps
    atlas_data = atlas_img.get_fdata()
    affine     = atlas_img.affine

    coords = []
    for idx in dmn_indices:
        if idx == 0:
            coords.append([0., 0., 0.])
            continue

        # Voxel indices where this label appears
        vox_indices = np.array(np.where(atlas_data == idx)).T  # (N, 3)

        if len(vox_indices) == 0:
            print(f"  Warning: label {idx} not found in atlas, using origin")
            coords.append([0., 0., 0.])
            continue

        # Centroid in voxel space, then apply affine to get MNI
        centroid_vox = vox_indices.mean(axis=0)
        centroid_mni = nib.affines.apply_affine(affine, centroid_vox)
        coords.append(centroid_mni.tolist())

    return np.array(coords)


ho_atlas   = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
ho_labels  = ho_atlas.labels

# Reuse the same DMN selection from earlier
DMN_KEYWORDS = [
    'Frontal Medial Cortex',
    'Precuneous Cortex',
    'Cingulate Gyrus, posterior',
    'Angular Gyrus',
    'Middle Frontal Gyrus',
    'Inferior Frontal Gyrus',
    'Parahippocampal Gyrus',
    'Temporal Pole',
    'Lateral Occipital Cortex',
]

dmn_indices = []
dmn_names   = []
for idx, label in enumerate(ho_labels):
    for kw in DMN_KEYWORDS:
        if kw.lower() in label.lower():
            dmn_indices.append(idx)
            dmn_names.append(label)
            break

dmn_coords = get_dmn_coords(ho_atlas, dmn_indices)
n_nodes    = len(dmn_coords)

# Short display labels for plots (truncate long names)
short_names = []
for name in dmn_names:
    name = name.replace('Gyrus', 'Gy').replace('Cortex', 'Cx')
    name = name.replace('division', 'div').replace('Cingulate', 'Cing')
    name = name.replace('Parahippocampal', 'ParaHipp')
    name = name.replace('Occipital', 'Occ').replace('Frontal', 'Front')
    name = name.replace('Inferior', 'IFG').replace('Superior', 'Sup')
    name = name.replace('triangularis', 'tri').replace('opercularis', 'oper')
    short_names.append(name[:30])

print(f"Loaded {n_nodes} DMN nodes with MNI coordinates")


# ============================================================
# 2. Helper: build adjacency + triangle arrays from complexes
# ============================================================

def complexes_to_adjacency(complexes, n_nodes, mode='persistence'):
    """
    Convert list of per-window complexes to an (n x n) adjacency matrix.
    mode='persistence': edge value = fraction of windows it appears in
    mode='any':         binary, 1 if edge appears in any window
    """
    adj = np.zeros((n_nodes, n_nodes))
    n_win = len(complexes)
    for c in complexes:
        for edge in c['edges']:
            i, j = sorted(edge)
            if i < n_nodes and j < n_nodes:
                adj[i, j] += 1
                adj[j, i] += 1
    if mode == 'persistence':
        adj /= n_win
    elif mode == 'any':
        adj = (adj > 0).astype(float)
    return adj


def get_triangle_persistence(complexes, n_nodes):
    """
    Returns dict: frozenset -> persistence (fraction of windows).
    """
    counts = {}
    n_win  = len(complexes)
    for c in complexes:
        for tri in c['triangles']:
            counts[tri] = counts.get(tri, 0) + 1
    return {t: v / n_win for t, v in counts.items()}


# ============================================================
# 3. Plot 1 — Persistence connectome: ASD vs Control side by side
# ============================================================

def plot_persistence_comparison(complexes_asd, complexes_ctrl,
                                 dmn_coords, short_names,
                                 threshold=0.3, save_path=None):
    """
    Side-by-side glass brain showing edge persistence for ASD and Control.
    Edge color = persistence value. Only edges above threshold shown.
    Node size = weighted degree (how many persistent edges touch each node).
    """
    adj_asd  = complexes_to_adjacency(complexes_asd,  len(dmn_coords), 'persistence')
    adj_ctrl = complexes_to_adjacency(complexes_ctrl, len(dmn_coords), 'persistence')

    deg_asd  = adj_asd.sum(axis=1)
    deg_ctrl = adj_ctrl.sum(axis=1)

    # Shared normalization across both subjects for fair comparison
    max_deg  = max(deg_asd.max(), deg_ctrl.max(), 1e-6)

    cmap_degree = plt.cm.plasma         # node color: degree
    cmap_edges  = plt.cm.viridis        # edge color: persistence
    norm_degree = Normalize(vmin=0,   vmax=max_deg)
    norm_edges  = Normalize(vmin=0.0, vmax=1.0)

    colors_asd  = [cmap_degree(norm_degree(d)) for d in deg_asd]
    colors_ctrl = [cmap_degree(norm_degree(d)) for d in deg_ctrl]

    # gridspec: 2 rows (ASD / ctrl) × 3 cols (brain | edge cbar | degree cbar)
    # The two colorbar columns are slim; the brain column is wide
    fig = plt.figure(figsize=(16, 10), facecolor='white')
    gs  = gridspec.GridSpec(
        2, 3,
        width_ratios=[22, 1, 1],   # brain | edge persistence | weighted degree
        hspace=0.40,
        wspace=0.08
    )

    for row, (adj, node_colors, title) in enumerate([
        (adj_asd,  colors_asd,  'ASD'),
        (adj_ctrl, colors_ctrl, 'Control'),
    ]):
        brain_ax  = fig.add_subplot(gs[row, 0])
        # colorbar axes are added once (row 0) and span both rows
        # — handled below after the loop

        plotting.plot_connectome(
            adjacency_matrix=adj,
            node_coords=dmn_coords,
            node_color=node_colors,
            node_size=30,
            edge_cmap='viridis',
            edge_vmin=0.0,
            edge_vmax=1.0,
            edge_threshold=threshold,
            display_mode='ortho',
            colorbar=False,          # suppress auto colorbar entirely
            title=f'{title}  —  edge persistence (threshold ≥ {threshold})',
            axes=brain_ax,
            figure=fig,
            annotate=True,
            alpha=0.85,
        )

    # Shared edge persistence colorbar — spans both brain rows
    cbar_edge_ax = fig.add_subplot(gs[:, 1])
    sm_edge = ScalarMappable(cmap=cmap_edges, norm=norm_edges)
    sm_edge.set_array([])
    cb_edge = fig.colorbar(sm_edge, cax=cbar_edge_ax)
    cb_edge.set_label('Edge persistence', fontsize=9, labelpad=6)
    cb_edge.ax.tick_params(labelsize=8)

    # Shared weighted degree colorbar — spans both brain rows
    cbar_deg_ax = fig.add_subplot(gs[:, 2])
    sm_deg = ScalarMappable(cmap=cmap_degree, norm=norm_degree)
    sm_deg.set_array([])
    cb_deg = fig.colorbar(sm_deg, cax=cbar_deg_ax)
    cb_deg.set_label('Weighted degree', fontsize=9, labelpad=6)
    cb_deg.ax.tick_params(labelsize=8)

    # Small legend clarifying what each colormap encodes
    fig.text(
        0.98, 0.02,
        'Edge color → persistence    Node color → weighted degree',
        ha='right', va='bottom', fontsize=8,
        color='dimgray', style='italic'
    )

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()
    return fig


# ============================================================
# 4. Plot 2 — Difference map: ASD persistence minus Control
# ============================================================

def plot_difference_map(complexes_asd, complexes_ctrl,
                         dmn_coords, short_names,
                         save_path=None):
    """
    Shows edges where ASD > Control (red) and Control > ASD (blue).
    Symmetric colormap centered at 0.
    """
    adj_asd  = complexes_to_adjacency(complexes_asd,  len(dmn_coords), 'persistence')
    adj_ctrl = complexes_to_adjacency(complexes_ctrl, len(dmn_coords), 'persistence')
    diff     = adj_asd - adj_ctrl

    fig, ax = plt.subplots(1, 1, figsize=(12, 5), facecolor='white')

    plotting.plot_connectome(
        adjacency_matrix=diff,
        node_coords=dmn_coords,
        node_color='dimgray',
        node_size=30,
        edge_cmap='RdBu_r',
        edge_vmin=-1.0,
        edge_vmax=1.0,
        edge_threshold=0.1,
        display_mode='ortho',
        colorbar=True,
        title='Edge Persistence Difference  (Red = more ASD,  Blue = more Control)',
        axes=ax,
        figure=fig,
        annotate=True,
        alpha=0.8,
    )

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()
    return fig


# ============================================================
# 5. Plot 3 — Triangle overlay on glass brain
#    Triangles drawn as colored patches over the connectome
# ============================================================

def get_node_positions_from_display(display, dmn_coords, n_nodes):
    """
    Instead of guessing which MNI columns nilearn projects onto each panel,
    we read back the actual (x, y) scatter positions nilearn placed the nodes at.
    This is robust to any display_mode and any nilearn version.
    """
    panel_node_positions = {}
    for panel_name, panel in display.axes.items():
        ax = panel.ax
        # Find the scatter collection nilearn used to draw nodes
        # It's the PathCollection with the most points matching n_nodes
        for coll in ax.collections:
            offsets = coll.get_offsets()
            if len(offsets) == n_nodes:
                panel_node_positions[panel_name] = np.array(offsets)
                break
    return panel_node_positions

def plot_triangles_on_brain(complexes_asd, complexes_ctrl,
                             dmn_coords, short_names,
                             min_persistence=0.4, save_path=None):
    """
    Draws the most persistent triangles as filled polygons.
    Uses matplotlib patches overlaid on the glass brain projection.
    The z-projection (axial view) is most readable for this.
    """
    tri_asd  = get_triangle_persistence(complexes_asd,  len(dmn_coords))
    tri_ctrl = get_triangle_persistence(complexes_ctrl, len(dmn_coords))

    cmap_tri = plt.cm.YlOrRd
    norm_tri = Normalize(vmin=min_persistence, vmax=1.0)
    n_nodes  = len(dmn_coords)

    def _draw_subject(tri_pers, adj, title, tri_color, save_path_sub):

        fig = plt.figure(figsize=(16, 7), facecolor='white')
        gs  = gridspec.GridSpec(1, 2, width_ratios=[20, 1], wspace=0.06)
        brain_ax = fig.add_subplot(gs[0, 0])
        cbar_ax  = fig.add_subplot(gs[0, 1])

        display = plotting.plot_connectome(
            adjacency_matrix=adj,
            node_coords=dmn_coords,
            node_color='slategray',
            node_size=35,
            edge_cmap='Greys',
            edge_vmin=0, edge_vmax=1,
            edge_threshold=0.25,
            display_mode='ortho',     # reliable 3-panel: sagittal, coronal, axial
            colorbar=False,
            title=f'{title}  —  triangles (persistence ≥ {min_persistence})',
            axes=brain_ax,
            figure=fig,
            alpha=0.55,
        )

        # Read node positions back from each panel's scatter artist
        panel_positions = get_node_positions_from_display(display, dmn_coords, n_nodes)

        persistent_tris = {t: p for t, p in tri_pers.items()
                           if p >= min_persistence}

        # Draw triangles on every panel using the actual scatter coordinates
        for panel_name, node_xy in panel_positions.items():
            ax = display.axes[panel_name].ax

            for tri, pers in sorted(persistent_tris.items(),
                                    key=lambda x: x[1]):
                nodes = sorted(tri)
                if any(i >= len(dmn_coords) for i in nodes):
                    continue

                # Use the actual positions nilearn placed the nodes at
                pts = node_xy[nodes]   # shape (3, 2)

                # Skip degenerate triangles (nodes too close together in this projection)
                v1   = pts[1] - pts[0]
                v2   = pts[2] - pts[0]
                area = abs(v1[0]*v2[1] - v1[1]*v2[0])
                if area < 5.0:
                    continue

                alpha = 0.12 + 0.50 * norm_tri(pers)
                patch = mpatches.Polygon(
                    pts, closed=True,
                    facecolor=cmap_tri(norm_tri(pers)),
                    edgecolor=tri_color,
                    linewidth=1.5,
                    alpha=alpha,
                    transform=ax.transData,
                    zorder=3
                )
                ax.add_patch(patch)

        # Colorbar
        sm = ScalarMappable(cmap=cmap_tri, norm=norm_tri)
        sm.set_array([])
        cb = fig.colorbar(sm, cax=cbar_ax)
        cb.set_label('Triangle persistence', fontsize=9, labelpad=6)
        cb.ax.tick_params(labelsize=8)

        n_shown = len(persistent_tris)
        fig.suptitle(
            f'{title}  —  {n_shown} triangles  (persistence ≥ {min_persistence})',
            fontsize=11, y=1.01
        )

        plt.tight_layout()
        if save_path_sub:
            fig.savefig(save_path_sub, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path_sub}")
        plt.show()
        return fig

    adj_asd  = complexes_to_adjacency(complexes_asd,  len(dmn_coords), 'persistence')
    adj_ctrl = complexes_to_adjacency(complexes_ctrl, len(dmn_coords), 'persistence')

    path_asd  = save_path.replace('.png', '_asd.png')  if save_path else None
    path_ctrl = save_path.replace('.png', '_ctrl.png') if save_path else None

    fig_asd  = _draw_subject(tri_asd,  adj_asd,  'ASD',     '#d62728', path_asd)
    fig_ctrl = _draw_subject(tri_ctrl, adj_ctrl, 'Control', '#2ca02c', path_ctrl)

    return fig_asd, fig_ctrl


# ============================================================
# 6. Plot 4 — Time evolution: how the complex changes per window
# ============================================================

def plot_temporal_evolution(complexes_asd, complexes_ctrl,
                             n_nodes, dmn_names, save_path=None):
    """
    For each window, plots:
    - Top row: number of edges and triangles over time
    - Bottom rows: heatmap of which edges are active per window
    """
    def build_edge_list(complexes, n):
        all_edges = sorted({e for c in complexes for e in c['edges']},
                           key=lambda e: sorted(e))
        return all_edges

    edges_union = build_edge_list(complexes_asd + complexes_ctrl, n_nodes)
    n_win = len(complexes_asd)
    n_edges_u = len(edges_union)

    # Build binary activation matrices: (n_windows x n_edges)
    def activation_matrix(complexes, edges_union):
        mat = np.zeros((len(complexes), len(edges_union)))
        edge_idx = {e: i for i, e in enumerate(edges_union)}
        for w, c in enumerate(complexes):
            for e in c['edges']:
                if e in edge_idx:
                    mat[w, edge_idx[e]] = 1
        return mat

    mat_asd  = activation_matrix(complexes_asd,  edges_union)
    mat_ctrl = activation_matrix(complexes_ctrl, edges_union)

    # Time series of edge/triangle counts
    n_edges_asd  = [len(c['edges'])     for c in complexes_asd]
    n_tris_asd   = [len(c['triangles']) for c in complexes_asd]
    n_edges_ctrl = [len(c['edges'])     for c in complexes_ctrl]
    n_tris_ctrl  = [len(c['triangles']) for c in complexes_ctrl]

    fig = plt.figure(figsize=(18, 12), facecolor='white')
    gs  = gridspec.GridSpec(3, 2, hspace=0.45, wspace=0.3,
                             height_ratios=[1.2, 2, 2])

    # --- Top: edge/triangle count over time ---
    for col, (n_e, n_t, label, color) in enumerate([
        (n_edges_asd,  n_tris_asd,  'ASD',     '#d62728'),
        (n_edges_ctrl, n_tris_ctrl, 'Control', '#1f77b4'),
    ]):
        ax = fig.add_subplot(gs[0, col])
        windows = np.arange(len(n_e))
        ax.plot(windows, n_e, 'o-', color=color,
                label='Edges', linewidth=2, markersize=5)
        ax.plot(windows, n_t, 's--', color=color, alpha=0.6,
                label='Triangles', linewidth=2, markersize=5)
        ax.set_xlabel('Window index')
        ax.set_ylabel('Count')
        ax.set_title(f'{label}  —  Complex size over time')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # --- Bottom: edge activation heatmap per window ---
    edge_labels = []
    for e in edges_union:
        i, j = sorted(e)
        ni = dmn_names[i][:18] if i < len(dmn_names) else str(i)
        nj = dmn_names[j][:18] if j < len(dmn_names) else str(j)
        edge_labels.append(f'{ni} ↔ {nj}')

    for row, (mat, label, cmap_name) in enumerate([
        (mat_asd,  'ASD',     'Reds'),
        (mat_ctrl, 'Control', 'Blues'),
    ], start=1):
        ax = fig.add_subplot(gs[row, :])
        im = ax.imshow(mat.T, aspect='auto', cmap=cmap_name,
                       interpolation='nearest', vmin=0, vmax=1)
        ax.set_xlabel('Window index', fontsize=10)
        ax.set_ylabel('Edge', fontsize=10)
        ax.set_title(f'{label}  —  Edge activation per window '
                     f'(white=absent, colored=present)', fontsize=11)
        ax.set_xticks(np.arange(mat.shape[0]))
        ax.set_xticklabels([str(i) for i in range(mat.shape[0])],
                           fontsize=7)
        ax.set_yticks(np.arange(len(edge_labels)))
        ax.set_yticklabels(edge_labels, fontsize=6)
        plt.colorbar(im, ax=ax, shrink=0.4)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()
    return fig


# ============================================================
# 7. Plot 5 — Interactive 3D HTML viewer (open in browser)
# ============================================================

def plot_interactive_3d(complexes_asd, complexes_ctrl,
                         dmn_coords, threshold=0.3,
                         save_dir='.'):
    """
    Generates interactive HTML connectomes via nilearn's view_connectome.
    Opens in browser or saves as standalone HTML files.
    """
    adj_asd  = complexes_to_adjacency(complexes_asd,  len(dmn_coords), 'persistence')
    adj_ctrl = complexes_to_adjacency(complexes_ctrl, len(dmn_coords), 'persistence')

    for adj, label in [(adj_asd, 'ASD'), (adj_ctrl, 'Control')]:
        view = plotting.view_connectome(
            adjacency_matrix=adj,
            node_coords=dmn_coords,
            edge_threshold=threshold,
            edge_cmap='hot_r',
            symmetric_cmap=False,
            linewidth=8.0,
            node_size=6.0,
            title=f'{label} DMN Simplicial Complex — Edge Persistence'
        )
        path = f'{save_dir}/dmn_complex_{label.lower()}.html'
        view.save_as_html(path)
        print(f"Saved interactive viewer: {path}")

    # Difference view
    diff = adj_asd - adj_ctrl
    view_diff = plotting.view_connectome(
        adjacency_matrix=diff,
        node_coords=dmn_coords,
        edge_threshold=0.1,
        edge_cmap='RdBu_r',
        symmetric_cmap=True,
        linewidth=8.0,
        node_size=6.0,
        title='ASD − Control  (Red = more ASD,  Blue = more Control)'
    )
    path = f'{save_dir}/dmn_complex_difference.html'
    view_diff.save_as_html(path)
    print(f"Saved interactive viewer: {path}")


def plot_triangle_persistence_comparison(complexes_asd, complexes_ctrl,
                                          dmn_coords, dmn_names,
                                          threshold=0.3, save_path=None):
    """
    Side-by-side triangle persistence comparison, analogous to
    plot_persistence_comparison() for edges.

    - One row per subject (ASD / Control)
    - Triangle fill color = persistence value (viridis)
    - Only triangles above threshold shown
    - Underlying edges shown faintly for anatomical context
    - Single shared colorbar on the right
    - Node color = triangle-weighted degree (how many persistent
      triangles each node participates in), colormap plasma
    """
    tri_asd  = get_triangle_persistence(complexes_asd,  len(dmn_coords))
    tri_ctrl = get_triangle_persistence(complexes_ctrl, len(dmn_coords))

    adj_asd  = complexes_to_adjacency(complexes_asd,  len(dmn_coords), 'persistence')
    adj_ctrl = complexes_to_adjacency(complexes_ctrl, len(dmn_coords), 'persistence')

    cmap_tri    = plt.cm.viridis
    cmap_degree = plt.cm.plasma
    norm_tri    = Normalize(vmin=threshold, vmax=1.0)

    # Triangle-weighted degree: for each node count how many persistent
    # triangles it participates in (weighted by persistence)
    def triangle_node_degree(tri_pers, n, threshold):
        deg = np.zeros(n)
        for tri, pers in tri_pers.items():
            if pers >= threshold:
                for i in sorted(tri):
                    if i < n:
                        deg[i] += pers
        return deg

    deg_asd  = triangle_node_degree(tri_asd,  len(dmn_coords), threshold)
    deg_ctrl = triangle_node_degree(tri_ctrl, len(dmn_coords), threshold)
    max_deg  = max(deg_asd.max(), deg_ctrl.max(), 1e-6)
    norm_deg = Normalize(vmin=0, vmax=max_deg)

    colors_asd  = [cmap_degree(norm_deg(d)) for d in deg_asd]
    colors_ctrl = [cmap_degree(norm_deg(d)) for d in deg_ctrl]

    # gridspec: 2 rows × 3 cols (brain | tri persistence cbar | degree cbar)
    fig = plt.figure(figsize=(16, 10), facecolor='white')
    gs  = gridspec.GridSpec(
        2, 3,
        width_ratios=[22, 1, 1],
        hspace=0.40,
        wspace=0.08
    )

    for row, (tri_pers, adj, node_colors, title) in enumerate([
        (tri_asd,  adj_asd,  colors_asd,  'ASD'),
        (tri_ctrl, adj_ctrl, colors_ctrl, 'Control'),
    ]):
        brain_ax = fig.add_subplot(gs[row, 0])

        # Base connectome — edges shown faintly for anatomical reference
        display = plotting.plot_connectome(
            adjacency_matrix=adj,
            node_coords=dmn_coords,
            node_color=node_colors,
            node_size=30,
            edge_cmap='Greys',
            edge_vmin=0, edge_vmax=1,
            edge_threshold=0.25,
            display_mode='ortho',
            colorbar=False,
            title=f'{title}  —  triangle persistence (threshold ≥ {threshold})',
            axes=brain_ax,
            figure=fig,
            alpha=0.6,
        )

        # Read node scatter positions from each panel
        panel_positions = get_node_positions_from_display(
            display, dmn_coords, len(dmn_coords)
        )

        # Filter and sort triangles so most persistent are drawn on top
        persistent_tris = {t: p for t, p in tri_pers.items()
                           if p >= threshold}

        for panel_name, node_xy in panel_positions.items():
            ax = display.axes[panel_name].ax

            for tri, pers in sorted(persistent_tris.items(),
                                    key=lambda x: x[1]):   # low persistence first
                nodes = sorted(tri)
                if any(i >= len(dmn_coords) for i in nodes):
                    continue

                pts  = node_xy[nodes]
                v1   = pts[1] - pts[0]
                v2   = pts[2] - pts[0]
                area = abs(v1[0]*v2[1] - v1[1]*v2[0])
                if area < 5.0:
                    continue

                alpha = 0.15 + 0.55 * norm_tri(pers)
                patch = mpatches.Polygon(
                    pts, closed=True,
                    facecolor=cmap_tri(norm_tri(pers)),
                    edgecolor=cmap_tri(norm_tri(pers)),
                    linewidth=1.0,
                    alpha=alpha,
                    transform=ax.transData,
                    zorder=3
                )
                ax.add_patch(patch)

    # Shared triangle persistence colorbar — spans both rows
    cbar_tri_ax = fig.add_subplot(gs[:, 1])
    sm_tri = ScalarMappable(cmap=cmap_tri, norm=norm_tri)
    sm_tri.set_array([])
    cb_tri = fig.colorbar(sm_tri, cax=cbar_tri_ax)
    cb_tri.set_label('Triangle persistence', fontsize=9, labelpad=6)
    cb_tri.ax.tick_params(labelsize=8)

    # Shared triangle-weighted degree colorbar — spans both rows
    cbar_deg_ax = fig.add_subplot(gs[:, 2])
    sm_deg = ScalarMappable(cmap=cmap_degree, norm=norm_deg)
    sm_deg.set_array([])
    cb_deg = fig.colorbar(sm_deg, cax=cbar_deg_ax)
    cb_deg.set_label('Triangle-weighted degree', fontsize=9, labelpad=6)
    cb_deg.ax.tick_params(labelsize=8)

    fig.text(
        0.98, 0.02,
        'Triangle fill → persistence    Node color → triangle-weighted degree',
        ha='right', va='bottom', fontsize=8,
        color='dimgray', style='italic'
    )

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()
    return fig

def plot_triangle_difference_map(complexes_asd, complexes_ctrl,
                                  dmn_coords, dmn_names,
                                  min_persistence=0.2,
                                  save_path=None):
    """
    Triangle persistence difference map: ASD minus Control.
    - Red triangles appear more persistently in ASD
    - Blue triangles appear more persistently in Control
    - Triangles below min_persistence in both groups are hidden
    - Underlying edges shown faintly for anatomical context
    - Single diverging colorbar (RdBu_r), shared normalization
    """
    tri_asd  = get_triangle_persistence(complexes_asd,  len(dmn_coords))
    tri_ctrl = get_triangle_persistence(complexes_ctrl, len(dmn_coords))
    adj_asd  = complexes_to_adjacency(complexes_asd,  len(dmn_coords), 'persistence')
    adj_ctrl = complexes_to_adjacency(complexes_ctrl, len(dmn_coords), 'persistence')

    # Union of all triangles that appear in either group
    all_tris = set(tri_asd.keys()) | set(tri_ctrl.keys())

    # Compute signed difference for each triangle
    tri_diff = {}
    for tri in all_tris:
        p_asd  = tri_asd.get(tri,  0.0)
        p_ctrl = tri_ctrl.get(tri, 0.0)
        # Only include if at least one group exceeds min_persistence
        if max(p_asd, p_ctrl) >= min_persistence:
            tri_diff[tri] = p_asd - p_ctrl   # positive = more ASD, negative = more ctrl

    print(f"Triangles in ASD only:     "
          f"{sum(1 for t,d in tri_diff.items() if tri_ctrl.get(t,0)==0)}")
    print(f"Triangles in Control only: "
          f"{sum(1 for t,d in tri_diff.items() if tri_asd.get(t,0)==0)}")
    print(f"Triangles in both:         "
          f"{sum(1 for t in tri_diff if tri_asd.get(t,0)>0 and tri_ctrl.get(t,0)>0)}")
    print(f"Total shown:               {len(tri_diff)}")

    # Symmetric colormap centered at 0
    max_diff = max(abs(d) for d in tri_diff.values()) if tri_diff else 1.0
    cmap_diff = plt.cm.RdBu_r
    norm_diff = Normalize(vmin=-max_diff, vmax=max_diff)

    # Average adjacency for faint background edges
    adj_mean = (adj_asd + adj_ctrl) / 2.0

    # Node color: net triangle bias per node
    # Positive = more ASD triangles, negative = more control triangles
    node_bias = np.zeros(len(dmn_coords))
    for tri, diff in tri_diff.items():
        for i in sorted(tri):
            if i < len(dmn_coords):
                node_bias[i] += diff
    max_bias   = max(abs(node_bias).max(), 1e-6)
    norm_bias  = Normalize(vmin=-max_bias, vmax=max_bias)
    node_colors = [cmap_diff(norm_bias(b)) for b in node_bias]

    # gridspec: 1 row × 3 cols (brain | diff cbar | node bias cbar)
    fig = plt.figure(figsize=(16, 6), facecolor='white')
    gs  = gridspec.GridSpec(
        1, 3,
        width_ratios=[22, 1, 1],
        wspace=0.08
    )
    brain_ax    = fig.add_subplot(gs[0, 0])
    cbar_tri_ax = fig.add_subplot(gs[0, 1])
    cbar_nod_ax = fig.add_subplot(gs[0, 2])

    display = plotting.plot_connectome(
        adjacency_matrix=adj_mean,
        node_coords=dmn_coords,
        node_color=node_colors,
        node_size=30,
        edge_cmap='Greys',
        edge_vmin=0, edge_vmax=1,
        edge_threshold=0.25,
        display_mode='ortho',
        colorbar=False,
        title='Triangle persistence difference  '
              '(Red = more ASD,  Blue = more Control)',
        axes=brain_ax,
        figure=fig,
        alpha=0.5,
    )

    panel_positions = get_node_positions_from_display(
        display, dmn_coords, len(dmn_coords)
    )

    # Draw triangles sorted by |diff| ascending so strongest on top
    for panel_name, node_xy in panel_positions.items():
        ax = display.axes[panel_name].ax

        for tri, diff in sorted(tri_diff.items(),
                                 key=lambda x: abs(x[1])):
            nodes = sorted(tri)
            if any(i >= len(dmn_coords) for i in nodes):
                continue

            pts  = node_xy[nodes]
            v1   = pts[1] - pts[0]
            v2   = pts[2] - pts[0]
            area = abs(v1[0]*v2[1] - v1[1]*v2[0])
            if area < 5.0:
                continue

            color = cmap_diff(norm_diff(diff))
            # Alpha scales with magnitude of difference, not raw persistence
            alpha = 0.12 + 0.55 * (abs(diff) / max_diff)
            patch = mpatches.Polygon(
                pts, closed=True,
                facecolor=color,
                edgecolor=color,
                linewidth=0.8,
                alpha=alpha,
                transform=ax.transData,
                zorder=3
            )
            ax.add_patch(patch)

    # Triangle difference colorbar
    sm_tri = ScalarMappable(cmap=cmap_diff, norm=norm_diff)
    sm_tri.set_array([])
    cb_tri = fig.colorbar(sm_tri, cax=cbar_tri_ax)
    cb_tri.set_label('Persistence difference\n(ASD − Control)',
                     fontsize=9, labelpad=6)
    cb_tri.ax.tick_params(labelsize=8)
    cb_tri.ax.axhline(0, color='black', linewidth=0.8, linestyle='--')

    # Node bias colorbar
    sm_nod = ScalarMappable(cmap=cmap_diff, norm=norm_bias)
    sm_nod.set_array([])
    cb_nod = fig.colorbar(sm_nod, cax=cbar_nod_ax)
    cb_nod.set_label('Node triangle bias\n(ASD − Control)',
                     fontsize=9, labelpad=6)
    cb_nod.ax.tick_params(labelsize=8)
    cb_nod.ax.axhline(0, color='black', linewidth=0.8, linestyle='--')

    fig.text(
        0.98, 0.01,
        'Triangle fill → ASD−ctrl difference    '
        'Node color → net triangle bias per region',
        ha='right', va='bottom', fontsize=8,
        color='dimgray', style='italic'
    )

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()
    return fig


# ============================================================
# 8. Run all plots
#    Assumes complexes_asd, complexes_ctrl, dmn_names
#    are already computed from the previous script
# ============================================================

print("Plotting persistence comparison...")
plot_persistence_comparison(
    complexes_asd, complexes_ctrl,
    dmn_coords, short_names,
    threshold=0.3,
    save_path='dmn_persistence_comparison.png'
)

print("Plotting difference map...")
plot_difference_map(
    complexes_asd, complexes_ctrl,
    dmn_coords, short_names,
    save_path='dmn_difference_map.png'
)

print("Plotting triangles on brain...")
plot_triangles_on_brain(
    complexes_asd, complexes_ctrl,
    dmn_coords, short_names,
    min_persistence=0.4,
    save_path='dmn_triangles.png'
)

print("Plotting temporal evolution...")
plot_temporal_evolution(
    complexes_asd, complexes_ctrl,
    n_nodes, dmn_names,
    save_path='dmn_temporal_evolution.png'
)

print("Generating interactive 3D HTML viewers...")
plot_interactive_3d(
    complexes_asd, complexes_ctrl,
    dmn_coords,
    threshold=0.3,
    save_dir='.'
)

plot_triangle_persistence_comparison(
    complexes_asd, complexes_ctrl,
    dmn_coords, dmn_names,
    threshold=0.3,
    save_path='dmn_triangle_persistence_comparison.png'
)

plot_triangle_difference_map(
    complexes_asd, complexes_ctrl,
    dmn_coords, dmn_names,
    min_persistence=0.2,
    save_path='dmn_triangle_difference.png'
)


print("\nDone. Open dmn_complex_asd.html in a browser for 3D interaction.")