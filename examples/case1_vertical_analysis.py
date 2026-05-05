# -*- coding: utf-8 -*-
"""
Case 1 – Vertical analysis: detailed decomposition for a single vertical
candidate (index CANDIDATE_IDX = 1).

Produces a three-panel figure showing:
  (a) hit/miss counts across all prior realizations
  (b) mean prior model for the representative subset + hole trace
  (c) full slice matrix (ore indicator vs. realization index)

Outputs
-------
fig_case1_vertical_analysis_drillhole_{CANDIDATE_IDX}.png
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from discretize import TensorMesh

from voi_drill import simulate_trajectory_2d
from voi_drill.metrics import _sample_slices

# Path to the input data directory
DATA_DIR = pathlib.Path(__file__).parent.parent / "inputs"
OUT_DIR  = pathlib.Path(__file__).parent.parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)

plt.rcParams['font.size'] = 18
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

np.random.seed(10000)

# ------------------------------------------------------------------
# Mesh and data
# ------------------------------------------------------------------
dh = 10
hx = [(dh, 130)]
hz = [(dh, 60)]
mesh = TensorMesh([hx, hz], "00")
EXTENT = [0, 130 * dh, 0, 60 * dh]

true_model   = np.loadtxt(DATA_DIR / "true_model_binary_2d.txt").T
prior_models = np.loadtxt(DATA_DIR / "prior_model_geometry.txt")
prior_models_permutation = np.random.permutation(prior_models.T)

n_models, n_cells = prior_models_permutation.shape

# ------------------------------------------------------------------
# Candidate drillholes – vertical pattern
# ------------------------------------------------------------------
n_holes = 10
x     = np.linspace(100, 1200, n_holes)
z     = np.full(n_holes, 599.5)
depth = np.full(n_holes, 500.0)
dip   = np.full(n_holes, 90.0)
parameters = np.column_stack([x, z, depth, dip])

x_all, z_all = simulate_trajectory_2d(parameters, n_segments=60)

CANDIDATE_IDX = 1
coords = np.c_[x_all[CANDIDATE_IDX], z_all[CANDIDATE_IDX]]

# ------------------------------------------------------------------
# Sample slices at this candidate
# ------------------------------------------------------------------
slices = _sample_slices(prior_models_permutation, coords, mesh)

# ------------------------------------------------------------------
# Unique-trace decomposition
# ------------------------------------------------------------------
unique_slices, inverse, counts = np.unique(
    slices, axis=0, return_inverse=True, return_counts=True
)
K = unique_slices.shape[0]

targets = np.linspace(0, n_models - 1, K)
rep_idx = np.empty(K, dtype=int)
for k in range(K):
    inds       = np.flatnonzero(inverse == k)
    rep_idx[k] = inds[np.argmin(np.abs(inds - targets[k]))]

rep_idx = np.unique(rep_idx)
rep_idx.sort()

# ------------------------------------------------------------------
# Mean prior model for the representative subset
# ------------------------------------------------------------------
mean_subset_2d = prior_models_permutation[rep_idx, :].mean(0).reshape(
    mesh.shape_cells[0], mesh.shape_cells[1], order='F'
)

# ------------------------------------------------------------------
# Hit / miss statistics
# ------------------------------------------------------------------
hit_mask = slices.any(axis=1)
n_hit    = int(hit_mask.sum())
n_miss   = n_models - n_hit
p_hit    = n_hit  / n_models
p_miss   = n_miss / n_models

# ------------------------------------------------------------------
# Three-panel figure
# ------------------------------------------------------------------
fig = plt.figure(figsize=(18, 3), constrained_layout=True)
gs  = GridSpec(1, 3, figure=fig, wspace=0.1, hspace=0.1)

# Panel (a) – hit / miss bar chart
axD = fig.add_subplot(gs[0, 0])
labels = ['Hit (ore found)', 'Miss (no ore)']
vals   = [n_hit, n_miss]
probs  = [p_hit, p_miss]
bars   = axD.bar(labels, vals, width=0.5, color=['#FDE725', '#00224E'])
offset = max(vals) * 0.03 if max(vals) > 0 else 1
for i, b in enumerate(bars):
    axD.text(b.get_x() + b.get_width() / 2, b.get_height() + offset,
             f"n={vals[i]}, p={probs[i]:.2f}", ha="center", va="bottom")
axD.set_ylabel("Number of realizations")
axD.set_title("(a)", loc="left")
axD.set_ylim(0, 7500)

# Panel (b) – mean subset model + hole trace
axC = fig.add_subplot(gs[0, 1])
axC.contour(true_model, levels=[0.8], colors="w", linewidths=2, extent=EXTENT)
imC = axC.imshow(
    mean_subset_2d.T, origin='lower', cmap='cividis', vmin=0, vmax=1, extent=EXTENT
)
axC.plot(x_all[CANDIDATE_IDX, :], z_all[CANDIDATE_IDX, :], '--', c="r",
         linewidth=1.5, alpha=1)
axC.set_xlabel("Easting (m)")
axC.set_ylabel("Elevation (m)")
axC.set_title("(b)", loc="left")
cbarC = fig.colorbar(imC, ax=axC, location='right', shrink=0.5)
cbarC.set_label("Mean orebody", rotation=270, labelpad=15)
cbarC.set_ticks([0, 1])

# Panel (c) – full slice matrix
axA = fig.add_subplot(gs[0, 2])
imA = axA.imshow(
    slices.T, aspect='auto', interpolation='nearest',
    vmin=0, vmax=1, extent=[0, 6130, 0, 60 * dh], cmap='cividis'
)
axA.set_xlabel("Realization index")
axA.set_ylabel(" ")
axA.set_title("(c)", loc="left")
axA.set_yticks([0, 200, 400, 600])
axA.set_xticks([0, 1200, 2400, 3600, 4800, 6000])
cbarA = fig.colorbar(imA, ax=axA, location='right', shrink=0.5)
cbarA.set_label("Orebody", rotation=270, labelpad=15)
cbarA.set_ticks([0, 1])

plt.savefig(
    OUT_DIR / f'fig_case1_vertical_analysis_drillhole_{CANDIDATE_IDX}.png',
    bbox_inches='tight'
)
plt.close(fig)
