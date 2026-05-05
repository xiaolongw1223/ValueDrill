# -*- coding: utf-8 -*-
"""
Case 1 – Vertical pattern: 10 vertical drillholes (90° dip, 500 m depth)
distributed across the model domain at uniform easting spacing.

Outputs
-------
fig_case1_vertical_model_drillhole_trace.png
fig_case1_vertical_voi_thickness.png
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt

from discretize import TensorMesh

from voi_drill import (
    simulate_trajectory_2d,
    value_function_action,
    utility_linear,
    VOI,
    grade_thickness,
)

# Path to the input data directory
DATA_DIR = pathlib.Path(__file__).parent.parent / "inputs"
OUT_DIR  = pathlib.Path(__file__).parent.parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)

plt.rcParams['font.size'] = 18
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

# ------------------------------------------------------------------
# Mesh and data
# ------------------------------------------------------------------
dh = 10
hx = [(dh, 130)]
hz = [(dh, 60)]
mesh = TensorMesh([hx, hz], "00")
EXTENT = [0, 130 * dh, 0, 60 * dh]

drillhole    = np.loadtxt(DATA_DIR / "drillhole_2d.txt")
true_model   = np.loadtxt(DATA_DIR / "true_model_binary_2d.txt").T
prior_models = np.loadtxt(DATA_DIR / "prior_model_geometry.txt").T
mean_2d      = np.mean(prior_models, 0).reshape(
    mesh.shape_cells[0], mesh.shape_cells[1], order='F'
)

# ------------------------------------------------------------------
# Candidate drillholes – vertical pattern
# ------------------------------------------------------------------
n_holes = 10
x     = np.linspace(100, 1200, n_holes)
z     = np.full(n_holes, 599.5)
depth = np.full(n_holes, 500.0)
dip   = np.full(n_holes, 90.0)
parameters = np.column_stack([x, z, depth, dip])

x_all, z_all = simulate_trajectory_2d(parameters, n_segments=50)
trajectories = [np.c_[x_all[i], z_all[i]] for i in range(n_holes)]

# ------------------------------------------------------------------
# VOI – risk-neutral
# ------------------------------------------------------------------
voi_calc = VOI(
    value_fn=value_function_action,
    value_kwargs={'hit_value': 500, 'miss_value': -100},
    utility_fn=utility_linear,
    utility_kwargs={'coefficient': 1.0, 'cutoff': 0.0},
    no_action_value=0.0,
)
results = voi_calc.compute_for_candidates(trajectories, prior_models, mesh)

# ------------------------------------------------------------------
# Grade-thickness
# ------------------------------------------------------------------
gt_scores = grade_thickness(trajectories, prior_models, mesh, segment_length=10.0)

# ------------------------------------------------------------------
# Figure (a) – model + drillhole traces
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))
ax.contour(true_model, levels=[0.8], colors="w", linewidths=2, extent=EXTENT)
im = ax.imshow(mean_2d.T, origin='lower', cmap='cividis', vmin=0, vmax=1, extent=EXTENT)
ax.imshow(drillhole.T, origin='lower', cmap='cool', extent=EXTENT)
for i in range(n_holes):
    ax.plot(x_all[i, :], z_all[i, :], '--', c="r", linewidth=1.5, alpha=1)
ax.set_xlabel('Easting (m)')
ax.set_ylabel('Elevation (m)')
ax.set_title("(a)", loc="left")
ax.set_yticks([0, 200, 400, 600])
ax.set_xticks([0, 500, 1000])
cbar = fig.colorbar(im, ax=ax, location='right', shrink=0.4)
cbar.set_ticks([0, 1])
cbar.set_label('Mean orebody', rotation=270, labelpad=20)
plt.savefig(OUT_DIR / 'fig_case1_vertical_model_drillhole_trace.png', bbox_inches='tight')
plt.close(fig)

# ------------------------------------------------------------------
# Figure (b/c) – grade-thickness and VOI profiles
# ------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(
    nrows=2, ncols=1, sharex=False, figsize=(8, 4),
    gridspec_kw={"hspace": 0.3}
)

ax1.plot(x, gt_scores, marker='s', color='black', linestyle='--',
         markersize=6, linewidth=2)
ax1.set_ylabel("Thickness (m)")
ax1.set_title("(b)", loc="left")
ax1.set_xticklabels([])
ax1.set_xlabel('')
ax1.set_xlim(0, 1300)
ax1.set_ylim(-25, 400)
ax1.set_yticks([0, 200, 400])
ax1.set_xticks([0, 500, 1000])

ax2.plot(x, results['voi'], marker='s', color='black', linestyle='--',
         markersize=6, linewidth=2)
ax2.set_ylim(-5, 100)
ax2.set_xlim(0, 1300)
ax2.set_yticks([0, 50, 100])
ax2.set_xticks([0, 500, 1000])
ax2.set_xlabel('Easting (m)')
ax2.set_ylabel("VOI ($100K)")
ax2.set_title("(c)", loc="left")

plt.savefig(OUT_DIR / 'fig_case1_vertical_voi_thickness.png', bbox_inches='tight')
plt.close(fig)
