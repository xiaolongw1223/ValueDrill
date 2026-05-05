# -*- coding: utf-8 -*-
"""
Case 2 – Cone pattern: 200 drillholes fanning from a single collar with
random dip (20–160°) and near-constant depth (~500 m).

Outputs
-------
fig_case2_cone_model_drillhole_trace.png
fig_case2_cone_voi.png
fig_case2_cone_thickness.png
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import numpy as np
import matplotlib as mpl
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

np.random.seed(99)

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
# Candidate drillholes – cone pattern
# ------------------------------------------------------------------
n_holes = 200
x     = np.linspace(649.5, 650.5, n_holes)
z     = np.full(n_holes, 599.5)
depth = np.random.uniform(499, 500, n_holes)
dip   = np.random.uniform(20, 160, n_holes)
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
    ax.plot(x_all[i, :], z_all[i, :], '--', c="r", linewidth=1.5, alpha=0.2)
ax.set_xlabel('Easting (m)')
ax.set_ylabel('Elevation (m)')
ax.set_xticks([0, 500, 1000])
ax.set_title("(a)", loc="left")
cbar = fig.colorbar(im, ax=ax, location='right', shrink=0.4)
cbar.set_ticks([0, 1])
cbar.set_label('Mean orebody', rotation=270, labelpad=20)
plt.savefig(OUT_DIR / 'fig_case2_cone_model_drillhole_trace.png', bbox_inches='tight')
plt.close(fig)

# ------------------------------------------------------------------
# Figure (b) – VOI per trace
# ------------------------------------------------------------------
voi_vals = np.asarray(results["voi"], dtype=float)
cmap     = plt.cm.jet
norm     = mpl.colors.Normalize(vmin=np.nanmin(voi_vals), vmax=np.nanmax(voi_vals))

fig, ax = plt.subplots(figsize=(8, 6))
ax.imshow(np.zeros_like(mean_2d.T), origin='lower', cmap='binary', vmin=0, vmax=1,
          extent=EXTENT)
for i in range(n_holes):
    ax.plot(x_all[i, :], z_all[i, :], '--', color=cmap(norm(voi_vals[i])),
            linewidth=1.5, alpha=1)
ax.set_xlabel('Easting (m)')
ax.set_ylabel('Elevation (m)')
ax.set_xticks([0, 500, 1000])
ax.set_title("(b)", loc="left")
cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                    location='right', shrink=0.4)
cbar.set_ticks([0, int(np.max(voi_vals))])
cbar.set_label('VOI ($100K)', rotation=270, labelpad=20)
plt.savefig(OUT_DIR / 'fig_case2_cone_voi.png', bbox_inches='tight')
plt.close(fig)

# ------------------------------------------------------------------
# Figure (c) – grade-thickness per trace
# ------------------------------------------------------------------
norm = mpl.colors.Normalize(vmin=np.nanmin(gt_scores), vmax=np.nanmax(gt_scores))

fig, ax = plt.subplots(figsize=(8, 6))
ax.imshow(np.zeros_like(mean_2d.T), origin='lower', cmap='binary', vmin=0, vmax=1,
          extent=EXTENT)
for i in range(n_holes):
    ax.plot(x_all[i, :], z_all[i, :], '--', color=cmap(norm(gt_scores[i])),
            linewidth=1.5, alpha=1)
ax.set_xlabel('Easting (m)')
ax.set_ylabel('Elevation (m)')
ax.set_xticks([0, 500, 1000])
ax.set_title("(c)", loc="left")
cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                    location='right', shrink=0.4)
cbar.set_ticks([int(np.min(gt_scores)), int(np.max(gt_scores))])
cbar.set_label('Thickness (m)', rotation=270, labelpad=20)
plt.savefig(OUT_DIR / 'fig_case2_cone_thickness.png', bbox_inches='tight')
plt.close(fig)
