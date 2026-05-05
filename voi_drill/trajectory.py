# -*- coding: utf-8 -*-

import numpy as np


def simulate_trajectory_2d(params, n_segments=120):
    """
    Simulate 2D drillhole trajectory with constant dip angle.

    Parameters
    ----------
    params : array-like
        Input parameters as:
        - 1D array [x0, z0, max_depth, dip] for single trajectory
        - 2D array with shape (n_trajectories, 4) where columns are:
          * x0: Starting X coordinate (collar position)
          * z0: Starting Z coordinate (collar position, Z+ is upward)
          * max_depth: Maximum measured depth along the hole
          * dip: Dip angle in degrees (measured from horizontal, 90° = vertical)
    n_segments : int, optional
        Number of trajectory segments (default: 120).

    Returns
    -------
    x : np.ndarray
        X coordinates along trajectory/trajectories.
        - Shape (n_segments,) for single trajectory
        - Shape (n_trajectories, n_segments) for multiple trajectories
    z : np.ndarray
        Z coordinates along trajectory/trajectories.
        - Shape (n_segments,) for single trajectory
        - Shape (n_trajectories, n_segments) for multiple trajectories

    Raises
    ------
    ValueError
        If input array doesn't have 4 columns.

    Examples
    --------
    >>> # Single trajectory
    >>> params = [100, 600, 500, 90]
    >>> x, z = simulate_trajectory_2d(params, n_segments=50)
    >>>
    >>> # Multiple trajectories
    >>> params = np.array([[100, 600, 500, 90],
    ...                    [200, 600, 500, 90]])
    >>> x, z = simulate_trajectory_2d(params, n_segments=50)
    """
    params = np.atleast_2d(params)
    if params.shape[1] != 4:
        raise ValueError("Input array must have 4 columns: [x0, z0, max_depth, dip]")

    x0_arr = params[:, 0]
    z0_arr = params[:, 1]
    max_depth_arr = params[:, 2]
    dip_arr = params[:, 3]

    n_trajectories = params.shape[0]
    dip_rad = np.radians(dip_arr)
    md = max_depth_arr / (n_segments - 1)
    dx = md * np.cos(dip_rad)
    dz = md * np.sin(dip_rad)

    segment_indices = np.arange(n_segments)
    x_all = x0_arr[:, np.newaxis] + dx[:, np.newaxis] * segment_indices
    z_all = z0_arr[:, np.newaxis] - dz[:, np.newaxis] * segment_indices

    if n_trajectories == 1:
        return x_all.squeeze(), z_all.squeeze()
    return x_all, z_all
