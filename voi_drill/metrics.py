# -*- coding: utf-8 -*-


import numpy as np
from scipy.spatial import KDTree

from .utilities import utility_linear


def _sample_slices(prior_models, coords, mesh):
    """
    Map trajectory coordinates to nearest mesh cells and return model slices.

    Parameters
    ----------
    prior_models : np.ndarray, shape (n_models, n_cells)
        Ensemble of model realizations.
    coords : np.ndarray, shape (n_segments, 2)
        (x, z) coordinates along the trajectory.
    mesh : TensorMesh
        Mesh whose ``cell_centers`` are used for nearest-neighbour lookup.

    Returns
    -------
    np.ndarray, shape (n_models, n_segments)
    """
    indices = KDTree(mesh.cell_centers).query(coords)[1]
    return prior_models[:, indices]


def _apply_utility(values, utility_fn, utility_kwargs):
    """Vectorised utility application over an array of raw values."""
    return np.array(
        [utility_fn(v, **utility_kwargs) for v in values], dtype=float
    )


# ---------------------------------------------------------------------------
# VOI class
# ---------------------------------------------------------------------------

class VOI:
    """
    Value of Information for candidate drillhole locations.

    VOI measures the expected gain in decision quality obtained by drilling
    a candidate hole *before* committing to a production decision:

    .. math::
        \\text{VOI} = \\mathbb{E}_{d}[\\max_a \\bar{u}(a \\mid d)]
                    - \\max_a \\mathbb{E}[u(a)]

    where :math:`d` is the observed data, :math:`a` is the chosen action,
    and :math:`\\bar{u}` is the posterior expected utility.

    Parameters
    ----------
    value_fn : callable
        Maps a 1-D binary trace ``(n_segments,)`` to a raw monetary value.
        Signature: ``value_fn(trace, **value_kwargs) -> float``.
    value_kwargs : dict, optional
        Keyword arguments forwarded to *value_fn*.
    utility_fn : callable, optional
        Maps a raw value to a utility score.
        Signature: ``utility_fn(v, **utility_kwargs) -> float``.
        Defaults to :func:`~voi_drill.utilities.utility_linear` with
        ``coefficient=1, cutoff=0`` (risk-neutral).
    utility_kwargs : dict, optional
        Keyword arguments forwarded to *utility_fn*.
    no_action_value : float
        Utility of the do-nothing action. Default ``0.0``.

    Examples
    --------
    Risk-neutral VOI (standard case):

    >>> from voi_drill import VOI, value_function_action
    >>> voi = VOI(value_fn=value_function_action,
    ...           value_kwargs={'hit_value': 500, 'miss_value': -100})
    >>> result = voi.compute(slices)

    Risk-adjusted VOI:

    >>> from voi_drill import VOI, value_function_action, utility_exponential
    >>> voi = VOI(value_fn=value_function_action,
    ...           value_kwargs={'hit_value': 500, 'miss_value': -100},
    ...           utility_fn=utility_exponential,
    ...           utility_kwargs={'alpha': 20, 'beta': -30, 'gamma': 0.1})
    >>> result = voi.compute(slices)
    """

    def __init__(self, value_fn, value_kwargs=None,
                 utility_fn=None, utility_kwargs=None,
                 no_action_value=0.0):
        self.value_fn = value_fn
        self.value_kwargs = value_kwargs or {}
        self.utility_fn = utility_fn if utility_fn is not None else utility_linear
        self.utility_kwargs = utility_kwargs or {}
        self.no_action_value = float(no_action_value)

    def compute(self, slices):
        """
        Compute VOI from pre-sampled model slices at one candidate location.

        Parameters
        ----------
        slices : np.ndarray, shape (n_models, n_segments)
            Binary ore indicators along the drillhole for every realization.

        Returns
        -------
        dict with keys:
            ``voi``          – Value of Information.
            ``eu_with_data`` – Expected utility after observing the hole.
            ``eu_prior``     – Expected utility without any new data.
        """
        n_models = slices.shape[0]

        # prior expected utility (no new data)
        u_prior = _apply_utility(
            [self.value_fn(s, **self.value_kwargs) for s in slices],
            self.utility_fn, self.utility_kwargs
        )
        eu_prior = max(float(u_prior.mean()), self.no_action_value)

        # posterior expected utility (after observing this hole)
        # group realizations that produce the same observable trace
        _, inverse, counts = np.unique(
            slices, axis=0, return_inverse=True, return_counts=True
        )
        probs = counts / n_models

        eu_with_data = 0.0
        for k, p in enumerate(probs):
            ensemble = slices[inverse == k]
            u_vals = _apply_utility(
                [self.value_fn(s, **self.value_kwargs) for s in ensemble],
                self.utility_fn, self.utility_kwargs
            )
            eu_with_data += p * max(float(u_vals.mean()), self.no_action_value)

        voi = eu_with_data - eu_prior

        return {'voi': voi, 'eu_with_data': eu_with_data, 'eu_prior': eu_prior}

    def compute_for_candidates(self, trajectories, prior_models, mesh):
        """
        Compute VOI for every candidate trajectory.

        Parameters
        ----------
        trajectories : list of np.ndarray, each shape (n_segments, 2)
            (x, z) coordinate arrays for each candidate drillhole.
        prior_models : np.ndarray, shape (n_models, n_cells)
            Ensemble of prior model realizations.
        mesh : TensorMesh
            Mesh used for nearest-neighbour cell lookup.

        Returns
        -------
        dict with keys:
            ``voi``          – np.ndarray of VOI values.
            ``eu_with_data`` – np.ndarray of posterior expected utilities.
            ``eu_prior``     – np.ndarray of prior expected utilities.

        Examples
        --------
        >>> voi = VOI(value_fn=value_function_action,
        ...           value_kwargs={'hit_value': 500, 'miss_value': -100})
        >>> results = voi.compute_for_candidates(trajectories, prior_models, mesh)
        >>> best_idx = np.argmax(results['voi'])
        """
        keys = ('voi', 'eu_with_data', 'eu_prior')
        accum = {k: [] for k in keys}

        for coords in trajectories:
            slices = _sample_slices(prior_models, coords, mesh)
            result = self.compute(slices)
            for k in keys:
                accum[k].append(result[k])

        return {k: np.array(v) for k, v in accum.items()}


# ---------------------------------------------------------------------------
# Grade-thickness metric
# ---------------------------------------------------------------------------

def grade_thickness(trajectories, prior_models, mesh, segment_length=10.0):
    """
    Expected ore-thickness intercept for each candidate drillhole.

    For every trajectory the function computes the mean cumulative ore
    length across all prior realizations:

    .. math::
        T_i = \\mathbb{E}\\left[\\sum_{j} s_{ij} \\cdot \\Delta l\\right]

    where :math:`s_{ij} \\in \\{0,1\\}` is the ore indicator at segment *j*
    of realization *i* and :math:`\\Delta l` is the segment length.

    Parameters
    ----------
    trajectories : list of np.ndarray, each shape (n_segments, 2)
        (x, z) coordinate arrays for each candidate drillhole.
    prior_models : np.ndarray, shape (n_models, n_cells)
        Ensemble of prior model realizations.
    mesh : TensorMesh
        Mesh used for nearest-neighbour cell lookup.
    segment_length : float
        Physical length (metres) of each trajectory segment. Default ``10.0``.

    Returns
    -------
    np.ndarray, shape (n_candidates,)
        Expected ore thickness in metres for each candidate.

    Examples
    --------
    >>> gt = grade_thickness(trajectories, prior_models, mesh, segment_length=10)
    >>> best_idx = np.argmax(gt)
    """
    scores = []
    for coords in trajectories:
        slices = _sample_slices(prior_models, coords, mesh)
        scores.append(float(np.mean(segment_length * slices.sum(axis=1))))
    return np.array(scores)
