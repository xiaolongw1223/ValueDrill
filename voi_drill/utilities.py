# -*- coding: utf-8 -*-


import numpy as np


# ---------------------------------------------------------------------------
# Value function
# ---------------------------------------------------------------------------

def value_function_action(trace, hit_value=10.0, miss_value=-10.0):
    """
    Binary hit/miss value function for a single drillhole realization.

    Returns *hit_value* if any ore (1) is present along the trace,
    otherwise *miss_value*.

    Parameters
    ----------
    trace : array-like of {0, 1}
        Binary ore indicators sampled along the drillhole.
    hit_value : float
        Payoff when ore is encountered. Default ``10.0``.
    miss_value : float
        Payoff when no ore is encountered. Default ``-10.0``.

    Returns
    -------
    float

    Examples
    --------
    >>> value_function_action([0, 0, 1, 0], hit_value=500, miss_value=-100)
    500
    >>> value_function_action([0, 0, 0, 0], hit_value=500, miss_value=-100)
    -100
    """
    return hit_value if np.any(np.asarray(trace) == 1) else miss_value


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def utility_linear(v, coefficient=1.0, cutoff=0.0):
    """
    Linear (risk-neutral) utility function.

    .. math:: u = \\text{coefficient} \\times (v - \\text{cutoff})

    Parameters
    ----------
    v : float or np.ndarray
        Raw value(s).
    coefficient : float
        Scaling factor. Default ``1.0``.
    cutoff : float
        Value offset before scaling. Default ``0.0``.

    Returns
    -------
    float or np.ndarray

    Examples
    --------
    >>> utility_linear(100.0)
    100.0
    >>> utility_linear(100.0, coefficient=0.5, cutoff=20.0)
    40.0
    """
    return coefficient * (np.asarray(v, dtype=float) - cutoff)


def utility_exponential(v, alpha=0.0, beta=1.0, gamma=0.2):
    """
    Exponential utility function encoding constant absolute risk aversion.

    .. math:: u = \\alpha + \\beta \\, e^{-\\gamma v}

    Risk attitude is determined by the sign combination of *beta* and *gamma*:

    +------------------+---------+---------+
    | Risk attitude    |  beta   |  gamma  |
    +==================+=========+=========+
    | Risk-averse      |  < 0    |  > 0    |
    +------------------+---------+---------+
    | Risk-seeking     |  > 0    |  < 0    |
    +------------------+---------+---------+


    Parameters
    ----------
    v : float or np.ndarray
        Raw value(s).
    alpha : float
        Utility offset. Default ``0.0``.
    beta : float
        Scale parameter. Default ``1.0``.
    gamma : float
        Curvature (risk aversion coefficient). Default ``0.2``.

    Returns
    -------
    float or np.ndarray

    Examples
    --------
    >>> # Risk-averse (concave)
    >>> utility_exponential(50.0, alpha=20, beta=-30, gamma=0.1)

    >>> # Risk-seeking (convex)
    >>> utility_exponential(50.0, alpha=-50, beta=1e-6, gamma=-0.1)
    """
    return alpha + beta * np.exp(-gamma * np.asarray(v, dtype=float))
