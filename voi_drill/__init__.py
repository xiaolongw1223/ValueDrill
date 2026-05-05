# -*- coding: utf-8 -*-
"""
voi_drill
=========
Value of Information (VOI) analysis for candidate drillhole siting in
2-D mineral exploration settings.

Public API
----------
simulate_trajectory_2d  -- 2-D drillhole trajectory with constant dip
value_function_action   -- binary hit/miss value function
utility_linear          -- linear (risk-neutral) utility
utility_exponential     -- exponential (CARA) utility
VOI                     -- VOI calculator class
grade_thickness         -- expected ore-thickness metric
"""

from .utilities import value_function_action, utility_linear, utility_exponential
from .trajectory import simulate_trajectory_2d
from .metrics import VOI, grade_thickness

__all__ = [
    "value_function_action",
    "utility_linear",
    "utility_exponential",
    "simulate_trajectory_2d",
    "VOI",
    "grade_thickness",
]
