<img src="inputs/logo.png" width=15% align="left">

# ValueDrill
Smart Drilling Design for Mineral Exploration

Python implementation of **Value of Information (VOI)** analysis for candidate drillhole siting in 2D mineral exploration settings.

---

## Overview

Given an ensemble of prior geological model realizations, this package computes the VOI for each candidate drillhole — the expected gain in decision quality from drilling that hole before committing to a production decision. An alternative grade-thickness metric is also provided for comparison.

**Key components**

| Module | Contents |
|---|---|
| `voi_drill/utilities.py` | Value function and utility functions (linear, exponential) |
| `voi_drill/trajectory.py` | 2D drillhole trajectory simulation |
| `voi_drill/metrics.py` | `VOI` class and `grade_thickness` function |

---

## Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/xiaolongw1223/voi_drill.git
cd voi_drill
pip install -r requirements.txt
```

The example scripts run without installation — they add the package to `sys.path` automatically. To install the package into your environment instead, run `pip install -e .` from inside `voi_drill_package/`.

### Dependencies

| Package | Minimum version |
|---|---|
| numpy | 1.21 |
| scipy | 1.7 |
| matplotlib | 3.4 |
| discretize | 0.8 |

---

## Repository structure

```
voi_drill_package/
├── inputs/                           # input data (place files here before running)
│   ├── drillhole_2d.txt              # existing drillhole observations
│   ├── true_model_binary_2d.txt      # binary true model (for reference)
│   └── prior_model_geometry.txt      # prior model ensemble (n_models × n_cells)
├── outputs/                          # all output figures saved here (auto-created)
├── examples/                         # runnable example scripts
│   ├── case1_vertical.py
│   ├── case1_vertical_analysis.py
│   ├── case2_cone.py
│   └── case3_random.py
├── voi_drill/                    # core package
│   ├── __init__.py
│   ├── utilities.py
│   ├── trajectory.py
│   └── metrics.py
├── requirements.txt
└── pyproject.toml
```

---

## Usage

### Minimal example

```python
import numpy as np
from discretize import TensorMesh
from voi_drill import (
    simulate_trajectory_2d,
    value_function_action,
    utility_linear,
    VOI,
    grade_thickness,
)

# 130 × 60 cell mesh, 10 m cell size
dh = 10
hx = [(dh, 130)]
hz = [(dh, 60)]
mesh = TensorMesh([hx, hz], "00")

# prior_models: shape (n_models, n_cells), binary ore indicators
prior_models = np.loadtxt("inputs/prior_model_geometry.txt").T

# Define 5 vertical candidate drillholes
x = np.linspace(200, 1000, 5)
params = np.column_stack([x, np.full(5, 599.5), np.full(5, 500.0), np.full(5, 90.0)])
x_all, z_all = simulate_trajectory_2d(params, n_segments=50)
trajectories = [np.c_[x_all[i], z_all[i]] for i in range(5)]

# Compute risk-neutral VOI
voi_calc = VOI(
    value_fn=value_function_action,
    value_kwargs={"hit_value": 500, "miss_value": -100},
    utility_fn=utility_linear,
    utility_kwargs={"coefficient": 1.0, "cutoff": 0.0},
    no_action_value=0.0,
)
results = voi_calc.compute_for_candidates(trajectories, prior_models, mesh)
print("VOI:", results["voi"])

# Compute expected ore thickness
gt = grade_thickness(trajectories, prior_models, mesh, segment_length=10.0)
print("Grade-thickness (m):", gt)
```

### Risk-adjusted VOI

Pass a non-linear utility function to encode risk aversion:

```python
from voi_drill import utility_exponential

voi_calc = VOI(
    value_fn=value_function_action,
    value_kwargs={"hit_value": 500, "miss_value": -100},
    utility_fn=utility_exponential,
    utility_kwargs={"alpha": 20, "beta": -30, "gamma": 0.1},  # risk-averse
    no_action_value=0.0,
)
```

---

## Reproducing the paper figures

Place the input data files in `inputs/` then run any script from the `examples/` directory. Figures are saved automatically to `outputs/`.

```bash
cd examples
python case1_vertical.py
```

| Script | Output figures |
|---|---|
| `case1_vertical.py` | `fig_case1_vertical_model_drillhole_trace.png`, `fig_case1_vertical_voi_thickness.png` |
| `case1_vertical_analysis.py` | `fig_case1_vertical_analysis_drillhole_1.png` |
| `case2_cone.py` | `fig_case2_cone_model_drillhole_trace.png`, `fig_case2_cone_voi.png`, `fig_case2_cone_thickness.png` |
| `case3_random.py` | `fig_case3_random_model_drillhole_trace.png`, `fig_case3_random_voi_thickness.png` |

---

## License

[MIT](LICENSE)
