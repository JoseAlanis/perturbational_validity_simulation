"""Simulate and evaluate perturbational validity across participant models.

This is a translation of a reference MATLAB implementation and keeps its
numerical conventions: standard deviations are sample estimates, per-condition
seeds are derived from the calibration size, and indices are zero-based unless a
field documents otherwise.

Every reported default lives in `config.py`. The generating system and its
integration are in `dynamics.py`, the shared network and baseline in `model.py`,
the metrics in `metrics.py` and `evaluate.py`, the robustness sweeps in
`robustness.py`, and the figure layouts in `figures.py`.
"""

from .config import (
    DEFAULT_CONFIG,
    DEFAULT_NET,
    DEFAULT_OPT,
    DEFAULT_PALETTE,
    METRIC_NAMES,
    Config,
    NetConfig,
    OptConfig,
    Palette,
)

__all__ = [
    "Config",
    "NetConfig",
    "OptConfig",
    "Palette",
    "DEFAULT_CONFIG",
    "DEFAULT_NET",
    "DEFAULT_OPT",
    "DEFAULT_PALETTE",
    "METRIC_NAMES",
]
