"""Structural, distributional, and interventional evaluation metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Config
from .dynamics import DriftFn, rollout_many, simulate_ensemble


def cumulative_trapezoid(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Compute a cumulative trapezoidal integral starting at zero."""
    y = np.asarray(y, dtype=float).ravel()
    x = np.asarray(x, dtype=float).ravel()
    increments = 0.5 * (y[1:] + y[:-1]) * np.diff(x)
    return np.concatenate([[0.0], np.cumsum(increments)])


def recover_potential(drift_fn: DriftFn, x_grid: np.ndarray) -> np.ndarray:
    """Integrate an unforced drift estimate into an energy landscape."""
    drift = drift_fn(x_grid, np.zeros_like(x_grid))
    v = -cumulative_trapezoid(drift, x_grid)
    return v - v.min()


def align_potential_offset_only(v: np.ndarray) -> np.ndarray:
    """Remove the arbitrary additive constant; shape is left untouched."""
    v = np.asarray(v).ravel()
    return v - v.min()


@dataclass(frozen=True)
class LandscapeFeatures:
    """Minima, saddle and barrier height of an energy landscape."""

    minima_x: np.ndarray
    saddle_x: float
    barrier: float

    @property
    def is_bistable(self) -> bool:
        """Whether two minima and an intervening saddle were recovered."""
        return self.minima_x.size == 2 and np.isfinite(self.saddle_x)


def landscape_features(x_grid: np.ndarray, v: np.ndarray) -> LandscapeFeatures:
    """Locate the two deepest minima and the barrier between them."""
    v = np.asarray(v).ravel()
    dv = np.diff(v)

    # Interior local minima: the slope changes from negative to non-negative.
    min_idx = np.flatnonzero((dv[:-1] < 0) & (dv[1:] >= 0)) + 1
    if min_idx.size == 0:
        min_idx = np.array([int(np.argmin(v))])

    order = np.argsort(v[min_idx], kind="stable")
    min_idx = np.sort(min_idx[order[: min(2, order.size)]])

    if min_idx.size >= 2:
        between = np.arange(min_idx[0], min_idx[-1] + 1)
        saddle_idx = int(between[int(np.argmax(v[between]))])
    else:
        return LandscapeFeatures(
            minima_x=np.asarray(x_grid)[min_idx],
            saddle_x=float("nan"),
            barrier=float("nan"),
        )

    return LandscapeFeatures(
        minima_x=np.asarray(x_grid)[min_idx],
        saddle_x=float(np.asarray(x_grid)[saddle_idx]),
        barrier=float(v[saddle_idx] - np.mean(v[min_idx])),
    )


def matched_attractor_error(estimated: np.ndarray, truth: np.ndarray) -> float:
    """Mean absolute attractor displacement, penalising a wrong attractor count."""
    estimated = np.sort(np.asarray(estimated).ravel())
    truth = np.sort(np.asarray(truth).ravel())
    if estimated.size == 0 or truth.size == 0:
        return float("nan")
    k = min(estimated.size, truth.size)
    return float(
        np.mean(np.abs(estimated[:k] - truth[:k]))
        + 0.5 * abs(estimated.size - truth.size)
    )


def basin_transition(x: np.ndarray, saddle: float) -> bool:
    """Whether a trajectory ends in the opposite basin from where it started."""
    x = np.asarray(x).ravel()
    initial_side = np.sign(x[0] - saddle)
    final_side = np.sign(np.mean(x[max(0, x.size - 61) :]) - saddle)

    if initial_side == 0:
        initial_side = -1.0
    if final_side == 0:
        final_side = initial_side
    return bool(initial_side != final_side)


def count_basin_transitions(x: np.ndarray, saddle: float) -> int:
    """Number of basin crossings in a trajectory."""
    side = np.sign(np.asarray(x).ravel() - saddle)
    side[side == 0] = 1
    return int(np.sum(np.abs(np.diff(side)) > 0))


def ensemble_transition_probability(x: np.ndarray, saddle: float) -> float:
    """Fraction of an ensemble that ends in the opposite basin."""
    initial_side = np.sign(x[0, :] - saddle)
    initial_side[initial_side == 0] = -1.0

    final_side = np.sign(np.mean(x[max(0, x.shape[0] - 61) :, :], axis=0) - saddle)
    final_side = np.where(final_side == 0, initial_side, final_side)

    return float(np.mean(initial_side != final_side))


def empirical_probability(
    samples: np.ndarray, edges: np.ndarray, pseudo_count: float
) -> np.ndarray:
    """Histogram of samples over ``edges``, smoothed and normalised."""
    counts, _ = np.histogram(np.asarray(samples).ravel(), bins=edges)
    prob = counts.astype(float) + pseudo_count
    return prob / prob.sum()


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=float).ravel()
    q = np.asarray(q, dtype=float).ravel()
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=float).ravel()
    q = np.asarray(q, dtype=float).ravel()
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))


def mean_response_js_divergence(
    true_x: np.ndarray,
    model_x: np.ndarray,
    edges: np.ndarray,
    stride: int,
    pseudo_count: float,
) -> float:
    """Jensen--Shannon divergence between response distributions, averaged over time."""
    n_steps = true_x.shape[0]
    idx = np.unique(np.append(np.arange(0, n_steps, stride), n_steps - 1))

    values = [
        js_divergence(
            empirical_probability(true_x[k, :], edges, pseudo_count),
            empirical_probability(model_x[k, :], edges, pseudo_count),
        )
        for k in idx
    ]
    return float(np.mean(values))


def transition_curve_deterministic(
    drift_fn: DriftFn, cfg: Config, initial_state: float, saddle: float
) -> np.ndarray:
    """Deterministic transition indicator as a function of pulse amplitude."""
    amplitudes = cfg.test_amplitude_grid
    inputs = np.column_stack([cfg.pulse_input(a) for a in amplitudes])
    trajectories = rollout_many(drift_fn, initial_state, inputs, cfg)
    return np.array(
        [
            float(basin_transition(trajectories[:, k], saddle))
            for k in range(amplitudes.size)
        ]
    )


def transition_probability_curve(
    drift_fn: DriftFn,
    cfg: Config,
    initial_state: float,
    saddle: float,
    sigma: float,
    n_replicates: int,
    seed: int,
) -> np.ndarray:
    """Stochastic transition probability as a function of pulse amplitude."""
    curve = np.zeros(cfg.test_amplitude_grid.size)
    for k, amplitude in enumerate(cfg.test_amplitude_grid):
        ensemble = simulate_ensemble(
            drift_fn,
            cfg.pulse_input(amplitude),
            cfg,
            initial_state,
            n_replicates,
            sigma,
            seed + k + 1,
        )
        curve[k] = ensemble_transition_probability(ensemble, saddle)
    return curve


def ensemble_band(x: np.ndarray):
    """Return the ensemble mean and Hazen 90% interval."""
    mu = np.nanmean(x, axis=1)
    lo = np.percentile(x, 5, axis=1, method="hazen")
    hi = np.percentile(x, 95, axis=1, method="hazen")
    return mu, lo, hi
