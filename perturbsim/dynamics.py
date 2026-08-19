"""Ground-truth bistable dynamics, inputs, and stochastic integration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .config import Config

#: A drift model: maps (state, input) arrays to a drift array, both unscaled.
DriftFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclass(frozen=True)
class SubjectParams:
    """Participant-specific parameters of the generating system."""

    a: float
    c: float
    d: float
    b: float
    sigma: float


def true_drift(x, u, p: SubjectParams):
    """Controlled drift f(x, u) of the generating system."""
    return -(p.a * np.asarray(x) ** 3 - p.c * np.asarray(x) - p.d) + p.b * np.asarray(u)


def true_potential(x, p: SubjectParams):
    """Unforced potential V(x), with f(x, 0) = -dV/dx."""
    x = np.asarray(x)
    return p.a * x**4 / 4 - p.c * x**2 / 2 - p.d * x


def initial_state_from_params(p: SubjectParams) -> float:
    """Left-basin starting state used for every evaluation rollout."""
    return -np.sqrt(max(p.c / p.a, 0.05))


def _uniform_sample(rng: np.random.Generator, span: tuple[float, float]) -> float:
    lo, hi = span
    if lo == hi:
        return lo
    return lo + rng.random() * (hi - lo)


def sample_subject_parameters(
    n: int, cfg: Config, rng: np.random.Generator
) -> list[SubjectParams]:
    """Draw ``n`` participants from the population family."""
    return [
        SubjectParams(
            a=_uniform_sample(rng, cfg.a_range),
            c=_uniform_sample(rng, cfg.c_range),
            d=_uniform_sample(rng, cfg.d_range),
            b=_uniform_sample(rng, cfg.b_range),
            sigma=_uniform_sample(rng, cfg.sigma_range),
        )
        for _ in range(n)
    ]


def make_random_pulse_input(
    n_steps: int, cfg: Config, rng: np.random.Generator
) -> np.ndarray:
    """Generate population-pretraining noise with sparse random pulses."""
    u = cfg.train_input_noise_sd * rng.standard_normal(n_steps)

    t = 0
    while t < n_steps:
        if rng.random() < cfg.train_pulse_probability:
            lo, hi = cfg.train_pulse_duration_range
            duration = int(rng.integers(lo, hi + 1))
            amplitude = _uniform_sample(rng, cfg.train_pulse_amplitude_range)

            stop = min(n_steps, t + duration)
            u[t:stop] += amplitude
            t = stop
        else:
            t += 1

    limit = max(abs(np.asarray(cfg.train_pulse_amplitude_range)))
    return np.clip(u, -limit, limit)


def make_calibration_input(
    n_steps: int, cfg: Config, rng: np.random.Generator
) -> np.ndarray:
    """Generate calibration noise with a fixed number of random pulses."""
    u = cfg.cal_input_noise_sd * rng.standard_normal(n_steps)

    n_candidates = max(1, n_steps - 100)
    n_pulses = min(cfg.cal_pulse_count, n_candidates)
    starts = rng.permutation(n_candidates)[:n_pulses]

    lo, hi = cfg.cal_pulse_duration_range
    for start in starts:
        duration = int(rng.integers(lo, hi + 1))
        amplitude = _uniform_sample(rng, cfg.cal_pulse_amplitude_range)
        stop = min(n_steps, int(start) + duration)
        u[int(start) : stop] += amplitude

    limit = max(abs(np.asarray(cfg.cal_pulse_amplitude_range)))
    return np.clip(u, -limit, limit)


def simulate_subject(
    p: SubjectParams,
    u: np.ndarray,
    cfg: Config,
    initial_basin: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Stochastic trajectory of the true system under input ``u``."""
    n_steps = u.size
    x = np.zeros(n_steps)

    base = np.sqrt(max(p.c / p.a, 0.05))
    if initial_basin < 0:
        x[0] = -base + 0.08 * rng.standard_normal()
    elif initial_basin > 0:
        x[0] = base + 0.08 * rng.standard_normal()
    else:
        x[0] = 0.15 * rng.standard_normal()

    noise = p.sigma * np.sqrt(cfg.dt) * rng.standard_normal(n_steps - 1)
    for t in range(n_steps - 1):
        drift = true_drift(x[t], u[t], p)
        x[t + 1] = np.clip(
            x[t] + cfg.dt * drift + noise[t], -cfg.state_clip, cfg.state_clip
        )
    return x


def rollout(
    drift_fn: DriftFn, initial_state: float, u: np.ndarray, cfg: Config
) -> np.ndarray:
    """Deterministic rollout of a learned or true drift model."""
    n_steps = u.size
    x = np.zeros(n_steps)
    x[0] = initial_state
    for t in range(n_steps - 1):
        drift = float(np.reshape(drift_fn(np.array([x[t]]), np.array([u[t]])), -1)[0])
        x[t + 1] = np.clip(
            x[t] + cfg.dt * drift, -cfg.state_clip, cfg.state_clip
        )
    return x


def rollout_many(
    drift_fn: DriftFn, initial_state: float, inputs: np.ndarray, cfg: Config
) -> np.ndarray:
    """Return batched rollouts for ``inputs`` shaped (steps, sequences)."""
    n_steps, n_seq = inputs.shape
    x = np.zeros((n_steps, n_seq))
    x[0, :] = initial_state
    for t in range(n_steps - 1):
        drift = np.reshape(drift_fn(x[t, :], inputs[t, :]), n_seq)
        x[t + 1, :] = np.clip(
            x[t, :] + cfg.dt * drift, -cfg.state_clip, cfg.state_clip
        )
    return x


def simulate_ensemble(
    drift_fn: DriftFn,
    u: np.ndarray,
    cfg: Config,
    initial_state: float,
    n_replicates: int,
    sigma: float,
    seed: int,
) -> np.ndarray:
    """Return a stochastic ensemble shaped (steps, replicates)."""
    rng = np.random.default_rng(seed)
    n_steps = u.size
    x = np.zeros((n_steps, n_replicates))
    x[0, :] = initial_state

    scale = sigma * np.sqrt(cfg.dt)
    for t in range(n_steps - 1):
        drift = np.reshape(
            drift_fn(x[t, :], np.full(n_replicates, u[t])), n_replicates
        )
        x[t + 1, :] = np.clip(
            x[t, :] + cfg.dt * drift + scale * rng.standard_normal(n_replicates),
            -cfg.state_clip,
            cfg.state_clip,
        )
    return x


def true_drift_fn(p: SubjectParams) -> DriftFn:
    """Drift callable for the ground-truth system."""
    return lambda x, u: true_drift(x, u, p)


@dataclass
class Dataset:
    """Flattened (state, input, drift target, subject) training samples."""

    x: np.ndarray
    u: np.ndarray
    y: np.ndarray
    subject: np.ndarray
    xs: np.ndarray | None = None
    us: np.ndarray | None = None
    ys: np.ndarray | None = None

    def __len__(self) -> int:
        return self.x.size


@dataclass(frozen=True)
class Scale:
    """Standardisation constants estimated on the perturbational population."""

    x_mean: float
    x_std: float
    u_mean: float
    u_std: float
    y_mean: float
    y_std: float

    @classmethod
    def from_dataset(cls, data: Dataset) -> "Scale":
        eps = np.finfo(float).eps
        # Sample standard deviation; ddof=1 is deliberate.
        return cls(
            x_mean=float(np.mean(data.x)),
            x_std=float(np.std(data.x, ddof=1) + eps),
            u_mean=float(np.mean(data.u)),
            u_std=float(np.std(data.u, ddof=1) + eps),
            y_mean=float(np.mean(data.y)),
            y_std=float(np.std(data.y, ddof=1) + eps),
        )


def standardise(data: Dataset, scale: Scale) -> Dataset:
    """Attach standardised copies of the state, input and target."""
    data.xs = (data.x - scale.x_mean) / scale.x_std
    data.us = (data.u - scale.u_mean) / scale.u_std
    data.ys = (data.y - scale.y_mean) / scale.y_std
    return data


def generate_population_dataset(
    params: list[SubjectParams],
    steps_per_subject: int,
    cfg: Config,
    with_perturbations: bool,
    rng: np.random.Generator,
) -> Dataset:
    """Simulate the pretraining population, passively or under perturbation."""
    regime = "perturbational" if with_perturbations else "passive"
    return generate_population_dataset_regime(
        params, steps_per_subject, cfg, regime, rng
    )


def generate_population_dataset_regime(
    params: list[SubjectParams],
    steps_per_subject: int,
    cfg: Config,
    regime: str,
    rng: np.random.Generator,
) -> Dataset:
    """Simulate a passive, perturbational, or identification-control dataset."""
    valid_regimes = {
        "passive",
        "perturbational",
        "state_coverage",
        "input_excitation",
    }
    if regime not in valid_regimes:
        raise ValueError(f"unknown pretraining regime: {regime}")

    samples_per_subject = steps_per_subject - 1
    n_subjects = len(params)
    total = n_subjects * samples_per_subject

    all_x = np.zeros(total)
    all_u = np.zeros(total)
    all_y = np.zeros(total)
    all_subject = np.zeros(total, dtype=int)
    total_steps = steps_per_subject + cfg.burn_in

    for s, p in enumerate(params):
        drives_state = regime in {"perturbational", "state_coverage"}
        trajectory_u = (
            make_random_pulse_input(total_steps, cfg, rng)
            if drives_state
            else np.zeros(total_steps)
        )
        if cfg.randomise_initial_basin:
            initial_basin = 2.0 * (rng.random() > 0.5) - 1.0
        else:
            initial_basin = -1.0

        x = simulate_subject(p, trajectory_u, cfg, initial_basin, rng)[cfg.burn_in :]
        trajectory_u = trajectory_u[cfg.burn_in :]
        x_now = x[:-1]

        if regime == "input_excitation":
            model_u = make_random_pulse_input(total_steps, cfg, rng)[cfg.burn_in:-1]
        elif regime == "state_coverage":
            model_u = np.zeros(samples_per_subject)
        else:
            model_u = trajectory_u[:-1]

        if cfg.target_mode.lower() == "truedrift":
            y_now = true_drift(x_now, model_u, p)
        elif regime in {"state_coverage", "input_excitation"}:
            raise ValueError("identification controls require analytical drift targets")
        else:
            y_now = np.diff(x) / cfg.dt

        sl = slice(s * samples_per_subject, (s + 1) * samples_per_subject)
        all_x[sl] = x_now
        all_u[sl] = model_u
        all_y[sl] = y_now
        all_subject[sl] = s

    return Dataset(x=all_x, u=all_u, y=all_y, subject=all_subject)
