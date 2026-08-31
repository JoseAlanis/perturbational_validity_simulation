"""Robustness analyses for the bistable-system experiment."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import Config, NetConfig, OptConfig
from .dynamics import (
    Dataset,
    Scale,
    generate_population_dataset_regime,
    initial_state_from_params,
    make_calibration_input,
    sample_subject_parameters,
    simulate_ensemble,
    simulate_subject,
    standardise,
    true_drift,
    true_drift_fn,
    true_potential,
)
from .evaluate import sample_calibration_indices
from .metrics import (
    align_potential_offset_only,
    landscape_features,
    mean_response_js_divergence,
    recover_potential,
)
from .model import (
    adapt_embedding,
    adapt_full_model,
    fit_cubic_scratch_model,
    initialise_shared_model,
    cubic_drift_fn,
    shared_drift_fn,
    train_shared_model,
)

REGIMES = ("passive", "state_coverage", "input_excitation", "perturbational")
MODEL_LABELS = {
    "scratch": "Cubic scratch",
    "passive_frozen": "Passive, embedding only",
    "passive_full": "Passive, full adaptation",
    "state_coverage": "State coverage only",
    "input_excitation": "Input excitation only",
    "perturbational": "Perturbational",
}
METRICS = (
    "landscapeRMSE",
    "flowRMSE",
    "responseJSInterpolation",
    "responseJSExtrapolation",
    "transitionProbabilityErrorInterpolation",
    "transitionProbabilityErrorExtrapolation",
)


@dataclass(frozen=True)
class RobustnessCondition:
    """One pretraining and evaluation configuration."""

    name: str
    cfg: Config
    net: NetConfig
    opt: OptConfig


def _calibration_data(
    p, cfg: Config, scale: Scale, subject_index: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(1000 + 100_000 * cfg.seed + subject_index)
    xs, us, ys = [], [], []
    for episode in range(cfg.calibration_episodes):
        u = make_calibration_input(cfg.test_steps, cfg, rng)
        basin = -1.0 if episode % 2 == 0 else 1.0
        x = simulate_subject(p, u, cfg, basin, rng)
        xs.append(x[:-1])
        us.append(u[:-1])
        ys.append(true_drift(x[:-1], u[:-1], p))
    return np.concatenate(xs), np.concatenate(us), np.concatenate(ys)


def _response_js(
    truth_fn, model_fn, amplitude: float, p, cfg: Config, subject_index: int, seed_tag: int
) -> float:
    u = cfg.pulse_input(amplitude)
    initial = initial_state_from_params(p)
    true_ensemble = simulate_ensemble(
        truth_fn, u, cfg, initial, cfg.distribution_replicates, p.sigma,
        seed_tag + 10_000 * cfg.seed + subject_index,
    )
    model_ensemble = simulate_ensemble(
        model_fn, u, cfg, initial, cfg.distribution_replicates, p.sigma,
        seed_tag + 10_000 * cfg.seed + subject_index,  # shared innovations
    )
    return mean_response_js_divergence(
        true_ensemble, model_ensemble, cfg.density_edges,
        cfg.response_eval_stride, cfg.density_pseudo_count,
    )


def _transition_probability_error(
    truth_fn, model_fn, amplitude: float, p, cfg: Config, subject_index: int, seed_tag: int
) -> float:
    u = cfg.pulse_input(amplitude)
    initial = initial_state_from_params(p)
    truth = simulate_ensemble(
        truth_fn, u, cfg, initial, cfg.distribution_replicates, p.sigma,
        seed_tag + 30_000 * cfg.seed + subject_index,
    )
    model = simulate_ensemble(
        model_fn, u, cfg, initial, cfg.distribution_replicates, p.sigma,
        seed_tag + 30_000 * cfg.seed + subject_index,  # shared innovations
    )
    true_v = true_potential(cfg.x_grid, p)
    saddle = landscape_features(cfg.x_grid, true_v).saddle_x
    true_probability = np.mean(truth[-1] > saddle)
    model_probability = np.mean(model[-1] > saddle)
    return float(abs(true_probability - model_probability))


def _score_model(
    model_fn, p, cfg: Config, subject_index: int, extrapolation_amplitude: float
) -> dict[str, float]:
    x_grid = cfg.x_grid
    truth_fn = true_drift_fn(p)
    true_v = true_potential(x_grid, p)
    true_v -= true_v.min()
    estimated_v = align_potential_offset_only(recover_potential(model_fn, x_grid))
    flow_x, flow_u = np.meshgrid(x_grid, cfg.flow_input_grid, indexing="ij")
    true_flow = true_drift(flow_x, flow_u, p)
    estimated_flow = model_fn(flow_x.ravel(), flow_u.ravel()).reshape(flow_x.shape)

    return {
        "landscapeRMSE": float(np.sqrt(np.mean((estimated_v - true_v) ** 2))),
        "flowRMSE": float(np.sqrt(np.mean((estimated_flow - true_flow) ** 2))),
        "responseJSInterpolation": _response_js(
            truth_fn, model_fn, cfg.test_pulse_amplitude, p, cfg, subject_index, 100_000
        ),
        "responseJSExtrapolation": _response_js(
            truth_fn, model_fn, extrapolation_amplitude, p, cfg, subject_index, 200_000
        ),
        "transitionProbabilityErrorInterpolation": _transition_probability_error(
            truth_fn, model_fn, cfg.test_pulse_amplitude, p, cfg, subject_index, 300_000
        ),
        "transitionProbabilityErrorExtrapolation": _transition_probability_error(
            truth_fn, model_fn, extrapolation_amplitude, p, cfg, subject_index, 400_000
        ),
    }


def run_condition(
    condition: RobustnessCondition,
    calibration_sizes: tuple[int, ...],
    include_ablations: bool = True,
    verbose: bool = True,
) -> list[dict]:
    """Run one robustness condition and return subject-level metric rows."""
    cfg, net, opt = condition.cfg, condition.net, condition.opt
    seed_sequence = np.random.SeedSequence(cfg.seed)
    param_seed, base_seed, test_seed, *data_seeds, train_seed = seed_sequence.spawn(8)
    parameter_rng = np.random.default_rng(param_seed)
    train_params = sample_subject_parameters(cfg.train_subjects, cfg, parameter_rng)

    regimes = REGIMES if include_ablations else ("passive", "perturbational")
    datasets = {}
    for regime, child_seed in zip(REGIMES, data_seeds):
        if regime in regimes:
            datasets[regime] = generate_population_dataset_regime(
                train_params, cfg.train_steps_per_subject, cfg, regime,
                np.random.default_rng(child_seed),
            )

    scale = Scale.from_dataset(datasets["perturbational"])
    datasets = {name: standardise(data, scale) for name, data in datasets.items()}
    base_model = initialise_shared_model(
        net, cfg.train_subjects, np.random.default_rng(base_seed)
    )
    models = {}
    training_rng = np.random.default_rng(train_seed)
    for regime in regimes:
        if verbose:
            print(f"  training {regime}", flush=True)
        models[regime] = train_shared_model(
            base_model, datasets[regime], opt, regime, training_rng, verbose=False
        )

    test_params = sample_subject_parameters(
        cfg.test_subjects, cfg, np.random.default_rng(test_seed)
    )
    extrapolation_amplitude = max(3.0, 1.25 * max(abs(v) for v in cfg.train_pulse_amplitude_range))
    rows = []

    for subject_index, p in enumerate(test_params, start=1):
        all_x, all_u, all_y = _calibration_data(p, cfg, scale, subject_index)
        for ci, requested in enumerate(calibration_sizes):
            n_cal = min(requested, all_x.size)
            idx = sample_calibration_indices(
                all_u, n_cal, cfg.cal_perturbed_fraction,
                5000 + 100_000 * cfg.seed + 100 * subject_index + ci,
            )
            cal = standardise(
                Dataset(
                    x=all_x[idx], u=all_u[idx], y=all_y[idx],
                    subject=np.zeros(idx.size, dtype=int),
                ),
                scale,
            )
            beta = fit_cubic_scratch_model(cal)
            drift_fns = {"scratch": cubic_drift_fn(beta, scale)}
            for regime, model in models.items():
                embedding = adapt_embedding(model, cal, opt)
                label = f"{regime}_frozen" if regime == "passive" else regime
                drift_fns[label] = shared_drift_fn(model, embedding, scale)
            full_model, full_embedding = adapt_full_model(models["passive"], cal, opt)
            drift_fns["passive_full"] = shared_drift_fn(
                full_model, full_embedding, scale
            )

            for model_name, drift_fn in drift_fns.items():
                scores = _score_model(
                    drift_fn, p, cfg, subject_index, extrapolation_amplitude
                )
                rows.append({
                    "condition": condition.name,
                    "trainingSeed": cfg.seed,
                    "subject": subject_index,
                    "calibrationSize": n_cal,
                    "model": model_name,
                    "extrapolationAmplitude": extrapolation_amplitude,
                    **scores,
                })
        if verbose:
            print(f"  evaluated subject {subject_index}/{cfg.test_subjects}", flush=True)
    return rows


def write_rows(rows: list[dict], path: Path) -> Path:
    """Write subject-level robustness results as CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def summarise_rows(rows: list[dict]) -> list[dict]:
    """Summarise subject-level results within each training run."""
    groups = {}
    for row in rows:
        key = (
            row["condition"], row["trainingSeed"],
            row["calibrationSize"], row["model"],
        )
        groups.setdefault(key, []).append(row)

    summary = []
    for key, group in sorted(groups.items()):
        item = dict(zip(("condition", "trainingSeed", "calibrationSize", "model"), key))
        for metric in METRICS:
            values = np.array([row[metric] for row in group], dtype=float)
            item[f"{metric}Mean"] = float(np.mean(values))
            item[f"{metric}SEM"] = float(np.std(values, ddof=1) / np.sqrt(values.size))
        summary.append(item)
    return summary
