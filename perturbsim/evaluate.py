"""Evaluate held-out participants across calibration sizes and models."""

from __future__ import annotations

import numpy as np

from .config import METRIC_NAMES, Config, OptConfig
from .dynamics import (
    Dataset,
    Scale,
    SubjectParams,
    initial_state_from_params,
    make_calibration_input,
    rollout,
    simulate_ensemble,
    simulate_subject,
    standardise,
    true_drift,
    true_drift_fn,
    true_potential,
)
from .metrics import (
    align_potential_offset_only,
    basin_transition,
    count_basin_transitions,
    empirical_probability,
    js_divergence,
    kl_divergence,
    landscape_features,
    matched_attractor_error,
    mean_response_js_divergence,
    recover_potential,
    transition_curve_deterministic,
    transition_probability_curve,
)
from .model import (
    SharedModel,
    adapt_embedding,
    cubic_drift_fn,
    fit_cubic_scratch_model,
    shared_drift_fn,
)

N_MODELS = 3


def sample_calibration_indices(
    u: np.ndarray, n_cal: int, perturbed_fraction: float, seed: int
) -> np.ndarray:
    """Draw calibration samples, half of them from perturbed periods where possible."""
    rng = np.random.default_rng(seed)

    perturbed = np.flatnonzero(np.abs(u) > 0.15)
    unperturbed = np.flatnonzero(np.abs(u) <= 0.15)

    n_pert = min(int(round(n_cal * perturbed_fraction)), perturbed.size)
    n_unpert = min(n_cal - n_pert, unperturbed.size)

    picks = []
    if n_pert > 0:
        picks.append(rng.permutation(perturbed)[:n_pert])
    if n_unpert > 0:
        picks.append(rng.permutation(unperturbed)[:n_unpert])
    idx = np.concatenate(picks) if picks else np.array([], dtype=int)

    if idx.size < n_cal:
        remaining = np.setdiff1d(np.arange(u.size), idx)
        n_add = min(n_cal - idx.size, remaining.size)
        if n_add > 0:
            idx = np.concatenate([idx, rng.permutation(remaining)[:n_add]])

    return np.sort(idx)


def evaluate_one_subject(
    subject_index: int,
    p: SubjectParams,
    passive_model: SharedModel,
    perturb_model: SharedModel,
    cfg: Config,
    opt: OptConfig,
    scale: Scale,
    keep_example: bool = False,
) -> tuple[dict, np.ndarray, dict | None]:
    """Return metrics, coverage, and optional figure data for one system."""
    n_cal_sizes = len(cfg.calibration_sizes)
    metrics = {
        name: np.full((n_cal_sizes, N_MODELS), np.nan) for name in METRIC_NAMES
    }
    coverage = np.full(5, np.nan)

    rng = np.random.default_rng(1000 + subject_index)

    # --- Calibration episodes -------------------------------------------------
    all_x, all_u, all_y = [], [], []
    episode_states, episode_inputs = [], []

    for episode in range(cfg.calibration_episodes):
        u_cal = make_calibration_input(cfg.test_steps, cfg, rng)
        initial_basin = -1.0 if episode % 2 == 0 else 1.0
        x_cal = simulate_subject(p, u_cal, cfg, initial_basin, rng)

        x_now, u_now = x_cal[:-1], u_cal[:-1]
        if cfg.target_mode.lower() == "truedrift":
            y_now = true_drift(x_now, u_now, p)
        else:
            y_now = np.diff(x_cal) / cfg.dt

        all_x.append(x_now)
        all_u.append(u_now)
        all_y.append(y_now)
        episode_states.append(x_cal)
        episode_inputs.append(u_cal)

    all_x = np.concatenate(all_x)
    all_u = np.concatenate(all_u)
    all_y = np.concatenate(all_y)

    # --- Ground-truth structure ----------------------------------------------
    x_grid = cfg.x_grid
    true_v = true_potential(x_grid, p)
    true_v = true_v - true_v.min()
    true_features = landscape_features(x_grid, true_v)

    flow_x, flow_u = np.meshgrid(x_grid, cfg.flow_input_grid, indexing="ij")
    true_flow = true_drift(flow_x, flow_u, p)

    truth_drift = true_drift_fn(p)
    initial_state = initial_state_from_params(p)

    true_invariant = simulate_ensemble(
        truth_drift,
        np.zeros(cfg.invariant_steps),
        cfg,
        initial_state,
        cfg.distribution_replicates,
        p.sigma,
        31000 + subject_index,
    )
    true_invariant_prob = empirical_probability(
        true_invariant[cfg.invariant_burn_in :, :],
        cfg.density_edges,
        cfg.density_pseudo_count,
    )

    u_response = cfg.pulse_input(cfg.test_pulse_amplitude)
    true_response_ensemble = simulate_ensemble(
        truth_drift,
        u_response,
        cfg,
        initial_state,
        cfg.distribution_replicates,
        p.sigma,
        32000 + subject_index,
    )

    # The reference dose--transition curve does not depend on the calibration
    # size, so it is simulated once. Re-drawing it per calibration size would
    # make part of every learning curve Monte-Carlo jitter in the reference.
    true_prob_curve = transition_probability_curve(
        truth_drift,
        cfg,
        initial_state,
        true_features.saddle_x,
        p.sigma,
        cfg.distribution_replicates,
        61000 + subject_index,
    )

    # --- Calibration-set composition -----------------------------------------
    coverage[0] = np.mean(all_x < true_features.saddle_x - 0.15)
    coverage[1] = np.mean(np.abs(all_x - true_features.saddle_x) <= 0.15)
    coverage[2] = np.mean(all_x > true_features.saddle_x + 0.15)
    coverage[3] = np.mean(np.abs(all_u) > 0.15)
    coverage[4] = count_basin_transitions(all_x, true_features.saddle_x)

    # --- Reference trajectories ----------------------------------------------
    u_passive = np.zeros(cfg.test_steps)
    u_test = cfg.pulse_input(cfg.test_pulse_amplitude)

    x_passive_true = simulate_subject(
        p, u_passive, cfg, initial_state, np.random.default_rng(10000 + subject_index)
    )
    x_perturb_true = simulate_subject(
        p, u_test, cfg, initial_state, np.random.default_rng(20000 + subject_index)
    )
    true_transition = basin_transition(x_perturb_true, true_features.saddle_x)

    # Shared deterministic reference curve.
    true_curve = transition_curve_deterministic(
        truth_drift, cfg, initial_state, true_features.saddle_x
    )

    example = None

    for ci, requested in enumerate(cfg.calibration_sizes):
        n_cal = min(requested, all_x.size)

        idx = sample_calibration_indices(
            all_u,
            n_cal,
            cfg.cal_perturbed_fraction,
            5000 + 100 * subject_index + (ci + 1),
        )

        cal = standardise(
            Dataset(
                x=all_x[idx],
                u=all_u[idx],
                y=all_y[idx],
                subject=np.zeros(idx.size, dtype=int),
            ),
            scale,
        )

        e_passive = adapt_embedding(passive_model, cal, opt)
        e_perturb = adapt_embedding(perturb_model, cal, opt)
        cubic_beta = fit_cubic_scratch_model(cal)

        drift_fns = [
            cubic_drift_fn(cubic_beta),
            shared_drift_fn(passive_model, e_passive, scale),
            shared_drift_fn(perturb_model, e_perturb, scale),
        ]

        pred_passive = [rollout(f, initial_state, u_passive, cfg) for f in drift_fns]
        pred_perturb = [rollout(f, initial_state, u_test, cfg) for f in drift_fns]
        potentials = [recover_potential(f, x_grid) for f in drift_fns]
        estimated_features = [landscape_features(x_grid, v) for v in potentials]

        pred_curves = [
            transition_curve_deterministic(f, cfg, initial_state, feat.saddle_x)
            if feat.is_bistable else np.full(cfg.test_amplitude_grid.size, np.nan)
            for f, feat in zip(drift_fns, estimated_features)
        ]

        model_invariants, model_responses, model_prob_curves = [], [], []

        for m, drift_fn in enumerate(drift_fns):
            metrics["passiveRMSE"][ci, m] = np.sqrt(
                np.mean((pred_passive[m] - x_passive_true) ** 2)
            )
            metrics["perturbRMSE"][ci, m] = np.sqrt(
                np.mean((pred_perturb[m] - x_perturb_true) ** 2)
            )

            v_hat = align_potential_offset_only(potentials[m])
            metrics["landscapeRMSE"][ci, m] = np.sqrt(np.mean((v_hat - true_v) ** 2))
            metrics["barrierError"][ci, m] = abs(
                estimated_features[m].barrier - true_features.barrier
            )
            metrics["attractorError"][ci, m] = matched_attractor_error(
                estimated_features[m].minima_x, true_features.minima_x
            )

            pred_transition = basin_transition(pred_perturb[m], true_features.saddle_x)
            metrics["transitionCorrect"][ci, m] = float(
                true_transition == pred_transition
            )
            metrics["amplitudeCurveRMSEModelSaddle"][ci, m] = np.sqrt(
                np.mean((pred_curves[m] - true_curve) ** 2)
            )

            # 1) Dynamical structure: controlled vector-field error.
            estimated_flow = np.reshape(
                drift_fn(flow_x.ravel(), flow_u.ravel()), true_flow.shape
            )
            metrics["flowRMSE"][ci, m] = np.sqrt(
                np.mean((estimated_flow - true_flow) ** 2)
            )

            # 2) Dynamical structure: invariant-occupancy divergence.
            model_invariant = simulate_ensemble(
                drift_fn,
                np.zeros(cfg.invariant_steps),
                cfg,
                initial_state,
                cfg.distribution_replicates,
                p.sigma,
                31000 + subject_index,  # shared innovations with the truth
            )
            metrics["invariantJS"][ci, m] = js_divergence(
                true_invariant_prob,
                empirical_probability(
                    model_invariant[cfg.invariant_burn_in :, :],
                    cfg.density_edges,
                    cfg.density_pseudo_count,
                ),
            )

            # 3) Perturbational response: distributional divergence over time.
            model_response = simulate_ensemble(
                drift_fn,
                u_response,
                cfg,
                initial_state,
                cfg.distribution_replicates,
                p.sigma,
                32000 + subject_index,  # shared innovations with the truth
            )
            metrics["responseJS"][ci, m] = mean_response_js_divergence(
                true_response_ensemble,
                model_response,
                cfg.density_edges,
                cfg.response_eval_stride,
                cfg.density_pseudo_count,
            )

            # 4) Intervention outcome: stochastic dose--transition curve.
            if estimated_features[m].is_bistable:
                model_relative_curve = transition_probability_curve(
                    drift_fn, cfg, initial_state, estimated_features[m].saddle_x,
                    p.sigma, cfg.distribution_replicates, 61000 + subject_index,
                )
                metrics["transitionProbabilityRMSEModelSaddle"][ci, m] = np.sqrt(
                    np.mean((model_relative_curve - true_prob_curve) ** 2)
                )
            model_prob_curve = transition_probability_curve(
                drift_fn,
                cfg,
                initial_state,
                true_features.saddle_x,
                p.sigma,
                cfg.distribution_replicates,
                61000 + subject_index,  # shared innovations with the truth
            )
            metrics["transitionProbabilityRMSECommonSaddle"][ci, m] = np.sqrt(
                np.mean((model_prob_curve - true_prob_curve) ** 2)
            )

            # Diagnostic: how close this model came to the imposed state
            # boundary. Rollouts and ensembles are clipped at cfg.state_clip,
            # so a value at the clip means the boundary shaped the result.
            metrics["maxAbsState"][ci, m] = float(
                max(
                    np.max(np.abs(pred_passive[m])),
                    np.max(np.abs(pred_perturb[m])),
                    np.max(np.abs(model_response)),
                    np.max(np.abs(model_invariant)),
                )
            )

            if keep_example and ci == cfg.example_calibration_index:
                model_invariants.append(model_invariant)
                model_responses.append(model_response)
                model_prob_curves.append(model_prob_curve)

        if keep_example and ci == cfg.example_calibration_index:
            example = {
                "params": p,
                "calibrationX": episode_states[0],
                "calibrationU": episode_inputs[0],
                "truePerturb": x_perturb_true,
                "predPerturb": pred_perturb,
                "Vtrue": true_v,
                "Vset": [align_potential_offset_only(v) for v in potentials],
                "trueFeatures": true_features,
                "nCal": n_cal,
                "trueCurve": true_curve,
                "predCurves": pred_curves,
                "trueFlow": true_flow,
                "flowX": flow_x,
                "flowU": flow_u,
                "trueInvariantProb": true_invariant_prob,
                "trueResponseEnsemble": true_response_ensemble,
                "trueProbCurve": true_prob_curve,
                "modelProbCurves": model_prob_curves,
                "modelInvariantSamples": model_invariants,
                "flowPredictions": [
                    np.reshape(f(flow_x.ravel(), flow_u.ravel()), flow_x.shape)
                    for f in drift_fns
                ],
                "modelResponseEnsembles": model_responses,
                "selectedX": all_x[idx],
                "selectedU": all_u[idx],
                "selectedIdx": idx,
            }

    return metrics, coverage, example


def monte_carlo_floor(
    cfg: Config, params: list[SubjectParams], n_subjects: int = 8
) -> dict:
    """Scores a perfect model attains on the stochastic metrics.

    Both sides use the true drift, so any non-zero value is finite-sample noise
    rather than model error. ``shared`` repeats the truth with the innovations
    the evaluation actually uses; ``independent`` draws fresh innovations and so
    measures what those metrics would report for a perfect model if truth and
    model were simulated separately.
    """
    out = {k: [] for k in ("responseJSShared", "responseJSIndependent",
                           "doseRMSEShared", "doseRMSEIndependent")}
    u = cfg.pulse_input(cfg.test_pulse_amplitude)

    for si, p in enumerate(params[:n_subjects]):
        truth = true_drift_fn(p)
        x0 = initial_state_from_params(p)
        v = true_potential(cfg.x_grid, p)
        saddle = landscape_features(cfg.x_grid, v - v.min()).saddle_x

        reference = simulate_ensemble(truth, u, cfg, x0, cfg.distribution_replicates,
                                     p.sigma, 32000 + si)
        for tag, seed in (("Shared", 32000 + si), ("Independent", 99000 + si)):
            repeat = simulate_ensemble(truth, u, cfg, x0, cfg.distribution_replicates,
                                       p.sigma, seed)
            out[f"responseJS{tag}"].append(
                mean_response_js_divergence(reference, repeat, cfg.density_edges,
                                            cfg.response_eval_stride,
                                            cfg.density_pseudo_count)
            )

        ref_curve = transition_probability_curve(
            truth, cfg, x0, saddle, p.sigma, cfg.distribution_replicates, 61000 + si)
        for tag, seed in (("Shared", 61000 + si), ("Independent", 88000 + si)):
            curve = transition_probability_curve(
                truth, cfg, x0, saddle, p.sigma, cfg.distribution_replicates, seed)
            out[f"doseRMSE{tag}"].append(
                float(np.sqrt(np.mean((np.asarray(ref_curve) - np.asarray(curve)) ** 2)))
            )

    return {k: (float(np.mean(v)), float(np.std(v))) for k, v in out.items()}


def summarise_results(results: dict, cfg: Config) -> dict:
    """Mean and standard error across held-out subjects, per calibration size."""
    summary = {}
    for name in METRIC_NAMES:
        values = results[name]  # (n_subjects, n_calibration_sizes, n_models)
        counts = np.sum(np.isfinite(values), axis=0)
        means = np.divide(
            np.nansum(values, axis=0), counts,
            out=np.full(values.shape[1:], np.nan), where=counts > 0,
        )
        squared = np.nansum((values - means[None, :, :]) ** 2, axis=0)
        sem = np.sqrt(np.divide(
            squared, counts * (counts - 1),
            out=np.full_like(squared, np.nan), where=counts > 1,
        ))
        summary[name] = {
            "mean": means,
            "sem": sem,
        }
    return summary
