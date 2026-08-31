#!/usr/bin/env python3
"""Numerical and behavioral checks for the simulation package."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from perturbsim.config import Config, NetConfig, OptConfig
from perturbsim.dynamics import (
    Dataset,
    Scale,
    SubjectParams,
    is_bistable,
    sample_subject_parameters,
    generate_population_dataset_regime,
    initial_state_from_params,
    rollout,
    simulate_ensemble,
    standardise,
    true_drift,
    true_drift_fn,
    true_potential,
)
from perturbsim.metrics import (
    basin_transition,
    cumulative_trapezoid,
    ensemble_band,
    js_divergence,
    kl_divergence,
    landscape_features,
    matched_attractor_error,
    recover_potential,
)
from perturbsim.model import (
    _shared_loss_and_gradients,
    adapt_embedding,
    adapt_full_model,
    fit_cubic_scratch_model,
    initialise_shared_model,
    shared_drift_fn,
)

CFG = Config()
FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "ok  " if condition else "FAIL"
    print(f"[{status}] {name}{'  ' + detail if detail else ''}")
    if not condition:
        FAILURES.append(name)


def test_potential_matches_drift() -> None:
    """f(x, 0) = -dV/dx, so integrating the drift must recover the potential."""
    p = SubjectParams(a=1.0, c=1.1, d=0.12, b=1.0, sigma=0.14)
    x = CFG.x_grid

    v_true = true_potential(x, p)
    v_true = v_true - v_true.min()
    v_recovered = recover_potential(true_drift_fn(p), x)

    err = np.max(np.abs(v_recovered - v_true))
    check("potential recovered from drift", err < 1e-4, f"max error {err:.2e}")


def test_cumulative_trapezoid() -> None:
    x = np.linspace(0, 3, 501)
    y = np.sin(x)
    integral = cumulative_trapezoid(y, x)
    exact = 1 - np.cos(x)
    check(
        "cumulative_trapezoid matches the analytic integral",
        np.max(np.abs(integral - exact)) < 1e-5,
    )
    check("cumulative_trapezoid starts at zero", integral[0] == 0.0)


def test_landscape_features() -> None:
    """A symmetric double well has minima at +-sqrt(c/a) and a saddle at zero."""
    p = SubjectParams(a=1.0, c=1.0, d=0.0, b=1.0, sigma=0.14)
    x = CFG.x_grid
    v = true_potential(x, p)
    v = v - v.min()

    features = landscape_features(x, v)
    expected = np.array([-1.0, 1.0])

    check(
        "landscape_features locates both minima",
        np.allclose(np.sort(features.minima_x), expected, atol=5e-3),
        f"found {np.sort(features.minima_x)}",
    )
    check("landscape_features locates the saddle", abs(features.saddle_x) < 5e-3)
    check(
        "barrier height matches the analytic value",
        abs(features.barrier - 0.25) < 1e-3,
        f"got {features.barrier:.4f}",
    )


def test_matched_attractor_error() -> None:
    check(
        "attractor error is zero for identical sets",
        matched_attractor_error(np.array([-1.0, 1.0]), np.array([1.0, -1.0])) == 0.0,
    )
    check(
        "attractor error penalises a missing attractor",
        matched_attractor_error(np.array([-1.0]), np.array([-1.0, 1.0])) == 0.5,
    )


def test_divergences() -> None:
    rng = np.random.default_rng(0)
    p = rng.random(40)
    p /= p.sum()
    q = rng.random(40)
    q /= q.sum()

    check("KL of a distribution with itself is zero", abs(kl_divergence(p, p)) < 1e-12)
    check("JS of a distribution with itself is zero", abs(js_divergence(p, p)) < 1e-12)
    check("JS is symmetric", abs(js_divergence(p, q) - js_divergence(q, p)) < 1e-12)
    check("JS is bounded by log 2", js_divergence(p, q) <= np.log(2) + 1e-12)


def test_percentile_convention() -> None:
    """Hazen percentiles place sorted values at ``(i - 0.5) / n``."""
    x = np.arange(1.0, 11.0).reshape(1, 10)  # 1..10
    _, lo, hi = ensemble_band(x)
    # Hazen percentiles clamp at the ends: p5 of 1..10 is 1, p95 is 10.
    check("5th percentile clamps to the lowest value", abs(lo[0] - 1.0) < 1e-12, f"got {lo[0]}")
    check("95th percentile clamps to the highest value", abs(hi[0] - 10.0) < 1e-12, f"got {hi[0]}")

    # The Hazen 25th percentile interpolates halfway between the first two values.
    y = np.arange(1.0, 5.0).reshape(1, 4)  # p25 of 1..4 is 1.5
    q = np.percentile(y, 25, axis=1, method="hazen")
    check("interior percentile interpolates", abs(q[0] - 1.5) < 1e-12, f"got {q[0]}")


def test_shared_gradients() -> None:
    """Hand-written gradients must agree with central finite differences."""
    rng = np.random.default_rng(7)
    net = NetConfig()
    opt = OptConfig()
    model = initialise_shared_model(net, 4, rng)

    n = 32
    x = rng.standard_normal(n)
    u = rng.standard_normal(n)
    y = rng.standard_normal(n)
    subjects = rng.integers(0, 4, n)

    _, grads = _shared_loss_and_gradients(model, x, u, y, subjects, opt)

    step = 1e-6
    worst = 0.0
    for name in ("W1", "b1", "W2", "b2", "W3", "b3", "E"):
        param = getattr(model, name)
        flat = np.asarray(param).ravel()
        for k in rng.choice(flat.size, size=min(6, flat.size), replace=False):
            original = flat[k]

            flat[k] = original + step
            loss_plus, _ = _shared_loss_and_gradients(model, x, u, y, subjects, opt)
            flat[k] = original - step
            loss_minus, _ = _shared_loss_and_gradients(model, x, u, y, subjects, opt)
            flat[k] = original

            numeric = (loss_plus - loss_minus) / (2 * step)
            analytic = np.asarray(grads[name]).ravel()[k]
            worst = max(worst, abs(numeric - analytic))

    check("analytic gradients match finite differences", worst < 1e-6, f"max |diff| {worst:.2e}")


def test_embedding_adaptation_recovers_a_known_subject() -> None:
    """Adapting on data generated at a known embedding should move towards it."""
    rng = np.random.default_rng(3)
    net = NetConfig()
    opt = OptConfig()
    model = initialise_shared_model(net, 1, rng)
    scale = Scale(0.0, 1.0, 0.0, 1.0, 0.0, 1.0)

    target = np.array([[0.8], [-0.4], [0.2]])
    x = rng.uniform(-1.5, 1.5, 200)
    u = rng.uniform(-2.0, 2.0, 200)
    y = shared_drift_fn(model, target, scale)(x, u)

    cal = standardise(Dataset(x=x, u=u, y=y, subject=np.zeros(x.size, int)), scale)
    estimate = adapt_embedding(model, cal, opt)

    residual = np.sqrt(
        np.mean((shared_drift_fn(model, estimate, scale)(x, u) - y) ** 2)
    )
    check(
        "embedding adaptation fits the generating drift",
        residual < 1e-2,
        f"residual RMSE {residual:.2e}",
    )


def test_full_adaptation_updates_a_copy() -> None:
    """Full adaptation should improve fit without changing the source model."""
    rng = np.random.default_rng(17)
    model = initialise_shared_model(NetConfig(), 1, rng)
    original = model.W1.copy()
    scale = Scale(0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
    x = rng.uniform(-1.5, 1.5, 200)
    u = rng.uniform(-2.0, 2.0, 200)
    p = SubjectParams(a=1.0, c=1.1, d=0.1, b=1.2, sigma=0.14)
    y = true_drift(x, u, p)
    cal = standardise(Dataset(x=x, u=u, y=y, subject=np.zeros(x.size, int)), scale)

    initial_e = np.zeros(model.embedding_dim)
    initial_error = np.sqrt(
        np.mean((shared_drift_fn(model, initial_e, scale)(x, u) - y) ** 2)
    )
    adapted, embedding = adapt_full_model(model, cal, OptConfig())
    adapted_error = np.sqrt(
        np.mean((shared_drift_fn(adapted, embedding, scale)(x, u) - y) ** 2)
    )

    check(
        "full adaptation improves calibration fit",
        adapted_error < initial_error,
        f"RMSE {initial_error:.2e} to {adapted_error:.2e}",
    )
    check("full adaptation preserves the source model", np.array_equal(model.W1, original))


def test_cubic_baseline_recovers_the_true_coefficients() -> None:
    """With unit standardisation the cubic fit should return the true parameters."""
    p = SubjectParams(a=1.0, c=1.05, d=0.1, b=1.2, sigma=0.14)
    scale = Scale(0.0, 1.0, 0.0, 1.0, 0.0, 1.0)

    rng = np.random.default_rng(11)
    x = rng.uniform(-1.8, 1.8, 500)
    u = rng.uniform(-2.0, 2.0, 500)
    y = true_drift(x, u, p)

    cal = standardise(Dataset(x=x, u=u, y=y, subject=np.zeros(x.size, int)), scale)
    beta = fit_cubic_scratch_model(cal)

    expected = np.array([p.d, p.c, 0.0, -p.a, p.b])  # [1, x, x^2, x^3, u]
    err = np.max(np.abs(beta - expected))
    check("cubic baseline recovers the true drift", err < 1e-3, f"max error {err:.2e}")


def test_identification_control_datasets() -> None:
    """Ablation datasets should isolate state coverage and input excitation."""
    cfg = Config(train_input_noise_sd=0.0)
    params = [SubjectParams(1.0, 1.0, 0.0, 1.0, 0.05) for _ in range(4)]
    state_data = generate_population_dataset_regime(
        params, 500, cfg, "state_coverage", np.random.default_rng(21)
    )
    input_data = generate_population_dataset_regime(
        params, 500, cfg, "input_excitation", np.random.default_rng(21)
    )

    check("state-coverage control has zero model input", np.all(state_data.u == 0))
    check("input-excitation control has varied model input", np.any(input_data.u != 0))
    expected = np.concatenate([
        true_drift(state_data.x[state_data.subject == s], 0.0, p)
        for s, p in enumerate(params)
    ])
    check("state-coverage targets are unforced drift", np.allclose(state_data.y, expected))


def test_rollout_matches_euler_maruyama_without_noise() -> None:
    p = SubjectParams(a=1.0, c=1.0, d=0.0, b=1.0, sigma=0.0)
    u = np.zeros(300)
    x0 = initial_state_from_params(p)

    trajectory = rollout(true_drift_fn(p), x0, u, CFG)
    ensemble = simulate_ensemble(true_drift_fn(p), u, CFG, x0, 3, 0.0, seed=1)

    check(
        "noise-free ensemble equals the deterministic rollout",
        np.allclose(ensemble[:, 0], trajectory),
    )
    check("unforced trajectory stays in its basin", not basin_transition(trajectory, 0.0))


def test_strong_pulse_causes_a_transition() -> None:
    """The held-out pulse must be able to drive the true system across the barrier."""
    p = SubjectParams(a=1.0, c=1.0, d=0.0, b=1.2, sigma=0.0)
    x0 = initial_state_from_params(p)

    weak = rollout(true_drift_fn(p), x0, CFG.pulse_input(0.2), CFG)
    strong = rollout(true_drift_fn(p), x0, CFG.pulse_input(CFG.test_pulse_amplitude), CFG)

    check("a weak pulse leaves the system in place", not basin_transition(weak, 0.0))
    check("the held-out pulse drives a transition", basin_transition(strong, 0.0))


def test_sampled_population_is_bistable() -> None:
    """Every drawn participant has two wells; the known bad corner is rejected."""
    cfg = Config()
    params = sample_subject_parameters(2000, cfg, np.random.default_rng(7))
    check(
        "every sampled participant is bistable",
        all(is_bistable(p) for p in params),
        f"{sum(not is_bistable(p) for p in params)} of {len(params)} monostable",
    )

    # The draw that the shipped seeds used to produce: 27 d^2 = 2.02 > 4 c^3 = 1.84.
    monostable = SubjectParams(a=1.0, c=0.772, d=0.274, b=1.0, sigma=0.14)
    check(
        "the known monostable corner is rejected",
        not is_bistable(monostable),
        "predicate accepted a single-well parameter set",
    )

    minima = landscape_features(
        cfg.x_grid, true_potential(cfg.x_grid, monostable)
    ).minima_x
    check(
        "the rejected corner really has one well",
        minima.size == 1,
        f"found {minima.size} minima",
    )


def main() -> int:
    for test in (
        test_potential_matches_drift,
        test_cumulative_trapezoid,
        test_landscape_features,
        test_sampled_population_is_bistable,
        test_matched_attractor_error,
        test_divergences,
        test_percentile_convention,
        test_shared_gradients,
        test_embedding_adaptation_recovers_a_known_subject,
        test_full_adaptation_updates_a_copy,
        test_cubic_baseline_recovers_the_true_coefficients,
        test_identification_control_datasets,
        test_rollout_matches_euler_maruyama_without_noise,
        test_strong_pulse_causes_a_transition,
    ):
        test()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) failed: {', '.join(FAILURES)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
