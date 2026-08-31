"""Configuration for the perturbational foundation-model simulation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Config:
    """Simulation settings."""

    seed: int = 11

    dt: float = 0.02

    train_subjects: int = 60
    test_subjects: int = 20
    train_steps_per_subject: int = 1500
    test_steps: int = 1400
    burn_in: int = 150

    calibration_sizes: tuple[int, ...] = (2, 3, 5, 8, 12, 20, 35, 60, 100)
    learning_curve_yscale: str = "log"
    calibration_episodes: int = 2

    # Ground-truth subject family.
    a_range: tuple[float, float] = (1.00, 1.00)
    c_range: tuple[float, float] = (0.75, 1.30)
    d_range: tuple[float, float] = (-0.28, 0.28)
    b_range: tuple[float, float] = (0.60, 1.45)
    sigma_range: tuple[float, float] = (0.14, 0.14)

    # Population-pretraining perturbations.
    train_pulse_probability: float = 0.035
    train_pulse_duration_range: tuple[int, int] = (35, 90)
    train_pulse_amplitude_range: tuple[float, float] = (-2.30, 2.30)
    train_input_noise_sd: float = 0.10
    randomise_initial_basin: bool = True

    # New-subject calibration perturbations.
    cal_pulse_count: int = 18
    cal_pulse_duration_range: tuple[int, int] = (30, 75)
    cal_pulse_amplitude_range: tuple[float, float] = (-1.70, 1.70)
    cal_input_noise_sd: float = 0.08
    cal_perturbed_fraction: float = 0.50

    # Held-out test perturbation.
    test_pulse_start: int = 430  # one-based; `pulse_window` converts
    test_pulse_duration: int = 120
    test_pulse_amplitude: float = 2.25

    # Distributional and structural evaluation.
    invariant_steps: int = 5000
    invariant_burn_in: int = 1000
    distribution_replicates: int = 40
    response_eval_stride: int = 20
    density_pseudo_count: float = 1e-8

    target_mode: str = "trueDrift"

    state_clip: float = 2.3
    example_subject: int = 0
    example_calibration_index: int = 5  # selects 20 calibration samples

    @property
    def test_amplitude_grid(self) -> np.ndarray:
        return np.linspace(0.5, 2.8, 12)

    @property
    def flow_input_grid(self) -> np.ndarray:
        return np.unique(
            np.array([-2.0, -1.0, 0.0, 1.0, 2.0, self.test_pulse_amplitude])
        )

    @property
    def density_edges(self) -> np.ndarray:
        return np.linspace(-2.3, 2.3, 81)

    @property
    def x_grid(self) -> np.ndarray:
        return np.linspace(-1.9, 1.9, 450)

    def pulse_window(self, n_steps: int) -> slice:
        """Return the held-out pulse range as a half-open slice."""
        start = self.test_pulse_start - 1
        stop = min(n_steps, start + self.test_pulse_duration)
        return slice(start, stop)

    def pulse_input(self, amplitude: float, n_steps: int | None = None) -> np.ndarray:
        """Zero input with a single held-out pulse of the given amplitude."""
        n = self.test_steps if n_steps is None else n_steps
        u = np.zeros(n)
        u[self.pulse_window(n)] = amplitude
        return u


@dataclass(frozen=True)
class NetConfig:
    """Shared drift-network dimensions."""

    hidden1: int = 24
    hidden2: int = 24
    embedding_dim: int = 3

    @property
    def input_dim(self) -> int:
        return 2 + self.embedding_dim


@dataclass(frozen=True)
class OptConfig:
    """Optimisation settings."""

    pretrain_epochs: int = 55
    pretrain_batch_size: int = 2048
    pretrain_learning_rate: float = 2e-3
    adaptation_epochs: int = 120
    embedding_learning_rate: float = 3e-2
    full_adaptation_learning_rate: float = 3e-3
    weight_decay: float = 1e-5
    embedding_penalty: float = 1e-3
    gradient_clip: float = 5.0


@dataclass(frozen=True)
class Palette:
    """Plotting colours, labels, and model order."""

    ground_truth: tuple[float, float, float] = (0.173, 0.188, 0.220)  # #2C3038
    scratch: tuple[float, float, float] = (0.000, 0.447, 0.698)  # blue
    passive: tuple[float, float, float] = (0.835, 0.369, 0.000)  # orange
    perturb: tuple[float, float, float] = (0.000, 0.620, 0.451)  # green
    pulse_shade: tuple[float, float, float] = (0.878, 0.635, 0.235)  # #E0A23C
    population_alpha: float = 0.55

    #: Model order used throughout: scratch, passive pretraining, perturbational.
    model_names: tuple[str, str, str] = (
        "CubicScratch",
        "PassivePretraining",
        "PerturbationalPretraining",
    )
    model_labels: tuple[str, str, str] = (
        "Scratch",
        "Passive pretraining",
        "Perturbational pretraining",
    )

    @property
    def model_colors(self):
        return (self.scratch, self.passive, self.perturb)


DEFAULT_CONFIG = Config()
DEFAULT_NET = NetConfig()
DEFAULT_OPT = OptConfig()
DEFAULT_PALETTE = Palette()

#: Metric names, in the order used by `summarise_results` and the summary CSV.
METRIC_NAMES: tuple[str, ...] = (
    "passiveRMSE",
    "perturbRMSE",
    "landscapeRMSE",
    "barrierError",
    "attractorError",
    "transitionCorrect",
    "amplitudeCurveRMSEModelSaddle",
    "flowRMSE",
    "invariantJS",
    "responseJS",
    "transitionProbabilityRMSEModelSaddle",
    "transitionProbabilityRMSECommonSaddle",
    "maxAbsState",
)
