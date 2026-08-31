"""Create publication figures and export them in vector and raster formats."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
import numpy as np

from .config import Config, Palette
from .dynamics import (
    make_calibration_input, simulate_subject, true_drift, true_potential,
)
from .metrics import ensemble_band

CM = 1 / 2.54
FIGURE_SIZE = (18.3 * CM, 13.6 * CM)
#: Figure 4 is six plain panels and reads better with more height
METRICS_FIGURE_SIZE = (18.3 * CM, 15.6 * CM)
PANEL_BG = "#FCFDFE"
PANEL_EDGE = "#8C949D"
#: Opacity for overlapping pretraining series.
SERIES_ALPHA = 0.72
#: x ticks for the learning curves, on a log axis spanning 2 to 100 samples
LEARNING_CURVE_TICKS = (2, 5, 10, 20, 50, 100)
#: Shared state-axis ticks and limits.
STATE_TICKS = (-2, -1, 0, 1, 2)
STATE_LIM = (-2.08, 2.08)
#: Fixed energy and response axes.
ENERGY_TICKS, ENERGY_LIM = (0, 1, 2, 3), (-0.12, 3.12)
RESPONSE_TICKS, RESPONSE_LIM = (-2, -1, 0, 1, 2), (-2.15, 2.15)


def apply_style() -> None:
    """Apply global publication styling to Matplotlib."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Arial",
                "Helvetica",
                "Liberation Sans",
                "Nimbus Sans",
                "DejaVu Sans",
            ],
            "font.size": 8.5,
            "axes.linewidth": 0.9,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.edgecolor": PANEL_EDGE,
            "axes.facecolor": PANEL_BG,
            "axes.labelsize": 9.5,
            "axes.titlesize": 9.5,
            "axes.titleweight": "bold",
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "xtick.major.width": 0.9,
            "ytick.major.width": 0.9,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "legend.frameon": False,
            "legend.fontsize": 8.5,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "mathtext.default": "it",
            "pdf.fonttype": 42,
        }
    )


def export_figure(fig, output_dir: Path, base_name: str, dpi: int = 400) -> None:
    """Write vector and raster versions of a figure."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig.savefig(output_dir / f"{base_name}.pdf")
    fig.savefig(output_dir / f"{base_name}.png", dpi=dpi)
    try:
        # LZW keeps the TIFF a tenth of the size of the uncompressed default.
        fig.savefig(
            output_dir / f"{base_name}.tif",
            dpi=dpi,
            pil_kwargs={"compression": "tiff_lzw"},
        )
    except Exception as exc:  # Pillow missing or TIFF writer unavailable
        print(f"  TIFF export skipped for {base_name}: {exc}")
    plt.close(fig)


#: Nice step sizes to choose from when forcing a fixed number of y ticks.
_NICE_STEPS = (0.002, 0.005, 0.01, 0.02, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.5)


def fixed_yticks(ax, data_max: float, n_ticks: int = 6) -> None:
    """Set ``n_ticks`` round-valued y ticks from zero through ``data_max``."""
    intervals = n_ticks - 1
    step = next(s for s in _NICE_STEPS if s * intervals >= data_max)
    ax.set_ylim(0, step * intervals)
    ax.set_yticks(np.linspace(0, step * intervals, n_ticks))


#: Within-decade tick subdivisions tried when a log panel spans few decades.
_LOG_SUBS = ((1.0,), (1.0, 3.0), (1.0, 2.0, 5.0))


def _log_ytick_candidates(lo: float, hi: float):
    """Yield candidate log tick sets, from sparse decades to fine subdivisions.

    Panels span anywhere from one to nine decades, so no single locator gives a
    comparable number of labels everywhere: decade ticks leave a one-decade
    panel with a single label, and a nine-decade panel with nine. Candidates
    vary both the decade stride and the within-decade steps, and the caller
    keeps whichever lands closest to the requested count.
    """
    lo_exp = int(np.floor(np.log10(lo)))
    hi_exp = int(np.ceil(np.log10(hi)))

    for stride in (1, 2, 3, 4):
        for offset in range(stride):
            subs_options = _LOG_SUBS if stride == 1 else ((1.0,),)
            for subs in subs_options:
                values = sorted(
                    s * 10.0**e
                    for e in range(lo_exp + offset, hi_exp + 1, stride)
                    for s in subs
                )
                kept = [v for v in values if lo <= v <= hi]
                if len(kept) >= 2:
                    yield kept, stride, len(subs)


def _format_log_tick(value: float, plain: bool) -> str:
    """Label a log tick as a trimmed decimal or as a power of ten."""
    if plain:
        return f"{value:.10f}".rstrip("0").rstrip(".")
    exponent = int(np.floor(np.log10(value) + 1e-9))
    mantissa = value / 10.0**exponent
    power = f"10^{{{exponent}}}"
    # \mathdefault keeps the digits in the figure's text font, as matplotlib's
    # own log formatter does; bare mathtext would switch to the math font.
    if abs(mantissa - 1.0) < 1e-6:
        return f"$\\mathdefault{{{power}}}$"
    return f"$\\mathdefault{{{mantissa:g}\\times{power}}}$"


def log_yticks(ax, n_ticks: int = 5) -> None:
    """Give a log y axis close to ``n_ticks`` labelled ticks.

    Keeps power-of-ten labels on wide-spanning panels and switches to plain
    decimals on narrow ones, where subdivided ticks would otherwise read as
    ``5\\times10^{-2}`` instead of ``0.05``.
    """
    lo, hi = ax.get_ylim()
    if not (lo > 0 and hi > lo):
        return

    best = min(
        _log_ytick_candidates(lo, hi),
        key=lambda c: (abs(len(c[0]) - n_ticks), c[1], c[2]),
        default=None,
    )
    if best is None:
        return

    ticks = best[0]
    plain = max(abs(np.log10(t)) for t in ticks) <= 2.0
    ax.set_yticks(ticks)
    ax.set_yticklabels([_format_log_tick(t, plain) for t in ticks])
    ax.set_ylim(lo, hi)


def panel_title(ax, letter: str, title: str) -> None:
    """Place the panel anchor and centered title on one fixed baseline."""
    heading_y = 1.075
    ax.set_title("")
    ax._panel_anchor = ax.text(
        -0.16, heading_y, letter, transform=ax.transAxes,
        ha="left", va="bottom", fontsize=12.0, fontweight="bold",
        color="black", clip_on=False,
    )
    ax._panel_heading = ax.text(
        0.5, heading_y, title, transform=ax.transAxes,
        ha="center", va="bottom", fontsize=9.5, fontweight="bold",
        color="black", clip_on=False,
    )


def make_main_figure(
    cfg: Config,
    train_params,
    example: dict,
    palette: Palette,
    output_dir: Path,
) -> None:
    """Figure 3: mechanistic validation in a single held-out system."""
    fig = plt.figure(figsize=(18.3 * CM, 15.6 * CM), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, height_ratios=(1.24, 1.28))
    axes = np.empty((2, 3), dtype=object)
    axes[0, 0] = fig.add_subplot(grid[0, 0])
    calibration_grid = grid[0, 1].subgridspec(
        2, 1, height_ratios=(2.15, 1.0), hspace=0.035
    )
    axes[0, 1] = fig.add_subplot(calibration_grid[0, 0])
    ax_calibration_input = fig.add_subplot(
        calibration_grid[1, 0], sharex=axes[0, 1]
    )
    axes[0, 2] = fig.add_subplot(grid[0, 2])
    for column in range(3):
        axes[1, column] = fig.add_subplot(grid[1, column])
    fig.set_constrained_layout_pads(
        w_pad=6 / 72, h_pad=8 / 72, wspace=0.20, hspace=0.34
    )
    x_grid = cfg.x_grid

    # A: population family --------------------------------------------------
    ax = axes[0, 0]
    for k in np.unique(np.linspace(0, len(train_params) - 1, 7).round().astype(int)):
        v = true_potential(x_grid, train_params[k])
        ax.plot(x_grid, v - v.min(), color=(0.55, 0.55, 0.55), lw=1.0)
    ax.set_xlabel("State, $x$")
    ax.set_ylabel("Potential, $V(x)$")
    ax.set_xticks(STATE_TICKS)
    ax.set_xlim(*STATE_LIM)
    panel_title(ax, "A", "Dynamical Family")

    # B: calibration episode ------------------------------------------------
    ax = axes[0, 1]
    # Recreate both calibration episodes to align selected samples.
    calibration_rng = np.random.default_rng(1000 + cfg.example_subject + 1)
    episode_gap = 1.5
    episode_span = cfg.test_steps * cfg.dt
    fit_episode_length = cfg.test_steps - 1
    episode_states = []
    episode_inputs = []
    episode_times = []
    for episode in range(cfg.calibration_episodes):
        u_episode = make_calibration_input(cfg.test_steps, cfg, calibration_rng)
        initial_basin = -1.0 if episode % 2 == 0 else 1.0
        x_episode = simulate_subject(
            example["params"], u_episode, cfg, initial_basin, calibration_rng
        )
        offset = episode * (episode_span + episode_gap)
        episode_times.append(offset + np.arange(cfg.test_steps) * cfg.dt)
        episode_states.append(x_episode)
        episode_inputs.append(u_episode)

    h_state = None
    for episode_time, x_episode in zip(episode_times, episode_states):
        (line,) = ax.plot(
            episode_time, x_episode, color=palette.ground_truth, lw=1.6
        )
        h_state = h_state or line

    selected_episode = example["selectedIdx"] // fit_episode_length
    selected_within = example["selectedIdx"] % fit_episode_length
    selected_time = (
        selected_episode * (episode_span + episode_gap)
        + selected_within * cfg.dt
    )
    h_samples = ax.scatter(
        selected_time,
        example["selectedX"],
        s=18,
        color=palette.perturb,
        edgecolors="none",
        zorder=3,
    )
    state_values = np.concatenate(episode_states)
    state_span = np.ptp(state_values)
    state_lower = min(state_values.min() - 0.14 * state_span, -2.1)
    state_upper = max(state_values.max() + 0.14 * state_span, 2.1)
    ax.set_ylim(state_lower, state_upper)
    ax.set_yticks([-2, 0, 2])
    ax.set_ylabel("State, $x$")
    ax.tick_params(axis="x", labelbottom=True, pad=1.0)

    ax_input = ax_calibration_input
    h_input = None
    for episode_time, u_episode in zip(episode_times, episode_inputs):
        (line,) = ax_input.step(
            episode_time, u_episode, where="post",
            color=palette.pulse_shade, lw=1.1,
        )
        h_input = h_input or line
    input_values = np.concatenate(episode_inputs)
    input_span = np.ptp(input_values)
    ax_input.set_ylim(
        input_values.min() - 0.14 * input_span,
        input_values.max() + 0.14 * input_span,
    )
    ax_input.set_ylabel("Input, $u$")
    ax_input.set_xlabel("$t$")

    gap_start = episode_times[0][-1]
    gap_end = episode_times[1][0]
    ax.axvspan(gap_start, gap_end, color="#EEF1F4", zorder=0)
    ax_input.axvspan(gap_start, gap_end, color="#EEF1F4", zorder=0)
    for number, episode_time in enumerate(episode_times, start=1):
        ax.text(
            np.mean([episode_time[0], episode_time[-1]]), 0.98,
            f"Episode {number}", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=7.5, color="#666D75",
        )

    panel_title(ax, "B", "Few-Shot Calibration")
    calibration_legend = ax_input.legend(
        [h_state, h_samples, h_input],
        ["State", "Samples", "Input"],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.62),
        ncol=3,
        fontsize=7.7,
        frameon=False,
        handlelength=1.4,
        columnspacing=0.8,
        handletextpad=0.4,
    )
    # Exclude the nested legend from constrained layout.
    calibration_legend.set_in_layout(False)

    # C: controlled vector fields -------------------------------------------
    ax = axes[0, 2]
    line_styles = ["-", (0, (5, 2))]
    for q, u0 in enumerate([0.0, cfg.test_pulse_amplitude]):
        true_f = true_drift(x_grid, u0, example["params"])
        u_col = int(np.argmin(np.abs(cfg.flow_input_grid - u0)))
        passive_f = example["flowPredictions"][1][:, u_col]
        perturb_f = example["flowPredictions"][2][:, u_col]

        ax.plot(
            x_grid, true_f, color=palette.ground_truth, ls=line_styles[q],
            lw=2.3, alpha=0.84,
        )
        ax.plot(
            x_grid, passive_f, color=palette.passive, ls=line_styles[q],
            lw=2.2, alpha=SERIES_ALPHA,
        )
        ax.plot(
            x_grid, perturb_f, color=palette.perturb, ls=line_styles[q],
            lw=2.3, alpha=SERIES_ALPHA,
        )
    ax.axhline(0, color=(0.75, 0.75, 0.75), lw=0.9, zorder=0)
    ax.set_xlabel("State, $x$")
    ax.set_ylabel("Flow, $f(x,u)$")
    ax.set_xticks(STATE_TICKS)
    ax.set_xlim(*STATE_LIM)
    panel_title(ax, "C", "Controlled Vector Fields")
    ax.text(
        0.50, 0.055, "passive predictions overlap",
        transform=ax.transAxes, ha="center", va="bottom",
        fontsize=7.3, color=palette.passive,
        bbox={"boxstyle": "round,pad=0.20", "facecolor": "white",
              "edgecolor": palette.passive, "linewidth": 0.7, "alpha": 0.92},
        zorder=10,
    )
    input_condition_legend = ax.legend(
        handles=[
            Line2D([0], [0], color="#555555", lw=1.5, ls="-", label="$u=0$"),
            Line2D([0], [0], color="#555555", lw=1.5, ls=(0, (5, 2)),
                   label="Test $u$"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.24),
        ncol=2,
        fontsize=8.0,
        frameon=False,
    )
    input_condition_legend.set_in_layout(False)

    # D: estimated energy landscapes ----------------------------------------
    ax = axes[1, 0]
    ax.plot(x_grid, example["Vtrue"], "-", color=palette.ground_truth, lw=2.4)
    ax.plot(x_grid, example["Vset"][0], "--", color=palette.scratch, lw=1.9)
    ax.plot(x_grid, example["Vset"][1], ":", color=palette.passive, lw=2.2,
            alpha=SERIES_ALPHA)
    ax.plot(x_grid, example["Vset"][2], "-", color=palette.perturb, lw=2.3,
            alpha=SERIES_ALPHA)
    ax.set_xlabel("State, $x$")
    ax.set_ylabel("Energy, $V(x)$")
    ax.set_xticks(STATE_TICKS)
    ax.set_xlim(*STATE_LIM)
    ax.set_yticks(ENERGY_TICKS)
    ax.set_ylim(*ENERGY_LIM)
    panel_title(ax, "D", "Estimated Landscapes")

    # E: held-out perturbational response ------------------------------------
    ax = axes[1, 1]
    time = np.arange(cfg.test_steps) * cfg.dt
    bands = [
        (ensemble_band(example["trueResponseEnsemble"]), (0.75, 0.75, 0.75), 0.32),
        (ensemble_band(example["modelResponseEnsembles"][1]), palette.passive, 0.12),
        (ensemble_band(example["modelResponseEnsembles"][2]), palette.perturb, 0.12),
    ]
    for (_, lo, hi), color, alpha in bands:
        ax.fill_between(time, lo, hi, color=color, alpha=alpha, lw=0)
    for (mu, _, _), color, lw in zip(
        [b[0] for b in bands],
        [palette.ground_truth, palette.passive, palette.perturb],
        [2.3, 2.1, 2.3],
    ):
        ax.plot(time, mu, color=color, lw=lw,
                alpha=1.0 if color == palette.ground_truth else SERIES_ALPHA)

    ax.axvline(cfg.test_pulse_start * cfg.dt, ls=":", color=palette.pulse_shade)
    ax.axvline(
        (cfg.test_pulse_start + cfg.test_pulse_duration) * cfg.dt,
        ls=":",
        color=palette.pulse_shade,
    )
    ax.set_xlabel("Time, $t$")
    ax.set_ylabel("State, $x$")
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.set_yticks(RESPONSE_TICKS)
    ax.set_ylim(*RESPONSE_LIM)
    panel_title(ax, "E", "Test-Pulse Response")

    # F: stochastic dose-transition curve -------------------------------------
    ax = axes[1, 2]
    amplitudes = cfg.test_amplitude_grid
    ax.plot(amplitudes, example["trueProbCurve"], "-", color=palette.ground_truth, lw=2.4)
    for curve, color, style, lw in zip(
        example["modelProbCurves"],
        palette.model_colors,
        ["--o", ":s", "-d"],
        [1.7, 2.0, 2.3],
    ):
        ax.plot(amplitudes, curve, style, color=color, lw=lw, ms=4.2, mfc="none",
                alpha=SERIES_ALPHA)
    ax.set_xlabel("Pulse amplitude, $u$")
    ax.set_ylabel("Transition probability")
    ax.set_ylim(-0.03, 1.03)
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    panel_title(ax, "F", "Intervention Outcome")

    method_handles = [
        Line2D([0], [0], color=palette.ground_truth, lw=2.0),
        Line2D([0], [0], color=palette.scratch, lw=2.0, ls="--"),
        Line2D([0], [0], color=palette.passive, lw=2.0, ls=":"),
        Line2D([0], [0], color=palette.perturb, lw=2.0),
    ]
    for panel_ax in axes.flat:
        panel_ax.tick_params(labelsize=9.0)
        panel_ax.xaxis.label.set_size(10.5)
        panel_ax.yaxis.label.set_size(10.5)
    ax_input.tick_params(labelsize=9.0)
    ax_input.xaxis.label.set_size(10.5)
    ax_input.yaxis.label.set_size(10.5)

    fig.legend(
        method_handles, ["Ground truth", *palette.model_labels],
        loc="outside lower center", ncol=4, fontsize=8.0,
        handlelength=2.3, columnspacing=1.5,
    )

    # Align panel A with panel B after resolving constrained layout.
    fig.canvas.draw()
    panel_a_position = axes[0, 0].get_position()
    panel_c_position = axes[0, 2].get_position()
    panel_b_state_position = axes[0, 1].get_position()
    panel_b_input_position = ax_input.get_position()
    fig.set_layout_engine("none")
    shared_top = panel_b_state_position.y1
    shared_bottom = panel_b_input_position.y0
    shared_height = shared_top - shared_bottom
    axes[0, 0].set_position([
        panel_a_position.x0,
        shared_bottom,
        panel_a_position.width,
        shared_height,
    ])
    axes[0, 2].set_position([
        panel_c_position.x0,
        shared_bottom,
        panel_c_position.width,
        shared_height,
    ])

    # Align top-row headings in figure coordinates.
    heading_baseline = shared_top + 0.018
    for top_ax in axes[0, :]:
        position = top_ax.get_position()
        top_ax._panel_anchor.set_transform(fig.transFigure)
        top_ax._panel_anchor.set_position((position.x0 - 0.022, heading_baseline))
        top_ax._panel_heading.set_transform(fig.transFigure)
        top_ax._panel_heading.set_position(
            (position.x0 + position.width / 2, heading_baseline)
        )

    # Align legends from axes with different internal layouts.
    legend_top = shared_bottom - 0.064
    panel_b_position = axes[0, 1].get_position()
    panel_c_position = axes[0, 2].get_position()
    calibration_legend.set_bbox_to_anchor(
        (panel_b_position.x0 + panel_b_position.width / 2, legend_top),
        transform=fig.transFigure,
    )
    input_condition_legend.set_bbox_to_anchor(
        (panel_c_position.x0 + panel_c_position.width / 2, legend_top),
        transform=fig.transFigure,
    )

    export_figure(fig, output_dir, "Figure_3_mechanistic_validation")


def _plot_learning_curve(ax, cfg: Config, metric: dict, palette: Palette) -> None:
    """Plot a log-log learning curve with positive error-bar bounds."""
    styles = ["--o", ":s", "-d"]
    widths = [2.0, 2.2, 2.5]
    faces = ["white", "white", palette.perturb]

    for m in range(3):
        y = np.maximum(metric["mean"][:, m], 1e-8)
        sem = np.maximum(metric["sem"][:, m], 0.0)
        lower = np.minimum(sem, 0.80 * y)
        ax.errorbar(
            cfg.calibration_sizes,
            y,
            yerr=np.vstack([lower, sem]),
            fmt=styles[m],
            color=palette.model_colors[m],
            lw=widths[m],
            ms=5.5,
            mfc=faces[m],
            capsize=3.5,
            alpha=SERIES_ALPHA,
        )

    ax.set_xscale("log")
    ax.set_yscale(cfg.learning_curve_yscale)
    # Use sparse round-number ticks on the log axis.
    ticks = [t for t in LEARNING_CURVE_TICKS
             if min(cfg.calibration_sizes) <= t <= max(cfg.calibration_sizes)]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.minorticks_off()
    ax.set_xlim(min(cfg.calibration_sizes) * 0.85, max(cfg.calibration_sizes) * 1.15)
    if cfg.learning_curve_yscale == "log":
        log_yticks(ax)


def make_metrics_figure(
    cfg: Config, summary: dict, palette: Palette, output_dir: Path
) -> None:
    """Figure 4: few-shot transfer across complementary validation criteria."""
    panels = [
        ("passiveRMSE", "Trajectory Accuracy", "Passive rollout RMSE"),
        ("flowRMSE", "Controlled Flow", "Controlled flow RMSE"),
        ("invariantJS", "State Occupancy", "Finite-run occupancy JS"),
        ("responseJS", "Response Spread", "Response-distribution JS"),
        (
            "transitionProbabilityRMSE",
            "Intervention Outcome",
            "Dose\u2013transition RMSE",  # en dash
        ),
        ("attractorError", "Attractor Geometry", "Attractor-location error"),
    ]

    fig, axes = plt.subplots(
        2, 3, figsize=METRICS_FIGURE_SIZE, constrained_layout=True)
    fig.set_constrained_layout_pads(
        w_pad=6 / 72, h_pad=8 / 72, wspace=0.20, hspace=0.12
    )

    for k, (name, group, ylabel) in enumerate(panels):
        ax = axes.flat[k]
        _plot_learning_curve(ax, cfg, summary[name], palette)
        ax.set_xlabel("Calibration samples")
        ax.set_ylabel(ylabel)
        panel_title(ax, chr(ord("A") + k), group)

    metric_handles = [
        Line2D([0], [0], color=palette.scratch, lw=2.0, ls="--", marker="o",
               markerfacecolor="white"),
        Line2D([0], [0], color=palette.passive, lw=2.0, ls=":", marker="s",
               markerfacecolor="white"),
        Line2D([0], [0], color=palette.perturb, lw=2.0, ls="-", marker="d",
               markerfacecolor=palette.perturb),
    ]
    fig.legend(
        metric_handles, palette.model_labels, loc="outside lower center",
        ncol=3, fontsize=8.0, handlelength=2.3, columnspacing=1.7,
    )

    export_figure(fig, output_dir, "Figure_4_few_shot_metrics")


def make_coverage_figure(
    cfg: Config,
    passive_x: np.ndarray,
    perturb_x: np.ndarray,
    perturb_u: np.ndarray,
    coverage_stats: np.ndarray,
    palette: Palette,
    output_dir: Path,
) -> None:
    """Figure 2: coverage and excitation diagnostics."""
    fig, axes = plt.subplots(
        1, 3, figsize=(18.3 * CM, 8.8 * CM), constrained_layout=True
    )
    fig.set_constrained_layout_pads(w_pad=6 / 72, wspace=0.10)
    title_size = 10.0

    ax = axes[0]
    edges = cfg.density_edges
    # Draw both coverage distributions with equal visual weight.
    counts = {}
    for values, colour, label in (
        (passive_x, palette.passive, "Passive"),
        (perturb_x, palette.perturb, "Perturbational"),
    ):
        weights = np.full(values.size, 1 / values.size)
        counts[label], _, _ = ax.hist(
            values, bins=edges, weights=weights, color=colour, alpha=0.32,
            lw=0, label=label,
        )
    passive_counts, perturb_counts = counts["Passive"], counts["Perturbational"]
    ax.set_xlabel("State, $x$")
    ax.set_xticks(STATE_TICKS)
    ax.set_ylabel("Probability")
    fixed_yticks(ax, float(max(passive_counts.max(), perturb_counts.max())))
    panel_title(ax, "A", "State-Space Coverage")
    coverage_handles = ax.get_legend_handles_labels()

    ax = axes[1]
    input_counts, _, _ = ax.hist(
        perturb_u,
        bins=60,
        weights=np.full(perturb_u.size, 1 / perturb_u.size),
        color=(0.55, 0.55, 0.55),
    )
    ax.set_xlabel("Input, $u$")
    ax.set_xticks(STATE_TICKS)
    ax.set_ylabel("Probability")
    fixed_yticks(ax, float(input_counts.max()))
    panel_title(ax, "B", "Input-Space Coverage")

    ax = axes[2]
    means = np.nanmean(coverage_stats[:, :4], axis=0)
    sems = np.nanstd(coverage_stats[:, :4], axis=0, ddof=1) / np.sqrt(
        coverage_stats.shape[0]
    )
    positions = np.arange(4)
    ax.bar(positions, means, color=(0.65, 0.65, 0.65))
    ax.errorbar(positions, means, yerr=sems, fmt="k.", lw=1.1, capsize=3)
    ax.set_xticks(positions)
    ax.set_xticklabels(
        ["Left", "Boundary", "Right", "Input on"], fontsize=9.0, rotation=20, ha="right"
    )
    ax.set_ylabel("Fraction of candidate-pool samples")
    fixed_yticks(ax, float((means + sems).max()))
    panel_title(ax, "C", "Candidate Calibration Pool")

    for panel_ax in axes:
        panel_ax.tick_params(labelsize=9.0)
        panel_ax.xaxis.label.set_size(10.5)
        panel_ax.yaxis.label.set_size(10.5)

    # Use one shared legend for both coverage regimes.
    fig.legend(
        *coverage_handles, loc="outside lower center", ncol=2, fontsize=8.5,
        handlelength=2.0, columnspacing=1.8,
    )

    export_figure(fig, output_dir, "Figure_2_coverage")


def write_summary_table(cfg: Config, summary: dict, output_dir: Path) -> Path:
    """Write the per-calibration-size, per-model metric means as CSV."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "summary_metrics_visual.csv"

    columns = [
        ("PassiveRMSE", "passiveRMSE"),
        ("PerturbRMSE", "perturbRMSE"),
        ("LandscapeRMSE", "landscapeRMSE"),
        ("BarrierError", "barrierError"),
        ("AttractorError", "attractorError"),
        ("TransitionAccuracy", "transitionCorrect"),
        ("AmplitudeCurveRMSE", "amplitudeCurveRMSE"),
        ("FlowRMSE", "flowRMSE"),
        ("InvariantJS", "invariantJS"),
        ("ResponseJS", "responseJS"),
        ("TransitionProbabilityRMSE", "transitionProbabilityRMSE"),
        ("MaxAbsState", "maxAbsState"),
    ]

    palette = Palette()
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["CalibrationSamples", "Model", *[c[0] for c in columns]])
        for ci, n_cal in enumerate(cfg.calibration_sizes):
            for m, model_name in enumerate(palette.model_names):
                writer.writerow(
                    [
                        n_cal,
                        model_name,
                        *[summary[key]["mean"][ci, m] for _, key in columns],
                    ]
                )
    return path


__all__ = [
    "apply_style",
    "export_figure",
    "make_main_figure",
    "make_metrics_figure",
    "make_coverage_figure",
    "write_summary_table",
]
