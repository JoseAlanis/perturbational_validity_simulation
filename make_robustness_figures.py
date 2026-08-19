#!/usr/bin/env python3
"""Create supplementary figures from robustness-analysis CSV files."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from perturbsim.config import Palette
from perturbsim.figures import CM, apply_style, export_figure, panel_title

MODEL_ORDER = (
    "scratch", "passive_frozen", "passive_full", "state_coverage",
    "input_excitation", "perturbational",
)
MODEL_LABELS = {
    "scratch": "Cubic scratch",
    "passive_frozen": "Passive, embedding only",
    "passive_full": "Passive, full adaptation",
    "state_coverage": "State coverage only",
    "input_excitation": "Input excitation only",
    "perturbational": "Perturbational",
}
PALETTE = Palette()
COLORS = {
    "scratch": PALETTE.scratch,
    "passive_frozen": PALETTE.passive,
    "passive_full": "#CC79A7",
    "state_coverage": "#8C8C8C",
    "input_excitation": "#56B4E9",
    "perturbational": PALETTE.perturb,
}
MARKERS = ("o", "s", "^", "v", "P", "D")
LINESTYLES = ("--", ":", "-.", "--", ":", "-")


def read_rows(path: Path) -> list[dict]:
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def grouped(rows: list[dict], keys: tuple[str, ...]) -> dict:
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    return groups


def make_controls(rows: list[dict], output_dir: Path) -> None:
    metrics = (
        ("flowRMSEMean", "Controlled-flow RMSE"),
        ("responseJSInterpolationMean", "Response JS, interpolation"),
        ("responseJSExtrapolationMean", "Response JS, extrapolation"),
        ("transitionProbabilityErrorExtrapolationMean",
         "Transition-probability error,\nextrapolation"),
    )
    groups = grouped(rows, ("calibrationSize", "model"))
    titles = (
        "Controlled Flow", "Interpolation Response",
        "Extrapolation Response", "Extrapolation Outcome",
    )
    fig, axes = plt.subplots(
        2, 2, figsize=(18.3 * CM, 15.6 * CM), constrained_layout=True
    )
    fig.set_constrained_layout_pads(
        w_pad=6 / 72, h_pad=8 / 72, wspace=0.20, hspace=0.18
    )
    for ax, (metric, ylabel), title, letter in zip(
        axes.ravel(), metrics, titles, "ABCD"
    ):
        for model, marker, linestyle in zip(MODEL_ORDER, MARKERS, LINESTYLES):
            sizes = sorted({int(row["calibrationSize"]) for row in rows})
            positions = np.arange(len(sizes))
            means, errors = [], []
            for size in sizes:
                values = np.array([float(row[metric]) for row in groups[(str(size), model)]])
                means.append(values.mean())
                errors.append(values.std(ddof=1))
            ax.errorbar(
                positions, means, yerr=errors, color=COLORS[model], marker=marker,
                ls=linestyle, lw=1.8, ms=4.8, capsize=3.0,
                mfc="white" if model != "perturbational" else COLORS[model],
                alpha=0.82, label=MODEL_LABELS[model],
            )
        ax.set_xticks(positions, [str(value) for value in sizes])
        ax.set_xlim(-0.25, len(sizes) - 0.75)
        ax.set_xlabel("Calibration evaluations")
        ax.set_ylabel(ylabel)
        panel_title(ax, letter, title)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="outside lower center", ncol=3,
        fontsize=8.0, handlelength=2.3, columnspacing=1.4,
    )
    export_figure(fig, output_dir, "Figure_S1_robustness_controls")


def make_sensitivity(rows: list[dict], output_dir: Path) -> None:
    labels = {
        "baseline": "Baseline",
        "training_population_20": "Training systems: 20",
        "training_population_100": "Training systems: 100",
        "diffusion_0.10": "Diffusion: 0.10",
        "diffusion_0.20": "Diffusion: 0.20",
        "hidden_width_12": "Hidden width: 12",
        "hidden_width_48": "Hidden width: 48",
        "embedding_dim_1": "Embedding dimension: 1",
        "embedding_dim_6": "Embedding dimension: 6",
        "calibration_perturbed_0.25": "Perturbed calibration: 25%",
        "calibration_perturbed_0.75": "Perturbed calibration: 75%",
        "pretraining_amplitude_1.20": "Pretraining amplitude: 1.20",
        "pretraining_amplitude_3.00": "Pretraining amplitude: 3.00",
        "pretraining_probability_0.015": "Pulse probability: 0.015",
        "pretraining_probability_0.070": "Pulse probability: 0.070",
    }
    order = tuple(labels)
    groups = grouped(rows, ("condition", "trainingSeed", "model"))
    metrics = (
        ("flowRMSEMean", "Controlled-flow\nRMSE ratio"),
        ("responseJSInterpolationMean", "Interpolation-response\nJS ratio"),
        ("responseJSExtrapolationMean", "Extrapolation-response\nJS ratio"),
    )
    fig, axes = plt.subplots(
        1, 3, figsize=(18.3 * CM, 13.6 * CM), sharey=True,
        constrained_layout=True,
    )
    fig.set_constrained_layout_pads(w_pad=6 / 72, h_pad=8 / 72, wspace=0.16)
    y = np.arange(len(order))
    seeds = sorted({row["trainingSeed"] for row in rows})
    for ax, (metric, title), letter in zip(axes, metrics, "ABC"):
        means, errors = [], []
        for condition in order:
            ratios = []
            for seed in seeds:
                perturb = float(groups[(condition, seed, "perturbational")][0][metric])
                passive = float(groups[(condition, seed, "passive_full")][0][metric])
                ratios.append(perturb / passive)
            means.append(np.mean(ratios))
            errors.append(np.std(ratios, ddof=1))
        ax.errorbar(
            means, y, xerr=errors, fmt="o", color=PALETTE.perturb,
            ms=5.0, mfc=PALETTE.perturb, lw=1.8, capsize=3.0,
        )
        ax.axvline(1.0, color="#666666", lw=1.1, ls="--")
        ax.set_xlabel("Perturbational / passive full")
        panel_title(ax, letter, title)
        ax._panel_anchor.set_position((-0.13, 1.18))
        ax._panel_heading.set_y(1.06)
    axes[0].set_yticks(y, [labels[name] for name in order])
    axes[0].invert_yaxis()
    export_figure(fig, output_dir, "Figure_S2_sensitivity")


def write_across_run_summary(rows: list[dict], output: Path) -> None:
    groups = grouped(rows, ("condition", "calibrationSize", "model"))
    metrics = [key[:-4] for key in rows[0] if key.endswith("Mean")]
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["condition", "calibrationSize", "model"]
    for metric in metrics:
        fields.extend((f"{metric}AcrossRunMean", f"{metric}AcrossRunSD"))
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for key, group in sorted(groups.items()):
            row = dict(zip(fields[:3], key))
            for metric in metrics:
                values = np.array([float(item[f"{metric}Mean"]) for item in group])
                row[f"{metric}AcrossRunMean"] = values.mean()
                row[f"{metric}AcrossRunSD"] = values.std(ddof=1)
            writer.writerow(row)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--sensitivity", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    apply_style()
    core_rows = read_rows(args.core)
    sensitivity_rows = read_rows(args.sensitivity)
    make_controls(core_rows, args.output_dir)
    make_sensitivity(sensitivity_rows, args.output_dir)
    write_across_run_summary(core_rows, args.output_dir / "robustness_across_runs.csv")
    write_across_run_summary(
        sensitivity_rows, args.output_dir / "sensitivity_across_runs.csv"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
