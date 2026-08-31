#!/usr/bin/env python3
"""Run, cache, evaluate, and plot the perturbational model simulation."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from perturbsim import METRIC_NAMES, Config, NetConfig, OptConfig, Palette
from perturbsim.dynamics import (
    Scale,
    generate_population_dataset,
    sample_subject_parameters,
    standardise,
)
from perturbsim.evaluate import (
    N_MODELS,
    evaluate_one_subject,
    monte_carlo_floor,
    summarise_results,
)
from perturbsim.figures import (
    apply_style,
    make_coverage_figure,
    make_main_figure,
    make_metrics_figure,
    write_summary_table,
)
from perturbsim.model import initialise_shared_model, train_shared_model
from perturbsim.storage import RESULTS_FILENAME, load_results, save_results

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "results" / "simulation"


def _worker(task):
    """Evaluate one held-out subject; top level so it can be sent to a process pool."""
    index, params, passive_model, perturb_model, cfg, opt, scale, keep_example = task
    return index, evaluate_one_subject(
        index, params, passive_model, perturb_model, cfg, opt, scale, keep_example
    )


def simulate(
    cfg: Config,
    net: NetConfig,
    opt: OptConfig,
    output_dir: Path,
    workers: int | None,
) -> dict:
    """Run the whole pipeline and cache the results."""
    started = time.time()
    rng = np.random.default_rng(cfg.seed)

    print("Generating population data...", flush=True)
    train_params = sample_subject_parameters(cfg.train_subjects, cfg, rng)

    passive_data = generate_population_dataset(
        train_params, cfg.train_steps_per_subject, cfg, False, rng
    )
    perturb_data = generate_population_dataset(
        train_params, cfg.train_steps_per_subject, cfg, True, rng
    )
    print(f"Passive samples: {len(passive_data)}")
    print(f"Perturbational samples: {len(perturb_data)}")

    # Share perturbational-population scaling between both models.
    scale = Scale.from_dataset(perturb_data)
    passive_data = standardise(passive_data, scale)
    perturb_data = standardise(perturb_data, scale)

    base_model = initialise_shared_model(net, cfg.train_subjects, rng)

    print("\nTraining passive population model...", flush=True)
    passive_model = train_shared_model(base_model, passive_data, opt, "Passive", rng)

    print("\nTraining perturbational population model...", flush=True)
    perturb_model = train_shared_model(
        base_model, perturb_data, opt, "Perturbational", rng
    )

    test_params = sample_subject_parameters(cfg.test_subjects, cfg, rng)

    results = {
        name: np.full(
            (cfg.test_subjects, len(cfg.calibration_sizes), N_MODELS), np.nan
        )
        for name in METRIC_NAMES
    }
    coverage_stats = np.full((cfg.test_subjects, 5), np.nan)
    example = None

    tasks = [
        (
            s + 1,  # one-based seed offset
            test_params[s],
            passive_model,
            perturb_model,
            cfg,
            opt,
            scale,
            s == cfg.example_subject,
        )
        for s in range(cfg.test_subjects)
    ]

    print(f"\nEvaluating {cfg.test_subjects} unseen subjects...", flush=True)

    if workers and workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            completed = pool.map(_worker, tasks)
            for done, (index, payload) in enumerate(completed, start=1):
                _collect(results, coverage_stats, index, payload)
                if payload[2] is not None:
                    example = payload[2]
                print(f"  completed subject {done}/{cfg.test_subjects}", flush=True)
    else:
        for done, task in enumerate(tasks, start=1):
            index, payload = _worker(task)
            _collect(results, coverage_stats, index, payload)
            if payload[2] is not None:
                example = payload[2]
            print(f"  completed subject {done}/{cfg.test_subjects}", flush=True)

    if example is None:
        raise RuntimeError(
            "no example subject was recorded; check cfg.example_subject and "
            "cfg.example_calibration_index"
        )

    path = save_results(
        Path(output_dir) / RESULTS_FILENAME,
        cfg,
        results,
        coverage_stats,
        train_params,
        test_params,
        example,
        passive_data.x,
        perturb_data.x,
        perturb_data.u,
    )
    print(f"\nSaved results to {path} ({time.time() - started:.0f} s)")

    return {
        "results": results,
        "coverageStats": coverage_stats,
        "trainParams": train_params,
        "testParams": test_params,
        "example": example,
        "populationPassiveX": passive_data.x,
        "populationPerturbX": perturb_data.x,
        "populationPerturbU": perturb_data.u,
    }


def _collect(results: dict, coverage_stats: np.ndarray, index: int, payload) -> None:
    metrics, coverage, _ = payload
    for name in METRIC_NAMES:
        results[name][index - 1] = metrics[name]
    coverage_stats[index - 1] = coverage


def draw_figures(cfg: Config, bundle: dict, output_dir: Path) -> None:
    """Draw all three figures and write the summary table."""
    apply_style()
    summary = summarise_results(bundle["results"], cfg)

    make_main_figure(
        cfg, bundle["trainParams"], bundle["example"], Palette(), output_dir
    )
    make_metrics_figure(cfg, summary, Palette(), output_dir)
    make_coverage_figure(
        cfg,
        bundle["populationPassiveX"],
        bundle["populationPerturbX"],
        bundle["populationPerturbU"],
        bundle["coverageStats"],
        Palette(),
        output_dir,
    )
    table = write_summary_table(cfg, summary, output_dir)
    print(f"Figures and {table.name} written to {output_dir}")

    # What a perfect model scores: any non-zero value is finite-sample noise.
    floor = monte_carlo_floor(cfg, bundle["testParams"])
    floor_path = output_dir / "monte_carlo_floor.csv"
    with floor_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["quantity", "mean", "sd"])
        for name, (mean, sd) in floor.items():
            writer.writerow([name, f"{mean:.6f}", f"{sd:.6f}"])
    print(
        f"Perfect-model floor: responseJS shared "
        f"{floor['responseJSShared'][0]:.4f}, independent "
        f"{floor['responseJSIndependent'][0]:.4f}; dose RMSE shared "
        f"{floor['doseRMSEShared'][0]:.4f}, independent "
        f"{floor['doseRMSEIndependent'][0]:.4f} -> {floor_path.name}"
    )


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("all", "simulate", "figures"),
        default="all",
        help="run the simulation, draw the figures, or both (default: both)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"where results and figures are written (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="processes for the held-out evaluation; 0 or 1 runs serially",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="small run for checking the pipeline end to end, not for the manuscript",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    cfg = Config()
    net = NetConfig()
    opt = OptConfig()

    if args.quick:
        cfg = replace(
            cfg,
            train_subjects=6,
            test_subjects=3,
            train_steps_per_subject=400,
            test_steps=400,
            calibration_sizes=(2, 8, 20),
            invariant_steps=600,
            invariant_burn_in=100,
            distribution_replicates=8,
            example_calibration_index=2,
            test_pulse_start=120,
        )
        opt = replace(opt, pretrain_epochs=6, adaptation_epochs=30)
        print("Quick mode: reduced settings, results are not publication values.\n")

    output_dir = Path(args.output_dir)

    if args.stage in ("all", "simulate"):
        bundle = simulate(cfg, net, opt, output_dir, args.workers)
    else:
        bundle = load_results(output_dir / RESULTS_FILENAME)

    if args.stage in ("all", "figures"):
        draw_figures(cfg, bundle, output_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
