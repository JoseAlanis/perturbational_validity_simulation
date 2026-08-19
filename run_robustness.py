#!/usr/bin/env python3
"""Run repeated-seed controls and sensitivity analyses."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from perturbsim import Config, NetConfig, OptConfig
from perturbsim.robustness import (
    RobustnessCondition,
    run_condition,
    summarise_rows,
    write_rows,
)

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "results" / "robustness"
DEFAULT_SEEDS = (11, 23, 37, 51, 71)


def sensitivity_conditions(seed: int, quick: bool) -> list[RobustnessCondition]:
    """Return one-factor-at-a-time sensitivity conditions."""
    cfg = Config(
        seed=seed,
        test_subjects=3 if quick else 10,
        train_steps_per_subject=300 if quick else 1000,
        test_steps=300 if quick else 1000,
        invariant_steps=400 if quick else 2500,
        invariant_burn_in=80 if quick else 500,
        distribution_replicates=6 if quick else 20,
        response_eval_stride=50,
        calibration_sizes=(5,),
        example_calibration_index=0,
        test_pulse_start=90 if quick else 300,
    )
    net = NetConfig()
    opt = OptConfig(
        pretrain_epochs=4 if quick else 40,
        adaptation_epochs=12 if quick else 80,
    )
    specs = [
        ("baseline", cfg, net),
        ("training_population_20", replace(cfg, train_subjects=20), net),
        ("training_population_100", replace(cfg, train_subjects=100), net),
        ("diffusion_0.10", replace(cfg, sigma_range=(0.10, 0.10)), net),
        ("diffusion_0.20", replace(cfg, sigma_range=(0.20, 0.20)), net),
        ("hidden_width_12", cfg, replace(net, hidden1=12, hidden2=12)),
        ("hidden_width_48", cfg, replace(net, hidden1=48, hidden2=48)),
        ("embedding_dim_1", cfg, replace(net, embedding_dim=1)),
        ("embedding_dim_6", cfg, replace(net, embedding_dim=6)),
        ("calibration_perturbed_0.25", replace(cfg, cal_perturbed_fraction=0.25), net),
        ("calibration_perturbed_0.75", replace(cfg, cal_perturbed_fraction=0.75), net),
        (
            "pretraining_amplitude_1.20",
            replace(cfg, train_pulse_amplitude_range=(-1.20, 1.20)),
            net,
        ),
        (
            "pretraining_amplitude_3.00",
            replace(cfg, train_pulse_amplitude_range=(-3.00, 3.00)),
            net,
        ),
        (
            "pretraining_probability_0.015",
            replace(cfg, train_pulse_probability=0.015),
            net,
        ),
        (
            "pretraining_probability_0.070",
            replace(cfg, train_pulse_probability=0.070),
            net,
        ),
    ]
    return [RobustnessCondition(name, condition_cfg, condition_net, opt)
            for name, condition_cfg, condition_net in specs]


def core_condition(seed: int, quick: bool) -> RobustnessCondition:
    """Return the repeated-seed control condition."""
    cfg = Config(seed=seed)
    opt = OptConfig()
    if quick:
        cfg = replace(
            cfg, train_subjects=6, test_subjects=3,
            train_steps_per_subject=300, test_steps=300,
            invariant_steps=400, invariant_burn_in=80,
            distribution_replicates=6, response_eval_stride=50,
            test_pulse_start=90,
        )
        opt = replace(opt, pretrain_epochs=4, adaptation_epochs=12)
    return RobustnessCondition("core", cfg, NetConfig(), opt)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", choices=("core", "sensitivity", "all"), default="all")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    rows = []
    for seed in args.seeds:
        if args.analysis in ("core", "all"):
            print(f"Core robustness analysis, seed {seed}", flush=True)
            rows.extend(run_condition(
                core_condition(seed, args.quick),
                (2, 5) if args.quick else (2, 5, 20),
                include_ablations=True,
            ))
        if args.analysis in ("sensitivity", "all"):
            for condition in sensitivity_conditions(seed, args.quick):
                print(f"Sensitivity {condition.name}, seed {seed}", flush=True)
                rows.extend(run_condition(
                    condition, (5,), include_ablations=False
                ))

    subject_path = write_rows(rows, args.output_dir / "robustness_subject_level.csv")
    summary_path = write_rows(
        summarise_rows(rows), args.output_dir / "robustness_by_training_run.csv"
    )
    print(f"Wrote {subject_path}")
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
