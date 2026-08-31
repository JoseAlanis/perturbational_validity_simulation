"""Save and load simulation results in compressed NumPy archives."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .config import METRIC_NAMES, Config
from .dynamics import SubjectParams
from .metrics import LandscapeFeatures

RESULTS_FILENAME = "simulation_results.npz"

#: Example fields that are lists of one array per model class.
_EXAMPLE_STACKED = (
    "predPerturb",
    "Vset",
    "predCurves",
    "modelProbCurves",
    "modelInvariantSamples",
    "modelResponseEnsembles",
    "flowPredictions",
)

#: Example fields that are a single array.
_EXAMPLE_PLAIN = (
    "calibrationX",
    "calibrationU",
    "truePerturb",
    "Vtrue",
    "trueCurve",
    "trueFlow",
    "flowX",
    "flowU",
    "trueInvariantProb",
    "trueResponseEnsemble",
    "trueProbCurve",
    "selectedX",
    "selectedU",
    "selectedIdx",
)


def params_to_array(params: list[SubjectParams]) -> np.ndarray:
    return np.array([[p.a, p.c, p.d, p.b, p.sigma] for p in params], dtype=float)


def params_from_array(values: np.ndarray) -> list[SubjectParams]:
    return [SubjectParams(*row) for row in np.atleast_2d(values)]


def save_results(
    path: Path,
    cfg: Config,
    results: dict,
    coverage_stats: np.ndarray,
    train_params: list[SubjectParams],
    test_params: list[SubjectParams],
    example: dict,
    passive_x: np.ndarray,
    perturb_x: np.ndarray,
    perturb_u: np.ndarray,
) -> Path:
    """Write one results bundle."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, np.ndarray] = {
        f"results__{name}": results[name] for name in METRIC_NAMES
    }
    payload["coverageStats"] = coverage_stats
    payload["trainParams"] = params_to_array(train_params)
    payload["testParams"] = params_to_array(test_params)
    payload["calibrationSizes"] = np.asarray(cfg.calibration_sizes)
    payload["populationPassiveX"] = passive_x
    payload["populationPerturbX"] = perturb_x
    payload["populationPerturbU"] = perturb_u
    payload["configJson"] = np.asarray(json.dumps(asdict(cfg), sort_keys=True))

    for key in _EXAMPLE_PLAIN:
        payload[f"example__{key}"] = np.asarray(example[key])
    for key in _EXAMPLE_STACKED:
        payload[f"example__{key}"] = np.stack([np.asarray(v) for v in example[key]])

    payload["example__params"] = params_to_array([example["params"]])[0]
    payload["example__nCal"] = np.asarray(example["nCal"])
    features = example["trueFeatures"]
    payload["example__minimaX"] = np.asarray(features.minima_x)
    payload["example__saddleX"] = np.asarray(features.saddle_x)
    payload["example__barrier"] = np.asarray(features.barrier)

    np.savez_compressed(path, **payload)
    return path


def load_results(path: Path, expected_cfg: Config | None = None) -> dict:
    """Read a results bundle and optionally verify its simulation settings."""
    with np.load(Path(path)) as data:
        if "configJson" not in data:
            raise ValueError(
                "results cache predates configuration metadata; rerun simulation"
            )
        config_json = str(data["configJson"])
        if expected_cfg is not None:
            expected = json.dumps(asdict(expected_cfg), sort_keys=True)
            if config_json != expected:
                raise ValueError("results cache configuration does not match this run")
        results = {name: data[f"results__{name}"] for name in METRIC_NAMES}

        example: dict = {}
        for key in _EXAMPLE_PLAIN:
            example[key] = data[f"example__{key}"]
        for key in _EXAMPLE_STACKED:
            example[key] = list(data[f"example__{key}"])

        example["params"] = SubjectParams(*data["example__params"])
        example["nCal"] = int(data["example__nCal"])
        example["trueFeatures"] = LandscapeFeatures(
            minima_x=data["example__minimaX"],
            saddle_x=float(data["example__saddleX"]),
            barrier=float(data["example__barrier"]),
        )

        return {
            "results": results,
            "coverageStats": data["coverageStats"],
            "trainParams": params_from_array(data["trainParams"]),
            "testParams": params_from_array(data["testParams"]),
            "calibrationSizes": data["calibrationSizes"],
            "populationPassiveX": data["populationPassiveX"],
            "populationPerturbX": data["populationPerturbX"],
            "populationPerturbU": data["populationPerturbU"],
            "example": example,
        }
