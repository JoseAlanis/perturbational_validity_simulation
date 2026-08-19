"""Shared drift network, participant adaptation, and cubic baseline."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np

from .config import NetConfig, OptConfig
from .dynamics import Dataset, DriftFn, Scale

_ADAM_BETA1 = 0.9
_ADAM_BETA2 = 0.999
_ADAM_EPS = 1e-8
_PARAM_NAMES = ("W1", "b1", "W2", "b2", "W3", "b3", "E")


@dataclass
class SharedModel:
    """Population model: shared weights plus one embedding per training subject."""

    W1: np.ndarray
    b1: np.ndarray
    W2: np.ndarray
    b2: np.ndarray
    W3: np.ndarray
    b3: np.ndarray
    E: np.ndarray
    adam_m: dict = field(default_factory=dict)
    adam_v: dict = field(default_factory=dict)
    adam_step: int = 0

    def reset_adam_state(self) -> "SharedModel":
        self.adam_m = {n: np.zeros_like(np.atleast_1d(getattr(self, n))) for n in _PARAM_NAMES}
        self.adam_v = {n: np.zeros_like(np.atleast_1d(getattr(self, n))) for n in _PARAM_NAMES}
        self.adam_step = 0
        return self

    @property
    def embedding_dim(self) -> int:
        return self.E.shape[0]

    def to_dict(self) -> dict:
        return {n: np.asarray(getattr(self, n)) for n in _PARAM_NAMES}

    @classmethod
    def from_dict(cls, d: dict) -> "SharedModel":
        return cls(**{n: np.asarray(d[n]) for n in _PARAM_NAMES}).reset_adam_state()


def initialise_shared_model(
    net: NetConfig, n_subjects: int, rng: np.random.Generator
) -> SharedModel:
    r = 0.10
    model = SharedModel(
        W1=r * rng.standard_normal((net.hidden1, net.input_dim)),
        b1=np.zeros((net.hidden1, 1)),
        W2=r * rng.standard_normal((net.hidden2, net.hidden1)),
        b2=np.zeros((net.hidden2, 1)),
        W3=r * rng.standard_normal((1, net.hidden2)),
        b3=np.zeros((1, 1)),
        E=0.05 * rng.standard_normal((net.embedding_dim, n_subjects)),
    )
    return model.reset_adam_state()


def _forward(model: SharedModel, features: np.ndarray):
    """Return (h1, h2, prediction) for a (input_dim, n) feature matrix."""
    h1 = np.tanh(model.W1 @ features + model.b1)
    h2 = np.tanh(model.W2 @ h1 + model.b2)
    y_hat = model.W3 @ h2 + model.b3
    return h1, h2, y_hat


def _shared_loss_and_gradients(
    model: SharedModel,
    x: np.ndarray,
    u: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    opt: OptConfig,
):
    e_batch = model.E[:, subjects]
    features = np.vstack([x[None, :], u[None, :], e_batch])

    h1, h2, y_hat = _forward(model, features)

    n = features.shape[1]
    err = y_hat - y[None, :]

    loss = (
        float(np.mean(err**2))
        + opt.weight_decay
        * float(np.sum(model.W1**2) + np.sum(model.W2**2) + np.sum(model.W3**2))
        + opt.embedding_penalty * float(np.mean(e_batch**2))
    )

    d_y = 2 * err / n

    grads = {}
    grads["W3"] = d_y @ h2.T + 2 * opt.weight_decay * model.W3
    grads["b3"] = np.sum(d_y, axis=1, keepdims=True)

    d_z2 = (model.W3.T @ d_y) * (1 - h2**2)
    grads["W2"] = d_z2 @ h1.T + 2 * opt.weight_decay * model.W2
    grads["b2"] = np.sum(d_z2, axis=1, keepdims=True)

    d_z1 = (model.W2.T @ d_z2) * (1 - h1**2)
    grads["W1"] = d_z1 @ features.T + 2 * opt.weight_decay * model.W1
    grads["b1"] = np.sum(d_z1, axis=1, keepdims=True)

    d_features = model.W1.T @ d_z1
    d_embedding = d_features[2:, :] + 2 * opt.embedding_penalty * e_batch / e_batch.size

    grads["E"] = np.zeros_like(model.E)
    np.add.at(grads["E"], (slice(None), subjects), d_embedding)

    return loss, grads


def _clip_gradients(grads: dict, threshold: float) -> dict:
    global_norm = np.sqrt(sum(float(np.sum(g**2)) for g in grads.values()))
    if global_norm > threshold:
        factor = threshold / (global_norm + np.finfo(float).eps)
        grads = {name: g * factor for name, g in grads.items()}
    return grads


def _adam_update(model: SharedModel, grads: dict, lr: float) -> SharedModel:
    model.adam_step += 1
    t = model.adam_step

    for name, g in grads.items():
        model.adam_m[name] = _ADAM_BETA1 * model.adam_m[name] + (1 - _ADAM_BETA1) * g
        model.adam_v[name] = _ADAM_BETA2 * model.adam_v[name] + (1 - _ADAM_BETA2) * g**2

        m_hat = model.adam_m[name] / (1 - _ADAM_BETA1**t)
        v_hat = model.adam_v[name] / (1 - _ADAM_BETA2**t)

        setattr(model, name, getattr(model, name) - lr * m_hat / (np.sqrt(v_hat) + _ADAM_EPS))

    return model


def train_shared_model(
    base_model: SharedModel,
    data: Dataset,
    opt: OptConfig,
    label: str,
    rng: np.random.Generator,
    verbose: bool = True,
) -> SharedModel:
    """Pretrain a copy of ``base_model`` on population data."""
    model = copy.deepcopy(base_model).reset_adam_state()

    n = data.xs.size
    n_batches = int(np.ceil(n / opt.pretrain_batch_size))

    for epoch in range(1, opt.pretrain_epochs + 1):
        order = rng.permutation(n)
        epoch_loss = 0.0

        for b in range(n_batches):
            idx = order[b * opt.pretrain_batch_size : (b + 1) * opt.pretrain_batch_size]

            loss, grads = _shared_loss_and_gradients(
                model, data.xs[idx], data.us[idx], data.ys[idx], data.subject[idx], opt
            )
            grads = _clip_gradients(grads, opt.gradient_clip)
            model = _adam_update(model, grads, opt.pretrain_learning_rate)

            epoch_loss += loss * idx.size

        if verbose and (epoch == 1 or epoch % 5 == 0 or epoch == opt.pretrain_epochs):
            print(
                f"  {label} epoch {epoch:3d}/{opt.pretrain_epochs} | "
                f"loss {epoch_loss / n:.6f}",
                flush=True,
            )

    return model


def adapt_embedding(model: SharedModel, cal: Dataset, opt: OptConfig) -> np.ndarray:
    """Fit only the participant embedding on a calibration set; weights are frozen."""
    e = np.zeros((model.embedding_dim, 1))
    m = np.zeros_like(e)
    v = np.zeros_like(e)

    x = cal.xs[None, :]
    u = cal.us[None, :]
    y = cal.ys[None, :]
    n = x.shape[1]

    for epoch in range(1, opt.adaptation_epochs + 1):
        features = np.vstack([x, u, np.repeat(e, n, axis=1)])
        h1, h2, y_hat = _forward(model, features)

        err = y_hat - y
        d_y = 2 * err / n
        d_z2 = (model.W3.T @ d_y) * (1 - h2**2)
        d_z1 = (model.W2.T @ d_z2) * (1 - h1**2)
        d_features = model.W1.T @ d_z1

        grad_e = (
            np.sum(d_features[2:, :], axis=1, keepdims=True)
            + 2 * opt.embedding_penalty * e / e.size
        )

        norm = np.linalg.norm(grad_e)
        if norm > opt.gradient_clip:
            grad_e = grad_e * opt.gradient_clip / (norm + np.finfo(float).eps)

        m = _ADAM_BETA1 * m + (1 - _ADAM_BETA1) * grad_e
        v = _ADAM_BETA2 * v + (1 - _ADAM_BETA2) * grad_e**2

        m_hat = m / (1 - _ADAM_BETA1**epoch)
        v_hat = v / (1 - _ADAM_BETA2**epoch)

        e = e - opt.embedding_learning_rate * m_hat / (np.sqrt(v_hat) + _ADAM_EPS)

    return e


def adapt_full_model(
    model: SharedModel, cal: Dataset, opt: OptConfig
) -> tuple[SharedModel, np.ndarray]:
    """Fit all network parameters and one embedding on a calibration set."""
    adapted = copy.deepcopy(model)
    adapted.E = np.zeros((model.embedding_dim, 1))
    adapted.reset_adam_state()
    subjects = np.zeros(cal.xs.size, dtype=int)

    for _ in range(opt.adaptation_epochs):
        _, grads = _shared_loss_and_gradients(
            adapted, cal.xs, cal.us, cal.ys, subjects, opt
        )
        grads = _clip_gradients(grads, opt.gradient_clip)
        adapted = _adam_update(
            adapted, grads, opt.full_adaptation_learning_rate
        )

    return adapted, adapted.E[:, 0].copy()


def fit_cubic_scratch_model(cal: Dataset) -> np.ndarray:
    """Fit a ridge-regularised cubic drift model to one participant."""
    xs = np.asarray(cal.xs).ravel()
    us = np.asarray(cal.us).ravel()
    design = np.column_stack([np.ones(xs.size), xs, xs**2, xs**3, us])

    lam = 1e-4
    penalty = np.diag([0.0, 1.0, 1.0, 1.0, 1.0])
    return np.linalg.solve(
        design.T @ design + lam * penalty, design.T @ np.asarray(cal.ys).ravel()
    )


def _cubic_predict_standardised(beta: np.ndarray, xs: np.ndarray, us: np.ndarray):
    xs = np.asarray(xs).ravel()
    us = np.asarray(us).ravel()
    design = np.column_stack([np.ones(xs.size), xs, xs**2, xs**3, us])
    return design @ beta


def _shared_predict_standardised(
    model: SharedModel, e: np.ndarray, xs: np.ndarray, us: np.ndarray
):
    xs = np.asarray(xs).ravel()
    us = np.asarray(us).ravel()
    features = np.vstack(
        [xs[None, :], us[None, :], np.repeat(e.reshape(-1, 1), xs.size, axis=1)]
    )
    _, _, y_hat = _forward(model, features)
    return y_hat.ravel()


def cubic_drift_fn(beta: np.ndarray, scale: Scale) -> DriftFn:
    """Unscaled drift callable for the cubic baseline."""

    def drift(x, u):
        xs = (np.asarray(x).ravel() - scale.x_mean) / scale.x_std
        us = (np.asarray(u).ravel() - scale.u_mean) / scale.u_std
        return _cubic_predict_standardised(beta, xs, us) * scale.y_std + scale.y_mean

    return drift


def shared_drift_fn(model: SharedModel, e: np.ndarray, scale: Scale) -> DriftFn:
    """Unscaled drift callable for a pretrained model at embedding ``e``."""

    def drift(x, u):
        xs = (np.asarray(x).ravel() - scale.x_mean) / scale.x_std
        us = (np.asarray(u).ravel() - scale.u_mean) / scale.u_std
        return (
            _shared_predict_standardised(model, e, xs, us) * scale.y_std + scale.y_mean
        )

    return drift
