"""Readable float64 reference; one ndarray is one disjoint parameter block."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


@dataclass(frozen=True)
class Config:
    """Use radius=0 for the AdamW baseline, including identical moment math."""

    learning_rate: float = 6e-4
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    weight_decay: float = 0.1
    radius: float = 0.15
    guide_interval: int = 64
    guide_warmup: int = 128
    neutrality_tolerance: float = 1e-5

    def __post_init__(self) -> None:
        values = (
            self.learning_rate,
            self.beta1,
            self.beta2,
            self.epsilon,
            self.weight_decay,
            self.radius,
            self.neutrality_tolerance,
        )
        if not all(np.isfinite(x) for x in values):
            raise ValueError("Configuration values must be finite")
        if not 0 <= self.beta1 < 1 or not 0 <= self.beta2 < 1:
            raise ValueError("Moment decay must lie in [0, 1)")
        if self.learning_rate < 0 or self.weight_decay < 0 or self.epsilon <= 0:
            raise ValueError("Invalid learning rate, decay, or epsilon")
        if not 0 <= self.radius < 1 or self.neutrality_tolerance <= 0:
            raise ValueError("Radius must lie in [0, 1); tolerance must be positive")
        if self.guide_interval < 1 or self.guide_warmup < 0:
            raise ValueError("Invalid guidance schedule")

    def probe_due(self, count: int) -> bool:
        """count is the number of updates already completed, starting at zero."""
        return (
            self.radius > 0
            and count >= self.guide_warmup
            and (count - self.guide_warmup) % self.guide_interval == 0
        )


@dataclass(frozen=True)
class State:
    count: int
    first: tuple[Array, ...]
    second: tuple[Array, ...]
    guide_coefficients: Array
    last_probe: int


def init(params: tuple[Array, ...]) -> State:
    if not params:
        raise ValueError("At least one parameter block is required")
    return State(
        0,
        tuple(np.zeros_like(x, dtype=np.float64) for x in params),
        tuple(np.zeros_like(x, dtype=np.float64) for x in params),
        np.zeros(len(params), dtype=np.float64),
        -1,
    )


def normalize(x: Array) -> Array:
    """Positive rescaling changes neither the projection plane nor weights."""
    maximum = float(np.max(np.abs(x)))
    return x / maximum if maximum > 0 else np.zeros_like(x)


def project(a: Array, c: Array) -> Array:
    """Orthogonally remove training progress from guide coefficients."""
    a = normalize(a)
    norm_squared = float(a @ a)
    return c - a * float(a @ c) / norm_squared if norm_squared > 0 else c.copy()


def controller(
    a: Array, c: Array, radius: float, tolerance: float
) -> tuple[Array, bool]:
    """Reproject a possibly stale guide vector against CURRENT training progress."""
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(c)):
        return np.ones_like(a), True
    q = project(a, c)
    delta = radius * normalize(q)
    candidate = 1 + delta
    realized = candidate - 1
    # A floating-point implementation needs a certificate as well as a real proof.
    a_unit = normalize(a)
    error = abs(float(a_unit @ realized))
    budget = tolerance * max(float(np.sum(np.abs(a_unit))) * radius, 1e-30)
    valid = (
        np.all(np.isfinite(candidate)) and error <= budget and float(c @ realized) >= 0
    )
    return (candidate, False) if valid else (np.ones_like(a), True)


def step(
    params: tuple[Array, ...],
    gradients: tuple[Array, ...],
    state: State,
    config: Config,
    *,
    guide_gradients: tuple[Array, ...] | None = None,
    decay_mask: tuple[bool, ...] | None = None,
) -> tuple[tuple[Array, ...], State, dict]:
    """Apply one step; only training gradients enter Adam's two moment buffers.

    The caller computes a fresh guide gradient at the SAME pre-update parameters
    only when config.probe_due(state.count). Evaluation data has no interface here.
    """
    mask = decay_mask if decay_mask is not None else (True,) * len(params)
    if not len(params) == len(gradients) == len(state.first) == len(mask):
        raise ValueError("Block structures differ")
    if any(p.shape != g.shape for p, g in zip(params, gradients, strict=True)):
        raise ValueError("Gradient shapes differ from parameter shapes")
    if not all(np.all(np.isfinite(g)) for g in gradients):
        raise ValueError("Nonfinite training gradient; diagnose the training run")
    due = config.probe_due(state.count)
    if due != (guide_gradients is not None):
        raise ValueError("Guide gradient supplied outside the frozen probe schedule")
    if guide_gradients is not None and (
        len(guide_gradients) != len(params)
        or any(p.shape != h.shape for p, h in zip(params, guide_gradients, strict=True))
    ):
        raise ValueError("Guide gradient structure differs")

    count = state.count + 1
    first = tuple(
        config.beta1 * m + (1 - config.beta1) * g
        for m, g in zip(state.first, gradients, strict=True)
    )
    second = tuple(
        config.beta2 * v + (1 - config.beta2) * g**2
        for v, g in zip(state.second, gradients, strict=True)
    )
    adaptive = tuple(
        (m / (1 - config.beta1**count))
        / (np.sqrt(v / (1 - config.beta2**count)) + config.epsilon)
        for m, v in zip(first, second, strict=True)
    )
    a = np.array([np.sum(g * u) for g, u in zip(gradients, adaptive, strict=True)])
    c = state.guide_coefficients.copy()
    last = state.last_probe
    invalid_probe = False
    if guide_gradients is not None:
        raw = np.array(
            [np.sum(h * u) for h, u in zip(guide_gradients, adaptive, strict=True)]
        )
        invalid_probe = not np.all(np.isfinite(raw))
        c = np.zeros_like(raw) if invalid_probe else normalize(raw)
        last = state.count
    age = state.count - last
    radius = (
        config.radius * max(0.0, 1 - age / config.guide_interval) if last >= 0 else 0.0
    )
    weights, fallback = controller(a, c, radius, config.neutrality_tolerance)
    result = tuple(
        p - config.learning_rate * (w * u + config.weight_decay * p * use_decay)
        for p, u, w, use_decay in zip(params, adaptive, weights, mask, strict=True)
    )
    stats = {
        "weights": weights,
        "radius": radius,
        "probe": due,
        "guide_age": age if last >= 0 else None,
        "fallback": fallback or invalid_probe,
        "training_residual": float(a @ (weights - 1)),
        "proxy_gain": float(c @ (weights - 1)),
        "fresh_gain": float(raw @ (weights - 1)) if due and not invalid_probe else None,
    }
    return result, State(count, first, second, c, last), stats
