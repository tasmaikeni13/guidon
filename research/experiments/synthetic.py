"""Vectorized float64 synthetic apparatus; replayed against the public reference.

Each row is a separately initialized training seed. No evaluation arrays enter the
optimizer. The batch implementation is kept separate from its reference verifier.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Condition:
    id: str
    family: str = "quadratic"
    groups: int = 14
    condition_number: float = 100
    noise: float = 0.1
    covariance: float = 0
    batch_size: int = 32
    alignment: float = 1
    mismatch: float = 0
    interval: int = 16
    radius: float = 0.15
    guide_stream_size: int = 4096
    delay: int = 0
    beta1: float = 0.9
    rotation: float = 0
    learning_rate: float = 0.01
    steps: int = 32
    intervention: str = "ordinary"
    signal_floor: float = 0.01
    guide_batch_size: int = 8

    @property
    def dim(self) -> int:
        return max(2, self.groups)


def normalize(x):
    maximum = np.max(np.abs(x), axis=-1, keepdims=True)
    return np.divide(x, maximum, out=np.zeros_like(x), where=maximum > 0)


def coefficients(x, y, groups):
    products = x * y
    return products.sum(axis=-1, keepdims=True) if groups == 1 else products


def correction(a, c, radius, floor):
    unit = normalize(a)
    norm = np.sum(unit**2, axis=-1, keepdims=True)
    q = c - unit * np.divide(
        np.sum(unit * c, axis=-1, keepdims=True),
        norm,
        out=np.zeros_like(norm),
        where=norm > 0,
    )
    q = q - unit * np.divide(
        np.sum(unit * q, axis=-1, keepdims=True),
        norm,
        out=np.zeros_like(norm),
        where=norm > 0,
    )
    weights = 1 + radius * q / np.maximum(
        np.max(np.abs(q), axis=-1, keepdims=True), floor
    )
    realized = weights - 1
    residual = np.abs(np.sum(unit * realized, axis=-1))
    budget = 1e-5 * np.maximum(radius * np.sum(np.abs(unit), axis=-1), 1e-30)
    products = c * realized
    gain_budget = (
        4 * a.shape[-1] * np.finfo(np.float64).eps * np.sum(np.abs(products), axis=-1)
    )
    good = (
        np.isfinite(weights).all(axis=-1)
        & (residual <= budget)
        & (np.sum(products, axis=-1) >= gain_budget)
    )
    return np.where(good[:, None], weights, 1), ~good


def moments(g, first, second, count, beta1):
    first = beta1 * first + (1 - beta1) * g
    second = 0.95 * second + 0.05 * g**2
    u = (first / (1 - beta1**count)) / (np.sqrt(second / (1 - 0.95**count)) + 1e-8)
    return first, second, u


def center(dim: int, t: int, rotation: float):
    target = np.resize(np.array([0.75, -0.75]), dim).copy()
    angle = rotation * t
    x, y = target[:2].copy()
    target[0] = np.cos(angle) * x - np.sin(angle) * y
    target[1] = np.sin(angle) * x + np.cos(angle) * y
    return target


def problem_gradient(theta, condition, t, noise, x=None, y=None, *, guide=False):
    if condition.family in ("regression", "classification"):
        score = np.einsum("nbd,nd->nb", x, theta)
        if condition.family == "classification":
            prediction = 1 / (1 + np.exp(-np.clip(score, -40, 40)))
        else:
            prediction = score
        return np.einsum("nb,nbd->nd", prediction - y, x) / x.shape[1]
    eigen = np.geomspace(1, condition.condition_number, condition.dim)
    target = np.zeros(condition.dim)
    if condition.family == "rotating":
        target = 0.25 * center(condition.dim, t, condition.rotation)
    if guide:
        target = (
            condition.alignment
            * (1 - 2 * condition.mismatch)
            * center(condition.dim, t - condition.delay, condition.rotation)
        )
    return (theta - target) * eigen + noise


def measured_loss(theta, condition, target, x=None, y=None):
    if condition.family in ("regression", "classification"):
        scores = np.einsum("nbd,nd->nb", x, theta)
        if condition.family == "classification":
            return np.mean(np.logaddexp(0, scores) - y * scores, axis=-1)
        return 0.5 * np.mean((scores - y) ** 2, axis=-1)
    eigen = np.geomspace(1, condition.condition_number, condition.dim)
    return 0.5 * np.mean(eigen * (theta - target) ** 2, axis=-1)


def feature_batch(rng, n, batch, correlation, family):
    z = rng.normal(size=(n, batch, 2))
    x = z.copy()
    x[:, :, 1] = correlation * z[:, :, 0] + np.sqrt(1 - correlation**2) * z[:, :, 1]
    response = x[:, :, 0] + 0.2 * rng.normal(size=(n, batch))
    return x, (response > 0).astype(float) if family == "classification" else response


def noise_batch(rng, n, dim, sigma, covariance):
    independent = rng.normal(size=(n, dim))
    shared = rng.normal(size=(n, 1))
    return sigma * (
        np.sqrt(1 - covariance) * independent + np.sqrt(covariance) * shared
    )


def simulate(
    condition: Condition, seeds: np.ndarray, *, changed_evaluation=False, capture=False
) -> tuple[list[dict], dict]:
    """Return per-draw observations and optional full replay tapes.

    Per-draw SeedSequences give rows stable identity regardless of condition order.
    Stream zero is initialization, one training, two guidance, three evaluation.
    """
    n, dim = len(seeds), condition.dim
    rngs = [
        [
            np.random.default_rng(np.random.SeedSequence([int(seed), stream]))
            for seed in seeds
        ]
        for stream in range(4)
    ]
    initial = np.stack([rng.uniform(0.5, 1.5, dim) for rng in rngs[0]])
    if condition.family in ("regression", "classification"):
        initial[:] = 0
    theta = np.broadcast_to(initial, (3, n, dim)).copy()
    first, second = np.zeros_like(theta), np.zeros_like(theta)
    c = np.zeros((n, condition.groups))
    last = -1
    paths = []
    tapes = []
    residual_max = np.zeros(n)
    relative_max = np.zeros(n)
    proxy_min = np.zeros(n)
    fresh_min = np.zeros(n)
    current_gain = np.zeros(n)
    test_gain = np.zeros(n)
    actual_guide = np.zeros(n)
    actual_train = np.zeros(n)
    fallback_count = np.zeros(n)
    norms = np.zeros(n)
    regret = np.zeros((3, n))
    raw_coefficients = []
    guide_examples = 0
    fresh_probes = 0
    null_checks = np.ones(n, dtype=bool)
    # Finite source population uncertainty, separate from fresh per-probe batches.
    population_offset = np.stack(
        [
            rng.normal(size=dim)
            * condition.noise
            / np.sqrt(condition.guide_stream_size)
            for rng in rngs[2]
        ]
    )
    eval_x = eval_y = None
    diagnostic_train = diagnostic_guide = None
    if condition.family in ("regression", "classification"):
        evaluation_rng = [
            np.random.default_rng(
                np.random.SeedSequence([int(s), 999 if changed_evaluation else 3])
            )
            for s in seeds
        ]
        evaluation = [
            feature_batch(
                rng, 1, 256, 0.8 if changed_evaluation else 0, condition.family
            )
            for rng in evaluation_rng
        ]
        eval_x = np.concatenate([p[0] for p in evaluation])
        eval_y = np.concatenate([p[1] for p in evaluation])
        # Fixed population diagnostics are independent of optimization batches.
        diagnostic_train = [
            feature_batch(
                np.random.default_rng(np.random.SeedSequence([int(s), 4])),
                1,
                256,
                0.95,
                condition.family,
            )
            for s in seeds
        ]
        diagnostic_guide = [
            feature_batch(
                np.random.default_rng(np.random.SeedSequence([int(s), 5])),
                1,
                256,
                0.95 * condition.mismatch,
                condition.family,
            )
            for s in seeds
        ]
        diagnostic_train = tuple(
            np.concatenate([p[k] for p in diagnostic_train]) for k in range(2)
        )
        diagnostic_guide = tuple(
            np.concatenate([p[k] for p in diagnostic_guide]) for k in range(2)
        )
    for t in range(condition.steps):
        due = t % condition.interval == 0
        train_noise = np.stack(
            [
                noise_batch(
                    rng,
                    1,
                    dim,
                    condition.noise / np.sqrt(condition.batch_size),
                    condition.covariance,
                )[0]
                for rng in rngs[1]
            ]
        )
        guide_noise = None
        train_x = train_y = guide_x = guide_y = None
        if condition.family in ("regression", "classification"):
            samples = [
                feature_batch(rng, 1, condition.batch_size, 0.95, condition.family)
                for rng in rngs[1]
            ]
            train_x = np.concatenate([p[0] for p in samples])
            train_y = np.concatenate([p[1] for p in samples])
        g = np.stack(
            [
                problem_gradient(p, condition, t, train_noise, train_x, train_y)
                for p in theta
            ]
        )
        h = None
        if due:
            guide_noise = (
                np.stack(
                    [
                        noise_batch(
                            rng,
                            1,
                            dim,
                            condition.noise / np.sqrt(condition.guide_batch_size),
                            condition.covariance,
                        )[0]
                        for rng in rngs[2]
                    ]
                )
                + population_offset
            )
            if condition.family in ("regression", "classification"):
                samples = [
                    feature_batch(
                        rng,
                        1,
                        condition.guide_batch_size,
                        0.95 * condition.mismatch,
                        condition.family,
                    )
                    for rng in rngs[2]
                ]
                guide_x = np.concatenate([p[0] for p in samples])
                guide_y = np.concatenate([p[1] for p in samples])
            h = np.stack(
                [
                    problem_gradient(
                        p, condition, t, guide_noise, guide_x, guide_y, guide=True
                    )
                    for p in theta
                ]
            )
            guide_examples += condition.guide_batch_size
            if guide_examples > condition.guide_stream_size:
                raise ValueError(
                    "Fresh guidance stream exhausted; no recycling permitted"
                )
        combined = g[2].copy()
        if due:
            combined = (
                condition.batch_size * g[2] + condition.guide_batch_size * h[2]
            ) / (condition.batch_size + condition.guide_batch_size)
        first[0], second[0], u0 = moments(
            g[0], first[0], second[0], t + 1, condition.beta1
        )
        first[1], second[1], u = moments(
            g[1], first[1], second[1], t + 1, condition.beta1
        )
        first[2], second[2], u2 = moments(
            combined, first[2], second[2], t + 1, condition.beta1
        )
        a = coefficients(g[1], u, condition.groups)
        fresh = None
        if due:
            fresh = coefficients(h[1], u, condition.groups)
            raw_coefficients.append(fresh.copy())
            if condition.intervention == "reverse":
                fresh = -fresh
            elif condition.intervention == "shuffle":
                fresh = np.roll(fresh, 1, axis=-1)
            elif condition.intervention == "collinear":
                fresh = a.copy()
            elif condition.intervention == "remove":
                fresh = np.zeros_like(a)
            c = normalize(fresh)
            last = t
            fresh_probes += 1
        age_radius = condition.radius * max(0, 1 - (t - last) / condition.interval)
        weights, failed = correction(a, c, age_radius, condition.signal_floor)
        delta = weights - 1
        expanded = np.repeat(delta, dim, axis=-1) if condition.groups == 1 else delta
        d = (1 + expanded) * u
        baseline_here = theta[1] - condition.learning_rate * u
        guided_here = theta[1] - condition.learning_rate * d
        raw_residual = np.sum(a * delta, axis=-1)
        residual_max = np.maximum(residual_max, np.abs(raw_residual))
        an = normalize(a)
        relative_max = np.maximum(
            relative_max,
            np.abs(np.sum(an * delta, axis=-1))
            / np.maximum(age_radius * np.sum(np.abs(an), axis=-1), 1e-30),
        )
        proxy_min = np.minimum(proxy_min, np.sum(c * delta, axis=-1))
        if due and condition.intervention == "ordinary":
            fresh_min = np.minimum(fresh_min, np.sum(fresh * delta, axis=-1))
        if condition.family in ("regression", "classification"):
            true_h = problem_gradient(
                theta[1], condition, t, None, *diagnostic_guide, guide=True
            )
            eval_gradient = problem_gradient(
                theta[1], condition, t, None, eval_x, eval_y
            )
            guide_base = measured_loss(
                baseline_here, condition, None, *diagnostic_guide
            )
            guide_after = measured_loss(guided_here, condition, None, *diagnostic_guide)
            train_base = measured_loss(
                baseline_here, condition, None, *diagnostic_train
            )
            train_after = measured_loss(guided_here, condition, None, *diagnostic_train)
        else:
            target = (
                condition.alignment
                * (1 - 2 * condition.mismatch)
                * center(dim, t, condition.rotation)
            )
            true_h = problem_gradient(
                theta[1], condition, t + condition.delay, np.zeros_like(u), guide=True
            )
            eigen = np.geomspace(1, condition.condition_number, dim)
            eval_target = (
                -center(dim, t, condition.rotation)
                if changed_evaluation
                else center(dim, t, condition.rotation)
            )
            eval_gradient = (theta[1] - eval_target) * eigen
            guide_base = measured_loss(baseline_here, condition, target)
            guide_after = measured_loss(guided_here, condition, target)
            train_target = (
                0.25 * center(dim, t, condition.rotation)
                if condition.family == "rotating"
                else np.zeros(dim)
            )
            train_base = measured_loss(baseline_here, condition, train_target)
            train_after = measured_loss(guided_here, condition, train_target)
        current_gain += np.sum(true_h * expanded * u, axis=-1)
        test_gain += np.sum(eval_gradient * expanded * u, axis=-1)
        actual_guide += guide_base - guide_after
        actual_train += train_base - train_after
        fallback_count += failed
        norms += np.linalg.norm(d, axis=-1)
        theta[0] -= condition.learning_rate * u0
        theta[1] = guided_here
        theta[2] -= condition.learning_rate * u2
        if condition.radius == 0 or condition.intervention == "remove":
            null_checks &= np.all(theta[0] == theta[1], axis=-1)
        if condition.family in ("regression", "classification"):
            # Logistic/regression loss has no exact known finite-sample oracle here.
            pass
        else:
            target = center(dim, t, condition.rotation)
            for arm in range(3):
                regret[arm] += measured_loss(theta[arm], condition, target)
        paths.append(theta.copy())
        if capture:
            tapes.append(
                {
                    "g": g.copy(),
                    "h": None if h is None else h.copy(),
                    "weights": weights.copy(),
                    "c": c.copy(),
                }
            )
    if condition.family in ("regression", "classification"):
        train_losses = np.stack(
            [measured_loss(p, condition, None, *diagnostic_train) for p in theta]
        )
        guide_losses = np.stack(
            [measured_loss(p, condition, None, *diagnostic_guide) for p in theta]
        )
        eval_losses = np.stack(
            [measured_loss(p, condition, None, eval_x, eval_y) for p in theta]
        )
    else:
        target = center(dim, condition.steps - 1, condition.rotation)
        guide_target = condition.alignment * (1 - 2 * condition.mismatch) * target
        eval_target = -target if changed_evaluation else target
        train_target = (
            0.25 * target if condition.family == "rotating" else np.zeros(dim)
        )
        train_losses = np.stack(
            [measured_loss(p, condition, train_target) for p in theta]
        )
        guide_losses = np.stack(
            [measured_loss(p, condition, guide_target) for p in theta]
        )
        eval_losses = np.stack(
            [measured_loss(p, condition, eval_target) for p in theta]
        )
    paths = np.stack(paths)
    coefficient_variance = np.var(np.stack(raw_coefficients), axis=0).mean(axis=-1)
    rows = []
    for i, seed in enumerate(seeds):
        rows.append(
            {
                "seed": int(seed),
                "condition": condition.id,
                "train_loss": train_losses[:, i].tolist(),
                "guide_loss": guide_losses[:, i].tolist(),
                "evaluation_loss": eval_losses[:, i].tolist(),
                "oracle_regret": regret[:, i].tolist()
                if condition.family not in ("regression", "classification")
                else None,
                "update_norm_mean": float(norms[i] / condition.steps),
                "fallback_fraction": float(fallback_count[i] / condition.steps),
                "coefficient_time_variance": float(coefficient_variance[i]),
                "current_guide_linear_gain": float(current_gain[i] / condition.steps),
                "evaluation_linear_gain": float(test_gain[i] / condition.steps),
                "actual_guide_advantage": float(actual_guide[i] / condition.steps),
                "actual_train_advantage": float(actual_train[i] / condition.steps),
                "max_training_residual": float(residual_max[i]),
                "max_relative_neutrality": float(relative_max[i]),
                "min_proxy_gain": float(proxy_min[i]),
                "min_fresh_gain": float(fresh_min[i]),
                "guide_examples": guide_examples,
                "fresh_probes": fresh_probes,
                "adamw_limit_bitwise": bool(null_checks[i]),
                "final_parameters": theta[:, i].tolist(),
                "trajectory_sha256": hashlib.sha256(
                    paths[:, :, i].tobytes()
                ).hexdigest(),
                "evaluation_sha256": hashlib.sha256(
                    (eval_x[i].tobytes() + eval_y[i].tobytes())
                    if eval_x is not None
                    else eval_target.tobytes()
                ).hexdigest(),
            }
        )
    return rows, {"initial": initial, "paths": paths, "tapes": tapes}
