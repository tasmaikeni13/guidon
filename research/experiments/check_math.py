"""Independent invariant checker: never imports the GUIDON controller."""

from __future__ import annotations

from fractions import Fraction

import numpy as np


def check_weights(a, c, weights, radius: float, tolerance: float) -> dict:
    """Check the realized correction, using long-double reductions on CPU."""
    a, c, w = (np.asarray(x, dtype=np.longdouble) for x in (a, c, weights))
    delta = w - 1
    scale = np.max(np.abs(a))
    unit = a / scale if scale else np.zeros_like(a)
    relative = abs(np.dot(unit, delta)) / max(
        np.longdouble(radius) * np.sum(np.abs(unit)), np.longdouble("1e-30")
    )
    gain = np.dot(c, delta)
    bound_error = max(float(np.max(np.abs(delta))) - radius, 0.0)
    finite = bool(np.isfinite(w).all())
    return {
        "relative_neutrality": float(relative),
        "gain": float(gain),
        "bound_error": bound_error,
        "finite": finite,
        "passed": bool(
            finite
            and relative <= tolerance
            and gain >= -1e-12 * max(float(np.sum(np.abs(c))), 1e-30)
            and bound_error <= 2e-7
        ),
    }


def oracle(a, c, radius: float, signal_floor: float = 1e-2) -> np.ndarray:
    """Orthogonal nullspace via SVD; independent from subtraction projection."""
    a, c = (np.asarray(x, dtype=np.float64) for x in (a, c))
    a_scale, c_scale = np.max(np.abs(a)), np.max(np.abs(c))
    if c_scale == 0:
        return np.ones_like(a)
    c = c / c_scale
    if a_scale:
        _, _, right = np.linalg.svd((a / a_scale)[None, :], full_matrices=True)
        basis = right[1:].T
        q = basis @ (basis.T @ c)
    else:
        q = c
    maximum = np.max(np.abs(q))
    return (
        1 + radius * q / max(maximum, signal_floor)
        if maximum > 1e-14
        else np.ones_like(a)
    )


def exact_witnesses() -> dict:
    """Rational witnesses; values remain strings to avoid decimal ambiguity."""
    r = Fraction(3, 20)
    delta = [r, -r]
    a, c = [Fraction(1), Fraction(1)], [Fraction(2), Fraction(0)]
    assert sum(x * y for x, y in zip(a, delta, strict=True)) == 0
    assert sum(x * y for x, y in zip(c, delta, strict=True)) == Fraction(3, 10)
    # F(x)=x1+x2+50(x1^2+x2^2), theta=0, D=(1+r,1-r), eta=1.
    train_before = Fraction(0)
    train_after = -2 + 50 * ((1 + r) ** 2 + (1 - r) ** 2)
    assert train_after > train_before
    # Baseline D0=(1,1) canceled by fixed decay at theta=(-1,-1).
    full_base = [Fraction(0), Fraction(0)]
    full_guided = delta
    assert full_base != full_guided
    # Zero-mean skew noise q=s*(1,-1): s=1/2 with p=2/3, -1 with p=1/3.
    assert Fraction(2, 3) * Fraction(1, 2) - Fraction(1, 3) == 0
    normalized_bias = (Fraction(2, 3) - Fraction(1, 3)) * r
    assert normalized_bias == Fraction(1, 20)
    # q=epsilon*(1,-1) has full radius for every epsilon>0, but q=0 has none.
    return {
        "two_block_delta": [str(x) for x in delta],
        "training_dot_delta": "0",
        "fresh_raw_gain": "3/10",
        "large_curvature_train_before": str(train_before),
        "large_curvature_train_after": str(train_after),
        "decay_cancelled_base": [str(x) for x in full_base],
        "decay_cancelled_guided": [str(x) for x in full_guided],
        "skew_zero_mean_normalization_bias": str(normalized_bias),
        "near_zero_one_sided_delta_limit": [str(x) for x in delta],
        "q_zero_delta": ["0", "0"],
        "stale_current_gain_with_current_c_minus_stored_c": "-3/10",
        "adverse_momentum_a": ["-1", "-1"],
        "opposite_population_dot_correction": "-3/10",
        "p10_train_losses": ["1", "0"],
        "p10_evaluation_losses": ["1", "4"],
    }


def self_check() -> dict:
    a, c = np.array([1.0, 1.0]), np.array([2.0, 0.0])
    assert check_weights(a, c, [1.15, 0.85], 0.15, 1e-5)["passed"]
    faults = {
        "wrong_sign": [0.85, 1.15],
        "no_projection": [1.15, 1.0],
        "independent_coordinate_clip": [1.15, 0.90],
        "excess_radius": [1.30, 0.70],
        "nonfinite_weights": [float("nan"), 1.0],
    }
    rejected = {
        key: not check_weights(a, c, w, 0.15, 1e-5)["passed"]
        for key, w in faults.items()
    }
    assert all(rejected.values())
    # A stale weight vector is feasible at its old plane and not at the new plane.
    assert not check_weights([1.0, 2.0], c, [1.15, 0.85], 0.15, 1e-5)["passed"]
    return {"faults_rejected": rejected, "exact": exact_witnesses()}
