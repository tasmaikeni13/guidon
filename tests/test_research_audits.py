"""Regression coverage for independently found optimizer/audit failures."""

import json
from pathlib import Path

import numpy as np
import pytest

from guidon.reference import Config, controller, init
from research.experiments.check_math import check_weights, self_check
from research.experiments.synthetic import Condition
from research.experiments.verify_simulation import manufactured_checks, reference_replay


def test_checker_rejects_mutated_optimizer_behaviors():
    assert all(self_check()["faults_rejected"].values())


def test_signal_floor_removes_near_zero_full_radius_jump():
    a = np.ones(2)
    for magnitude in (1e-3, 1e-6, 1e-9):
        c = np.array([magnitude, -magnitude])
        weights, _ = controller(a, c, 0.15, 1e-5)
        assert np.max(np.abs(weights - 1)) <= 0.15 * magnitude / 0.01 + 1e-15
        assert check_weights(a, c, weights, 0.15, 1e-5)["passed"]


def test_empty_and_tied_reference_coordinates_are_rejected():
    p = np.ones(3)
    with pytest.raises(ValueError, match="once"):
        init((p, p))
    with pytest.raises(ValueError, match="coordinates"):
        init((np.zeros(0),))


def test_synthetic_apparatus_matches_manufactured_and_reference_problems():
    assert manufactured_checks()["finite_difference_gradient"]
    for family in ("quadratic", "rotating", "regression", "classification"):
        case = Condition("test", family=family, groups=2, steps=4, rotation=0.1)
        result = reference_replay(case, np.array([173, 419], dtype=np.uint64))
        assert result["evaluation_noninterference_bitwise"]


def test_compiled_certificate_checks_returned_rounded_weights():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    from guidon import jax_optimizer as jo

    cfg = Config(guide_warmup=0, weight_decay=0)
    update = jo.make_step((0, 1), (False, False), cfg)

    def single(g, h):
        p = (jnp.zeros(1), jnp.zeros(1))
        _, _, metrics = update(
            p, (g[:1], g[1:]), jo.init(p, (0, 1), (False, False)), (h[:1], h[1:])
        )
        return metrics

    rng = np.random.default_rng(19072)
    g = rng.uniform(0.25, 1, (256, 2)).astype(np.float32)
    h = (g * (1 + rng.normal(size=g.shape) * 1e-7)).astype(np.float32)
    metrics = jax.tree.map(
        np.asarray, jax.jit(jax.vmap(single))(jnp.asarray(g), jnp.asarray(h))
    )
    for a, c, w in zip(
        metrics["train_coefficients"],
        metrics["guide_coefficients"],
        metrics["weights"],
        strict=True,
    ):
        assert check_weights(a, c, w, cfg.radius, cfg.neutrality_tolerance)["passed"]
    assert metrics["numerics_ok"].all()


def test_float32_moment_overflow_is_a_visible_run_failure():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    from guidon import jax_optimizer as jo

    p, ids, mask = (jnp.zeros(1),), (0,), (False,)
    cfg = Config(radius=0)
    update = jax.jit(jo.make_step(ids, mask, cfg))
    _, _, metrics = update(p, (jnp.array([1e30]),), jo.init(p, ids, mask))
    assert not bool(metrics["numerics_ok"])


def test_accumulation_then_shared_clipping_preserves_the_supplied_plane():
    micro = [np.array([2.0, -1.0]), np.array([1.0, 3.0])]
    gradient = np.mean(micro, axis=0)
    clipped = gradient / max(np.linalg.norm(gradient), 1)
    adaptive = np.array([-0.2, 0.6])
    weights, _ = controller(clipped * adaptive, np.array([1.0, -0.5]), 0.15, 1e-5)
    assert abs((gradient * adaptive) @ (weights - 1)) < 1e-12


def test_registered_design_covers_required_factors_and_draws():
    path = Path(__file__).resolve().parents[1] / "research/phase02/protocol.json"
    protocol = json.loads(path.read_text())
    assert protocol["draws_per_primary_condition_per_stream"] >= 1000
    conditions = protocol["conditions"]
    for key, required in {
        "groups": {1, 2, 14, 64},
        "interval": {1, 16, 64, 256},
        "radius": {0, 0.05, 0.15, 0.3},
        "condition_number": {1, 10000},
    }.items():
        assert required <= {c[key] for c in conditions}


def test_zero_coefficients_adverse_momentum_and_fixed_decay_witnesses():
    from dataclasses import replace

    from guidon.reference import State, step

    weights, failed = controller(np.zeros(2), np.array([-0.5, 1.0]), 0.15, 1e-5)
    assert not failed
    np.testing.assert_allclose(weights, [0.925, 1.15])
    np.testing.assert_array_equal(controller(np.ones(2), np.zeros(2), 0.15, 1e-5)[0], 1)
    p = (np.zeros(1), np.zeros(1))
    g = (np.ones(1), np.ones(1))
    state = State(
        0, (-np.ones(1), -np.ones(1)), (np.ones(1), np.ones(1)), np.zeros(2), -1
    )
    cfg = Config(radius=0, weight_decay=0)
    changed, _, _ = step(p, g, state, cfg)
    assert sum(float(x[0]) for x in changed) > 0  # F=x1+x2 ascends.
    cfg = Config(beta1=0, beta2=0, weight_decay=1, guide_warmup=0)
    u = 1 / (1 + cfg.epsilon)
    p = (-u * np.ones(1), -u * np.ones(1))
    baseline, _, _ = step(p, g, init(p), replace(cfg, radius=0))
    guided, _, _ = step(
        p, g, init(p), cfg, guide_gradients=(2 * np.ones(1), np.zeros(1))
    )
    for before, after in zip(p, baseline, strict=True):
        np.testing.assert_array_equal(before, after)
    assert any(not np.array_equal(x, y) for x, y in zip(p, guided, strict=True))
