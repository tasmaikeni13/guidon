from dataclasses import replace

import numpy as np
import pytest

from guidon.reference import Config, controller, init, project, step


def test_hand_computed_projection_and_limits():
    a, c = np.array([1.0, 1.0]), np.array([2.0, 0.0])
    np.testing.assert_allclose(project(a, c), [1, -1])
    w, failed = controller(a, c, 0.15, 1e-5)
    np.testing.assert_allclose(w, [1.15, 0.85])
    assert not failed
    assert a @ (w - 1) == pytest.approx(0)
    assert c @ (w - 1) == pytest.approx(0.3)
    np.testing.assert_array_equal(controller(a, c, 0, 1e-5)[0], [1, 1])
    np.testing.assert_array_equal(controller(a, a, 0.15, 1e-5)[0], [1, 1])
    np.testing.assert_allclose(project(np.zeros(2), c), c)


def test_random_projection_with_independent_invariants():
    rng = np.random.default_rng(981)
    for _ in range(200):
        number = int(rng.integers(2, 65))
        a = rng.normal(size=number) * 10.0 ** rng.uniform(-120, 120)
        c = rng.normal(size=number)
        rho = float(rng.uniform(0, 0.9))
        w, failed = controller(a, c, rho, 1e-5)
        assert not failed
        assert abs((a / np.max(np.abs(a))) @ (w - 1)) < 1e-12
        assert np.max(np.abs(w - 1)) <= rho + 1e-15
        assert c @ (w - 1) >= -1e-12


def test_guidance_changes_weights_but_never_moments():
    p = (np.array([1.0, -0.5]), np.array([0.2, 0.3]))
    g = (np.array([1.0, 1.0]), np.array([1.0, -1.0]))
    h = (np.array([2.0, 2.0]), np.array([-1.0, 1.0]))
    config = Config(guide_warmup=0, guide_interval=3)
    guided, s, stats = step(p, g, init(p), config, guide_gradients=h)
    baseline, b, _ = step(p, g, init(p), replace(config, radius=0))
    for m, n in zip(s.first + s.second, b.first + b.second, strict=True):
        np.testing.assert_array_equal(m, n)
    assert any(not np.array_equal(x, y) for x, y in zip(guided, baseline, strict=True))
    assert stats["fresh_gain"] > 0


def test_current_training_gradient_reprojects_stale_controller():
    p = (np.array([0.1]), np.array([0.2]), np.array([0.3]))
    config = Config(guide_warmup=0, guide_interval=3)
    s = init(p)
    for t in range(7):
        g = tuple(np.array([x]) for x in (1 + t, 2 - t, 0.5 + 0.3 * t))
        h = tuple(np.array([x]) for x in (2, -3, 1)) if config.probe_due(t) else None
        p, s, stats = step(p, g, s, config, guide_gradients=h)
        assert abs(stats["training_residual"]) < 1e-12
        assert stats["proxy_gain"] >= -1e-12
        assert (stats["fresh_gain"] is not None) == (h is not None)


def test_schedule_and_nonfinite_probe_are_visible():
    p = (np.array([0.1]), np.array([0.2]))
    g = (np.array([1.0]), np.array([2.0]))
    config = Config(guide_warmup=0)
    with pytest.raises(ValueError, match="schedule"):
        step(p, g, init(p), config)
    result, _, stats = step(
        p,
        g,
        init(p),
        config,
        guide_gradients=(np.array([np.nan]), np.array([1.0])),
    )
    expected, _, _ = step(p, g, init(p), replace(config, radius=0))
    assert stats["fallback"]
    for a, b in zip(result, expected, strict=True):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"radius": 1},
        {"beta1": 1},
        {"epsilon": 0},
        {"guide_interval": 0},
        {"learning_rate": float("nan")},
    ],
)
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        Config(**kwargs)
