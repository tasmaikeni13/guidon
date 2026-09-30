from dataclasses import replace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
optax = pytest.importorskip("optax")

from guidon import jax_optimizer as jo  # noqa: E402
from guidon.reference import Config, init, step  # noqa: E402


def test_adamw_matches_independent_optax_for_multiple_steps():
    rng = np.random.default_rng(24)
    p = {
        "block": jnp.asarray(rng.normal(size=(4, 8)), jnp.float32),
        "bias": jnp.zeros(8),
    }
    ids, mask = {"block": 0, "bias": 0}, {"block": True, "bias": False}
    config = Config(radius=0, learning_rate=2e-3)
    s = jo.init(p, ids, mask)
    update = jax.jit(jo.make_step(ids, mask, config))
    tx = optax.adamw(
        config.learning_rate,
        b1=config.beta1,
        b2=config.beta2,
        eps=config.epsilon,
        weight_decay=config.weight_decay,
        mask=mask,
    )
    b, bs = p, tx.init(p)
    for _ in range(25):
        g = jax.tree.map(lambda x: jnp.asarray(rng.normal(size=x.shape), x.dtype), p)
        p, s, metrics = update(p, g, s)
        updates, bs = tx.update(g, bs, b)
        b = optax.apply_updates(b, updates)
        assert bool(metrics["schedule_ok"])
    for x, y in zip(jax.tree.leaves(p), jax.tree.leaves(b), strict=True):
        np.testing.assert_allclose(x, y, atol=3e-6, rtol=2e-5)


def test_jit_guidon_matches_float64_reference_and_probe_schedule():
    rng = np.random.default_rng(38)
    ps = tuple(rng.normal(size=(4, 8)) for _ in range(3))
    p = tuple(jnp.asarray(x, jnp.float32) for x in ps)
    ids, mask = (0, 1, 2), (True, False, True)
    config = Config(learning_rate=1e-3, guide_warmup=2, guide_interval=4)
    s, ns = jo.init(p, ids, mask), init(ps)
    update = jax.jit(jo.make_step(ids, mask, config))
    for t in range(15):
        g = tuple(rng.normal(size=x.shape) for x in ps)
        h = tuple(rng.normal(size=x.shape) for x in ps) if config.probe_due(t) else None
        p, s, metrics = update(
            p,
            tuple(jnp.asarray(x, jnp.float32) for x in g),
            s,
            None if h is None else tuple(jnp.asarray(x, jnp.float32) for x in h),
        )
        ps, ns, stats = step(ps, g, ns, config, guide_gradients=h, decay_mask=mask)
        assert bool(metrics["schedule_ok"])
        assert not bool(metrics["fallback"])
        np.testing.assert_allclose(metrics["weights"], stats["weights"], atol=2e-6)
        assert float(metrics["proxy_gain"]) >= -1e-6
        for x, y in zip(p, ps, strict=True):
            np.testing.assert_allclose(x, y, atol=3e-6, rtol=2e-5)


def test_group_reductions_cover_all_leaves_and_keep_decay_fixed():
    p = {"a": jnp.array([0.1, 0.2]), "b": jnp.array([0.3]), "c": jnp.array([0.4])}
    g = jax.tree.map(jnp.ones_like, p)
    h = {"a": jnp.array([2.0, 2.0]), "b": jnp.array([2.0]), "c": jnp.array([-2.0])}
    ids, mask = {"a": 0, "b": 0, "c": 1}, {"a": True, "b": False, "c": True}
    config = Config(guide_warmup=0)
    update = jax.jit(jo.make_step(ids, mask, config))
    guided, s, metrics = update(p, g, jo.init(p, ids, mask), h)
    base_update = jax.jit(jo.make_step(ids, mask, replace(config, radius=0)))
    baseline, b, _ = base_update(p, g, jo.init(p, ids, mask))
    for x, y in zip(jax.tree.leaves(s.first), jax.tree.leaves(b.first), strict=True):
        np.testing.assert_array_equal(x, y)
    # Three coordinates in group zero versus one coordinate in group one.
    assert float(
        3 * (metrics["weights"][0] - 1) + metrics["weights"][1] - 1
    ) == pytest.approx(0, abs=1e-6)
    assert any(
        not np.array_equal(x, y)
        for x, y in zip(jax.tree.leaves(guided), jax.tree.leaves(baseline), strict=True)
    )


def test_missing_probe_flags_protocol_violation_and_falls_back():
    p, g = (jnp.ones(4), jnp.ones(4)), (jnp.ones(4), -jnp.ones(4))
    ids, mask = (0, 1), (True, True)
    config = Config(guide_warmup=0)
    _, _, metrics = jax.jit(jo.make_step(ids, mask, config))(
        p, g, jo.init(p, ids, mask)
    )
    assert not bool(metrics["schedule_ok"])
    assert bool(metrics["fallback"])
    np.testing.assert_array_equal(metrics["weights"], [1, 1])
