import jax
import jax.numpy as jnp
import numpy as np
import pytest

from guidon import jax_optimizer as jo
from guidon.kernels import make_step
from guidon.reference import Config
from research.experiments.check_math import check_weights


@pytest.mark.parametrize("kind", ["packed_xla", "pallas"])
@pytest.mark.parametrize("radius", [0.0, 0.15])
def test_padded_tiles_masks_groups_and_probe_age_match_oracle(kind, radius):
    rng = np.random.default_rng(91)
    params = {
        "a": jnp.asarray(rng.normal(size=(7, 19)), jnp.float32),
        "b": jnp.asarray(rng.normal(size=1031), jnp.float32),
        "bias": jnp.zeros(7, jnp.float32),
    }
    groups = {"a": 0, "b": 1, "bias": 0}
    masks = {"a": True, "b": True, "bias": False}
    config = Config(radius=radius, guide_warmup=0, guide_interval=2)
    oracle = jax.jit(jo.make_step(groups, masks, config))
    candidate = jax.jit(
        make_step(params, groups, masks, config, kind=kind, interpret=True)
    )
    expected, actual = params, params
    state_expected, state_actual = (
        jo.init(params, groups, masks),
        jo.init(params, groups, masks),
    )
    for count in range(4):
        g = jax.tree.map(
            lambda p: jnp.asarray(rng.normal(size=p.shape), jnp.float32), params
        )
        h = (
            jax.tree.map(
                lambda p: jnp.asarray(rng.normal(size=p.shape), jnp.float32), params
            )
            if config.probe_due(count)
            else None
        )
        expected, state_expected, em = oracle(expected, g, state_expected, h)
        actual, state_actual, am = candidate(actual, g, state_actual, h)
        for x, y in zip(
            jax.tree.leaves(expected), jax.tree.leaves(actual), strict=True
        ):
            np.testing.assert_allclose(x, y, rtol=2e-5, atol=3e-6)
        for x, y in zip(
            jax.tree.leaves(state_expected), jax.tree.leaves(state_actual), strict=True
        ):
            np.testing.assert_allclose(x, y, rtol=2e-5, atol=3e-6)
        assert am["schedule_ok"] and am["numerics_ok"]
        np.testing.assert_allclose(am["weights"], em["weights"], atol=2e-5)
        if radius:
            checked = check_weights(
                np.asarray(am["train_coefficients"], np.float64),
                np.asarray(am["guide_coefficients"], np.float64),
                np.asarray(am["weights"], np.float64),
                float(am["radius"]),
                config.neutrality_tolerance,
            )
            assert checked["passed"]
