import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
from jax.sharding import Mesh, NamedSharding  # noqa: E402
from jax.sharding import PartitionSpec as P  # noqa: E402

from guidon import jax_optimizer as jo  # noqa: E402
from guidon.reference import Config  # noqa: E402


@pytest.mark.skipif(jax.device_count() < 2, reason="Needs multiple JAX devices")
def test_global_sharded_reductions_match_single_device():
    rng = np.random.default_rng(142)
    number = jax.device_count()
    mesh = Mesh(np.array(jax.devices()), ("data",))
    sharding = NamedSharding(mesh, P("data"))
    p = tuple(
        jnp.asarray(rng.normal(size=(4 * number, 8)), jnp.float32) for _ in range(3)
    )
    g = tuple(jnp.asarray(rng.normal(size=x.shape), jnp.float32) for x in p)
    h = tuple(jnp.asarray(rng.normal(size=x.shape), jnp.float32) for x in p)
    ids, mask = (0, 1, 2), (True, True, True)
    config = Config(guide_warmup=0)
    update = jax.jit(jo.make_step(ids, mask, config))
    expected, _, em = update(p, g, jo.init(p, ids, mask), h)
    sp, sg, sh = (
        jax.tree.map(lambda x: jax.device_put(x, sharding), tree) for tree in (p, g, h)
    )
    actual, _, am = update(sp, sg, jo.init(sp, ids, mask), sh)
    for x, y in zip(actual, expected, strict=True):
        np.testing.assert_allclose(x, y, atol=2e-6, rtol=2e-5)
    np.testing.assert_allclose(am["weights"], em["weights"], atol=1e-6)
    assert bool(am["schedule_ok"])
