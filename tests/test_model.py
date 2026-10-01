from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from guidon.model import (
    ModelConfig,
    init,
    logits,
    loss_sums,
    optimizer_layout,
    parameter_count,
)


def tiny_config():
    return ModelConfig(
        layers=2,
        hidden_size=16,
        attention_heads=2,
        mlp_size=64,
        vocab_size=32,
        max_position_embeddings=16,
        compute_dtype="float32",
    )


def batch(ids, segments=None, mask=None):
    ids = jnp.asarray(ids, jnp.int32)
    return dict(
        input_ids=ids,
        labels=(ids + 1) % 32,
        position_ids=jnp.broadcast_to(jnp.arange(ids.shape[1]), ids.shape),
        segment_ids=jnp.zeros_like(ids) if segments is None else jnp.asarray(segments),
        loss_mask=jnp.ones_like(ids, dtype=jnp.float32)
        if mask is None
        else jnp.asarray(mask),
    )


def test_count_and_layout_match_registered_formula_without_duplicate_tie():
    config = tiny_config()
    params = init(jax.random.PRNGKey(0), config)
    h = config.hidden_size
    expected = (
        config.vocab_size * h
        + config.max_position_embeddings * h
        + config.layers * (12 * h * h + 13 * h)
        + 2 * h
    )
    assert parameter_count(params) == expected
    groups, masks = optimizer_layout(params)
    assert set(jax.tree.leaves(groups)) == set(range(config.layers + 2))
    assert masks["blocks"][0]["attention_norm"] == {"scale": False, "bias": False}
    assert masks["embedding"]
    actual = ModelConfig()
    assert (
        actual.vocab_size * 768 + 2048 * 768 + 12 * (12 * 768**2 + 13 * 768) + 2 * 768
        == 125_226_240
    )


def test_causal_and_document_isolation_for_logits():
    config = tiny_config()
    params = init(jax.random.PRNGKey(1), config)
    a = batch([[1, 2, 3, 4]], [[0, 0, 1, 1]])
    b = batch([[8, 9, 3, 4]], [[0, 0, 1, 1]])
    # Position IDs are document-local in the actual loader; keep them fixed here.
    out_a, out_b = logits(params, a, config), logits(params, b, config)
    np.testing.assert_array_equal(out_a[:, 2:], out_b[:, 2:])
    future = batch([[1, 2, 12, 13]], [[0, 0, 1, 1]])
    np.testing.assert_array_equal(out_a[:, :2], logits(params, future, config)[:, :2])


def test_masked_nll_sums_match_independent_numpy_and_padding_gradient():
    config = tiny_config()
    params = init(jax.random.PRNGKey(2), config)
    data = batch([[1, 2, 3, 4]], mask=[[1, 1, 0, 0]])
    scores = np.asarray(logits(params, data, config), dtype=np.float64)
    normalizer = np.log(
        np.exp(scores - scores.max(-1, keepdims=True)).sum(-1)
    ) + scores.max(-1)
    targets = np.take_along_axis(scores, np.asarray(data["labels"])[..., None], -1)[
        ..., 0
    ]
    total, count = loss_sums(params, data, config)
    np.testing.assert_allclose(
        float(total), ((normalizer - targets) * [[1, 1, 0, 0]]).sum(), rtol=1e-6
    )
    assert float(count) == 2
    bf16 = replace(config, compute_dtype="bfloat16")
    assert jnp.isfinite(loss_sums(params, data, bf16)[0])
