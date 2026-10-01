"""Shared GPT-2 and pinned GPT-NeoX decoder/loss for all optimizer arms."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class ModelConfig:
    family: str = "gpt2_decoder"
    layers: int = 12
    hidden_size: int = 768
    attention_heads: int = 12
    mlp_size: int = 3072
    vocab_size: int = 50257
    max_position_embeddings: int = 2048
    tie_embeddings: bool = True
    layer_norm_epsilon: float = 1e-5
    activation: str = "gelu_new"
    rotary_pct: float = 0.25
    rotary_base: float = 10000
    parallel_residual: bool = True
    compute_dtype: str = "bfloat16"
    attention_kernel: str = "xla"
    rematerialize: bool = True

    def __post_init__(self) -> None:
        if self.family not in {"gpt2_decoder", "gpt_neox"}:
            raise ValueError("Unsupported decoder family")
        if (
            min(self.layers, self.hidden_size, self.attention_heads, self.vocab_size)
            < 1
        ):
            raise ValueError("Model dimensions must be positive")
        if self.hidden_size % self.attention_heads:
            raise ValueError("Attention heads do not divide the hidden dimension")
        if self.compute_dtype not in {"float32", "bfloat16"}:
            raise ValueError("Matrix compute must be FP32 or bfloat16")
        if self.attention_kernel not in {"xla", "pallas"}:
            raise ValueError("Unsupported attention kernel")
        if self.activation not in {"gelu", "gelu_new"}:
            raise ValueError("Unsupported model activation")
        if self.family == "gpt_neox":
            width = int(self.hidden_size // self.attention_heads * self.rotary_pct)
            if width <= 0 or width % 2:
                raise ValueError("GPT-NeoX rotary width must be positive and even")

    @classmethod
    def from_scratch_protocol(
        cls, value: dict[str, Any], **overrides: Any
    ) -> ModelConfig:
        if value.get("dropout", 0) != 0 or not value.get("bias", True):
            raise ValueError("Only the registered bias/dropout decoder is implemented")
        fields = cls.__dataclass_fields__
        return cls(**({k: v for k, v in value.items() if k in fields} | overrides))

    @classmethod
    def from_neox_config(cls, value: dict[str, Any], **overrides: Any) -> ModelConfig:
        if value["model_type"] != "gpt_neox":
            raise ValueError("Source checkpoint is not GPT-NeoX")
        if value.get("attention_dropout", 0) or value.get("hidden_dropout", 0):
            raise ValueError(
                "Nonzero source dropout requires a registered implementation"
            )
        return cls(
            **(
                dict(
                    family="gpt_neox",
                    layers=value["num_hidden_layers"],
                    hidden_size=value["hidden_size"],
                    attention_heads=value["num_attention_heads"],
                    mlp_size=value["intermediate_size"],
                    vocab_size=value["vocab_size"],
                    max_position_embeddings=value["max_position_embeddings"],
                    tie_embeddings=value.get("tie_word_embeddings", False),
                    layer_norm_epsilon=value["layer_norm_eps"],
                    activation=value["hidden_act"],
                    rotary_pct=value["rotary_pct"],
                    rotary_base=value["rotary_emb_base"],
                    parallel_residual=value["use_parallel_residual"],
                )
                | overrides
            )
        )


def init(key: Any, config: ModelConfig) -> dict[str, Any]:
    h = config.hidden_size
    keys = iter(jax.random.split(key, 2 + config.layers * 4 + 1))

    def weight(shape: tuple[int, ...], scale: float = 0.02) -> Any:
        return jax.random.normal(next(keys), shape, dtype=jnp.float32) * scale

    def norm() -> dict[str, Any]:
        return dict(scale=jnp.ones(h, jnp.float32), bias=jnp.zeros(h, jnp.float32))

    def linear(width: int, output: int, scale: float = 0.02) -> dict[str, Any]:
        return dict(
            kernel=weight((width, output), scale), bias=jnp.zeros(output, jnp.float32)
        )

    params: dict[str, Any] = {"embedding": weight((config.vocab_size, h))}
    if config.family == "gpt2_decoder":
        params["position_embedding"] = weight((config.max_position_embeddings, h))
    blocks = []
    residual_scale = 0.02 / math.sqrt(2 * config.layers)
    for _ in range(config.layers):
        blocks.append(
            dict(
                attention_norm=norm(),
                qkv=linear(h, 3 * h),
                attention_out=linear(h, h, residual_scale),
                mlp_norm=norm(),
                mlp_in=linear(h, config.mlp_size),
                mlp_out=linear(config.mlp_size, h, residual_scale),
            )
        )
    params["blocks"] = tuple(blocks)
    params["final_norm"] = norm()
    if not config.tie_embeddings:
        params["output_embedding"] = weight((config.vocab_size, h))
    return params


def parameter_count(params: Any) -> int:
    return sum(int(p.size) for p in jax.tree.leaves(params))


def optimizer_layout(params: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Canonical ties occur once; block groups include matrices and nondecayed norms."""
    groups = jax.tree.map(lambda _: 0, params)
    groups["blocks"] = tuple(
        jax.tree.map(lambda _, group=i + 1: group, b)
        for i, b in enumerate(params["blocks"])
    )
    groups["final_norm"] = jax.tree.map(
        lambda _: len(params["blocks"]) + 1, params["final_norm"]
    )
    masks = jax.tree.map_with_path(
        lambda path, _: str(getattr(path[-1], "key", "")) not in {"bias", "scale"},
        params,
    )
    return groups, masks


def layer_norm(x: Any, params: dict[str, Any], epsilon: float) -> Any:
    x = x.astype(jnp.float32)
    mean = jnp.mean(x, axis=-1, keepdims=True)
    variance = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
    return (x - mean) * jax.lax.rsqrt(variance + epsilon) * params["scale"] + params[
        "bias"
    ]


def linear(x: Any, params: dict[str, Any], config: ModelConfig) -> Any:
    dtype = jnp.bfloat16 if config.compute_dtype == "bfloat16" else jnp.float32
    precision = (
        jax.lax.Precision.HIGHEST
        if config.compute_dtype == "float32"
        else jax.lax.Precision.DEFAULT
    )
    result = jnp.matmul(
        x.astype(dtype), params["kernel"].astype(dtype), precision=precision
    )
    return result.astype(jnp.float32) + params["bias"]


def rotary(q: Any, k: Any, positions: Any, config: ModelConfig) -> tuple[Any, Any]:
    width = int(q.shape[-1] * config.rotary_pct)
    inverse = 1 / (
        config.rotary_base ** (jnp.arange(0, width, 2, dtype=jnp.float32) / width)
    )
    frequencies = positions[..., None].astype(jnp.float32) * inverse
    angles = jnp.concatenate((frequencies, frequencies), axis=-1)[:, :, None, :]
    cosine, sine = jnp.cos(angles), jnp.sin(angles)

    def rotate(x: Any) -> Any:
        prefix, suffix = x[..., :width], x[..., width:]
        half = width // 2
        turned = jnp.concatenate((-prefix[..., half:], prefix[..., :half]), axis=-1)
        return jnp.concatenate((prefix * cosine + turned * sine, suffix), axis=-1)

    return rotate(q), rotate(k)


def attention(
    x: Any, params: dict[str, Any], batch: dict[str, Any], config: ModelConfig
) -> Any:
    n, length, _ = x.shape
    heads, width = config.attention_heads, config.hidden_size // config.attention_heads
    mixed = linear(x, params["qkv"], config)
    if config.family == "gpt_neox":
        mixed = mixed.reshape(n, length, heads, 3, width)
        q, k, v = (mixed[:, :, :, i, :] for i in range(3))
        q, k = rotary(q, k, batch["position_ids"], config)
    else:
        mixed = mixed.reshape(n, length, 3, heads, width)
        q, k, v = (mixed[:, :, i, :, :] for i in range(3))
    dtype = jnp.bfloat16 if config.compute_dtype == "bfloat16" else jnp.float32
    q, k, v = (a.astype(dtype) for a in (q, k, v))
    if config.attention_kernel == "pallas":
        from jax.experimental.pallas.ops.tpu.flash_attention import (
            SegmentIds,
            flash_attention,
        )

        ids = batch["segment_ids"]
        result = flash_attention(
            q.transpose(0, 2, 1, 3),
            k.transpose(0, 2, 1, 3),
            v.transpose(0, 2, 1, 3),
            segment_ids=SegmentIds(ids, ids),
            causal=True,
            sm_scale=1 / math.sqrt(width),
        ).transpose(0, 2, 1, 3)
    else:
        precision = (
            jax.lax.Precision.HIGHEST
            if config.compute_dtype == "float32"
            else jax.lax.Precision.DEFAULT
        )
        scores = jnp.einsum("bqhd,bkhd->bhqk", q, k, precision=precision).astype(
            jnp.float32
        ) / math.sqrt(width)
        ids = batch["segment_ids"]
        same = ids[:, :, None] == ids[:, None, :]
        causal = jnp.arange(length)[:, None] >= jnp.arange(length)[None, :]
        mask = same & causal[None, :, :]
        probabilities = jax.nn.softmax(
            jnp.where(mask[:, None, :, :], scores, -1e30), axis=-1
        )
        result = jnp.einsum(
            "bhqk,bkhd->bqhd", probabilities.astype(dtype), v, precision=precision
        )
    return linear(
        result.reshape(n, length, config.hidden_size), params["attention_out"], config
    )


def block(
    x: Any, params: dict[str, Any], batch: dict[str, Any], config: ModelConfig
) -> Any:
    normalized = layer_norm(x, params["attention_norm"], config.layer_norm_epsilon)
    attended = attention(normalized, params, batch, config)
    residual = x + attended
    mlp_input = (
        x if config.family == "gpt_neox" and config.parallel_residual else residual
    )
    normalized = layer_norm(mlp_input, params["mlp_norm"], config.layer_norm_epsilon)
    hidden = linear(normalized, params["mlp_in"], config)
    if config.activation == "gelu_new":
        hidden = (
            0.5
            * hidden
            * (1 + jnp.tanh(math.sqrt(2 / math.pi) * (hidden + 0.044715 * hidden**3)))
        )
    else:
        hidden = jax.nn.gelu(hidden, approximate=False)
    feedforward = linear(hidden, params["mlp_out"], config)
    if config.family == "gpt_neox" and config.parallel_residual:
        # Preserve the pinned source's FP32 addition order. Reassociating the
        # three parallel-residual terms changes a few full-vocabulary logits.
        combined = jax.lax.optimization_barrier(feedforward + attended)
        return combined + x
    return residual + feedforward


def logits(params: dict[str, Any], batch: dict[str, Any], config: ModelConfig) -> Any:
    x = params["embedding"][batch["input_ids"]]
    if config.family == "gpt2_decoder":
        x = x + params["position_embedding"][batch["position_ids"]]

    def step(x, p, b):
        return block(x, p, b, config)

    if config.rematerialize:
        # Blocks are statically unrolled. Under jit, CSE would otherwise retain
        # the forward attention intermediates instead of recomputing them.
        step = jax.checkpoint(step, prevent_cse=True)
    for params_block in params["blocks"]:
        x = step(x, params_block, batch)
    x = layer_norm(x, params["final_norm"], config.layer_norm_epsilon)
    output = (
        params["embedding"] if config.tie_embeddings else params["output_embedding"]
    )
    dtype = jnp.bfloat16 if config.compute_dtype == "bfloat16" else jnp.float32
    precision = (
        jax.lax.Precision.HIGHEST
        if config.compute_dtype == "float32"
        else jax.lax.Precision.DEFAULT
    )
    return jnp.matmul(
        x.astype(dtype), output.T.astype(dtype), precision=precision
    ).astype(jnp.float32)


def loss_sums(
    params: Any, batch: dict[str, Any], config: ModelConfig
) -> tuple[Any, Any]:
    """Return global FP32 NLL sum and actual positive-label count."""
    prediction = logits(params, batch, config)
    normalizer = jax.scipy.special.logsumexp(prediction, axis=-1)
    targets = jnp.take_along_axis(prediction, batch["labels"][..., None], axis=-1)[
        ..., 0
    ]
    mask = batch["loss_mask"].astype(jnp.float32)
    return jnp.sum((normalizer - targets) * mask), jnp.sum(mask)


def loss(params: Any, batch: dict[str, Any], config: ModelConfig) -> Any:
    total, count = loss_sums(params, batch, config)
    return total / jnp.maximum(count, 1)


def import_neox(source: dict[str, np.ndarray], config: ModelConfig) -> dict[str, Any]:
    """Map every source parameter explicitly; reject unknown/unmapped weights."""
    consumed: set[str] = set()

    def get(name: str, transpose: bool = False) -> Any:
        consumed.add(name)
        value = source[name].T if transpose else source[name]
        return jnp.asarray(value, jnp.float32)

    def norm(prefix: str) -> dict[str, Any]:
        return dict(scale=get(prefix + ".weight"), bias=get(prefix + ".bias"))

    def dense(prefix: str) -> dict[str, Any]:
        return dict(kernel=get(prefix + ".weight", True), bias=get(prefix + ".bias"))

    blocks = []
    for i in range(config.layers):
        prefix = f"gpt_neox.layers.{i}"
        # Older GPT-NeoX snapshots persist these deterministic buffers. Check
        # their contents against the imported architecture before regenerating.
        buffer_prefix = prefix + ".attention"
        name = buffer_prefix + ".bias"
        if name in source:
            width = config.max_position_embeddings
            expected = np.tril(np.ones((width, width), dtype=bool))[None, None]
            if not np.array_equal(source[name], expected):
                raise ValueError("Source causal-attention buffer differs")
            consumed.add(name)
        name = buffer_prefix + ".masked_bias"
        if name in source:
            value = source[name]
            # The pinned FP16 checkpoint saturates its -1e9 mask to -infinity.
            # Both represent a masked key; NaN/positive infinity are invalid.
            if value.size != 1 or np.isnan(value).any() or value.item() > -1e4:
                raise ValueError("Source masked-attention constant differs")
            consumed.add(name)
        name = buffer_prefix + ".rotary_emb.inv_freq"
        if name in source:
            width = int(
                config.hidden_size // config.attention_heads * config.rotary_pct
            )
            expected = 1 / (config.rotary_base ** (np.arange(0, width, 2) / width))
            np.testing.assert_allclose(
                source[name],
                expected,
                rtol=np.finfo(source[name].dtype).eps,
                atol=0,
            )
            consumed.add(name)
        blocks.append(
            dict(
                attention_norm=norm(prefix + ".input_layernorm"),
                qkv=dense(prefix + ".attention.query_key_value"),
                attention_out=dense(prefix + ".attention.dense"),
                mlp_norm=norm(prefix + ".post_attention_layernorm"),
                mlp_in=dense(prefix + ".mlp.dense_h_to_4h"),
                mlp_out=dense(prefix + ".mlp.dense_4h_to_h"),
            )
        )
    params = dict(
        embedding=get("gpt_neox.embed_in.weight"),
        blocks=tuple(blocks),
        final_norm=norm("gpt_neox.final_layer_norm"),
    )
    if not config.tie_embeddings:
        params["output_embedding"] = get("embed_out.weight")
    elif "embed_out.weight" in source:
        if not np.array_equal(
            source["embed_out.weight"], source["gpt_neox.embed_in.weight"]
        ):
            raise ValueError("Source checkpoint claims a tie with different weights")
        consumed.add("embed_out.weight")
    extras = set(source) - consumed
    if extras:
        raise ValueError(f"Unmapped source checkpoint tensors: {sorted(extras)}")
    return params
