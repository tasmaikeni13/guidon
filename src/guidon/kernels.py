"""Grouped XLA and two-pass fused Pallas optimizers, with the readable oracle kept.

Pallas stage one emits current moments/directions and tile coefficient partials;
the certified global controller runs before stage two applies block weights.
AdamW uses a single fused moment/parameter pass. Padding never contributes a
coordinate. Kernels are candidates until full-model TPU measurements select one.
"""

from __future__ import annotations

import math
from typing import Any

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from guidon import jax_optimizer as jo
from guidon.reference import Config

PARTIAL_SHAPE = (8, 128)


class GroupedLayout:
    """Pack by (block, decay), retaining the exact scalar decay exclusions."""

    def __init__(self, params: Any, groups: Any, masks: Any):
        leaves, self.structure = jax.tree.flatten(params)
        ids, decay, layout, number = jo._layout(groups, masks)
        if layout != self.structure:
            raise ValueError("Packed optimizer layout differs from parameters")
        self.shapes = [leaf.shape for leaf in leaves]
        self.sizes = [leaf.size for leaf in leaves]
        self.keys = sorted(set(zip(ids, decay, strict=True)))
        self.buckets = [
            [i for i, pair in enumerate(zip(ids, decay, strict=True)) if pair == key]
            for key in self.keys
        ]
        self.groups = tuple(key[0] for key in self.keys)
        self.masks = tuple(key[1] for key in self.keys)
        self.number = number

    def pack(self, tree: Any) -> tuple[Any, ...]:
        leaves, structure = jax.tree.flatten(tree)
        if structure != self.structure or [p.shape for p in leaves] != self.shapes:
            raise ValueError("Packed optimizer input shape/tree differs")
        return tuple(
            jnp.concatenate([leaves[i].reshape(-1) for i in bucket])
            for bucket in self.buckets
        )

    def unpack(self, packed: tuple[Any, ...]) -> Any:
        leaves: list[Any] = [None] * len(self.shapes)
        for values, bucket in zip(packed, self.buckets, strict=True):
            offset = 0
            for index in bucket:
                leaves[index] = values[offset : offset + self.sizes[index]].reshape(
                    self.shapes[index]
                )
                offset += self.sizes[index]
        return self.structure.unflatten(leaves)


def moment_tiles(
    p: Any,
    g: Any,
    m: Any,
    v: Any,
    h: Any,
    correction1: Any,
    correction2: Any,
    lr: Any,
    config: Config,
    *,
    decay: bool,
    guided: bool,
    probe: bool,
    tile: int = 8192,
    interpret: bool = False,
) -> tuple[Any, Any, Any, Any, Any, Any]:
    """Fused FP32 moments/direction; baseline also applies its parameter update."""
    size = p.size
    padded = math.ceil(size / tile) * tile
    vectors = [
        jnp.pad(x.astype(jnp.float32), (0, padded - size)) for x in (p, g, m, v, h)
    ]
    count = padded // tile

    def kernel(
        p_ref,
        g_ref,
        m_ref,
        v_ref,
        h_ref,
        b1_ref,
        b2_ref,
        lr_ref,
        first_ref,
        second_ref,
        result_ref,
        a_ref,
        c_ref,
        flag_ref,
    ):
        gradient, previous_m, previous_v = g_ref[:], m_ref[:], v_ref[:]
        first = config.beta1 * previous_m + (1 - config.beta1) * gradient
        second = config.beta2 * previous_v + (1 - config.beta2) * gradient**2
        direction = (first / b1_ref[0]) / (
            jnp.sqrt(second / b2_ref[0]) + config.epsilon
        )
        result = (
            direction
            if guided
            else p_ref[:]
            - lr_ref[0] * (direction + config.weight_decay * p_ref[:] * decay)
        )
        valid = jnp.arange(tile) + pl.program_id(0) * tile < size
        first_ref[:], second_ref[:], result_ref[:] = first, second, result
        a_value = (
            jnp.sum(jnp.where(valid, gradient * direction, 0))
            if guided
            else jnp.float32(0)
        )
        c_value = (
            jnp.sum(jnp.where(valid, h_ref[:] * direction, 0))
            if probe
            else jnp.float32(0)
        )
        a_ref[:] = jnp.broadcast_to(a_value, PARTIAL_SHAPE)
        c_ref[:] = jnp.broadcast_to(c_value, PARTIAL_SHAPE)
        finite = (
            jnp.isfinite(gradient)
            & jnp.isfinite(first)
            & jnp.isfinite(second)
            & jnp.isfinite(direction)
            & jnp.isfinite(result)
        )
        flag_ref[:] = jnp.broadcast_to(
            jnp.all(~valid | finite).astype(jnp.int32), PARTIAL_SHAPE
        )

    vector = pl.BlockSpec((tile,), lambda i: (i,))
    scalar = pl.BlockSpec((1,), lambda i: (0,), memory_space=pltpu.SMEM)
    # Native 8x128 output windows avoid XLA's changing small 1D tile layouts.
    # Replicate each scalar partial into a window and count one element only.
    partial = pl.BlockSpec(PARTIAL_SHAPE, lambda i: (i, 0))
    shapes = tuple(jax.ShapeDtypeStruct((padded,), jnp.float32) for _ in range(3)) + (
        jax.ShapeDtypeStruct((count * PARTIAL_SHAPE[0], PARTIAL_SHAPE[1]), jnp.float32),
        jax.ShapeDtypeStruct((count * PARTIAL_SHAPE[0], PARTIAL_SHAPE[1]), jnp.float32),
        jax.ShapeDtypeStruct((count * PARTIAL_SHAPE[0], PARTIAL_SHAPE[1]), jnp.int32),
    )
    result = pl.pallas_call(
        kernel,
        out_shape=shapes,
        grid=(count,),
        in_specs=(vector,) * 5 + (scalar,) * 3,
        out_specs=(vector,) * 3 + (partial,) * 3,
        interpret=interpret,
        name="guidon_fused_moments" if guided else "adamw_fused_update",
    )(
        *vectors,
        jnp.atleast_1d(jnp.asarray(correction1, jnp.float32)),
        jnp.atleast_1d(jnp.asarray(correction2, jnp.float32)),
        jnp.atleast_1d(jnp.asarray(lr, jnp.float32)),
    )
    first, second, direction_or_params, a, c, finite = result
    return (
        first[:size],
        second[:size],
        direction_or_params[:size],
        jnp.sum(a[:: PARTIAL_SHAPE[0], 0]),
        jnp.sum(c[:: PARTIAL_SHAPE[0], 0]),
        jnp.all(finite == 1),
    )


def apply_tiles(
    p: Any,
    u: Any,
    weight: Any,
    lr: Any,
    config: Config,
    *,
    decay: bool,
    tile: int = 8192,
    interpret: bool = False,
) -> tuple[Any, Any]:
    size = p.size
    padded = math.ceil(size / tile) * tile
    count = padded // tile

    def kernel(p_ref, u_ref, w_ref, lr_ref, result_ref, flag_ref):
        result = p_ref[:] - lr_ref[0] * (
            w_ref[0] * u_ref[:] + config.weight_decay * p_ref[:] * decay
        )
        valid = jnp.arange(tile) + pl.program_id(0) * tile < size
        result_ref[:] = result
        flag_ref[:] = jnp.broadcast_to(
            jnp.all(~valid | jnp.isfinite(result)).astype(jnp.int32),
            PARTIAL_SHAPE,
        )

    vector = pl.BlockSpec((tile,), lambda i: (i,))
    scalar = pl.BlockSpec((1,), lambda i: (0,), memory_space=pltpu.SMEM)
    partial = pl.BlockSpec(PARTIAL_SHAPE, lambda i: (i, 0))
    result, finite = pl.pallas_call(
        kernel,
        out_shape=(
            jax.ShapeDtypeStruct((padded,), jnp.float32),
            jax.ShapeDtypeStruct(
                (count * PARTIAL_SHAPE[0], PARTIAL_SHAPE[1]), jnp.int32
            ),
        ),
        grid=(count,),
        in_specs=(vector, vector, scalar, scalar),
        out_specs=(vector, partial),
        interpret=interpret,
        name="guidon_scaled_parameter_update",
    )(
        jnp.pad(p, (0, padded - size)),
        jnp.pad(u, (0, padded - size)),
        jnp.atleast_1d(jnp.asarray(weight, jnp.float32)),
        jnp.atleast_1d(jnp.asarray(lr, jnp.float32)),
    )
    return result[:size], jnp.all(finite == 1)


def certified_weights(
    a: Any,
    c: Any,
    radius: Any,
    config: Config,
    number: int,
    schedule_ok: Any,
    valid_probe: Any,
) -> tuple[Any, Any]:
    """Mirror the checked v0.2 controller on actual emitted rounded coefficients."""
    a_unit = jo._normalize(a)
    squared = jnp.sum(a_unit**2)
    divisor = jnp.where(squared > 0, squared, 1)
    q = c - a_unit * jnp.sum(a_unit * c) / divisor
    q = q - a_unit * jnp.sum(a_unit * q) / divisor
    delta = radius * q / jnp.maximum(jnp.max(jnp.abs(q)), config.signal_floor)
    candidate = jax.lax.optimization_barrier(1 + delta)
    realized = candidate - 1
    error = jnp.abs(jnp.sum(a_unit * realized))
    budget = config.neutrality_tolerance * jnp.maximum(
        jnp.sum(jnp.abs(a_unit)) * radius, 1e-30
    )
    products = c * realized
    roundoff = 4 * number * jnp.finfo(jnp.float32).eps * jnp.sum(jnp.abs(products))
    valid = (
        jnp.all(jnp.isfinite(candidate))
        & (error <= budget)
        & (jnp.sum(products) >= roundoff)
        & schedule_ok
        & valid_probe
    )
    return jnp.where(valid, candidate, jnp.ones(number, jnp.float32)), valid


def make_pallas_core(
    groups: tuple[int, ...], masks: tuple[bool, ...], config: Config, *, interpret: bool
):
    number = max(groups) + 1

    def update(params, gradients, state, guide_gradients=None, learning_rate=None):
        t = state.count + 1
        lr = config.learning_rate if learning_rate is None else learning_rate
        has_guide = guide_gradients is not None
        guided = config.radius > 0
        due = (
            (state.count >= config.guide_warmup)
            & ((state.count - config.guide_warmup) % config.guide_interval == 0)
            if guided
            else jnp.asarray(False)
        )
        schedule_ok = due == has_guide if guided else jnp.asarray(not has_guide)
        first, second, adaptive_or_params = [], [], []
        a, c_raw = jnp.zeros(number, jnp.float32), jnp.zeros(number, jnp.float32)
        finite = jnp.asarray(True)
        for i, (p, g, m, v, group, decay) in enumerate(
            zip(
                params, gradients, state.first, state.second, groups, masks, strict=True
            )
        ):
            h = jnp.zeros_like(g) if not has_guide else guide_gradients[i]
            mf, vf, u, aa, cc, ok = moment_tiles(
                p,
                g,
                m,
                v,
                h,
                1 - config.beta1**t,
                1 - config.beta2**t,
                lr,
                config,
                decay=decay,
                guided=guided,
                probe=guided and has_guide,
                interpret=interpret,
            )
            first.append(mf)
            second.append(vf)
            adaptive_or_params.append(u)
            a = a.at[group].add(aa)
            c_raw = c_raw.at[group].add(cc)
            finite = finite & ok
        c, last = state.guide_coefficients, state.last_probe
        weights = jnp.ones(number, jnp.float32)
        valid = jnp.asarray(True)
        radius = jnp.asarray(0.0, jnp.float32)
        if guided:
            valid_probe = jnp.all(jnp.isfinite(c_raw))
            if has_guide:
                candidate = jnp.where(valid_probe, jo._normalize(c_raw), 0)
                c = jnp.where(schedule_ok, candidate, c)
                last = jnp.where(schedule_ok, state.count, last)
            radius = jnp.where(
                last >= 0,
                config.radius
                * jnp.maximum(0.0, 1 - (state.count - last) / config.guide_interval),
                0.0,
            )
            weights, valid = certified_weights(
                a, c, radius, config, number, schedule_ok, valid_probe
            )
            result = []
            for p, u, group, decay in zip(
                params, adaptive_or_params, groups, masks, strict=True
            ):
                value, ok = apply_tiles(
                    p, u, weights[group], lr, config, decay=decay, interpret=interpret
                )
                result.append(value)
                finite = finite & ok
        else:
            result = adaptive_or_params
        new_state = jo.State(t, tuple(first), tuple(second), c, last)
        metrics = dict(
            weights=weights,
            probe_due=due,
            schedule_ok=schedule_ok,
            numerics_ok=finite,
            fallback=~valid if guided else jnp.asarray(False),
            radius=radius,
            training_residual=jnp.sum(a * (weights - 1)),
            proxy_gain=jnp.sum(c * (weights - 1)),
            fresh_gain=jnp.where(
                has_guide & valid, jnp.sum(c_raw * (weights - 1)), jnp.nan
            ),
            train_coefficients=a,
            guide_coefficients=c,
        )
        return tuple(result), new_state, metrics

    return update


def make_step(
    params: Any,
    groups: Any,
    masks: Any,
    config: Config,
    *,
    kind: str = "oracle",
    interpret: bool = False,
):
    if kind == "oracle":
        return jo.make_step(groups, masks, config)
    if kind not in {"packed_xla", "pallas"}:
        raise ValueError("Unknown optimizer kernel")
    layout = GroupedLayout(params, groups, masks)
    core = (
        jo.make_step(layout.groups, layout.masks, config)
        if kind == "packed_xla"
        else make_pallas_core(layout.groups, layout.masks, config, interpret=interpret)
    )
    if kind == "pallas" and not interpret:
        from jax.experimental.shard_map import shard_map
        from jax.sharding import PartitionSpec

        from guidon.runtime import data_mesh

        # Mosaic is not automatically partitionable. The outer loss graph
        # already averages the global gradient. Every replica applies the same
        # local parameter update; this map adds no replica coefficient reduction.
        core = shard_map(
            core,
            mesh=data_mesh(jax),
            in_specs=(PartitionSpec(),) * 5,
            out_specs=(PartitionSpec(),) * 3,
            check_rep=False,
        )

    def update(params, gradients, state, guide_gradients=None, learning_rate=None):
        packed_state = jo.State(
            state.count,
            layout.pack(state.first),
            layout.pack(state.second),
            state.guide_coefficients,
            state.last_probe,
        )
        value, following, metrics = core(
            layout.pack(params),
            layout.pack(gradients),
            packed_state,
            None if guide_gradients is None else layout.pack(guide_gradients),
            config.learning_rate if learning_rate is None else learning_rate,
        )
        following = jo.State(
            following.count,
            layout.unpack(following.first),
            layout.unpack(following.second),
            following.guide_coefficients,
            following.last_probe,
        )
        return layout.unpack(value), following, metrics

    return update
