"""JIT compatible GUIDON and AdamW steps over globally sharded JAX pytrees.

This is an XLA implementation. Pallas fusion and pod profiling are Phase 04;
no TPU speed result is implied by the CPU correctness checks.
"""

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from guidon.reference import Config


class State(NamedTuple):
    count: Any
    first: Any
    second: Any
    guide_coefficients: Any
    last_probe: Any


def _layout(group_ids: Any, decay_mask: Any) -> tuple[list[int], list[bool], Any, int]:
    groups, structure = jax.tree.flatten(group_ids)
    masks, mask_structure = jax.tree.flatten(decay_mask)
    if structure != mask_structure or not groups:
        raise ValueError("Group and decay mask pytrees must match and be nonempty")
    if any(not isinstance(x, int) or x < 0 for x in groups):
        raise ValueError("Group identifiers must be nonnegative integers")
    if any(not isinstance(x, bool) for x in masks):
        raise ValueError("Decay mask leaves must be booleans")
    number = max(groups) + 1
    if set(groups) != set(range(number)):
        raise ValueError("Group identifiers must be contiguous and nonempty")
    return groups, masks, structure, number


def init(params: Any, group_ids: Any, decay_mask: Any) -> State:
    """Moments and the O(number of groups) controller are stored in float32."""
    _, _, structure, number = _layout(group_ids, decay_mask)
    if jax.tree.structure(params) != structure:
        raise ValueError("Parameter and group pytrees differ")
    leaves = jax.tree.leaves(params)
    if any(p.size == 0 for p in leaves):
        raise ValueError("Parameter leaves must contain coordinates")
    if len({id(p) for p in leaves}) != len(leaves):
        raise ValueError("Tied parameters must occur only once in the pytree")
    zeros = jax.tree.map(lambda p: jnp.zeros(p.shape, jnp.float32), params)
    return State(
        jnp.asarray(0, jnp.int32),
        zeros,
        zeros,
        jnp.zeros(number, jnp.float32),
        jnp.asarray(-1, jnp.int32),
    )


def _normalize(x: Any) -> Any:
    maximum = jnp.max(jnp.abs(x))
    return x / jnp.where(maximum > 0, maximum, 1.0)


def make_step(group_ids: Any, decay_mask: Any, config: Config):
    """Return a pure step for jax.jit; compile ordinary and probe calls separately.

    A trainer MUST assert metrics['schedule_ok'] and metrics['numerics_ok'] and abort
    on False. Missing or unexpected probes use AdamW while exposing the violation.
    Reduce globally on global arrays; do not call this on unreduced pmap gradients.
    """
    groups, masks, structure, number = _layout(group_ids, decay_mask)

    def dots(left, right):
        totals = [jnp.asarray(0.0, jnp.float32) for _ in range(number)]
        for x, y, group in zip(left, right, groups, strict=True):
            totals[group] = totals[group] + jnp.sum(x.astype(jnp.float32) * y)
        return jnp.stack(totals)

    def update(params, gradients, state, guide_gradients=None, learning_rate=None):
        ps, ps_structure = jax.tree.flatten(params)
        gs, gs_structure = jax.tree.flatten(gradients)
        if ps_structure != structure or gs_structure != structure:
            raise ValueError(
                "Parameter or gradient tree differs from the frozen layout"
            )
        if any(p.shape != g.shape for p, g in zip(ps, gs, strict=True)):
            raise ValueError("Gradient shapes differ")
        ms = jax.tree.leaves(state.first)
        vs = jax.tree.leaves(state.second)
        t = state.count + 1
        first = [
            config.beta1 * m + (1 - config.beta1) * g.astype(jnp.float32)
            for m, g in zip(ms, gs, strict=True)
        ]
        second = [
            config.beta2 * v + (1 - config.beta2) * jnp.square(g.astype(jnp.float32))
            for v, g in zip(vs, gs, strict=True)
        ]
        adaptive = [
            (m / (1 - config.beta1**t))
            / (jnp.sqrt(v / (1 - config.beta2**t)) + config.epsilon)
            for m, v in zip(first, second, strict=True)
        ]
        has_guide = guide_gradients is not None
        c, last = state.guide_coefficients, state.last_probe
        weights = jnp.ones(number, jnp.float32)
        due = jnp.asarray(False)
        schedule_ok = jnp.asarray(not has_guide)
        fallback = jnp.asarray(False)
        a = jnp.zeros(number, jnp.float32)
        residual = proxy_gain = fresh_gain = radius = jnp.asarray(0.0, jnp.float32)
        if config.radius > 0:
            due = (state.count >= config.guide_warmup) & (
                (state.count - config.guide_warmup) % config.guide_interval == 0
            )
            schedule_ok = due == has_guide
            a = dots(gs, adaptive)
            raw_c = jnp.zeros_like(c)
            valid_probe = jnp.asarray(True)
            if has_guide:
                hs, hs_structure = jax.tree.flatten(guide_gradients)
                if hs_structure != structure or any(
                    p.shape != h.shape for p, h in zip(ps, hs, strict=True)
                ):
                    raise ValueError("Guide gradient tree or shapes differ")
                raw_c = dots(hs, adaptive)
                valid_probe = jnp.all(jnp.isfinite(raw_c))
                candidate = jnp.where(valid_probe, _normalize(raw_c), 0.0)
                c = jnp.where(schedule_ok, candidate, c)
                last = jnp.where(schedule_ok, state.count, last)
            radius = jnp.where(
                last >= 0,
                config.radius
                * jnp.maximum(0.0, 1.0 - (state.count - last) / config.guide_interval),
                0.0,
            )
            a_unit = _normalize(a)
            norm_squared = jnp.sum(jnp.square(a_unit))
            q = c - a_unit * jnp.sum(a_unit * c) / jnp.where(
                norm_squared > 0, norm_squared, 1.0
            )
            q = q - a_unit * jnp.sum(a_unit * q) / jnp.where(
                norm_squared > 0, norm_squared, 1.0
            )
            delta = radius * q / jnp.maximum(jnp.max(jnp.abs(q)), config.signal_floor)
            # XLA may otherwise cancel (1 + delta) - 1 across a rounding boundary.
            candidate_weights = jax.lax.optimization_barrier(1.0 + delta)
            realized_delta = candidate_weights - 1.0
            error = jnp.abs(jnp.sum(a_unit * realized_delta))
            budget = config.neutrality_tolerance * jnp.maximum(
                jnp.sum(jnp.abs(a_unit)) * radius, 1e-30
            )
            gain_products = c * realized_delta
            gain_roundoff = (
                4
                * number
                * jnp.finfo(jnp.float32).eps
                * jnp.sum(jnp.abs(gain_products))
            )
            valid = (
                jnp.all(jnp.isfinite(candidate_weights))
                & (error <= budget)
                & (jnp.sum(gain_products) >= gain_roundoff)
                & schedule_ok
                & valid_probe
            )
            fallback = ~valid
            weights = jnp.where(valid, candidate_weights, weights)
            residual = jnp.sum(a * (weights - 1))
            proxy_gain = jnp.sum(c * (weights - 1))
            fresh_gain = jnp.where(
                has_guide & valid, jnp.sum(raw_c * (weights - 1)), jnp.nan
            )
        lr = config.learning_rate if learning_rate is None else learning_rate
        result = [
            p - lr * (weights[group] * u + config.weight_decay * p * mask)
            for p, u, group, mask in zip(ps, adaptive, groups, masks, strict=True)
        ]
        numerics_ok = jnp.all(
            jnp.stack(
                [
                    jnp.all(jnp.isfinite(x))
                    for x in gs + first + second + adaptive + result
                ]
            )
        )
        new_state = State(
            t,
            structure.unflatten(first),
            structure.unflatten(second),
            c,
            last,
        )
        metrics = {
            "weights": weights,
            "probe_due": due,
            "schedule_ok": schedule_ok,
            "numerics_ok": numerics_ok,
            "fallback": fallback,
            "radius": radius,
            "training_residual": residual,
            "proxy_gain": proxy_gain,
            "fresh_gain": fresh_gain,
            "train_coefficients": a,
            "guide_coefficients": c,
        }
        return structure.unflatten(result), new_state, metrics

    return update


def environment() -> dict:
    """Hardware discovery: v4-32 names TensorCores; JAX device count is discovered."""
    return {
        "jax": jax.__version__,
        "numpy": np.__version__,
        "backend": jax.default_backend(),
        "process_count": jax.process_count(),
        "process_index": jax.process_index(),
        "device_count": jax.device_count(),
        "local_device_count": jax.local_device_count(),
        "devices": [str(d) for d in jax.devices()],
    }
