"""Registered full architecture and precision checks on synthetic software inputs."""

from __future__ import annotations

import json
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from guidon.artifacts import atomic_json, sha256_file
from guidon.runtime import data_mesh, environment, global_batch


def run(jax: Any, config_path: Path, destination: Path) -> dict[str, Any]:
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec

    from guidon import jax_optimizer as jo
    from guidon.model import ModelConfig, init, loss, optimizer_layout, parameter_count
    from guidon.profile import executable_record
    from guidon.train import make_graph

    protocol = json.loads(config_path.read_text())
    destination.parent.mkdir(parents=True, exist_ok=True)
    mesh = data_mesh(jax)
    placement = NamedSharding(mesh, PartitionSpec())
    model = ModelConfig.from_scratch_protocol(protocol["model"])
    params = jax.tree.map(
        lambda p: jax.device_put(p, placement),
        init(jax.random.PRNGKey(20261005), model),
    )
    jax.block_until_ready(params)
    count = parameter_count(params)
    if count != protocol["model"]["expected_parameters"]:
        raise ValueError(
            "Actual allocated architecture count differs from registration"
        )
    assert all(p.dtype == jnp.float32 for p in jax.tree.leaves(params))
    groups, masks = optimizer_layout(params)
    state = jax.tree.map(
        lambda p: jax.device_put(p, placement), jo.init(params, groups, masks)
    )
    state = state._replace(second=jax.tree.map(lambda p: p.copy(), state.second))
    assert all(
        p.dtype == jnp.float32 for p in jax.tree.leaves((state.first, state.second))
    )
    rng = np.random.default_rng(20261005)
    length = protocol["sequence_length"]

    def batch(rows: int) -> dict[str, Any]:
        ids = rng.integers(0, model.vocab_size, (rows, length), dtype=np.int32)
        labels = rng.integers(0, model.vocab_size, (rows, length), dtype=np.int32)
        positions = np.broadcast_to(np.arange(length, dtype=np.int32), ids.shape).copy()
        segments = np.repeat(np.arange(rows, dtype=np.int32)[:, None], length, axis=1)
        # Two documents per row exercise the same-document attention mask.
        positions[:, length // 2 :] -= length // 2
        segments[:, length // 2 :] += rows
        mask = np.ones(ids.shape, np.float32)
        mask[-1, -17:] = 0
        return global_batch(
            jax,
            mesh,
            dict(
                input_ids=ids,
                labels=labels,
                position_ids=positions,
                segment_ids=segments,
                loss_mask=mask,
            ),
        )

    guide = batch(protocol["optimizer_defaults"]["guide_batch_sequences"])
    fp32 = replace(model, compute_dtype="float32")
    gradients, precision_values = [], {}
    for label, specification in (("fp32", fp32), ("bf16", model)):
        evaluate = jax.jit(partial(loss, config=specification), out_shardings=placement)
        graph = jax.jit(
            jax.value_and_grad(evaluate), out_shardings=(placement, placement)
        )
        result, gradient = graph(params, guide)
        jax.block_until_ready((result, gradient))
        assert bool(
            jax.jit(
                lambda g: jnp.all(
                    jnp.stack([jnp.all(jnp.isfinite(a)) for a in jax.tree.leaves(g)])
                )
            )(gradient)
        )
        gradients.append(gradient)
        precision_values[label] = float(result)
    relative = jax.jit(
        lambda g, h: jnp.sqrt(
            sum(
                jnp.sum((a - b) ** 2)
                for a, b in zip(jax.tree.leaves(g), jax.tree.leaves(h), strict=True)
            )
            / sum(jnp.sum(a**2) for a in jax.tree.leaves(g))
        )
    )(*gradients)
    assert abs(precision_values["fp32"] - precision_values["bf16"]) <= 0.02
    assert float(relative) <= 0.03
    precision_values["gradient_relative_l2_error"] = float(relative)
    del gradients
    partial_record = dict(
        scope="allocated model and synthetic precision correctness; no training",
        parameter_count=count,
        precision=precision_values,
        environment=environment(jax),
        config_sha256=sha256_file(config_path),
    )
    atomic_json(
        destination.with_name(
            f"{destination.stem}-architecture-worker{jax.process_index()}.json"
        ),
        partial_record,
    )
    print(
        json.dumps(
            {
                "event": "architecture_precision_passed",
                "parameter_count": count,
                "precision": precision_values,
            }
        ),
        flush=True,
    )
    # Compile the actual main/probe graphs at the registered global shape.
    # These graphs are not executed here and provide no training-speed evidence.
    main = batch(protocol["global_batch_sequences"])
    configured = protocol | {"optimizer_kernel": "packed_xla"}
    compiler = {}
    for optimizer, due in (
        ("adamw", False),
        ("guidon", False),
        ("guidon", True),
        ("adamw_plus_guidance", True),
    ):
        graph = jax.jit(
            make_graph(model, params, configured, optimizer, has_guide=due),
            out_shardings=(placement, placement, placement),
            donate_argnums=(0, 1),
        )
        executable = graph.lower(
            params,
            state,
            main,
            guide if due else None,
            np.float32(protocol["optimizer_defaults"]["learning_rate"]),
        ).compile()
        name = optimizer + ("-probe" if due else "-ordinary")
        info = executable_record(executable)
        if jax.process_index() == 0:
            hlo = destination.parent / (destination.stem + "-" + name + ".hlo.txt")
            with hlo.open("x") as handle:
                handle.write(executable.as_text())
            info["hlo_sha256"] = sha256_file(hlo)
        compiler[name] = info
    result = dict(
        passed=True,
        scope=(
            "allocated full model, synthetic precision/shape checks "
            "and compiler estimates; no training or speed claim"
        ),
        config_sha256=sha256_file(config_path),
        environment=environment(jax),
        parameter_count=count,
        parameter_bytes=sum(p.size * p.dtype.itemsize for p in jax.tree.leaves(params)),
        parameter_and_moment_dtype="float32",
        model_compute_dtype="bfloat16",
        tokenizer_vocabulary=model.vocab_size,
        tied_embedding_occurrences=1,
        precision=precision_values,
        tolerances=dict(nll_absolute=0.02, gradient_relative_l2=0.03),
        positive_guide_labels=protocol["optimizer_defaults"]["guide_batch_sequences"]
        * length
        - 17,
        compiler_graphs=compiler,
    )
    path = destination.with_name(f"{destination.stem}-worker{jax.process_index()}.json")
    atomic_json(path, result)
    return result
