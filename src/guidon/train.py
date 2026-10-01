"""Exact-token shared decoder trainer; no evaluation reader or input channel."""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import signal
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from guidon.artifacts import atomic_json, object_sha256, sha256_file
from guidon.packing import exact_updates, training_readers
from guidon.runtime import data_mesh, environment, global_batch, initialize


def implementation_identity() -> dict[str, str]:
    """Bind resume to the code defining model, input, optimizer, and state."""
    package = Path(__file__).parent
    return {
        name: sha256_file(package / name)
        for name in (
            "artifacts.py",
            "checkpoint.py",
            "initialization.py",
            "jax_optimizer.py",
            "kernels.py",
            "model.py",
            "packing.py",
            "reference.py",
            "runtime.py",
            "train.py",
        )
    }


def learning_rate(config: dict[str, Any], completed_tokens: int) -> float:
    schedule = config["schedule"]
    peak = config["optimizer_defaults"]["learning_rate"]
    fraction = min(1.0, completed_tokens / config["training_loss_tokens"])
    warmup = schedule["warmup_fraction"]
    if warmup > 0 and fraction < warmup:
        return peak * fraction / warmup
    progress = (fraction - warmup) / max(1 - warmup, 1e-12)
    minimum = schedule["minimum_learning_rate_fraction"]
    return peak * (minimum + (1 - minimum) * 0.5 * (1 + math.cos(math.pi * progress)))


def optimizer_config(config: dict[str, Any], optimizer: str) -> Any:
    from guidon.reference import Config

    values = config["optimizer_defaults"]
    return Config(
        learning_rate=values["learning_rate"],
        beta1=values["beta1"],
        beta2=values["beta2"],
        epsilon=values["epsilon"],
        weight_decay=values["weight_decay"],
        radius=values["guidon_radius"] if optimizer == "guidon" else 0,
        guide_interval=values["guide_interval"],
        guide_warmup=values["guide_warmup_updates"],
        neutrality_tolerance=values["neutrality_tolerance"],
        signal_floor=values["signal_floor"],
    )


def make_graph(
    model_config: Any,
    params: Any,
    config: dict[str, Any],
    optimizer: str,
    *,
    has_guide: bool,
):
    import jax
    import jax.numpy as jnp

    from guidon import kernels
    from guidon.model import loss_sums, optimizer_layout

    groups, masks = optimizer_layout(params)
    opt = optimizer_config(config, optimizer)
    update = kernels.make_step(
        params,
        groups,
        masks,
        opt,
        kind=config.get("optimizer_kernel", "oracle"),
        interpret=config.get("pallas_interpret", False),
    )
    clip = config["optimizer_defaults"]["global_gradient_clip_norm"]

    def objective(p, batch):
        total, count = loss_sums(p, batch, model_config)
        return total / jnp.maximum(count, 1), (total, count)

    def graph(p, state, batch, guide_batch, lr):
        (train_loss, (total, labels)), gradients = jax.value_and_grad(
            objective, has_aux=True
        )(p, batch)
        guide_loss = jnp.asarray(0.0, jnp.float32)
        guide_labels = jnp.asarray(0.0, jnp.float32)
        guide_gradients = None
        guide_gradients_finite = jnp.asarray(True)
        if has_guide:
            (guide_loss, (_, guide_labels)), guide_gradients = jax.value_and_grad(
                objective, has_aux=True
            )(p, guide_batch)
            guide_gradients_finite = jnp.all(
                jnp.stack(
                    [jnp.all(jnp.isfinite(h)) for h in jax.tree.leaves(guide_gradients)]
                )
            )
            if optimizer == "adamw_plus_guidance":
                gradients = jax.tree.map(
                    lambda g, h: (labels * g + guide_labels * h)
                    / (labels + guide_labels),
                    gradients,
                    guide_gradients,
                )
        norm = jnp.sqrt(
            sum(jnp.sum(g.astype(jnp.float32) ** 2) for g in jax.tree.leaves(gradients))
        )
        scale = jnp.minimum(1.0, clip / jnp.maximum(norm, 1e-30))
        gradients = jax.tree.map(lambda g: g * scale, gradients)
        supplied = guide_gradients if optimizer == "guidon" and opt.radius > 0 else None
        new_params, new_state, metrics = update(
            p, gradients, state, supplied, learning_rate=lr
        )
        metrics.update(
            train_nll=train_loss,
            train_nll_sum=total,
            loss_tokens=labels,
            guide_nll=guide_loss,
            guide_loss_tokens=guide_labels,
            gradient_norm=norm,
            learning_rate=lr,
            guidance_backward_due=jnp.asarray(has_guide),
            guide_gradients_finite=guide_gradients_finite,
        )
        metrics["numerics_ok"] = (
            metrics["numerics_ok"]
            & jnp.isfinite(train_loss)
            & jnp.isfinite(guide_loss)
            & jnp.isfinite(norm)
            & guide_gradients_finite
        )
        return new_params, new_state, metrics

    return graph


def json_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for key, value in metrics.items():
        array = np.asarray(value)
        if array.dtype.kind == "f":
            clean = np.where(np.isfinite(array), array, np.nan)
            result[key] = (
                (None if not np.isfinite(clean.item()) else clean.item())
                if clean.ndim == 0
                else clean.tolist()
            )
        else:
            result[key] = array.item() if array.ndim == 0 else array.tolist()
    return result


def run(
    config_path: Path,
    optimizer: str,
    seed: int,
    capability_path: Path,
    run_dir: Path,
    *,
    resume: Path | None = None,
    max_updates: int | None = None,
    purpose: str = "training",
    distributed: bool = False,
    registration: Path | None = None,
    source_bundle: Path | None = None,
    interrupt_after_updates: int | None = None,
) -> dict[str, Any]:
    invocation_started = time.monotonic()
    config = json.loads(config_path.read_text())
    if optimizer not in {"adamw", "guidon", "adamw_plus_guidance"}:
        raise ValueError("Unknown registered optimizer arm")
    if seed not in config["seeds"]:
        raise ValueError("Seed is not registered in this run configuration")
    if purpose not in {"training", "verification", "profile"}:
        raise ValueError("Unknown execution purpose")
    if purpose != "training" and (max_updates is None or not 0 < max_updates <= 512):
        raise ValueError(
            "Phase 04 verification/profiling is bounded to at most 512 updates"
        )
    if interrupt_after_updates is not None and (
        purpose != "verification" or not 0 < interrupt_after_updates < max_updates
    ):
        raise ValueError("Signal injection requires a bounded verification run")
    if purpose == "training":
        from guidon.gates import verify

        if registration is None:
            raise ValueError("Training requires a hash-addressed registered run bundle")
        matrix = json.loads(registration.read_text())
        matches = [
            row
            for row in matrix["runs"]
            if row["optimizer"] == optimizer
            and row["seed"] == seed
            and row["config_sha256"] == sha256_file(config_path)
        ]
        if len(matches) != 1:
            raise ValueError(
                "Run config/arm/seed is absent or ambiguous in registration"
            )
        if config["regime"] == "hyperparameter_sweep":
            verify(["02", "03", "04"])
            if config.get("cohort") != "pilot":
                raise ValueError("Sweep training must use its reserved pilot cohort")
        else:
            verify(["05"] if config["regime"] == "from_scratch" else ["05", "06"])
            if not config["gates"]["frozen"] or not config["data"]["manifest_ready"]:
                raise ValueError("Confirmation config/data must be frozen and audited")
        max_updates = max_updates or exact_updates(
            config["training_loss_tokens"],
            config.get(
                "loss_tokens_per_update",
                config["global_batch_sequences"] * config["sequence_length"],
            ),
        )
    cohort = config.get("cohort", "confirmation")
    readers, access = training_readers(capability_path, cohort=cohort)
    if purpose == "training":
        # Passing a corpus gate cannot authorize a different role capability.
        # The registered run and gate must both name these exact frozen bytes.
        gate = json.loads(Path("research/gates/03.json").read_text())
        registered_data = gate["data"]
        regime = (
            "scratch"
            if config["regime"] in {"from_scratch", "hyperparameter_sweep"}
            else "continued"
        )
        granted = registered_data["training_capability_sha256"][regime][cohort]
        if (
            granted != sha256_file(capability_path)
            or matches[0].get("training_capability_sha256") != granted
            or registered_data["preparation_sha256"] != access["preparation_sha256"]
        ):
            raise ValueError("Run capability is not bound to the audited corpus")
    labels_per_update = config.get(
        "loss_tokens_per_update",
        config["global_batch_sequences"] * config["sequence_length"],
    )
    if labels_per_update > config["global_batch_sequences"] * config["sequence_length"]:
        raise ValueError("Registered positive labels exceed the physical batch")
    expected_updates = exact_updates(config["training_loss_tokens"], labels_per_update)
    if config.get("expected_updates", expected_updates) != expected_updates:
        raise ValueError("Registered update count differs from exact token arithmetic")
    if readers["train"].manifest["loss_tokens"] < min(
        config["training_loss_tokens"], max_updates * labels_per_update
    ):
        raise ValueError("Train tape cannot support the authorized verification budget")
    if not resume:
        run_dir.mkdir(parents=True, exist_ok=False)
    lock = (run_dir / ".run.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    jax = initialize(distributed)
    from jax.sharding import NamedSharding, PartitionSpec

    from guidon import checkpoint
    from guidon import jax_optimizer as jo
    from guidon.model import ModelConfig, init, optimizer_layout, parameter_count

    source_identity = None
    source_params = None
    if config["model"]["family"] == "gpt_neox":
        from guidon.initialization import pretrained_bundle

        if source_bundle is None:
            raise ValueError("Continued training requires --source-bundle")
        model_config, source_params, source_identity = pretrained_bundle(
            config, source_bundle
        )
        expected_parameters = source_identity["parameter_count"]
        for reader in readers.values():
            tape_tokenizer = (
                reader.readers[0].manifest["tokenizer"]
                if hasattr(reader, "readers")
                else reader.manifest["tokenizer"]
            )
            if (
                tape_tokenizer["revision"] != config["model"]["revision"]
                or tape_tokenizer["tokenizer_json_sha256"]
                != source_identity["tokenizer_sha256"]
            ):
                raise ValueError("Continued tape tokenizer differs from source model")
    else:
        model_config = ModelConfig.from_scratch_protocol(
            config["model"],
            compute_dtype=config["hardware"]["model_compute_dtype"],
            attention_kernel=config.get("attention_kernel", "xla"),
        )
        expected_parameters = config["model"]["expected_parameters"]
    mesh = data_mesh(jax)
    placement = NamedSharding(mesh, PartitionSpec())
    if config["global_batch_sequences"] % jax.device_count():
        raise ValueError("Batch sequence count must divide the discovered data mesh")
    identity = dict(
        config_sha256=object_sha256(config),
        data_sha256=access["data_sha256"],
        data_preparation_sha256=access["preparation_sha256"],
        optimizer=optimizer,
        seed=seed,
        model=asdict(model_config),
        purpose=purpose,
        implementation_sha256=implementation_identity(),
    )
    if source_identity is not None:
        identity["initialization"] = source_identity
    rng = jax.device_put(jax.random.PRNGKey(seed), placement)
    if resume:
        params, state, rng, cursors = checkpoint.load(resume, identity, placement)
    else:
        params = jax.tree.map(
            lambda p: jax.device_put(p, placement),
            init(rng, model_config) if source_params is None else source_params,
        )
        groups, masks = optimizer_layout(params)
        state = jax.tree.map(
            lambda s: jax.device_put(s, placement), jo.init(params, groups, masks)
        )
        # The pure oracle shares its initial all-zero first/second moment tree.
        # Donation requires separate storage for those two independent states.
        state = state._replace(
            second=jax.tree.map(lambda value: value.copy(), state.second)
        )
        cursors = dict(train=0, guidance=0, completed_loss_tokens=0, guide_probes=0)
        if parameter_count(params) != expected_parameters:
            raise ValueError(
                "Actual model parameter count differs from the registered architecture"
            )
        atomic_json(
            run_dir / "run.json",
            identity
            | dict(
                environment=environment(jax),
                access_audit=access,
                parameter_count=parameter_count(params),
                state="created",
            ),
        )
    graphs = {
        has_guide: jax.jit(
            make_graph(model_config, params, config, optimizer, has_guide=has_guide),
            out_shardings=(placement, placement, placement),
            donate_argnums=(0, 1),
        )
        for has_guide in (False, True)
    }
    compile_seconds = {}
    compiled_graphs = {}
    settings = config["optimizer_defaults"]
    interrupted = False

    def interrupt(_signal, _frame):
        nonlocal interrupted
        interrupted = True

    previous = signal.signal(signal.SIGTERM, interrupt)
    started = time.monotonic()
    trace_active = False
    memory_boundaries: set[str] = set()
    log_path = run_dir / f"updates-worker{jax.process_index()}.jsonl"

    def compile_graph(due, batch, guide_batch, lr):
        then = time.monotonic()
        executable = graphs[due].lower(params, state, batch, guide_batch, lr).compile()
        compiled_graphs[due] = executable
        compile_seconds[str(due)] = time.monotonic() - then
        if purpose == "profile":
            from guidon.profile import executable_record

            name = "probe" if due else "ordinary"
            info = executable_record(executable)
            if jax.process_index() == 0:
                hlo = run_dir / f"{name}.hlo.txt"
                with hlo.open("x") as handle:
                    handle.write(executable.as_text())
                info["hlo_sha256"] = sha256_file(hlo)
            atomic_json(run_dir / f"{name}-worker{jax.process_index()}.json", info)

    try:
        if purpose == "profile":
            # Compile from shapes before warmup, without reading/consuming a
            # guide example before its scheduled probe. Cold compilation is
            # included in end-to-end cost and excluded from steady timing.
            batch_placement = NamedSharding(mesh, PartitionSpec("data", None))

            def abstract_batch(sequences):
                return {
                    name: jax.ShapeDtypeStruct(
                        (sequences, config["sequence_length"]),
                        np.float32 if name == "loss_mask" else np.int32,
                        sharding=batch_placement,
                    )
                    for name in (
                        "input_ids",
                        "labels",
                        "position_ids",
                        "segment_ids",
                        "loss_mask",
                    )
                }

            main_shape = abstract_batch(config["global_batch_sequences"])
            compile_graph(False, main_shape, None, np.float32(0))
            if optimizer != "adamw" and max_updates > settings["guide_warmup_updates"]:
                compile_graph(
                    True,
                    main_shape,
                    abstract_batch(settings["guide_batch_sequences"]),
                    np.float32(0),
                )
        with log_path.open("a") as log:
            while (
                cursors["completed_loss_tokens"] < config["training_loss_tokens"]
                and int(state.count) < max_updates
            ):
                loop_started = time.monotonic()
                count = int(state.count)
                if (
                    interrupt_after_updates is not None
                    and count >= interrupt_after_updates
                    and jax.process_index() == 0
                    and not interrupted
                ):
                    os.kill(os.getpid(), signal.SIGTERM)
                if jax.process_count() > 1:
                    from jax.experimental import multihost_utils

                    # A signal to one controller must stop all peers before the
                    # next model collective. A signal during a step is handled at
                    # the next shared boundary, after every peer finishes it.
                    stopped = multihost_utils.process_allgather(
                        np.asarray(interrupted, dtype=np.uint8), tiled=False
                    )
                    interrupted = bool(np.any(stopped))
                if interrupted:
                    break
                labels = min(
                    labels_per_update,
                    config["training_loss_tokens"] - cursors["completed_loss_tokens"],
                )
                batch, train_cursor = readers["train"].batch(
                    cursors["train"],
                    labels,
                    config["global_batch_sequences"],
                    config["sequence_length"],
                )
                due = (
                    optimizer != "adamw"
                    and count >= settings["guide_warmup_updates"]
                    and (count - settings["guide_warmup_updates"])
                    % settings["guide_interval"]
                    == 0
                )
                guide_batch, guide_cursor = None, cursors["guidance"]
                guide_labels = (
                    settings["guide_batch_sequences"] * config["sequence_length"]
                    if due
                    else 0
                )
                if due:
                    guide_batch, guide_cursor = readers["guidance"].batch(
                        guide_cursor,
                        guide_labels,
                        settings["guide_batch_sequences"],
                        config["sequence_length"],
                    )
                    guide_batch = global_batch(jax, mesh, guide_batch)
                batch = global_batch(jax, mesh, batch)
                input_seconds = time.monotonic() - loop_started
                lr = np.float32(
                    learning_rate(config, cursors["completed_loss_tokens"] + labels)
                )
                if str(due) not in compile_seconds:
                    compile_graph(due, batch, guide_batch, lr)
                if purpose == "profile" and count == 32:
                    jax.profiler.start_trace(str(run_dir / "profile"))
                    trace_active = True
                then = time.monotonic()
                with jax.profiler.StepTraceAnnotation("train", step_num=count + 1):
                    params, state, metrics = compiled_graphs[due](
                        params, state, batch, guide_batch, lr
                    )
                    jax.block_until_ready((params, state, metrics))
                step_seconds = time.monotonic() - then
                host = json_metrics(metrics)
                if not host["schedule_ok"] or not host["numerics_ok"]:
                    raise FloatingPointError(
                        "Optimizer schedule/numerics certificate failed; "
                        "retain run and traceback"
                    )
                if (
                    int(host["loss_tokens"]) != labels
                    or int(host["guide_loss_tokens"]) != guide_labels
                ):
                    raise ValueError(
                        "Global loss-label accounting differs from data cursors"
                    )
                cursors = dict(
                    train=train_cursor,
                    guidance=guide_cursor,
                    completed_loss_tokens=cursors["completed_loss_tokens"] + labels,
                    guide_probes=cursors["guide_probes"] + int(due),
                )
                rng, _ = jax.random.split(rng)
                row = host | dict(
                    update=int(state.count),
                    cursors=cursors,
                    step_seconds=step_seconds,
                    input_seconds=input_seconds,
                    loop_seconds_before_log=time.monotonic() - loop_started,
                    elapsed_seconds=time.monotonic() - started,
                )
                log.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                log.flush()
                boundary = "probe" if due else "ordinary"
                if (
                    purpose == "profile"
                    and count >= 32
                    and boundary not in memory_boundaries
                ):
                    from guidon.profile import memory_snapshot

                    memory_snapshot(
                        jax,
                        run_dir / f"memory-{boundary}-worker{jax.process_index()}.json",
                    )
                    memory_boundaries.add(boundary)
            saved = checkpoint.save(
                run_dir / "checkpoints", params, state, rng, cursors, identity
            )
        if trace_active:
            jax.profiler.stop_trace()
            trace_active = False
        result = dict(
            purpose=purpose,
            completed_updates=int(state.count),
            cursors=cursors,
            checkpoint=str(saved),
            complete_training_budget=cursors["completed_loss_tokens"]
            == config["training_loss_tokens"],
            interrupted=interrupted,
            wall_seconds=time.monotonic() - started,
            end_to_end_seconds=time.monotonic() - invocation_started,
            compile_seconds=compile_seconds,
            access_audit=access,
            memory_boundaries=sorted(memory_boundaries),
        )
        atomic_json(
            run_dir
            / f"completion-{int(state.count):06d}-worker{jax.process_index()}.json",
            result,
        )
        return result
    finally:
        if trace_active:
            jax.profiler.stop_trace()
        signal.signal(signal.SIGTERM, previous)
        lock.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--optimizer", choices=["adamw", "guidon", "adamw_plus_guidance"], required=True
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--manifest", type=Path, required=True, help="Role-limited training capability"
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--purpose", choices=["training", "verification", "profile"], default="training"
    )
    parser.add_argument("--max-updates", type=int)
    parser.add_argument("--interrupt-after-updates", type=int)
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--registration", type=Path)
    parser.add_argument("--source-bundle", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        config = json.loads(args.config.read_text())
        print(
            json.dumps(
                dict(
                    config_sha256=object_sha256(config),
                    optimizer=args.optimizer,
                    seed=args.seed,
                    purpose=args.purpose,
                    training_loss_tokens=config["training_loss_tokens"],
                    expected_updates=exact_updates(
                        config["training_loss_tokens"],
                        config.get(
                            "loss_tokens_per_update",
                            config["global_batch_sequences"]
                            * config["sequence_length"],
                        ),
                    ),
                    expected_parameters=config["model"].get("expected_parameters"),
                    data_access="train_and_guidance_only",
                    executed=False,
                ),
                indent=2,
            )
        )
        return
    value = run(
        args.config,
        args.optimizer,
        args.seed,
        args.manifest,
        args.run_dir,
        resume=args.resume,
        max_updates=args.max_updates,
        purpose=args.purpose,
        distributed=args.distributed,
        registration=args.registration,
        source_bundle=args.source_bundle,
        interrupt_after_updates=args.interrupt_after_updates,
    )
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
