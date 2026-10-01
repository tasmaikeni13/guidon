"""TPU kernel correctness and registered full-decoder profiling summaries."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np

from guidon.artifacts import atomic_json, sha256_file
from guidon.runtime import data_mesh, environment, initialize


def collective_records(hlo: str) -> list[dict[str, Any]]:
    """Count every tuple member once; distinguish payloads from network traffic."""
    widths = {
        "pred": 1,
        "bf16": 2,
        "f16": 2,
        "f32": 4,
        "f64": 8,
        "s8": 1,
        "s16": 2,
        "s32": 4,
        "s64": 8,
        "u8": 1,
        "u16": 2,
        "u32": 4,
        "u64": 8,
    }
    completed = {}
    for line in hlo.splitlines():
        done = re.search(
            r"=\s*(.*?)\s+(?:all-reduce|all-gather|collective-permute)-done\((%[^,)]+)\)",
            line,
        )
        if done:
            completed[done[2]] = done[1]
    result = []
    for line in hlo.splitlines():
        operation = re.search(
            r"=\s*(.*?)\s+(all-reduce(?:-start)?|all-gather(?:-start)?|"
            r"reduce-scatter|collective-permute(?:-start)?)\(",
            line,
        )
        if not operation:
            continue
        shape_text = operation[1]
        if operation[2].endswith("-start"):
            # An asynchronous start returns input/output/context state. Its done
            # result carries the actual payload; summing start's state doubles it.
            shape_text = completed.get(line.split("=", 1)[0].strip(), "")
        shapes = re.findall(r"\b(\w+)\[([0-9,]*)\]", shape_text)
        known = bool(shapes) and all(dtype in widths for dtype, _ in shapes)
        size = (
            sum(
                math.prod(int(x) for x in dimensions.split(",") if x) * widths[dtype]
                for dtype, dimensions in shapes
            )
            if known
            else None
        )
        groups = re.search(r"replica_groups=(.*?), use_global_device_ids", line)
        result.append(
            dict(
                operation=operation[2],
                output_bytes=size,
                tuple_members=len(shapes),
                replica_groups=groups[1] if groups else None,
                instruction=line.strip(),
            )
        )
    return result


def memory_snapshot(jax: Any, destination: Path) -> dict[str, Any]:
    """Collect allocator peaks and a live-buffer profile without guessing values."""
    devices = []
    for device in jax.local_devices():
        values = device.memory_stats()
        devices.append(
            dict(
                id=device.id,
                process_index=device.process_index,
                allocator_statistics=(
                    {name: int(value) for name, value in values.items()}
                    if values is not None
                    else None
                ),
            )
        )
    result = dict(
        backend=jax.default_backend(),
        devices=devices,
        scope="allocator statistics since process initialization; live buffers "
        "at this synchronized boundary; compiler estimates are separate",
    )
    try:
        jax.profiler.save_device_memory_profile(str(destination.with_suffix(".prof")))
    except (NotImplementedError, RuntimeError) as error:
        result["live_buffer_profile_unavailable"] = str(error)
    else:
        result["live_buffer_profile_sha256"] = sha256_file(
            destination.with_suffix(".prof")
        )
    atomic_json(destination, result)
    return result


def executable_record(executable: Any) -> dict[str, Any]:
    """Record compiler estimates and explicit collective shapes, not a speed claim."""
    analysis = executable.cost_analysis()
    if isinstance(analysis, list):
        analysis = analysis[0]
    memory = executable.memory_analysis()
    memory_values = (
        {
            name: int(getattr(memory, name))
            for name in (
                "argument_size_in_bytes",
                "output_size_in_bytes",
                "alias_size_in_bytes",
                "temp_size_in_bytes",
                "generated_code_size_in_bytes",
                "host_argument_size_in_bytes",
                "host_output_size_in_bytes",
                "host_temp_size_in_bytes",
                "host_alias_size_in_bytes",
            )
            if hasattr(memory, name)
        }
        if memory is not None
        else {}
    )
    if memory_values:
        memory_values["estimated_peak_bytes"] = (
            memory_values["argument_size_in_bytes"]
            + memory_values["output_size_in_bytes"]
            - memory_values["alias_size_in_bytes"]
            + memory_values["temp_size_in_bytes"]
        )
    collectives = collective_records(executable.as_text())
    return dict(
        compiler_cost_estimates={key: float(value) for key, value in analysis.items()},
        compiler_memory_estimates=memory_values,
        collectives=collectives,
        communication_scope=(
            "per compiled partition output bytes; "
            "network traffic requires topology scaling"
        ),
    )


def kernel_audit(jax: Any, destination: Path) -> dict[str, Any]:
    """Compare actual compiled grouped/Pallas paths with the unfused FP32 oracle."""
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec

    from guidon import jax_optimizer as jo
    from guidon.kernels import make_step
    from guidon.reference import Config
    from research.experiments.check_math import check_weights

    mesh = data_mesh(jax)
    placement = NamedSharding(mesh, PartitionSpec())
    rng = np.random.default_rng(20261004)
    shapes, groups, masks = ((7, 19), (8199,), (7,)), (0, 1, 0), (True, True, False)
    original = tuple(
        jax.device_put(rng.normal(size=s).astype(np.float32), placement) for s in shapes
    )
    observations = []
    for radius in (0.0, 0.15):
        for interface in (jnp.float32, jnp.bfloat16):
            settings = Config(radius=radius, guide_interval=2, guide_warmup=0)
            for kind in ("packed_xla", "pallas"):
                actual = expected = original
                state = state_expected = jax.tree.map(
                    lambda x: jax.device_put(x, placement),
                    jo.init(original, groups, masks),
                )
                oracle = jo.make_step(groups, masks, settings)
                candidate = make_step(original, groups, masks, settings, kind=kind)
                graphs = {
                    due: jax.jit(
                        candidate,
                        out_shardings=(placement, placement, placement),
                    )
                    for due in (False, True)
                }
                for step in range(6):
                    gradient = tuple(
                        jax.device_put(
                            jnp.asarray(rng.normal(size=s), interface), placement
                        )
                        for s in shapes
                    )
                    due = settings.probe_due(step)
                    guide = (
                        tuple(
                            jax.device_put(
                                jnp.asarray(rng.normal(size=s), interface), placement
                            )
                            for s in shapes
                        )
                        if due
                        else None
                    )
                    expected, state_expected, _ = jax.jit(oracle)(
                        expected, gradient, state_expected, guide
                    )
                    actual, state, metrics = graphs[due](actual, gradient, state, guide)
                    jax.block_until_ready(metrics)
                    errors = []
                    for left, right in zip(
                        jax.tree.leaves((actual, state.first, state.second)),
                        jax.tree.leaves(
                            (expected, state_expected.first, state_expected.second)
                        ),
                        strict=True,
                    ):
                        la = np.asarray(left.addressable_data(0))
                        ra = np.asarray(right.addressable_data(0))
                        np.testing.assert_allclose(la, ra, rtol=4e-6, atol=3e-7)
                        errors.append(float(np.max(np.abs(la - ra))))
                    host = {
                        k: np.asarray(v.addressable_data(0)) for k, v in metrics.items()
                    }
                    assert bool(host["numerics_ok"]) and bool(host["schedule_ok"])
                    certificate = check_weights(
                        host["train_coefficients"],
                        host["guide_coefficients"],
                        host["weights"],
                        settings.radius,
                        settings.neutrality_tolerance,
                    )
                    assert certificate["passed"]
                    observations.append(
                        dict(
                            kernel=kind,
                            radius=radius,
                            interface=str(interface),
                            step=step,
                            probe=due,
                            maximum_leaf_error=max(errors),
                            independent_certificate=certificate,
                        )
                    )
    value = dict(
        passed=True,
        scope="compiled optimizer correctness; not full-model efficiency",
        environment=environment(jax),
        observations=observations,
        padding=8199,
        tied_parameter_policy="one canonical leaf",
        reduction="replicated elements counted once",
    )
    path = destination.with_name(f"{destination.stem}-worker{jax.process_index()}.json")
    atomic_json(path, value)
    return value


def summarize(run_dir: Path, destination: Path) -> dict[str, Any]:
    worker_dirs = sorted(run_dir.glob("worker[0-9]*")) or [run_dir]
    workers = []
    reference_rows = None
    reference_identity = None
    for directory in worker_dirs:
        completions = sorted(directory.glob("completion-*-worker*.json"))
        if len(completions) != 1:
            raise ValueError("Profile needs one complete, unrestarted run per worker")
        completion = json.loads(completions[0].read_text())
        logs = sorted(directory.glob("updates-worker*.jsonl"))
        if len(logs) != 1:
            raise ValueError("Profile needs exactly one worker update log")
        rows = [json.loads(line) for line in logs[0].read_text().splitlines()]
        steady = [row for row in rows if row["update"] > 32]
        if len(steady) < 200 or completion["purpose"] != "profile":
            raise ValueError(
                "Full profile requires at least 200 synchronized steady updates"
            )
        run = json.loads((directory / "run.json").read_text())
        env = run["environment"]
        if (
            env["backend"] != "tpu"
            or env["device_count"] != 16
            or env["process_count"] != 4
            or completion["interrupted"]
            or [row["update"] for row in rows] != list(range(1, len(rows) + 1))
            or completion["completed_updates"] != len(rows)
            or sum(row["loss_tokens"] for row in rows)
            != completion["cursors"]["completed_loss_tokens"]
            or sum(row["guide_loss_tokens"] for row in rows)
            != completion["cursors"]["guidance"]
            or not all(row["schedule_ok"] and row["numerics_ok"] for row in rows)
        ):
            raise ValueError(
                "Profile is not a complete valid run on the registered TPU"
            )
        identity = {
            key: value
            for key, value in run.items()
            if key not in {"environment", "access_audit"}
        }
        scientific = [
            {
                key: value
                for key, value in row.items()
                if key
                not in {
                    "step_seconds",
                    "input_seconds",
                    "loop_seconds_before_log",
                    "elapsed_seconds",
                }
            }
            for row in rows
        ]
        if reference_rows is None:
            reference_rows, reference_identity = scientific, identity
        elif scientific != reference_rows or identity != reference_identity:
            raise ValueError("Workers disagree on model/optimizer/loss/cursor identity")
        workers.append(
            dict(
                rank=env["process_index"],
                directory=directory,
                run=run,
                completion=completion,
                rows=rows,
            )
        )
    if {worker["rank"] for worker in workers} != {0, 1, 2, 3}:
        raise ValueError("Profile must retain every distinct JAX controller rank")
    workers.sort(key=lambda worker: worker["rank"])
    steady_rows = []
    for index in range(32, len(workers[0]["rows"])):
        row = workers[0]["rows"][index]
        steady_rows.append(
            dict(
                update=row["update"],
                probe=row["guidance_backward_due"],
                step_seconds=max(
                    worker["rows"][index]["step_seconds"] for worker in workers
                ),
                input_seconds=max(
                    worker["rows"][index]["input_seconds"] for worker in workers
                ),
                loop_seconds=max(
                    worker["rows"][index]["loop_seconds_before_log"]
                    for worker in workers
                ),
            )
        )
    ordinary = [row["step_seconds"] for row in steady_rows if not row["probe"]]
    probes = [row["step_seconds"] for row in steady_rows if row["probe"]]
    files = [
        path
        for directory in worker_dirs
        for path in directory.rglob("*")
        if path.is_file()
    ]
    value = dict(
        scope="actual full decoder on 16 TPU devices / 4 controllers; maximum controller time per synchronized update; I/O, cold compilation and checkpoint cost reported separately",
        identity=workers[0]["run"],
        steady_updates=len(steady_rows),
        ordinary_updates=len(ordinary),
        probe_updates=len(probes),
        ordinary_seconds=dict(
            mean=float(np.mean(ordinary)), median=float(np.median(ordinary))
        ),
        probe_seconds=(
            dict(mean=float(np.mean(probes)), median=float(np.median(probes)))
            if probes
            else None
        ),
        synchronized_steady_seconds=sum(row["step_seconds"] for row in steady_rows),
        steady_input_seconds=sum(row["input_seconds"] for row in steady_rows),
        steady_loop_seconds=sum(row["loop_seconds"] for row in steady_rows),
        compile_seconds={
            str(worker["rank"]): worker["completion"]["compile_seconds"]
            for worker in workers
        },
        end_to_end_seconds=max(
            worker["completion"]["end_to_end_seconds"] for worker in workers
        ),
        training_loop_seconds=max(
            worker["completion"]["wall_seconds"] for worker in workers
        ),
        cursors=workers[0]["completion"]["cursors"],
        raw_sha256={
            str(path.relative_to(run_dir)): sha256_file(path) for path in sorted(files)
        },
        compiler_graphs={
            path.stem: json.loads(path.read_text())
            for path in workers[0]["directory"].glob("*-worker0.json")
            if path.name.startswith(("ordinary-", "probe-"))
        },
        memory_snapshots=[
            json.loads(path.read_text())
            for directory in worker_dirs
            for path in directory.glob("memory-*.json")
        ],
        trace_files=[
            str(path.relative_to(run_dir)) for path in files if "profile" in path.parts
        ],
    )
    if not value["trace_files"]:
        raise ValueError("Full TPU profile must retain its actual profiler traces")
    atomic_json(destination, value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel-check", action="store_true")
    parser.add_argument("--model-check", action="store_true")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/pretrain_125m.json")
    )
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()
    if args.kernel_check or args.model_check:
        jax = initialize(args.distributed)
        if args.model_check:
            from guidon.model_audit import run

            value = run(jax, args.config, args.output)
        else:
            value = kernel_audit(jax, args.output)
        if jax.process_count() > 1:
            from jax.experimental import multihost_utils

            multihost_utils.sync_global_devices("optimizer-kernel-check-complete")
            jax.distributed.shutdown()
        print(
            json.dumps({"passed": value["passed"], "environment": value["environment"]})
        )
    elif args.run_dir:
        print(json.dumps(summarize(args.run_dir, args.output), indent=2))
    else:
        parser.error("choose --kernel-check or --run-dir")


if __name__ == "__main__":
    main()
