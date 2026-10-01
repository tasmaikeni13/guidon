"""Initialize controllers before devices; discover and audit global JAX reductions."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import socket
from pathlib import Path
from typing import Any

import numpy as np

from guidon.artifacts import atomic_json


def initialize(distributed: bool = False) -> Any:
    import jax

    coordinator = os.environ.get("GUIDON_COORDINATOR_ADDRESS")
    if coordinator:
        jax.distributed.initialize(
            coordinator_address=coordinator,
            num_processes=int(os.environ["GUIDON_PROCESS_COUNT"]),
            process_id=int(os.environ["GUIDON_PROCESS_ID"]),
            initialization_timeout=120,
        )
    elif distributed:
        jax.distributed.initialize(initialization_timeout=120)
    return jax


def data_mesh(jax: Any) -> Any:
    from jax.sharding import Mesh

    return Mesh(np.asarray(jax.devices()), ("data",))


def global_batch(jax: Any, mesh: Any, batch: dict[str, np.ndarray]) -> dict[str, Any]:
    """Each worker supplies its unique rows of the same global tape batch."""
    from jax.sharding import NamedSharding, PartitionSpec

    result = {}
    for key, value in batch.items():
        if value.shape[0] % jax.process_count():
            raise ValueError("Global batch does not divide controller count")
        per_host = value.shape[0] // jax.process_count()
        start = jax.process_index() * per_host
        result[key] = jax.make_array_from_process_local_data(
            NamedSharding(mesh, PartitionSpec("data", None)),
            value[start : start + per_host],
            global_shape=value.shape,
        )
    return result


def environment(jax: Any) -> dict[str, Any]:
    packages = {}
    for name in ("jax", "jaxlib", "libtpu", "numpy"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "hostname": socket.gethostname(),
        "packages": packages,
        "backend": jax.default_backend(),
        "process_count": jax.process_count(),
        "process_index": jax.process_index(),
        "device_count": jax.device_count(),
        "local_device_count": jax.local_device_count(),
        "devices": [
            {
                "id": d.id,
                "process_index": d.process_index,
                "kind": d.device_kind,
                "coordinates": list(d.coords) if hasattr(d, "coords") else None,
                "core_on_chip": getattr(d, "core_on_chip", None),
            }
            for d in jax.devices()
        ],
    }


def collective_audit(jax: Any) -> dict[str, Any]:
    """Compare masked global loss/gradient with an exact host oracle."""
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec

    mesh = data_mesh(jax)
    count = jax.device_count() * 4
    values = np.arange(1, count + 1, dtype=np.float32).reshape(count, 1)
    mask = (np.arange(count) % 3 != 0).astype(np.float32).reshape(count, 1)
    batch = global_batch(jax, mesh, {"values": values, "mask": mask})
    scalar = jax.device_put(np.float32(2), NamedSharding(mesh, PartitionSpec()))

    def objective(p: Any, batch: dict[str, Any]) -> Any:
        return jnp.sum((p * batch["values"]) ** 2 * batch["mask"]) / jnp.sum(
            batch["mask"]
        )

    value, grad = jax.jit(jax.value_and_grad(objective))(scalar, batch)
    labels = jax.jit(lambda mask: jnp.sum(mask))(batch["mask"])
    expected = float(np.sum(values**2 * mask, dtype=np.float64) / mask.sum())
    np.testing.assert_allclose(float(value), 4 * expected, rtol=2e-6)
    np.testing.assert_allclose(float(grad), 4 * expected, rtol=2e-6)
    assert int(labels) == int(mask.sum())
    return {
        "passed": True,
        "global_rows": count,
        "positive_labels": int(labels),
        "loss": float(value),
        "gradient": float(grad),
        "expected_loss_and_gradient": 4 * expected,
        "reduction": "global_array_sum_once_no_second_replica_psum",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    jax = initialize(args.distributed)
    value = {"environment": environment(jax), "collective_audit": collective_audit(jax)}
    if args.output:
        path = args.output.with_name(
            f"{args.output.stem}-worker{jax.process_index()}.json"
        )
        atomic_json(path, value)
    print(json.dumps(value, sort_keys=True), flush=True)
    if jax.process_count() > 1:
        from jax.experimental import multihost_utils

        multihost_utils.sync_global_devices("runtime-audit-complete")
        jax.distributed.shutdown()


if __name__ == "__main__":
    main()
