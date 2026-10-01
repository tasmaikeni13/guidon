"""Atomic, hash-verified replicated global checkpoints with complete run cursors.

The measured data-parallel mesh replicates model/controller state. Each worker
writes a full copy; a global preparation barrier checks logical identity before
any commit is published. A committed copy can restore all workers. Parameter
partitioning requires a different registered serializer, not silent host gathering.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import jax
import numpy as np

from guidon import jax_optimizer as jo
from guidon.artifacts import atomic_json, object_sha256, sha256_file


def encode_tree(value: Any, arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    if isinstance(value, jo.State):
        return {
            "kind": "optimizer_state",
            "children": [encode_tree(v, arrays) for v in value],
        }
    if isinstance(value, dict):
        return {
            "kind": "dict",
            "children": {k: encode_tree(v, arrays) for k, v in sorted(value.items())},
        }
    if isinstance(value, tuple | list):
        return {
            "kind": "tuple" if isinstance(value, tuple) else "list",
            "children": [encode_tree(v, arrays) for v in value],
        }
    if isinstance(value, jax.Array):
        if not value.is_fully_replicated:
            raise ValueError("Checkpoint serializer requires replicated global state")
        array = np.asarray(value.addressable_data(0))
    else:
        array = np.asarray(value)
    if array.dtype.hasobject:
        raise ValueError("Object arrays cannot be checkpointed")
    name = f"array-{len(arrays):05d}"
    arrays[name] = array
    return {
        "kind": "array",
        "name": name,
        "shape": list(array.shape),
        "dtype": str(array.dtype),
    }


def decode_tree(spec: dict[str, Any], arrays: Any, placement: Any) -> Any:
    kind = spec["kind"]
    if kind == "array":
        array = arrays[spec["name"]]
        if list(array.shape) != spec["shape"] or str(array.dtype) != spec["dtype"]:
            raise ValueError("Checkpoint leaf shape/dtype differs from its schema")
        return jax.device_put(array, placement)
    children = spec["children"]
    if kind == "dict":
        return {k: decode_tree(v, arrays, placement) for k, v in children.items()}
    items = [decode_tree(v, arrays, placement) for v in children]
    if kind == "optimizer_state":
        return jo.State(*items)
    if kind == "tuple":
        return tuple(items)
    if kind == "list":
        return items
    raise ValueError("Unknown checkpoint tree kind")


def save(
    root: Path,
    params: Any,
    state: jo.State,
    rng: Any,
    cursors: dict[str, int],
    identity: dict[str, Any],
    *,
    distributed: bool = True,
) -> Path:
    count = int(state.count)
    path = root / f"step-{count:06d}"
    path.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    arrays: dict[str, np.ndarray] = {}
    structure = encode_tree({"params": params, "optimizer": state, "rng": rng}, arrays)
    array_manifest = {
        name: {
            "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
            "shape": list(array.shape),
            "dtype": str(array.dtype),
        }
        for name, array in arrays.items()
    }
    logical_hash = object_sha256(
        {
            "arrays": array_manifest,
            "tree": structure,
            "cursors": cursors,
            "identity": identity,
        }
    )
    payload = path / "state.npz"
    with payload.open("xb") as handle:
        np.savez(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    record = dict(
        schema_version=1,
        format="replicated_global_npz",
        count=count,
        cursors=cursors,
        identity=identity,
        tree=structure,
        arrays=array_manifest,
        logical_state_sha256=logical_hash,
        payload_sha256=sha256_file(payload),
        process_count=jax.process_count(),
        worker=jax.process_index(),
    )
    atomic_json(path / "prepared.json", record)
    if distributed and jax.process_count() > 1:
        from jax.experimental import multihost_utils

        hashes = multihost_utils.process_allgather(
            np.frombuffer(bytes.fromhex(logical_hash), dtype=np.uint8), tiled=False
        )
        if not np.all(hashes == hashes[0]):
            raise ValueError(
                "Worker model/moment/RNG/cursor checkpoint identity differs"
            )
        multihost_utils.sync_global_devices(f"checkpoint-prepared-{count}")
    record["all_workers_prepared"] = True
    record["checkpoint_seconds"] = time.monotonic() - started
    atomic_json(path / "commit.json", record)
    if distributed and jax.process_count() > 1:
        multihost_utils.sync_global_devices(f"checkpoint-committed-{count}")
    return path


def load(
    path: Path, identity: dict[str, Any], placement: Any
) -> tuple[Any, jo.State, Any, dict[str, int]]:
    record = json.loads((path / "commit.json").read_text())
    if (
        not record.get("all_workers_prepared")
        or record["format"] != "replicated_global_npz"
    ):
        raise ValueError("Checkpoint transaction is incomplete or unsupported")
    if record["identity"] != identity:
        raise ValueError("Checkpoint config/data/optimizer/seed identity differs")
    if sha256_file(path / "state.npz") != record["payload_sha256"]:
        raise ValueError("Checkpoint payload is corrupt")
    with np.load(path / "state.npz", allow_pickle=False) as arrays:
        if set(arrays.files) != set(record["arrays"]):
            raise ValueError("Checkpoint leaves differ from the registered structure")
        for name, spec in record["arrays"].items():
            if hashlib.sha256(arrays[name].tobytes()).hexdigest() != spec["sha256"]:
                raise ValueError("Checkpoint logical tensor hash differs")
        value = decode_tree(record["tree"], arrays, placement)
    computed = object_sha256(
        {
            "arrays": record["arrays"],
            "tree": record["tree"],
            "cursors": record["cursors"],
            "identity": identity,
        }
    )
    if (
        computed != record["logical_state_sha256"]
        or int(value["optimizer"].count) != record["count"]
    ):
        raise ValueError("Checkpoint logical identity/count differs")
    return value["params"], value["optimizer"], value["rng"], record["cursors"]
