"""Load hash-verified pretrained artifacts without a source-framework runtime."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

from safetensors.numpy import load_file

from guidon.artifacts import checked_path, sha256_file
from guidon.model import ModelConfig, import_neox, parameter_count


def pretrained_bundle(
    protocol: dict[str, Any], root: Path
) -> tuple[ModelConfig, Any, dict[str, Any]]:
    """Reject stale parity, altered weights, and a different model/tokenizer pin."""
    manifest = root / "bundle.json"
    bundle = json.loads(manifest.read_text())
    model = protocol["model"]
    if (
        bundle.get("schema_version") != 1
        or bundle["model_id"] != model["id"]
        or bundle["revision"] != model["revision"]
    ):
        raise ValueError("Pretrained bundle differs from the registered model")
    required = {"config.json", "model.safetensors", "tokenizer.json", "parity.json"}
    if set(bundle["artifacts"]) != required:
        raise ValueError("Pretrained bundle artifact set differs")
    paths = {}
    for name, digest in bundle["artifacts"].items():
        path = checked_path(root, name)
        if sha256_file(path) != digest:
            raise ValueError(f"Pretrained bundle artifact is corrupt: {name}")
        paths[name] = path
    parity = json.loads(paths["parity.json"].read_text())
    if (
        not parity.get("passed")
        or parity["model_id"] != model["id"]
        or parity["revision"] != model["revision"]
        or parity["import_implementation_sha256"]
        != sha256_file(Path(inspect.getfile(import_neox)))
        or any(
            parity["artifact_sha256"][name] != bundle["artifacts"][name]
            for name in ("config.json", "model.safetensors", "tokenizer.json")
        )
    ):
        raise ValueError("Pretrained source-logit parity is missing or stale")
    architecture = json.loads(paths["config.json"].read_text())
    config = ModelConfig.from_neox_config(
        architecture,
        compute_dtype=protocol["hardware"]["model_compute_dtype"],
        attention_kernel=protocol.get("attention_kernel", "xla"),
    )
    params = import_neox(load_file(paths["model.safetensors"]), config)
    if parameter_count(params) != parity["parameter_count"]:
        raise ValueError("Imported model parameter count differs from source parity")
    return (
        config,
        params,
        {
            "bundle_sha256": sha256_file(manifest),
            "model_id": model["id"],
            "revision": model["revision"],
            "parameter_count": parity["parameter_count"],
            "tokenizer_sha256": bundle["artifacts"]["tokenizer.json"],
        },
    )
