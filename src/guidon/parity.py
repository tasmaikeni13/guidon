"""Verify the pinned Pythia checkpoint against its official Transformers model."""

from __future__ import annotations

import argparse
import inspect
import json
import shutil
from dataclasses import asdict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from huggingface_hub import HfApi, snapshot_download
from safetensors.numpy import load_file

from guidon.artifacts import atomic_json, sha256_file
from guidon.model import ModelConfig, import_neox, logits, parameter_count


def run(
    config_path: Path,
    destination: Path,
    cache_dir: Path,
    bundle_dir: Path | None = None,
) -> dict:
    import torch
    import transformers
    from transformers import GPTNeoXForCausalLM

    protocol = json.loads(config_path.read_text())
    model = protocol["model"]
    info = HfApi().model_info(model["id"], revision=model["revision"])
    if info.sha != model["revision"]:
        raise ValueError("Model revision did not resolve to its pinned commit")
    snapshot = Path(
        snapshot_download(
            model["id"],
            revision=model["revision"],
            cache_dir=str(cache_dir),
            allow_patterns=[
                "config.json",
                "model.safetensors",
                "tokenizer*",
                "special_tokens_map.json",
            ],
        )
    )
    architecture = json.loads((snapshot / "config.json").read_text())
    config = ModelConfig.from_neox_config(
        architecture, compute_dtype="float32", rematerialize=False
    )
    params = import_neox(load_file(snapshot / "model.safetensors"), config)
    torch.set_num_threads(4)
    source = GPTNeoXForCausalLM.from_pretrained(
        snapshot,
        torch_dtype=torch.float32,
        attn_implementation="eager",
        local_files_only=True,
    ).eval()
    counts = (parameter_count(params), sum(p.numel() for p in source.parameters()))
    if counts[0] != counts[1]:
        raise ValueError("Imported/source parameter counts differ")
    rng = np.random.default_rng(20261001)
    observations = []
    for length in (1, 7, 32):
        ids = rng.integers(0, config.vocab_size, size=(2, length), dtype=np.int32)
        batch = dict(
            input_ids=jnp.asarray(ids),
            position_ids=jnp.broadcast_to(jnp.arange(length), ids.shape),
            segment_ids=jnp.zeros(ids.shape, dtype=jnp.int32),
        )
        expected = jax.jit(lambda p, b: logits(p, b, config))(params, batch)
        with torch.no_grad():
            observed = source(torch.tensor(ids, dtype=torch.long)).logits.numpy()
        actual = np.asarray(expected)
        error = np.abs(actual - observed)
        np.testing.assert_allclose(actual, observed, rtol=3e-5, atol=3e-5)
        observations.append(
            dict(length=length, maximum_absolute_error=float(error.max()))
        )
    result = {
        "passed": True,
        "scope": "pinned pretrained GPT-NeoX FP32 full-vocabulary source logits",
        "model_id": model["id"],
        "revision": model["revision"],
        "parameter_count": counts[0],
        "source_parameter_count": counts[1],
        "config": asdict(config),
        "tolerance": {"rtol": 3e-5, "atol": 3e-5},
        "observations": observations,
        "artifact_sha256": {
            path.name: sha256_file(path)
            for path in snapshot.iterdir()
            if path.is_file()
        },
        "source_implementation_sha256": sha256_file(
            Path(inspect.getfile(GPTNeoXForCausalLM))
        ),
        "import_implementation_sha256": sha256_file(Path(inspect.getfile(import_neox))),
        "environment": {
            "jax": jax.__version__,
            "backend": jax.default_backend(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
    }
    if bundle_dir is not None:
        bundle_dir.mkdir(parents=True, exist_ok=False)
        for name in ("config.json", "model.safetensors", "tokenizer.json"):
            shutil.copyfile(snapshot / name, bundle_dir / name)
        atomic_json(bundle_dir / "parity.json", result)
        atomic_json(
            bundle_dir / "bundle.json",
            {
                "schema_version": 1,
                "model_id": model["id"],
                "revision": model["revision"],
                "artifacts": {
                    name: sha256_file(bundle_dir / name)
                    for name in (
                        "config.json",
                        "model.safetensors",
                        "tokenizer.json",
                        "parity.json",
                    )
                },
            },
        )
    atomic_json(destination, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/continued_pythia160m.json")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/model-cache"))
    parser.add_argument("--bundle-dir", type=Path)
    args = parser.parse_args()
    value = run(args.config, args.output, args.cache_dir, args.bundle_dir)
    print(
        json.dumps(
            {
                k: value[k]
                for k in ("passed", "parameter_count", "observations", "environment")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
