"""Independent development or externally authorized sealed-loss evaluator.

Training never imports this module. Final decryption runs in the evaluator's
environment with its private key, after authorization names every final checkpoint.
Only aggregate sums/counts are emitted; examples, labels and per-document scores
are not an output channel. This CLI does not create evaluation authorization.
"""

from __future__ import annotations

import argparse
import base64
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from guidon.artifacts import atomic_json, checked_path, object_sha256, sha256_file
from guidon.packing import MixtureTape, Tape
from guidon.runtime import data_mesh, environment, global_batch, initialize


def decrypt_tape(
    sealed_path: Path,
    target: Path,
    private_key: rsa.RSAPrivateKey,
    *,
    expected_sha256: str,
    role: str,
    cohort: str,
    preparation_sha256: str,
) -> Path:
    """Authenticate every artifact before publishing a plaintext tape locally."""
    if sha256_file(sealed_path) != expected_sha256:
        raise ValueError("Sealed tape differs from evaluator authorization")
    record = json.loads(sealed_path.read_text())
    context = record["context"]
    if (
        context["role"] != role
        or role not in {"validation", "test"}
        or context["cohort"] != cohort
        or context["preparation_sha256"] != preparation_sha256
    ):
        raise ValueError("Sealed role/cohort/preparation identity differs")
    target.mkdir(mode=0o700, parents=True, exist_ok=False)
    names = set()
    for item in record["files"]:
        if (
            item["context"] != context
            or item["context_sha256"] != object_sha256(context)
            or item["algorithm"] != "RSA-OAEP-SHA256/AES-256-GCM"
            or not item["path"].endswith(".aesgcm")
        ):
            raise ValueError("Sealed payload authentication metadata differs")
        source = checked_path(sealed_path.parent, item["path"])
        if sha256_file(source) != item["sha256"]:
            raise ValueError("Sealed ciphertext is corrupt")
        name = item["path"][: -len(".aesgcm")]
        if name in names:
            raise ValueError("Duplicate sealed plaintext filename")
        names.add(name)
        destination = checked_path(target, name)
        key = private_key.decrypt(
            base64.b64decode(item["wrapped_key_base64"], validate=True),
            padding.OAEP(
                mgf=padding.MGF1(hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None,
            ),
        )
        plaintext = AESGCM(key).decrypt(
            base64.b64decode(item["nonce_base64"], validate=True),
            source.read_bytes(),
            item["context_sha256"].encode(),
        )
        with destination.open("xb") as handle:
            handle.write(plaintext)
        if sha256_file(destination) != item["plaintext_sha256"]:
            raise ValueError("Decrypted token artifact hash differs")
    if "manifest.json" not in names:
        raise ValueError("Sealed tape has no token manifest")
    return target / "manifest.json"


def authorize_final(
    authorization: Path,
    checkpoint_path: Path,
    capability_path: Path,
    config_path: Path,
) -> dict[str, Any]:
    """Require the complete fixed-seed/fixed-arm final checkpoint inventory."""
    from guidon.gates import verify

    value = json.loads(authorization.read_text())
    config = json.loads(config_path.read_text())
    record = json.loads((checkpoint_path / "commit.json").read_text())
    identity = record["identity"]
    if (
        value.get("status") != "frozen_confirmation_evaluation"
        or value["config_sha256"] != sha256_file(config_path)
        or value["capability_sha256"] != sha256_file(capability_path)
        or identity["config_sha256"] != object_sha256(config)
        or identity["purpose"] != "training"
        or record["cursors"]["completed_loss_tokens"] != config["training_loss_tokens"]
        or not config["gates"]["frozen"]
        or not config["data"]["manifest_ready"]
        or config["seeds"] != [42, 43, 44]
    ):
        raise ValueError("Final evaluation requires a frozen exact-budget checkpoint")
    runs = value["final_checkpoints"]
    expected = {
        (arm, seed)
        for arm in ("adamw", "guidon", "adamw_plus_guidance")
        for seed in (42, 43, 44)
    }
    if (
        len(runs) != len(expected)
        or {(row["optimizer"], row["seed"]) for row in runs} != expected
    ):
        raise ValueError("Final authorization must retain every required arm and seed")
    matches = [
        row
        for row in runs
        if row["optimizer"] == identity["optimizer"] and row["seed"] == identity["seed"]
    ]
    if len(matches) != 1 or matches[0]["commit_sha256"] != sha256_file(
        checkpoint_path / "commit.json"
    ):
        raise ValueError("Checkpoint is absent from final evaluator authorization")
    verify(["05"] if config["regime"] == "from_scratch" else ["05", "06"])
    return value


def reader(path: Path, role: str, cohort: str, digest: str) -> Tape | MixtureTape:
    if sha256_file(path) != digest:
        raise ValueError("Evaluation manifest differs from its capability")
    value = json.loads(path.read_text())
    kind = (
        MixtureTape
        if value["packing"] == "cumulative_integer_loss_token_mixture"
        else Tape
    )
    return kind(path, role, cohort, expected_sha256=digest)


def run(
    config_path: Path,
    checkpoint_path: Path,
    capability_path: Path,
    destination: Path,
    *,
    purpose: str = "development",
    authorization: Path | None = None,
    evaluator_private_key: Path | None = None,
    distributed: bool = False,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text())
    capability = json.loads(capability_path.read_text())
    final = purpose == "confirmation"
    if purpose not in {"development", "confirmation"}:
        raise ValueError("Unknown evaluator purpose")
    if final:
        if authorization is None or evaluator_private_key is None:
            raise ValueError("Final evaluation needs external authorization and key")
        authorize_final(authorization, checkpoint_path, capability_path, config_path)
        if (
            capability["process_role"] != "sealed_evaluation"
            or capability["cohort"] != "confirmation"
        ):
            raise ValueError("Final evaluator capability is not sealed confirmation")
    elif (
        capability["process_role"] != "development"
        or capability["cohort"] != "pilot"
        or set(capability["tapes"]) != {"development"}
    ):
        raise ValueError("Development evaluator may open only pilot development")
    commit = json.loads((checkpoint_path / "commit.json").read_text())
    identity = commit["identity"]
    if identity["config_sha256"] != object_sha256(config) or (
        not final and config.get("cohort") != "pilot"
    ):
        raise ValueError("Evaluator checkpoint/config/cohort identity differs")
    if identity["data_preparation_sha256"] != capability["preparation_sha256"]:
        raise ValueError("Evaluator and checkpoint preparation identities differ")
    if identity["data_sha256"] == object_sha256(capability):
        raise ValueError("Evaluator cannot use a training capability")
    jax = initialize(distributed)
    from jax.sharding import NamedSharding, PartitionSpec

    from guidon import checkpoint
    from guidon.model import ModelConfig, loss_sums

    mesh = data_mesh(jax)
    placement = NamedSharding(mesh, PartitionSpec())
    params, _, _, _ = checkpoint.load(checkpoint_path, identity, placement)
    model = ModelConfig(**identity["model"])
    evaluate = jax.jit(
        lambda p, b: loss_sums(p, b, model), out_shardings=(placement, placement)
    )
    batch_size, length = config["global_batch_sequences"], config["sequence_length"]
    results, opened = {}, []
    with tempfile.TemporaryDirectory(prefix="guidon-evaluator-") as temporary:
        for name, item in capability["tapes"].items():
            if Path(name).name != name or name in {".", ".."}:
                raise ValueError("Evaluator tape name must be a flat identifier")
            role = item.get("role", name)
            if final:
                if role not in {"validation", "test"}:
                    raise ValueError("Unauthorized role in sealed evaluator capability")
                key = serialization.load_pem_private_key(
                    evaluator_private_key.read_bytes(), password=None
                )
                if not isinstance(key, rsa.RSAPrivateKey) or key.key_size < 3072:
                    raise ValueError("Unsupported external evaluator private key")
                source = capability_path.parent / item["sealed"]
                path = decrypt_tape(
                    source,
                    Path(temporary) / name,
                    key,
                    expected_sha256=item["sha256"],
                    role=role,
                    cohort=item["cohort"],
                    preparation_sha256=capability["preparation_sha256"],
                )
                tape = reader(path, role, item["cohort"], sha256_file(path))
            else:
                tape = reader(
                    capability_path.parent / item["manifest"],
                    role,
                    "pilot",
                    item["sha256"],
                )
            cursor, total, labels = 0, np.float64(0), 0
            while cursor < tape.manifest["loss_tokens"]:
                take = min(batch_size * length, tape.manifest["loss_tokens"] - cursor)
                batch, following = tape.batch(cursor, take, batch_size, length)
                loss_sum, count = evaluate(params, global_batch(jax, mesh, batch))
                jax.block_until_ready((loss_sum, count))
                if int(count) != take or not np.isfinite(float(loss_sum)):
                    raise FloatingPointError("Evaluation NLL/count is invalid")
                total += np.float64(loss_sum)
                labels += int(count)
                cursor = following
            results[name] = dict(
                nll_sum=float(total), loss_tokens=labels, mean_nll=float(total / labels)
            )
            opened.extend(tape.opens if not final else [str(source.resolve())])
    result = dict(
        purpose=purpose,
        checkpoint_sha256=sha256_file(checkpoint_path / "commit.json"),
        capability_sha256=sha256_file(capability_path),
        results=results,
        environment=environment(jax),
        access_audit=dict(
            process_role=capability["process_role"],
            opened_data_paths=opened,
            emitted="aggregate_sums_and_counts",
        ),
    )
    if authorization is not None:
        result["authorization_sha256"] = sha256_file(authorization)
    path = destination.with_name(f"{destination.stem}-worker{jax.process_index()}.json")
    atomic_json(path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "checkpoint", "manifest", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument(
        "--purpose", choices=["development", "confirmation"], default="development"
    )
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--evaluator-private-key", type=Path)
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        print(
            json.dumps(
                dict(
                    purpose=args.purpose,
                    executed=False,
                    output="aggregate_sums_and_counts",
                )
            )
        )
        return
    print(
        json.dumps(
            run(
                args.config,
                args.checkpoint,
                args.manifest,
                args.output,
                purpose=args.purpose,
                authorization=args.authorization,
                evaluator_private_key=args.evaluator_private_key,
                distributed=args.distributed,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
