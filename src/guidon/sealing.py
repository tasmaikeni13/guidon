"""Encrypt final payloads to an external evaluator's RSA public key.

The preparation process receives no private key. Ciphertext and public metadata
may be retained by research; decryption belongs to the independent evaluator.
"""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from guidon.artifacts import atomic_json, object_sha256, sha256_file


def public_key(path: Path) -> rsa.RSAPublicKey:
    key = serialization.load_pem_public_key(path.read_bytes())
    if not isinstance(key, rsa.RSAPublicKey) or key.key_size < 3072:
        raise ValueError(
            "Evaluator must supply an RSA public key of at least 3072 bits"
        )
    return key


def encrypt_file(
    source: Path,
    destination: Path,
    key: rsa.RSAPublicKey,
    context: dict[str, Any],
) -> dict[str, Any]:
    """Authenticate role/cohort/preparation identity with each encrypted payload."""
    if destination.exists():
        raise FileExistsError("Sealed payloads are immutable")
    aes_key = AESGCM.generate_key(bit_length=256)
    nonce = os.urandom(12)
    context_digest = object_sha256(context)
    encryptor = Cipher(algorithms.AES(aes_key), modes.GCM(nonce)).encryptor()
    encryptor.authenticate_additional_data(context_digest.encode())
    wrapped = key.encrypt(
        aes_key,
        padding.OAEP(
            mgf=padding.MGF1(hashes.SHA256()), algorithm=hashes.SHA256(), label=None
        ),
    )
    with source.open("rb") as plaintext, destination.open("xb") as handle:
        while chunk := plaintext.read(8 * 1024 * 1024):
            handle.write(encryptor.update(chunk))
        handle.write(encryptor.finalize())
        handle.write(encryptor.tag)
        handle.flush()
        os.fsync(handle.fileno())
    return {
        "path": destination.name,
        "sha256": sha256_file(destination),
        "plaintext_sha256": sha256_file(source),
        "nonce_base64": base64.b64encode(nonce).decode(),
        "wrapped_key_base64": base64.b64encode(wrapped).decode(),
        "context": context,
        "context_sha256": context_digest,
        "algorithm": "RSA-OAEP-SHA256/AES-256-GCM",
    }


def seal_directory(
    plaintext: Path,
    destination: Path,
    public_key_path: Path,
    context: dict[str, Any],
) -> dict[str, Any]:
    if context.get("role") not in {"validation", "test"}:
        raise ValueError("Only registered final evaluation roles may be sealed here")
    key = public_key(public_key_path)
    destination.mkdir(parents=True, exist_ok=False)
    records = []
    for source in sorted(plaintext.iterdir()):
        if not source.is_file() or source.is_symlink():
            raise ValueError(
                "Sealing requires a flat, verified token artifact directory"
            )
        records.append(
            encrypt_file(source, destination / (source.name + ".aesgcm"), key, context)
        )
    result = {
        "schema_version": 1,
        "context": context,
        "public_key_sha256": sha256_file(public_key_path),
        "files": records,
        "private_key_available_to_preparer": False,
    }
    atomic_json(destination / "sealed.json", result)
    # Remove only this preparation run's temporary final plaintext after the
    # complete authenticated encryption record is atomically durable.
    for source in plaintext.iterdir():
        source.unlink()
    plaintext.rmdir()
    return result
