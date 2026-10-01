"""Publish immutable artifact generations and verify their bytes by readback."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from guidon.artifacts import atomic_json, sha256_file
from guidon.sealing import encrypt_file, public_key


def publish_file(source: Path, uri: str, scratch: Path) -> dict[str, Any]:
    """GCS CRC checks plus an independent SHA256 readback of the pinned generation."""
    if not uri.startswith("gs://") or "#" in uri:
        raise ValueError("Archive needs an unversioned GCS destination object URI")
    digest = sha256_file(source)
    subprocess.run(
        [
            "gcloud",
            "storage",
            "cp",
            str(source),
            uri,
            "--if-generation-match=0",
            "--custom-metadata=sha256=" + digest,
            "--quiet",
        ],
        check=True,
    )
    return verify_published_file(source, uri, scratch)


def verify_published_file(source: Path, uri: str, scratch: Path) -> dict[str, Any]:
    """Check an already committed object without changing its bytes or generation."""
    digest = sha256_file(source)
    metadata = json.loads(
        subprocess.check_output(
            ["gcloud", "storage", "objects", "describe", uri, "--format=json"],
            text=True,
        )
    )
    if (
        metadata.get("custom_fields", {}).get("sha256") != digest
        or int(metadata["size"]) != source.stat().st_size
    ):
        raise ValueError("Published object metadata differs from the local bytes")
    version = uri + "#" + str(metadata["generation"])
    with tempfile.TemporaryDirectory(
        prefix="guidon-readback-", dir=scratch
    ) as temporary:
        downloaded = Path(temporary) / "artifact"
        subprocess.run(
            ["gcloud", "storage", "cp", version, str(downloaded), "--quiet"],
            check=True,
        )
        if sha256_file(downloaded) != digest:
            raise ValueError("Published artifact fails independent SHA256 readback")
    return dict(
        uri=uri,
        generation=str(metadata["generation"]),
        version_uri=version,
        sha256=digest,
        bytes=source.stat().st_size,
        independent_sha256_readback=True,
        object_metadata=metadata,
    )


def seal_archive(
    source: Path,
    uri: str,
    key_path: Path,
    context: dict[str, Any],
    record_path: Path,
    scratch: Path,
) -> dict[str, Any]:
    """Retain stopped preparation attempts under external custody before removal."""
    scratch.mkdir(parents=True, exist_ok=True)
    encrypted = scratch / (record_path.stem + ".aesgcm")
    encryption = encrypt_file(source, encrypted, public_key(key_path), context)
    # Preserve enough to resume publication of this exact ciphertext after any
    # network failure. No private/decrypted key is part of this public record.
    atomic_json(
        record_path.with_suffix(".pending.json"),
        dict(
            source=str(source),
            encryption=encryption,
            public_key_sha256=sha256_file(key_path),
            local_ciphertext=str(encrypted),
        ),
    )
    archive = publish_file(encrypted, uri, scratch)
    value = dict(
        schema_version=1,
        original_path=str(source),
        encryption=encryption,
        archive=archive,
        public_key_sha256=sha256_file(key_path),
        private_key_available_to_preparer=False,
        original_plaintext_removed_after_verified_publication=True,
    )
    atomic_json(record_path, value)
    source.unlink()
    encrypted.unlink()
    return value
