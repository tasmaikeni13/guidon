import json
from pathlib import Path

import numpy as np
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from test_training import inputs

from guidon.artifacts import sha256_file
from guidon.evaluate import decrypt_tape, run
from guidon.packing import TapeWriter, write_capability
from guidon.sealing import seal_directory
from guidon.train import run as train


def test_development_reader_count_and_rejection_of_training_final_channels(tmp_path):
    config, training = inputs(tmp_path)
    result = train(
        config,
        "adamw",
        101,
        training,
        tmp_path / "run",
        purpose="verification",
        max_updates=2,
    )
    writer = TapeWriter(tmp_path / "development", "development", "pilot", {}, 0)
    writer.add(dict(doc_id="development", split="development"), [1, 2, 3] * 17)
    writer.finish()
    capability = tmp_path / "development.json"
    write_capability(
        capability,
        {"development": tmp_path / "development/manifest.json"},
        process_role="development",
        cohort="pilot",
        preparation_sha256="fixture",
    )
    value = run(
        config, Path(result["checkpoint"]), capability, tmp_path / "result.json"
    )
    assert value["results"]["development"]["loss_tokens"] == 52
    assert np.isfinite(value["results"]["development"]["mean_nll"])
    assert all("development" in p for p in value["access_audit"]["opened_data_paths"])
    with pytest.raises(ValueError, match="only pilot development"):
        run(config, Path(result["checkpoint"]), training, tmp_path / "leak.json")
    with pytest.raises(ValueError, match="external authorization and key"):
        run(
            config,
            Path(result["checkpoint"]),
            capability,
            tmp_path / "final.json",
            purpose="confirmation",
        )


def test_authenticated_sealed_tape_role_cohort_and_corruption(tmp_path):
    private = rsa.generate_private_key(public_exponent=65537, key_size=3072)
    public = tmp_path / "public.pem"
    public.write_bytes(
        private.public_key().public_bytes(
            serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
        )
    )
    writer = TapeWriter(tmp_path / "plain", "test", "shift", {}, 0)
    writer.add(dict(doc_id="external-test-only", split="test"), [1, 2, 3])
    writer.finish()
    context = dict(role="test", cohort="shift", preparation_sha256="fixture")
    seal_directory(tmp_path / "plain", tmp_path / "sealed", public, context)
    record = tmp_path / "sealed/sealed.json"
    digest = sha256_file(record)
    path = decrypt_tape(
        record,
        tmp_path / "external",
        private,
        expected_sha256=digest,
        role="test",
        cohort="shift",
        preparation_sha256="fixture",
    )
    assert json.loads(path.read_text())["loss_tokens"] == 4
    with pytest.raises(ValueError, match="role/cohort"):
        decrypt_tape(
            record,
            tmp_path / "wrong-role",
            private,
            expected_sha256=digest,
            role="validation",
            cohort="shift",
            preparation_sha256="fixture",
        )
    item = json.loads(record.read_text())["files"][0]
    (record.parent / item["path"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="corrupt"):
        decrypt_tape(
            record,
            tmp_path / "corrupt",
            private,
            expected_sha256=digest,
            role="test",
            cohort="shift",
            preparation_sha256="fixture",
        )
