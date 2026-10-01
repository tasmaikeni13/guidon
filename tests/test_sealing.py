import base64

import pytest
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from guidon.sealing import seal_directory


def test_evaluator_key_and_authenticated_role_binding(tmp_path):
    # Test-only private key. Production preparation only accepts an external
    # public key and never creates/receives the evaluator's private key.
    private = rsa.generate_private_key(public_exponent=65537, key_size=3072)
    path = tmp_path / "public.pem"
    path.write_bytes(
        private.public_key().public_bytes(
            serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
        )
    )
    plain = tmp_path / "plain"
    plain.mkdir()
    payload = b"test-only sealed payload"
    (plain / "tokens.bin").write_bytes(payload)
    context = dict(role="test", cohort="confirmation", preparation_sha256="pinned")
    result = seal_directory(plain, tmp_path / "sealed", path, context)
    assert not plain.exists()
    entry = result["files"][0]
    key = private.decrypt(
        base64.b64decode(entry["wrapped_key_base64"]),
        padding.OAEP(
            mgf=padding.MGF1(hashes.SHA256()), algorithm=hashes.SHA256(), label=None
        ),
    )
    nonce = base64.b64decode(entry["nonce_base64"])
    ciphertext = (tmp_path / "sealed" / entry["path"]).read_bytes()
    assert (
        AESGCM(key).decrypt(nonce, ciphertext, entry["context_sha256"].encode())
        == payload
    )
    with pytest.raises(InvalidTag):
        AESGCM(key).decrypt(nonce, ciphertext, b"changed-role")
    assert result["private_key_available_to_preparer"] is False
