import hashlib
from pathlib import Path

import pytest

from guidon.data_boundary import audit, training_paths


def record(identifier, role):
    return {
        "doc_id": identifier,
        "content_sha256": hashlib.sha256(identifier.encode()).hexdigest(),
        "cluster_id": f"cluster-{identifier}",
        "source_group": f"source-{identifier}",
        "split": role,
    }


@pytest.mark.parametrize(
    "field", ["doc_id", "content_sha256", "cluster_id", "source_group"]
)
def test_all_identity_channels_reject_cross_split_overlap(field):
    a, b = record("a", "train"), record("b", "test")
    b[field] = a[field]
    with pytest.raises(ValueError, match="overlap"):
        audit([a, b])


def test_valid_manifest_and_training_reader_boundary():
    records = [
        record(role, role)
        for role in ("train", "guidance", "development", "validation", "test")
    ]
    assert audit(records) == {x["split"]: 1 for x in records}
    assert training_paths({"train": Path("t"), "guidance": Path("g")}) == {
        "train": Path("t"),
        "guidance": Path("g"),
    }
    with pytest.raises(ValueError, match="exactly"):
        training_paths({"train": Path("t"), "guidance": Path("g"), "test": Path("e")})


def test_empty_or_malformed_identity_cannot_pass():
    with pytest.raises(ValueError, match="Empty"):
        audit([])
    bad = record("a", "train")
    bad["content_sha256"] = "not-a-digest"
    with pytest.raises(ValueError, match="digest"):
        audit([bad])
