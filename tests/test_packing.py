import json

import numpy as np
import pytest

from guidon.artifacts import atomic_json, sha256_file
from guidon.packing import (
    MixtureTape,
    Tape,
    TapeWriter,
    training_readers,
    write_capability,
)


def make_tape(root, role="train", cohort="pilot"):
    writer = TapeWriter(root, role, cohort, {"revision": "pinned"}, 0, shard_tokens=4)
    writer.add(dict(doc_id="a", split=role), [1, 2, 3])
    writer.add(dict(doc_id="b", split=role), [4, 5])
    return writer.finish()


def test_partial_document_prefix_adds_no_synthetic_terminal_label(tmp_path):
    writer = TapeWriter(tmp_path / "partial", "train", "pilot", {}, 0)
    writer.add(dict(doc_id="a", split="train"), [1, 2, 3, 4], loss_limit=3)
    writer.finish()
    tape = Tape(tmp_path / "partial" / "manifest.json", "train", "pilot")
    batch, cursor = tape.batch(0, 3, 1, 4)
    assert batch["labels"][0].tolist() == [1, 2, 3, 0]
    assert cursor == 3 and tape.records[0]["truncated_document"]


def test_cumulative_mixture_quotas_resume_padding_and_source_isolation(tmp_path):
    sources = []
    for name, token, weight in (("math", 1, 9), ("retention", 2, 1)):
        writer = TapeWriter(tmp_path / name, "train", "pilot", {"revision": "pin"}, 9)
        writer.add(dict(doc_id=name, split="train"), [token] * 100)
        writer.finish()
        sources.append(
            dict(
                manifest=name + "/manifest.json",
                sha256=sha256_file(tmp_path / name / "manifest.json"),
                weight_numerator=weight,
                weight_denominator=10,
            )
        )
    path = tmp_path / "mixture.json"
    atomic_json(
        path,
        dict(
            role="train",
            cohort="pilot",
            loss_tokens=100,
            packing="cumulative_integer_loss_token_mixture",
            sources=sources,
        ),
    )
    tape = MixtureTape(path, "train", "pilot", expected_sha256=sha256_file(path))
    cursor = 0
    for take in (1, 11, 7, 13, 9):
        batch, following = tape.batch(cursor, take, 2, 8)
        mask = batch["loss_mask"].astype(bool)
        quotas = np.subtract(tape.quotas(following), tape.quotas(cursor))
        assert np.sum(batch["segment_ids"][mask] == 0) == quotas[0]
        assert np.sum(batch["segment_ids"][mask] == 1) == quotas[1]
        assert np.all(batch["input_ids"][~mask] == 9)
        assert int(mask.sum()) == take
        repeated, _ = tape.batch(cursor, take, 2, 8)
        for key in batch:
            np.testing.assert_array_equal(batch[key], repeated[key])
        cursor = following
    with pytest.raises(ValueError, match="physical batch"):
        tape.batch(0, 17, 2, 8)


def test_exact_masked_labels_document_boundaries_and_cross_shard_resume(tmp_path):
    root = tmp_path / "train"
    manifest = make_tape(root)
    assert manifest["loss_tokens"] == 7
    tape = Tape(root / "manifest.json", "train", "pilot")
    batch, cursor = tape.batch(0, 7, 2, 4)
    assert cursor == 7
    assert batch["loss_mask"].sum() == 7
    np.testing.assert_array_equal(batch["input_ids"], [[0, 1, 2, 3], [0, 4, 5, 0]])
    np.testing.assert_array_equal(batch["labels"], [[1, 2, 3, 0], [4, 5, 0, 0]])
    np.testing.assert_array_equal(batch["segment_ids"], [[0, 0, 0, 0], [1, 1, 1, -1]])
    _, first_cursor = tape.batch(0, 3, 1, 4)
    resumed, final_cursor = tape.batch(first_cursor, 4, 1, 4)
    assert final_cursor == 7
    np.testing.assert_array_equal(resumed["labels"], [[0, 4, 5, 0]])
    np.testing.assert_array_equal(resumed["segment_ids"], [[0, 1, 1, 1]])
    with pytest.raises(ValueError, match="exhausted"):
        tape.batch(cursor, 1, 1, 4)


def test_modified_shard_and_role_cohort_rejected(tmp_path):
    root = tmp_path / "guidance"
    make_tape(root, "guidance")
    for role, cohort in (("test", "pilot"), ("guidance", "confirmation")):
        with pytest.raises(ValueError):
            Tape(root / "manifest.json", role, cohort)
    path = root / "tokens-00000.bin"
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="Corrupt"):
        Tape(root / "manifest.json", "guidance", "pilot")


def test_reader_capability_cannot_include_final_evaluation(tmp_path):
    for role in ("train", "guidance", "test"):
        make_tape(tmp_path / role, role)
    tapes = {role: tmp_path / role / "manifest.json" for role in ("train", "guidance")}
    destination = tmp_path / "training.json"
    value = write_capability(
        destination,
        tapes,
        process_role="training",
        cohort="pilot",
        preparation_sha256="pinned",
    )
    readers, evidence = training_readers(destination, cohort="pilot")
    assert set(readers) == {"train", "guidance"}
    assert all("/test/" not in p for p in evidence["reader_opened_paths"])
    value["tapes"]["test"] = {"manifest": "test/manifest.json", "sha256": "untrusted"}
    destination.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="exactly"):
        training_readers(destination, cohort="pilot")


def test_path_traversal_rejected_even_with_correct_hash(tmp_path):
    root = tmp_path / "train"
    make_tape(root)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    outside = tmp_path / "outside.bin"
    outside.write_bytes((root / "tokens-00000.bin").read_bytes())
    manifest["shards"][0].update(path="../outside.bin", sha256=sha256_file(outside))
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="escapes"):
        Tape(manifest_path, "train", "pilot")


def test_immutable_record_does_not_replace_previous_evidence(tmp_path):
    path = tmp_path / "record.json"
    atomic_json(path, {"first": 1})
    with pytest.raises(FileExistsError):
        atomic_json(path, {"second": 2})
    assert json.loads(path.read_text()) == {"first": 1}
