"""Immutable document tapes and exact, document-isolated next-token batches."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from guidon.artifacts import atomic_json, checked_path, object_sha256, sha256_file


class TapeWriter:
    """Store each document as BOS(=EOS), content, EOS; never mix role tapes.

    Every document has len(content)+1 trainable labels, including its terminal
    EOS. The first content token is conditioned on its own BOS. No target or
    attention edge crosses a document boundary. A sequence window may truncate
    preceding context, but never repeats or drops its registered target labels.
    """

    def __init__(
        self,
        root: Path,
        role: str,
        cohort: str,
        tokenizer: dict[str, Any],
        eos: int,
        *,
        shard_tokens: int = 8_000_000,
    ):
        if root.exists():
            raise FileExistsError("Token tapes are immutable")
        root.mkdir(parents=True)
        if not 0 <= eos < 65536:
            raise ValueError("This tape format requires a uint16 tokenizer")
        self.root, self.role, self.cohort = root, role, cohort
        self.tokenizer, self.eos = tokenizer, eos
        self.shard_tokens = shard_tokens
        self.shards: list[dict[str, Any]] = []
        self.shard_handle: Any = None
        self.shard_count = 0
        self.offset = self.loss_tokens = self.documents = 0
        self.index = (root / "documents.jsonl").open("x")

    def _close_shard(self) -> None:
        if self.shard_handle is None:
            return
        self.shard_handle.flush()
        os.fsync(self.shard_handle.fileno())
        self.shard_handle.close()
        path = Path(self.shard_handle.name)
        self.shards.append(
            {
                "path": path.name,
                "sha256": sha256_file(path),
                "tokens": self.shard_count,
            }
        )
        self.shard_handle = None

    def add(
        self, metadata: dict[str, Any], ids: list[int], *, loss_limit: int | None = None
    ) -> None:
        if not ids:
            return
        content = np.asarray(ids)
        if (
            content.dtype.kind not in {"i", "u"}
            or content.ndim != 1
            or content.min() < 0
            or content.max() >= 65536
        ):
            raise ValueError("Tokenizer emitted an ID outside the registered dtype")
        if metadata["split"] != self.role:
            raise ValueError("Cross-role packing attempt")
        tokens = np.empty(content.size + 2, dtype="<u2")
        tokens[0] = tokens[-1] = self.eos
        tokens[1:-1] = content
        full_labels = len(tokens) - 1
        if loss_limit is not None:
            if loss_limit <= 0:
                raise ValueError("Partial-document loss limit must be positive")
            tokens = tokens[: loss_limit + 1]
        record = {key: value for key, value in metadata.items() if key != "text"}
        record.update(
            {
                "cohort": self.cohort,
                "token_offset": self.offset,
                "loss_token_offset": self.loss_tokens,
                "stored_tokens": len(tokens),
                "loss_tokens": len(tokens) - 1,
                "truncated_document": len(tokens) - 1 < full_labels,
            }
        )
        self.index.write(json.dumps(record, sort_keys=True) + "\n")
        position = 0
        while position < len(tokens):
            if self.shard_handle is None:
                self.shard_count = 0
                self.shard_handle = (
                    self.root / f"tokens-{len(self.shards):05d}.bin"
                ).open("xb")
            take = min(len(tokens) - position, self.shard_tokens - self.shard_count)
            self.shard_handle.write(tokens[position : position + take].tobytes())
            position += take
            self.shard_count += take
            if self.shard_count == self.shard_tokens:
                self._close_shard()
        self.offset += len(tokens)
        self.loss_tokens += len(tokens) - 1
        self.documents += 1

    def finish(self) -> dict[str, Any]:
        self._close_shard()
        self.index.flush()
        os.fsync(self.index.fileno())
        self.index.close()
        if not self.loss_tokens:
            raise ValueError("Cannot publish an empty token tape")
        manifest = {
            "schema_version": 1,
            "role": self.role,
            "cohort": self.cohort,
            "dtype": "uint16_little_endian",
            "tokenizer": self.tokenizer,
            "bos_token_id": self.eos,
            "eos_token_id": self.eos,
            "packing": "document_isolated_bos_content_eos",
            "boundary_attention": "same_document_and_causal",
            "boundary_loss": "within_document_including_first_content_and_eos",
            "stride": "no_target_repetition",
            "padding_loss_mask": 0,
            "partial_final_document": "prefix_without_synthetic_eos",
            "documents": self.documents,
            "stored_tokens": self.offset,
            "loss_tokens": self.loss_tokens,
            "document_index": {
                "path": "documents.jsonl",
                "sha256": sha256_file(self.root / "documents.jsonl"),
            },
            "shards": self.shards,
        }
        atomic_json(self.root / "manifest.json", manifest)
        return manifest


class Tape:
    def __init__(
        self,
        manifest_path: Path,
        expected_role: str,
        expected_cohort: str,
        *,
        expected_sha256: str | None = None,
    ):
        self.root = manifest_path.parent.resolve()
        if expected_sha256 and sha256_file(manifest_path) != expected_sha256:
            raise ValueError("Token manifest hash differs from the granted capability")
        self.manifest = json.loads(manifest_path.read_text())
        if self.manifest["role"] != expected_role:
            raise ValueError("Reader role differs from the token tape role")
        if self.manifest["cohort"] != expected_cohort:
            raise ValueError("Pilot and confirmation cohorts may not be interchanged")
        if self.manifest["packing"] != "document_isolated_bos_content_eos":
            raise ValueError("Unregistered packing convention")
        if self.manifest["dtype"] != "uint16_little_endian":
            raise ValueError("Unregistered tape dtype")
        self.opens: list[str] = [str(manifest_path.resolve())]
        index_info = self.manifest["document_index"]
        index = self._check(index_info)
        with index.open() as handle:
            self.records = [json.loads(line) for line in handle]
        self.offsets = np.asarray(
            [record["loss_token_offset"] for record in self.records], dtype=np.int64
        )
        offset = labels = 0
        for record in self.records:
            if record["split"] != expected_role or record["cohort"] != expected_cohort:
                raise ValueError("Cross-role/cohort document in a token tape")
            if (
                record["token_offset"] != offset
                or record["loss_token_offset"] != labels
            ):
                raise ValueError("Noncontiguous or forged document cursor")
            if record["stored_tokens"] != record["loss_tokens"] + 1:
                raise ValueError("Invalid next-token document accounting")
            offset += record["stored_tokens"]
            labels += record["loss_tokens"]
        if (
            labels != self.manifest["loss_tokens"]
            or offset != self.manifest["stored_tokens"]
        ):
            raise ValueError("Token budget differs from document accounting")
        arrays = []
        for shard in self.manifest["shards"]:
            path = self._check(shard)
            if path.stat().st_size != 2 * shard["tokens"]:
                raise ValueError("Token shard size differs from declared shape")
            arrays.append(np.memmap(path, mode="r", dtype="<u2"))
        self.arrays = arrays
        self.shard_ends = np.cumsum([len(a) for a in arrays])
        if not len(self.shard_ends) or self.shard_ends[-1] != offset:
            raise ValueError("Token shard total differs from the document index")

    def _check(self, entry: dict[str, Any]) -> Path:
        path = checked_path(self.root, entry["path"])
        if sha256_file(path) != entry["sha256"]:
            raise ValueError(f"Corrupt token artifact: {entry['path']}")
        self.opens.append(str(path))
        return path

    def _tokens(self, start: int, count: int) -> np.ndarray:
        pieces = []
        while count:
            index = int(np.searchsorted(self.shard_ends, start, side="right"))
            lower = int(self.shard_ends[index - 1]) if index else 0
            take = min(count, int(self.shard_ends[index]) - start)
            pieces.append(
                np.asarray(self.arrays[index][start - lower : start - lower + take])
            )
            count -= take
            start += take
        return np.concatenate(pieces)

    def batch(
        self, cursor: int, loss_tokens: int, batch_sequences: int, sequence_length: int
    ) -> tuple[dict[str, np.ndarray], int]:
        """Fill a fixed shape with exactly loss_tokens distinct target labels."""
        capacity = batch_sequences * sequence_length
        if min(batch_sequences, sequence_length, loss_tokens) <= 0:
            raise ValueError("Batch dimensions and requested labels must be positive")
        if loss_tokens > capacity:
            raise ValueError(
                "Positive loss-token count exceeds physical batch capacity"
            )
        if cursor < 0 or cursor + loss_tokens > self.manifest["loss_tokens"]:
            raise ValueError("Tape exhausted; recycling guidance is prohibited")
        shape = (batch_sequences, sequence_length)
        eos = self.manifest["eos_token_id"]
        batch = {
            "input_ids": np.full(shape, eos, dtype=np.int32),
            "labels": np.full(shape, eos, dtype=np.int32),
            "position_ids": np.zeros(shape, dtype=np.int32),
            "segment_ids": np.full(shape, -1, dtype=np.int32),
            "loss_mask": np.zeros(shape, dtype=np.float32),
        }
        filled = 0
        while filled < loss_tokens:
            document = int(np.searchsorted(self.offsets, cursor, side="right") - 1)
            record = self.records[document]
            inside = cursor - record["loss_token_offset"]
            row, col = divmod(filled, sequence_length)
            take = min(
                loss_tokens - filled,
                record["loss_tokens"] - inside,
                sequence_length - col,
            )
            ids = self._tokens(record["token_offset"] + inside, take + 1)
            region = (row, slice(col, col + take))
            batch["input_ids"][region] = ids[:-1]
            batch["labels"][region] = ids[1:]
            # A truncated continuation restarts positions at its visible context.
            batch["position_ids"][region] = np.arange(take, dtype=np.int32)
            batch["segment_ids"][region] = document
            batch["loss_mask"][region] = 1
            cursor += take
            filled += take
        return batch, cursor


def write_capability(
    destination: Path,
    tapes: dict[str, Path],
    *,
    process_role: str,
    cohort: str,
    preparation_sha256: str,
) -> dict[str, Any]:
    allowed = {
        "training": {"train", "guidance"},
        "development": {"development"},
        "evaluation": {"validation", "test"},
    }
    if process_role not in allowed or set(tapes) != allowed[process_role]:
        raise ValueError("Reader capability contains unauthorized roles")
    value = {
        "schema_version": 1,
        "process_role": process_role,
        "cohort": cohort,
        "preparation_sha256": preparation_sha256,
        "tapes": {
            role: {
                "manifest": os.path.relpath(
                    path.resolve(), destination.parent.resolve()
                ),
                "sha256": sha256_file(path),
            }
            for role, path in tapes.items()
        },
    }
    atomic_json(destination, value)
    return value


class MixtureTape:
    """Cumulative integer quotas give exact 90/10 accounting, including final masks."""

    def __init__(
        self, manifest_path: Path, role: str, cohort: str, *, expected_sha256: str
    ):
        if sha256_file(manifest_path) != expected_sha256:
            raise ValueError("Mixture manifest differs from its granted capability")
        self.root = manifest_path.parent.resolve()
        self.manifest = json.loads(manifest_path.read_text())
        if self.manifest["role"] != role or self.manifest["cohort"] != cohort:
            raise ValueError("Mixture role/cohort differs from its granted capability")
        if self.manifest["packing"] != "cumulative_integer_loss_token_mixture":
            raise ValueError("Unsupported mixture accounting")
        self.sources = self.manifest["sources"]
        if len(self.sources) != 2:
            raise ValueError("The registered continued mixture has two sources")
        if [s["weight_numerator"] for s in self.sources] != [9, 1] or any(
            s["weight_denominator"] != 10 for s in self.sources
        ):
            raise ValueError(
                "Continued training/guidance must use the exact 90/10 mixture"
            )
        self.readers = [
            Tape(
                checked_path(self.root, s["manifest"]),
                role,
                cohort,
                expected_sha256=s["sha256"],
            )
            for s in self.sources
        ]
        if (
            self.readers[0].manifest["tokenizer"]
            != self.readers[1].manifest["tokenizer"]
        ):
            raise ValueError("Mixture tokenizers differ")
        self.opens = [
            str(manifest_path.resolve()),
            *[p for r in self.readers for p in r.opens],
        ]
        for quota, reader in zip(
            self.quotas(self.manifest["loss_tokens"]), self.readers, strict=True
        ):
            if quota > reader.manifest["loss_tokens"]:
                raise ValueError("Mixture source capacity is insufficient")

    @staticmethod
    def quotas(cursor: int) -> list[int]:
        math_labels = cursor * 9 // 10
        return [math_labels, cursor - math_labels]

    def batch(
        self, cursor: int, loss_tokens: int, batch_sequences: int, sequence_length: int
    ):
        if min(loss_tokens, batch_sequences, sequence_length) <= 0:
            raise ValueError("Mixture batch dimensions and labels must be positive")
        if loss_tokens > batch_sequences * sequence_length:
            raise ValueError("Mixture labels exceed physical batch capacity")
        if cursor < 0 or cursor + loss_tokens > self.manifest["loss_tokens"]:
            raise ValueError("Mixture tape exhausted; labels may not be recycled")
        before, after = self.quotas(cursor), self.quotas(cursor + loss_tokens)
        target = None
        written = offset = 0
        for index, reader in enumerate(self.readers):
            take = after[index] - before[index]
            if not take:
                offset += len(reader.records)
                continue
            chunk, _ = reader.batch(
                before[index], take, batch_sequences, sequence_length
            )
            if target is None:
                target = {key: np.zeros_like(value) for key, value in chunk.items()}
                for key in ("input_ids", "labels"):
                    target[key].fill(reader.manifest["eos_token_id"])
                target["segment_ids"].fill(-1)
            for key, value in chunk.items():
                flattened = value.reshape(-1)[:take]
                if key == "segment_ids":
                    flattened = flattened + offset
                target[key].reshape(-1)[written : written + take] = flattened
            written += take
            offset += len(reader.records)
        if target is None or written != loss_tokens:
            raise ValueError("Invalid empty mixture batch")
        for row in range(batch_sequences):
            ids = target["segment_ids"][row]
            starts = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1]
            previous = np.repeat(starts, np.diff(np.r_[starts, sequence_length]))
            target["position_ids"][row] = np.arange(sequence_length) - previous
        return target, cursor + loss_tokens


def training_readers(
    capability_path: Path, *, cohort: str
) -> tuple[dict[str, Tape | MixtureTape], dict[str, Any]]:
    value = json.loads(capability_path.read_text())
    if value["process_role"] != "training" or set(value["tapes"]) != {
        "train",
        "guidance",
    }:
        raise ValueError("Training capability must grant exactly train and guidance")
    if value["cohort"] != cohort:
        raise ValueError("Run cohort differs from the granted training capability")
    readers = {}
    for role, item in value["tapes"].items():
        path = capability_path.parent / item["manifest"]
        if sha256_file(path) != item["sha256"]:
            raise ValueError("Capability manifest hash differs")
        entry = json.loads(path.read_text())
        reader = (
            MixtureTape
            if entry["packing"] == "cumulative_integer_loss_token_mixture"
            else Tape
        )
        readers[role] = reader(
            path,
            role,
            cohort,
            expected_sha256=item["sha256"],
        )
    return readers, {
        "capability_sha256": sha256_file(capability_path),
        "preparation_sha256": value["preparation_sha256"],
        "cohort": cohort,
        "reader_opened_paths": sorted(
            {p for tape in readers.values() for p in tape.opens}
        ),
        "data_sha256": object_sha256(value),
    }


def capacities(records: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        key = f"{record['split']}/{record['cohort']}/{record['dataset']}"
        counts[key] = counts.get(key, 0) + record["loss_tokens"]
    return counts


def exact_updates(tokens: int, labels_per_update: int) -> int:
    if tokens <= 0 or labels_per_update <= 0:
        raise ValueError("Token budgets must be positive")
    return math.ceil(tokens / labels_per_update)
