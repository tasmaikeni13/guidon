"""Ordered parallel tokenization inside the role-constrained preparation process."""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Iterator, Mapping
from typing import Any

from tokenizers import Tokenizer

from guidon.artifacts import object_sha256


def eligible_tapes(
    row: Mapping[str, Any],
    settings: Mapping[str, Mapping[str, Any]],
    remaining: Mapping[str, int],
) -> dict[str, list[str]]:
    """Preserve registered tape priority and whole-component source partitions."""
    result: dict[str, list[str]] = {}
    partition_value: int | None = None
    for name, spec in settings.items():
        if (
            spec["role"] != row["split"]
            or spec["dataset"] != row["dataset"]
            or remaining[name] <= 0
        ):
            continue
        partition = spec.get("source_partition")
        if partition:
            if partition_value is None:
                partition_value = int(
                    object_sha256({"cluster": row["cluster_id"], "seed": 20261003})[
                        :16
                    ],
                    16,
                )
            if partition_value % partition["modulus"] not in partition["remainders"]:
                continue
        result.setdefault(spec["tokenizer"], []).append(name)
    return result


def parallel_tokenize(
    rows: Iterable[dict[str, Any]],
    tokenizers: Mapping[str, Tokenizer],
    settings: Mapping[str, Mapping[str, Any]],
    remaining: Mapping[str, int],
    *,
    batch_size: int = 256,
) -> Iterator[tuple[dict[str, Any], dict[str, list[int]]]]:
    """Use Rust's ordered batch encoder while the caller advances exact budgets.

    `remaining` is the caller's live budget mapping. A prefetched batch may encode
    an extra document after a tape fills; it never assigns, writes, or counts that
    document. Allocation is decided again in order by the caller after each yield.
    """
    if batch_size < 1:
        raise ValueError("Tokenization batch size must be positive")
    iterator = iter(rows)
    while batch := list(itertools.islice(iterator, batch_size)):
        indices: dict[str, list[int]] = {}
        for index, row in enumerate(batch):
            for tokenizer in eligible_tapes(row, settings, remaining):
                indices.setdefault(tokenizer, []).append(index)
        encoded: list[dict[str, list[int]]] = [{} for _ in batch]
        for name, selected in indices.items():
            values = tokenizers[name].encode_batch(
                [batch[index]["text"] for index in selected], add_special_tokens=False
            )
            for index, value in zip(selected, values, strict=True):
                encoded[index][name] = value.ids
        yield from zip(batch, encoded, strict=True)
