"""Pinned Hub catalogs and payload readers used only by the preparation process."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

from guidon.artifacts import object_sha256, sha256_file


def catalog(
    repository: str, revision: str, prefix: str, limit: int | None = None
) -> dict[str, Any]:
    api = HfApi()
    info = api.dataset_info(repository, revision=revision, files_metadata=True)
    if info.sha != revision or len(revision) != 40:
        raise ValueError("Dataset must resolve to its exact registered commit")
    files = []
    for entry in sorted(info.siblings, key=lambda value: value.rfilename):
        if not entry.rfilename.startswith(prefix) or not entry.rfilename.endswith(
            ".parquet"
        ):
            continue
        lfs = entry.lfs
        if not lfs or not lfs.sha256:
            raise ValueError(
                "Dataset shard has no independently recorded payload checksum"
            )
        files.append(dict(path=entry.rfilename, sha256=lfs.sha256, bytes=entry.size))
    if limit is not None:
        files = files[:limit]
    if not files:
        raise ValueError("Pinned candidate shard selection is empty")
    return {
        "repository": repository,
        "revision": revision,
        "prefix": prefix,
        "files": files,
        "selection": "lexicographic_pinned_shard_prefix_before_model_outcomes",
        "card": info.card_data.to_dict() if info.card_data else {},
    }


def documents(source: dict[str, Any], cache_dir: Path) -> Iterator[dict[str, Any]]:
    """No nominal token count is used as a packing/accounting input."""
    for shard in source["files"]:
        path = Path(
            hf_hub_download(
                source["repository"],
                shard["path"],
                repo_type="dataset",
                revision=source["revision"],
                local_dir=cache_dir,
            )
        )
        if sha256_file(path) != shard["sha256"]:
            raise ValueError(
                "Downloaded corpus payload differs from the pinned catalog"
            )
        parquet = pq.ParquetFile(path)
        if not {"text", "url"} <= set(parquet.schema.names):
            raise ValueError("Corpus lacks text and auditable publisher metadata")
        offset = 0
        for batch in parquet.iter_batches(batch_size=512, columns=["text", "url"]):
            for row in batch.to_pylist():
                identity = {
                    "repository": source["repository"],
                    "revision": source["revision"],
                    "shard": shard["path"],
                    "row": offset,
                }
                yield dict(
                    doc_id=object_sha256(identity), dataset=source["name"], **row
                )
                offset += 1
        print(
            json.dumps(
                {
                    "event": "source_shard_complete",
                    "dataset": source["name"],
                    "shard_sha256": shard["sha256"],
                    "documents": offset,
                }
            ),
            flush=True,
        )
        # Only task-owned downloaded corpus payloads are removed after ingestion;
        # the pinned catalog suffices to reproduce them without retaining text.
        path.unlink()


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)
