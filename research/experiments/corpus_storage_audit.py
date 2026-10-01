"""Compare identical real-document indexing on disk and RAM before adoption."""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
import zlib
from pathlib import Path

from guidon.artifacts import atomic_json, object_sha256, sha256_file
from guidon.corpus import BenchmarkIndex, CorpusBuilder, DedupConfig
from guidon.prepare_data import parallel_signatures
from guidon.sources import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    connection = sqlite3.connect(f"file:{args.database.resolve()}?mode=ro", uri=True)
    rows = [
        {
            "doc_id": identifier,
            "dataset": dataset,
            "url": "https://" + source + "/",
            "text": zlib.decompress(text).decode(),
            "split": constraint,
        }
        for identifier, dataset, source, text, constraint in connection.execute(
            "SELECT doc_id,dataset,source_group,text,role_constraint "
            "FROM documents ORDER BY id LIMIT 5000"
        )
    ]
    connection.close()
    config = DedupConfig(**plan["dedup"])
    reference = Path(plan["benchmark_references"]["path"])
    if sha256_file(reference) != plan["benchmark_references"]["sha256"]:
        raise ValueError("Reference fingerprints differ")
    benchmarks = BenchmarkIndex(read_jsonl(reference), config)
    prepared = list(
        parallel_signatures(rows, plan["dedup"], plan["signature_workers"], benchmarks)
    )
    observations = []
    for name, path in (
        ("disk", Path("artifacts/phase03/storage-audit-v1.sqlite")),
        ("ram", Path("/dev/shm/guidon-storage-audit-v1.sqlite")),
    ):
        builder = CorpusBuilder(path, [], config)
        builder.benchmarks = benchmarks
        started = time.monotonic()
        for row, preprocessed in prepared:
            builder.add(row, prepared=preprocessed)
        result = builder.finish(plan["split_seed"], plan["train_fraction"])
        elapsed = time.monotonic() - started
        metadata = list(
            builder.db.execute(
                "SELECT doc_id,content_sha256,source_group,dataset,excluded,"
                "duplicate,cluster_id,split FROM documents ORDER BY doc_id"
            )
        )
        builder.close()
        observations.append(
            {
                "store": name,
                "path": str(path),
                "elapsed_seconds": elapsed,
                "metadata_sha256": object_sha256(metadata),
                "counts": result["counts"],
                "database_sha256": sha256_file(path),
            }
        )
    equal = observations[0]["metadata_sha256"] == observations[1]["metadata_sha256"]
    atomic_json(
        args.output,
        {
            "sample_documents": len(rows),
            "plan_sha256": sha256_file(args.plan),
            "source_database": str(args.database),
            "method_and_document_order_identical": True,
            "metadata_equal": equal,
            "observations": observations,
            "timing_scope": "parent indexing/role assignment only; same preprocessed "
            "real sample, warmed references, disk then RAM; not total corpus time",
            "scientific_outcomes_inspected": False,
        },
    )
    if not equal:
        raise ValueError("Storage changed exclusion/duplicate/component/role metadata")


if __name__ == "__main__":
    main()
