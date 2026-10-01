"""Construct audited joint corpus identities and immutable, role-limited tapes.

Production runs require complete benchmark reference coverage, actual token
capacities, and an external evaluator public key. --fixture uses only caller
supplied train/guidance/development examples for local verification; it cannot
publish a production-ready or sealed-evaluation gate.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import itertools
import json
import multiprocessing
import sqlite3
import time
import zlib
from collections import Counter
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download
from tokenizers import Tokenizer

from guidon.artifacts import atomic_json, object_sha256, sha256_file
from guidon.corpus import (
    BenchmarkIndex,
    CorpusBuilder,
    DedupConfig,
    PreparedDocument,
    canonicalize,
    false_negative_audit,
    shingles,
)
from guidon.data_boundary import audit
from guidon.packing import TapeWriter, write_capability
from guidon.sealing import encrypt_file, public_key, seal_directory
from guidon.sources import documents, read_jsonl

WORKER_BENCHMARKS: BenchmarkIndex | None = None


def initialize_preprocessing(benchmarks: BenchmarkIndex) -> None:
    global WORKER_BENCHMARKS
    WORKER_BENCHMARKS = benchmarks


def signature_batch(
    arguments: tuple[list[dict[str, Any]], dict[str, Any]],
) -> list[tuple[dict[str, Any], PreparedDocument]]:
    rows, settings = arguments
    config = DedupConfig(**settings)
    if WORKER_BENCHMARKS is None:
        raise RuntimeError("Preprocessing worker has no registered benchmark index")
    result = []
    for row in rows:
        text = canonicalize(row["text"])
        values = shingles(text, config.shingle_words)
        result.append(
            (
                row,
                PreparedDocument(
                    text=text,
                    compressed_text=zlib.compress(text.encode()),
                    signature=config.signature(values).hashvalues,
                    benchmark_excluded=bool(WORKER_BENCHMARKS.matches(values, text)),
                    lexical_content=bool(values),
                ),
            )
        )
    return result


def parallel_signatures(
    rows: Iterable[dict[str, Any]],
    settings: dict[str, Any],
    workers: int,
    benchmarks: BenchmarkIndex,
) -> Iterator[tuple[dict[str, Any], PreparedDocument | None]]:
    if workers <= 1:
        for row in rows:
            yield row, None
        return
    iterator = iter(rows)
    # This preparation CLI runs on the registered Linux hosts. Fork shares the
    # large immutable benchmark index; workers never open a role token tape.
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("fork"),
        initializer=initialize_preprocessing,
        initargs=(benchmarks,),
    ) as pool:
        queue = []
        for _ in range(workers * 2):
            batch = list(itertools.islice(iterator, 32))
            if not batch:
                break
            queue.append(pool.submit(signature_batch, (batch, settings)))
        while queue:
            future = queue.pop(0)
            yield from future.result()
            batch = list(itertools.islice(iterator, 32))
            if batch:
                queue.append(pool.submit(signature_batch, (batch, settings)))


def load_tokenizer(spec: dict[str, Any], cache_dir: Path) -> Tokenizer:
    info = HfApi().model_info(spec["repository"], revision=spec["revision"])
    if info.sha != spec["revision"] or len(info.sha) != 40:
        raise ValueError("Tokenizer revision did not resolve to its pinned commit")
    path = Path(
        hf_hub_download(
            spec["repository"],
            "tokenizer.json",
            revision=spec["revision"],
            local_dir=cache_dir,
        )
    )
    spec["tokenizer_json_sha256"] = sha256_file(path)
    tokenizer = Tokenizer.from_file(str(path))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    return tokenizer


def prepared_documents(path: Path):
    connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    try:
        for row in connection.execute(
            "SELECT doc_id,content_sha256,cluster_id,source_group,split,dataset,text "
            "FROM documents WHERE split!='quarantine' AND duplicate=0 ORDER BY doc_id"
        ):
            keys = (
                "doc_id",
                "content_sha256",
                "cluster_id",
                "source_group",
                "split",
                "dataset",
                "text",
            )
            yield dict(
                zip(keys, (*row[:-1], zlib.decompress(row[-1]).decode()), strict=True)
            )
    finally:
        connection.close()


def check_plan(plan: dict[str, Any], *, fixture: bool) -> None:
    if not plan.get("group_before_pack") or plan.get("guidance_reuse_within_run"):
        raise ValueError("Grouping before packing and fresh guidance are mandatory")
    if plan["split_seed"] != 20260930 and not fixture:
        raise ValueError(
            "Production global split seed differs from the registered contract"
        )
    roles = {item["role"] for item in plan["tapes"]}
    if fixture:
        if not roles <= {"train", "guidance", "development"}:
            raise ValueError("Fixture mode cannot open or emit final evaluation")
    else:
        if roles != {"train", "guidance", "development", "validation", "test"}:
            raise ValueError("Production preparation must reserve all five roles")
        coverage = plan.get("benchmark_coverage", {})
        required = {
            "lambada_openai",
            "arc_easy",
            "hellaswag",
            "piqa",
            "winogrande",
            "gsm8k",
            "math",
        }
        if set(coverage) != required or not all(
            v["complete"] for v in coverage.values()
        ):
            raise ValueError(
                "Production benchmark contamination coverage is incomplete"
            )
        if any(
            t["minimum_loss_tokens"] < 10_000_000
            for t in plan["tapes"]
            if t["role"] != "train"
        ):
            raise ValueError(
                "Production held-out reservations require at least 10M labels"
            )


def build_index(
    plan: dict[str, Any], output: Path, input_path: Path | None, *, fixture: bool
) -> dict[str, Any]:
    reference_path = Path(plan["benchmark_references"]["path"])
    if sha256_file(reference_path) != plan["benchmark_references"]["sha256"]:
        raise ValueError("Benchmark fingerprints differ from registered coverage")
    references = list(read_jsonl(reference_path))
    config = DedupConfig(**plan["dedup"])
    builder = CorpusBuilder(output / "preparation.sqlite", references, config)
    samples: list[str] = []
    sources = (
        read_jsonl(input_path)
        if input_path
        else itertools.chain.from_iterable(
            documents(source, output / ".source-cache") for source in plan["sources"]
        )
    )
    started = time.monotonic()
    try:
        for count, (row, prepared) in enumerate(
            parallel_signatures(
                sources,
                plan["dedup"],
                plan.get("signature_workers", 1),
                builder.benchmarks,
            ),
            1,
        ):
            if fixture and row.get("split") not in {"train", "guidance", "development"}:
                raise ValueError("Fixture preparation may receive only nonsealed roles")
            builder.add(row, prepared=prepared)
            if len(samples) < 128:
                samples.append(row["text"])
            if count % 10_000 == 0:
                print(
                    json.dumps(
                        {
                            "event": "indexed",
                            "documents": count,
                            "elapsed_seconds": time.monotonic() - started,
                        }
                    ),
                    flush=True,
                )
        result = builder.finish(plan["split_seed"], plan["train_fraction"])
        # Add deterministic edited copies to exercise actual-corpus signatures;
        # an absence of positives among arbitrary random pairs is not an audit.
        expanded = samples + [text + " additional audit words" for text in samples]
        result["false_negative_audit"] = false_negative_audit(expanded, config)
        if result["false_negative_audit"]["missed_pairs"]:
            raise ValueError("Independent sample-pair audit found missed duplicates")
        builder.close()
        result.update(
            fixture=fixture,
            elapsed_seconds=time.monotonic() - started,
            plan_sha256=object_sha256(plan),
            index_sha256=sha256_file(output / "preparation.sqlite"),
            source_sha256={
                str(path.relative_to(Path(__file__).resolve().parents[2])): sha256_file(
                    path
                )
                for path in (
                    Path(__file__).resolve(),
                    Path(__file__).with_name("corpus.py"),
                    Path(__file__).with_name("sources.py"),
                )
            },
        )
        atomic_json(output / "index-audit.json", result)
        return result
    finally:
        builder.close()


def materialize(
    plan: dict[str, Any], output: Path, *, fixture: bool, evaluator_key: Path | None
) -> dict[str, Any]:
    preparation_identity = object_sha256(plan)
    registered = json.loads((output / "registered-plan.json").read_text())
    index_audit = json.loads((output / "index-audit.json").read_text())
    if (
        object_sha256(registered) != preparation_identity
        or index_audit["plan_sha256"] != preparation_identity
        or index_audit["fixture"] != fixture
        or sha256_file(output / "preparation.sqlite") != index_audit["index_sha256"]
        or index_audit["false_negative_audit"]["missed_pairs"]
    ):
        raise ValueError("Corpus index/plan/audit identity differs")
    # Tokenizer checks enrich output records without mutating the frozen plan.
    plan = copy.deepcopy(plan)
    tokenizers = {
        key: load_tokenizer(spec, output / ".tokenizer-cache" / key)
        for key, spec in plan["tokenizers"].items()
    }
    writers = {}
    remaining = {}
    settings = {}
    for spec in plan["tapes"]:
        key = spec["name"]
        settings[key] = spec
        remaining[key] = spec["minimum_loss_tokens"]
        token_spec = plan["tokenizers"][spec["tokenizer"]]
        writers[key] = TapeWriter(
            output / "tapes" / key,
            spec["role"],
            spec["cohort"],
            token_spec,
            spec["eos_token_id"],
        )
    counts: Counter[str] = Counter()
    manifest_path = output / "identities.jsonl"
    with manifest_path.open("x") as manifest:
        # Global roles are fixed first. Within each role/dataset, deterministic
        # document order fills disjoint pilot/confirmation tapes for each tokenizer.
        for row in prepared_documents(output / "preparation.sqlite"):
            metadata = {k: v for k, v in row.items() if k != "text"}
            manifest.write(json.dumps(metadata, sort_keys=True) + "\n")
            by_tokenizer: dict[str, list[str]] = {}
            for key, spec in settings.items():
                partition = spec.get("source_partition")
                matches_partition = not partition or (
                    int(
                        object_sha256({"cluster": row["cluster_id"], "seed": 20261003})[
                            :16
                        ],
                        16,
                    )
                    % partition["modulus"]
                    in partition["remainders"]
                )
                if (
                    spec["role"] == row["split"]
                    and spec["dataset"] == row["dataset"]
                    and remaining[key] > 0
                    and matches_partition
                ):
                    by_tokenizer.setdefault(spec["tokenizer"], []).append(key)
            for tokenizer_name, keys in by_tokenizer.items():
                # Cohorts sharing this tokenizer consume whole documents once.
                key = keys[0]
                ids = (
                    tokenizers[tokenizer_name]
                    .encode(row["text"], add_special_tokens=False)
                    .ids
                )
                if not ids:
                    continue
                used = min(len(ids) + 1, remaining[key])
                writers[key].add(metadata, ids, loss_limit=used)
                remaining[key] -= used
                counts[key] += used
    deficits = {key: count for key, count in remaining.items() if count > 0}
    if deficits:
        atomic_json(
            output / "capacity-failure.json", {"deficits": deficits, "fixture": fixture}
        )
        raise ValueError(
            "Insufficient post-filter token capacity; "
            "extend registered candidates deterministically"
        )
    manifests = {}
    for key, writer in writers.items():
        manifests[key] = writer.finish()
        if writer.role in {"validation", "test"}:
            if evaluator_key is None:
                raise ValueError(
                    "External evaluator public key required for final payloads"
                )
            seal_directory(
                output / "tapes" / key,
                output / "sealed" / key,
                evaluator_key,
                dict(
                    role=writer.role,
                    cohort=writer.cohort,
                    preparation_sha256=preparation_identity,
                    tape=key,
                ),
            )
    with manifest_path.open() as handle:
        roles = audit(json.loads(line) for line in handle)
    for spec in plan.get("mixtures", []):
        paths = [output / "tapes" / name / "manifest.json" for name in spec["sources"]]
        for path in paths:
            if not path.exists():
                raise ValueError("Training mixture may not reference a sealed role")
        value = dict(
            schema_version=1,
            packing="cumulative_integer_loss_token_mixture",
            role=spec["role"],
            cohort=spec["cohort"],
            loss_tokens=spec["loss_tokens"],
            sources=[
                dict(
                    manifest=str(path.relative_to(output / "tapes")),
                    sha256=sha256_file(path),
                    weight_numerator=weight,
                    weight_denominator=10,
                )
                for path, weight in zip(paths, [9, 1], strict=True)
            ],
        )
        atomic_json(output / "tapes" / (spec["name"] + ".json"), value)
    capabilities = {}
    for spec in plan["capabilities"]:
        tapes = {
            role: output
            / "tapes"
            / (name if name.endswith(".json") else name + "/manifest.json")
            for role, name in spec["tapes"].items()
        }
        value = write_capability(
            output / spec["path"],
            tapes,
            process_role=spec["process_role"],
            cohort=spec["cohort"],
            preparation_sha256=preparation_identity,
        )
        capabilities[spec["path"]] = object_sha256(value)
    sealed_capabilities = {}
    if not fixture:
        for regime in ("scratch", "continued"):
            evaluation_tapes = {}
            for spec in plan["tapes"]:
                if spec["name"].startswith(regime) and spec["role"] in {
                    "validation",
                    "test",
                }:
                    sealed_path = output / "sealed" / spec["name"] / "sealed.json"
                    evaluation_tapes[spec["name"]] = {
                        "role": spec["role"],
                        "cohort": spec["cohort"],
                        "sealed": str(sealed_path.relative_to(output)),
                        "sha256": sha256_file(sealed_path),
                    }
            if not evaluation_tapes:
                raise ValueError("Regime has no sealed evaluator reservation")
            name = f"{regime}-sealed-evaluation.json"
            value = {
                "schema_version": 1,
                "process_role": "sealed_evaluation",
                "cohort": "confirmation",
                "preparation_sha256": preparation_identity,
                "tapes": evaluation_tapes,
            }
            atomic_json(output / name, value)
            sealed_capabilities[name] = object_sha256(value)
        # The joint preparation database also contains final text. Retain its
        # auditable original bytes under external encryption, then remove this
        # task's intermediate plaintext once every tape/capability is durable.
        source = output / "preparation.sqlite"
        context = {
            "role": "joint_preparation_intermediate",
            "preparation_sha256": preparation_identity,
            "index_sha256": index_audit["index_sha256"],
        }
        sealed_index = encrypt_file(
            source,
            output / "sealed" / "preparation.sqlite.aesgcm",
            public_key(evaluator_key),
            context,
        )
        atomic_json(output / "sealed" / "preparation-index.json", sealed_index)
        source.unlink()
        for suffix in ("-wal", "-shm"):
            (output / ("preparation.sqlite" + suffix)).unlink(missing_ok=True)
    result = dict(
        schema_version=1,
        fixture=fixture,
        production_ready=not fixture,
        manifest_sha256=sha256_file(manifest_path),
        preparation_sha256=preparation_identity,
        document_roles=roles,
        actual_loss_tokens=dict(counts),
        capabilities=capabilities,
        sealed_capabilities=sealed_capabilities,
        tape_manifests={key: object_sha256(value) for key, value in manifests.items()},
        final_payloads_sealed=not fixture,
        preparation_plaintext_removed=not fixture,
        tokenizer_records=plan["tokenizers"],
    )
    atomic_json(output / "audit.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--fixture", action="store_true")
    parser.add_argument("--index-only", action="store_true")
    parser.add_argument("--materialize-index", action="store_true")
    parser.add_argument("--evaluator-public-key", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    check_plan(plan, fixture=args.fixture)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "plan_sha256": object_sha256(plan),
                    "fixture": args.fixture,
                    "stages": [
                        "joint_index",
                        "audit",
                        "tokenize",
                        "seal",
                        "capabilities",
                    ],
                },
                indent=2,
            )
        )
        return
    if not args.fixture and not args.index_only:
        if args.evaluator_public_key is None:
            raise ValueError(
                "Production materialization requires an external evaluator public key"
            )
        public_key(args.evaluator_public_key)
    if not args.materialize_index:
        args.output.mkdir(parents=True, exist_ok=False)
        atomic_json(args.output / "registered-plan.json", plan)
        build_index(plan, args.output, args.input, fixture=args.fixture)
    if not args.index_only:
        result = materialize(
            plan,
            args.output,
            fixture=args.fixture,
            evaluator_key=args.evaluator_public_key,
        )
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
