"""Prepare answer-bearing benchmark fingerprints without exposing payloads to training.

This module is only a contamination-preparation reader. It produces references
outside Git and a small pinned coverage record; it never computes model scores.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

from guidon.artifacts import atomic_json, object_sha256, sha256_file
from guidon.corpus import canonicalize

SOURCES = {
    "lambada_openai": (
        "EleutherAI/lambada_openai",
        "default/test/",
        "900124bf3b8235c6daf21033af9948b3f07346c4",
    ),
    "arc_easy": (
        "allenai/ai2_arc",
        "ARC-Easy/",
        "210d026faf9955653af8916fad021475a3f00453",
    ),
    "hellaswag": (
        "Rowan/hellaswag",
        "data/",
        "218ec52e09a7e7462a5400043bb9a69a41d06b76",
    ),
    "winogrande": (
        "allenai/winogrande",
        "winogrande_xl/",
        "01e74176c63542e6b0bcb004dcdea22d94fb67b5",
    ),
    "gsm8k": ("openai/gsm8k", "main/", "740312add88f781978c0658806c59bc2815b9866"),
    "math": (
        "DigitalLearningGmbH/MATH-lighteval",
        "data/",
        "0530c78699ea5e8eb5530600900e1f328b48acad",
    ),
}


def variants(task: str, row: dict[str, Any]) -> list[str]:
    if task == "lambada_openai":
        return [row["text"]]
    if task == "arc_easy":
        return [
            row["question"],
            *[row["question"] + " " + text for text in row["choices"]["text"]],
        ]
    if task == "hellaswag":
        return [row["ctx"], *[row["ctx"] + " " + ending for ending in row["endings"]]]
    if task == "winogrande":
        return [
            row["sentence"],
            row["sentence"].replace("_", row["option1"]),
            row["sentence"].replace("_", row["option2"]),
        ]
    if task == "gsm8k":
        return [row["question"], row["question"] + " " + row["answer"]]
    if task == "math":
        return [row["problem"], row["problem"] + " " + row["solution"]]
    if task == "piqa":
        return [
            row["goal"],
            row["goal"] + " " + row["sol1"],
            row["goal"] + " " + row["sol2"],
        ]
    raise ValueError("Unknown registered benchmark")


def register(destination: Path, coverage_path: Path) -> dict[str, Any]:
    if destination.exists() or coverage_path.exists():
        raise FileExistsError("Benchmark preparation records are immutable")
    destination.mkdir(parents=True)
    references = destination / "references.jsonl"
    coverage = {}
    api = HfApi()
    with references.open("x") as output:
        for task, (repository, prefix, revision) in SOURCES.items():
            info = api.dataset_info(repository, revision=revision, files_metadata=True)
            if info.sha != revision:
                raise ValueError("Benchmark revision differs from pinned registry")
            selected = [
                entry
                for entry in info.siblings
                if entry.rfilename.startswith(prefix)
                and entry.rfilename.endswith(".parquet")
            ]
            rows = fingerprints = short_rows = 0
            artifacts = []
            for shard in sorted(selected, key=lambda entry: entry.rfilename):
                path = Path(
                    hf_hub_download(
                        repository,
                        shard.rfilename,
                        repo_type="dataset",
                        revision=revision,
                        local_dir=destination / task,
                    )
                )
                digest = sha256_file(path)
                if shard.lfs and digest != shard.lfs.sha256:
                    raise ValueError("Benchmark payload differs from source checksum")
                artifacts.append(dict(path=shard.rfilename, sha256=digest))
                for batch in pq.ParquetFile(path).iter_batches(batch_size=512):
                    for row in batch.to_pylist():
                        accepted = 0
                        for number, text in enumerate(variants(task, row)):
                            text = canonicalize(text)
                            if len(re.findall(r"\w+", text)) < 3:
                                continue
                            identity = dict(
                                task=task,
                                revision=revision,
                                shard=shard.rfilename,
                                row=rows,
                                variant=number,
                            )
                            output.write(
                                json.dumps(
                                    dict(
                                        doc_id=object_sha256(identity),
                                        text=text,
                                        task=task,
                                    ),
                                    sort_keys=True,
                                )
                                + "\n"
                            )
                            fingerprints += 1
                            accepted += 1
                        rows += 1
                        short_rows += accepted == 0
            coverage[task] = dict(
                repository=repository,
                revision=revision,
                artifacts=artifacts,
                rows=rows,
                fingerprints=fingerprints,
                uncovered_short_rows=short_rows,
                complete=bool(rows) and short_rows == 0,
                labels_forwarded_to_training=False,
                model_scores_computed=False,
            )
            print(
                json.dumps(
                    {
                        "task": task,
                        "rows": rows,
                        "fingerprints": fingerprints,
                        "complete": coverage[task]["complete"],
                    }
                ),
                flush=True,
            )
        # The pinned author Hub loader names the AI2 archive and author test URL.
        # Freeze their actual bytes, rather than inventing a GitHub source path.
        revision = "2e8ac2dffd59bac8c3c6714948f4c551a0848bb0"
        archive_url = "https://storage.googleapis.com/ai2-mosaic/public/physicaliqa/physicaliqa-train-dev.zip"
        test_url = "https://yonatanbisk.com/piqa/data/tests.jsonl"
        zipped = urllib.request.urlopen(archive_url, timeout=30).read()
        zipped_path = destination / "piqa-train-dev.zip"
        zipped_path.write_bytes(zipped)
        with zipfile.ZipFile(io.BytesIO(zipped)) as archive:
            splits = [
                (name, archive.read(name), archive_url)
                for name in sorted(archive.namelist())
                if name.endswith(".jsonl")
            ]
        splits.append(
            ("tests", urllib.request.urlopen(test_url, timeout=30).read(), test_url)
        )
        artifacts = []
        artifacts.append(dict(url=archive_url, sha256=sha256_file(zipped_path)))
        rows = fingerprints = short_rows = 0
        for split, payload, url in splits:
            path = destination / f"piqa-{Path(split).stem}.jsonl"
            path.write_bytes(payload)
            artifacts.append(dict(url=url, sha256=sha256_file(path)))
            for line in payload.decode().splitlines():
                row = json.loads(line)
                accepted = 0
                for number, text in enumerate(variants("piqa", row)):
                    text = canonicalize(text)
                    if len(re.findall(r"\w+", text)) < 3:
                        continue
                    identity = dict(
                        task="piqa",
                        revision=revision,
                        split=split,
                        row=rows,
                        variant=number,
                    )
                    output.write(
                        json.dumps(
                            dict(
                                doc_id=object_sha256(identity), text=text, task="piqa"
                            ),
                            sort_keys=True,
                        )
                        + "\n"
                    )
                    fingerprints += 1
                    accepted += 1
                rows += 1
                short_rows += accepted == 0
        coverage["piqa"] = dict(
            repository="ybisk/piqa",
            revision=revision,
            artifacts=artifacts,
            rows=rows,
            fingerprints=fingerprints,
            uncovered_short_rows=short_rows,
            complete=bool(rows) and short_rows == 0,
            labels_forwarded_to_training=False,
            model_scores_computed=False,
        )
    result = dict(
        schema_version=1,
        coverage=coverage,
        references=dict(path=str(references), sha256=sha256_file(references)),
        math_source=("pinned MATH mirror; original byte-equivalence unverified"),
        no_model_evaluation_performed=True,
    )
    atomic_json(coverage_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--coverage", type=Path, required=True)
    args = parser.parse_args()
    value = register(args.output, args.coverage)
    print(
        json.dumps(
            {
                "coverage": value["coverage"],
                "references_sha256": value["references"]["sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
