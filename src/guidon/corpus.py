"""Canonicalization, pinned publisher groups, and audited duplicate components.

The SQLite store is an intermediate artifact, never a trainer input. Its texts
must stay in the preparation process. MinHash retrieves candidate pairs; exact
word-shingle Jaccard, rather than the estimate, decides a near-duplicate edge.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import re
import sqlite3
import unicodedata
import zlib
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import numpy as np
from datasketch import MinHash, MinHashLSH
from tldextract import TLDExtract

from guidon.artifacts import object_sha256, sha256_file

RESOURCES = Path(__file__).parent / "resources"


@lru_cache(maxsize=8)
def minhash_permutations(number: int, seed: int) -> np.ndarray:
    return MinHash(num_perm=number, seed=seed).permutations


def canonicalize(text: str) -> str:
    if not isinstance(text, str):
        raise ValueError("Document text must be a string")
    text = unicodedata.normalize("NFKC", text).replace("\x00", "")
    return " ".join(text.split())


def shingles(text: str, width: int = 5) -> set[bytes]:
    words = re.findall(r"\w+", text.casefold(), flags=re.UNICODE)
    if not words:
        return set()
    if len(words) < width:
        return {" ".join(words).encode()}
    return {
        " ".join(words[i : i + width]).encode() for i in range(len(words) - width + 1)
    }


def jaccard(left: set[bytes], right: set[bytes]) -> float:
    union = len(left | right)
    return len(left & right) / union if union else 0.0


class SourceGroups:
    """Use ICANN and PRIVATE PSL rules, with no network or fallback snapshot."""

    def __init__(self, suffix_path: Path | None = None):
        self.path = suffix_path or RESOURCES / "public_suffix_list.dat"
        provenance = json.loads(
            (RESOURCES / "public_suffix_provenance.json").read_text()
        )
        if suffix_path is None and sha256_file(self.path) != provenance["sha256"]:
            raise ValueError("Pinned public-suffix list was modified")
        self.extract = TLDExtract(
            suffix_list_urls=(self.path.resolve().as_uri(),),
            cache_dir=None,
            fallback_to_snapshot=False,
            include_psl_private_domains=True,
        )

    def __call__(self, url: str) -> str:
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("Missing trustworthy HTTP source URL")
        host = parsed.hostname.rstrip(".").encode("idna").decode().lower()
        try:
            ipaddress.ip_address(host)
        except ValueError:
            pass
        else:
            raise ValueError("IP-address publishers require an explicit source audit")
        result = self.extract(host)
        group = result.top_domain_under_public_suffix
        if not group or not result.suffix:
            raise ValueError("Publisher has no registrable domain in the pinned PSL")
        return group


@dataclass(frozen=True)
class DedupConfig:
    shingle_words: int = 5
    permutations: int = 128
    bands: int = 24
    rows_per_band: int = 5
    seed: int = 20260930
    jaccard_threshold: float = 0.8
    benchmark_containment: float = 0.8
    benchmark_minimum_shingles: int = 3

    def __post_init__(self) -> None:
        if self.bands * self.rows_per_band > self.permutations:
            raise ValueError("More LSH rows than MinHash permutations")
        if min(self.shingle_words, self.bands, self.rows_per_band) < 1:
            raise ValueError("Invalid duplicate signature dimensions")
        if not 0 < self.jaccard_threshold <= 1:
            raise ValueError("Invalid near-duplicate threshold")

    def signature(self, values: set[bytes]) -> MinHash:
        signature = MinHash(
            num_perm=self.permutations,
            seed=self.seed,
            permutations=minhash_permutations(self.permutations, self.seed),
        )
        if values:
            signature.update_batch(sorted(values))
        return signature

    def theoretical_miss_probability(self) -> float:
        return (1 - self.jaccard_threshold**self.rows_per_band) ** self.bands


class Components:
    def __init__(self) -> None:
        self.parent: list[int] = []
        self.minimum: list[str] = []

    def add(self, identifier: str) -> int:
        index = len(self.parent)
        self.parent.append(index)
        self.minimum.append(identifier)
        return index

    def root(self, index: int) -> int:
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def join(self, left: int, right: int) -> None:
        a, b = self.root(left), self.root(right)
        if a == b:
            return
        if self.minimum[a] > self.minimum[b]:
            a, b = b, a
        self.parent[b] = a


class BenchmarkIndex:
    """Detect answer-bearing excerpts as well as full-document near copies.

    Short answers alone are deliberately insufficient fingerprints. References are
    question/context with every supplied answer/choice/solution; they stay in the
    preparation process. Missing reference coverage prevents a production signoff.
    """

    def __init__(self, records: Iterable[Mapping[str, str]], config: DedupConfig):
        self.config = config
        self.references: dict[str, set[bytes]] = {}
        self.inverted: dict[bytes, list[str]] = {}
        self.short_references: dict[str, set[bytes]] = {}
        self.short_inverted: dict[bytes, list[str]] = {}
        for row in records:
            identifier = row["doc_id"]
            if identifier in self.references:
                raise ValueError("Repeated benchmark reference ID")
            values = shingles(canonicalize(row["text"]), config.shingle_words)
            if len(values) < config.benchmark_minimum_shingles:
                short = shingles(canonicalize(row["text"]), 3)
                if len(re.findall(r"\w+", row["text"])) < 3:
                    raise ValueError("Benchmark fingerprint is too short to audit")
                self.short_references[identifier] = short
                for value in short:
                    self.short_inverted.setdefault(value, []).append(identifier)
                continue
            self.references[identifier] = values
            for value in values:
                self.inverted.setdefault(value, []).append(identifier)

    def matches(self, values: set[bytes], text: str = "") -> list[str]:
        intersections: Counter[str] = Counter()
        for value in values:
            for identifier in self.inverted.get(value, ()):
                intersections[identifier] += 1
        found = [
            identifier
            for identifier, count in intersections.items()
            if count >= self.config.benchmark_minimum_shingles
            and count / len(self.references[identifier])
            >= self.config.benchmark_containment
        ]
        if self.short_references:
            intersections.clear()
            for value in shingles(text, 3):
                for identifier in self.short_inverted.get(value, ()):
                    intersections[identifier] += 1
            found.extend(
                identifier
                for identifier, count in intersections.items()
                if count / len(self.short_references[identifier])
                >= self.config.benchmark_containment
            )
        return sorted(found)


@dataclass(frozen=True)
class PreparedDocument:
    """Internal worker result; canonicalization/exclusion run exactly once."""

    text: str
    compressed_text: bytes
    signature: np.ndarray
    benchmark_excluded: bool
    lexical_content: bool


class CorpusBuilder:
    """Build joint source/duplicate components before emitting any role tape."""

    def __init__(
        self,
        path: Path,
        benchmarks: Iterable[Mapping[str, str]],
        config: DedupConfig | None = None,
    ):
        if path.exists():
            raise FileExistsError("Corpus preparation runs are immutable")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute(
            "CREATE TABLE documents (id INTEGER PRIMARY KEY, doc_id TEXT UNIQUE, "
            "content_sha256 TEXT, source_group TEXT, dataset TEXT, text BLOB, "
            "signature BLOB, role_constraint TEXT, excluded INTEGER, duplicate INTEGER)"
        )
        self.db.execute("CREATE INDEX content_index ON documents(content_sha256)")
        config = config or DedupConfig()
        self.config = config
        self.sources = SourceGroups()
        self.benchmarks = BenchmarkIndex(benchmarks, config)
        self.lsh = MinHashLSH(
            num_perm=config.permutations,
            params=(config.bands, config.rows_per_band),
        )
        self.components = Components()
        self.source_first: dict[str, int] = {}
        self.counts: Counter[str] = Counter()
        self.quarantine: list[dict[str, str]] = []

    def add(
        self,
        record: Mapping[str, Any],
        prepared_signature: Any = None,
        prepared: PreparedDocument | None = None,
    ) -> None:
        text = canonicalize(record["text"]) if prepared is None else prepared.text
        identifier = str(record["doc_id"])
        if not text or not identifier:
            raise ValueError("Empty document text or ID")
        self.counts["candidates"] += 1
        values = shingles(text, self.config.shingle_words) if prepared is None else None
        if not values if prepared is None else not prepared.lexical_content:
            self.quarantine.append(
                {"doc_id": identifier, "reason": "No lexical shingles"}
            )
            self.counts["invalid_lexical_content"] += 1
            return
        try:
            source = self.sources(str(record.get("url", "")))
        except (ValueError, UnicodeError) as exc:
            self.quarantine.append({"doc_id": identifier, "reason": str(exc)})
            self.counts["invalid_source"] += 1
            return
        digest = hashlib.sha256(text.encode()).hexdigest()
        index = self.components.add(identifier)
        # Joining the known publisher first lets us omit only redundant edges
        # after one near duplicate has already established removal. Cross-source
        # candidate edges still receive the exact same Jaccard check.
        if source in self.source_first:
            self.components.join(index, self.source_first[source])
        else:
            self.source_first[source] = index
        exact = self.db.execute(
            "SELECT id FROM documents WHERE content_sha256=? LIMIT 1", (digest,)
        ).fetchone()
        if prepared is not None:
            prepared_signature = prepared.signature
        if prepared_signature is None:
            signature = self.config.signature(values)
        else:
            values_array = np.asarray(prepared_signature, dtype=np.uint64)
            if values_array.shape != (self.config.permutations,):
                raise ValueError("Prepared MinHash signature has an invalid shape")
            signature = MinHash(
                num_perm=self.config.permutations,
                seed=self.config.seed,
                hashvalues=values_array,
                permutations=minhash_permutations(
                    self.config.permutations, self.config.seed
                ),
            )
        excluded = (
            bool(self.benchmarks.matches(values, text))
            if prepared is None
            else prepared.benchmark_excluded
        )
        duplicate = exact is not None
        if exact:
            self.components.join(index, exact[0])
            self.counts["exact_duplicates"] += 1
        else:
            for other in sorted(self.lsh.query(signature)):
                if duplicate and self.components.root(index) == self.components.root(
                    other
                ):
                    self.counts["redundant_component_candidates_skipped"] += 1
                    continue
                if values is None:
                    values = shingles(text, self.config.shingle_words)
                payload = self.db.execute(
                    "SELECT text FROM documents WHERE id=?", (other,)
                ).fetchone()[0]
                other_values = shingles(
                    zlib.decompress(payload).decode(), self.config.shingle_words
                )
                if jaccard(values, other_values) >= self.config.jaccard_threshold:
                    self.components.join(index, other)
                    self.counts["near_duplicate_edges"] += 1
                    duplicate = True
            # Keep every non-exact signature: A~B and B~C need not imply A~C.
            self.lsh.insert(index, signature)
        constraint = record.get("split", "")
        if constraint not in {
            "",
            "train",
            "guidance",
            "development",
            "validation",
            "test",
        }:
            raise ValueError("Invalid declared source-role constraint")
        self.db.execute(
            "INSERT INTO documents VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                index,
                identifier,
                digest,
                source,
                str(record["dataset"]),
                zlib.compress(text.encode())
                if prepared is None
                else prepared.compressed_text,
                signature.hashvalues.astype("<u8").tobytes(),
                constraint,
                int(excluded),
                int(duplicate),
            ),
        )
        self.counts["benchmark_documents"] += excluded
        if index % 1000 == 0:
            self.db.commit()

    def finish(self, split_seed: int, train_fraction: float = 0.8) -> dict[str, Any]:
        """Reject conflicting components; hold out whole connected source groups."""
        if not 0 < train_fraction < 1:
            raise ValueError("Invalid training fraction")
        self.db.commit()
        constraints: dict[int, set[str]] = {}
        excluded: set[int] = set()
        for index, role, blocked in self.db.execute(
            "SELECT id,role_constraint,excluded FROM documents"
        ):
            root = self.components.root(index)
            if role:
                constraints.setdefault(root, set()).add(role)
            if blocked:
                excluded.add(root)
        conflicts = {r for r, roles in constraints.items() if len(roles) > 1}
        roles = ("guidance", "development", "validation", "test")
        self.db.execute("ALTER TABLE documents ADD COLUMN cluster_id TEXT")
        self.db.execute("ALTER TABLE documents ADD COLUMN split TEXT")
        for (index,) in self.db.execute("SELECT id FROM documents"):
            root = self.components.root(index)
            anchor = self.components.minimum[root]
            cluster = object_sha256({"component_anchor": anchor})
            if root in excluded or root in conflicts:
                role = "quarantine"
                self.counts["quarantined_component_documents"] += 1
            elif root in constraints:
                role = next(iter(constraints[root]))
            else:
                value = (
                    int(
                        object_sha256({"seed": split_seed, "component": cluster})[:16],
                        16,
                    )
                    / 2**64
                )
                role = (
                    "train"
                    if value < train_fraction
                    else roles[
                        min(
                            3,
                            int((value - train_fraction) / ((1 - train_fraction) / 4)),
                        )
                    ]
                )
            self.db.execute(
                "UPDATE documents SET cluster_id=?,split=? WHERE id=?",
                (cluster, role, index),
            )
        self.db.execute("CREATE INDEX split_index ON documents(split,dataset,doc_id)")
        self.db.commit()
        return {
            "counts": dict(self.counts),
            "invalid_source_records": self.quarantine,
            "conflicting_components": len(conflicts),
            "benchmark_excluded_components": len(excluded),
            "config": asdict(self.config),
            "theoretical_lsh_miss_probability_at_threshold": (
                self.config.theoretical_miss_probability()
            ),
            "signature_sha256": object_sha256(asdict(self.config)),
        }

    def documents(self) -> Iterable[dict[str, str]]:
        for row in self.db.execute(
            "SELECT doc_id,content_sha256,cluster_id,source_group,split,dataset,text "
            "FROM documents WHERE split!='quarantine' AND duplicate=0 ORDER BY doc_id"
        ):
            yield dict(
                zip(
                    (
                        "doc_id",
                        "content_sha256",
                        "cluster_id",
                        "source_group",
                        "split",
                        "dataset",
                        "text",
                    ),
                    (*row[:-1], zlib.decompress(row[-1]).decode()),
                    strict=True,
                )
            )

    def close(self) -> None:
        self.db.close()


def false_negative_audit(texts: list[str], config: DedupConfig) -> dict[str, Any]:
    """Exhaustively verify sample pairs against exact shingles, outside the LSH."""
    values = [shingles(canonicalize(t), config.shingle_words) for t in texts]
    signatures = [config.signature(s) for s in values]
    lsh = MinHashLSH(
        num_perm=config.permutations, params=(config.bands, config.rows_per_band)
    )
    for index, signature in enumerate(signatures):
        if values[index]:
            lsh.insert(index, signature)
    missed, positives = [], 0
    for i in range(len(texts)):
        found = set(lsh.query(signatures[i]))
        for j in range(i):
            if jaccard(values[i], values[j]) >= config.jaccard_threshold:
                positives += 1
                if j not in found:
                    missed.append([j, i])
    return {
        "sample_documents": len(texts),
        "exact_positive_pairs": positives,
        "missed_pairs": missed,
        "observed_miss_fraction": len(missed) / positives if positives else None,
        "scope": "exhaustive sample pair check; not a full-corpus absence proof",
    }
