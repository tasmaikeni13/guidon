import hashlib

import pytest

from guidon.corpus import (
    CorpusBuilder,
    DedupConfig,
    SourceGroups,
    canonicalize,
    false_negative_audit,
)
from guidon.data_boundary import audit


def record(identifier, text, url, split="", dataset="fineweb_edu"):
    return dict(doc_id=identifier, text=text, url=url, split=split, dataset=dataset)


def test_source_group_psl_private_idna_and_unknown():
    groups = SourceGroups()
    assert groups("https://a.school.co.uk/path") == "school.co.uk"
    assert groups("https://b.school.co.uk/other") == "school.co.uk"
    assert groups("https://tenant.github.io/path") == "tenant.github.io"
    assert groups("https://other.github.io/path") == "other.github.io"
    for url in ("not a url", "file:///local", "https://127.0.0.1", "https://a.invalid"):
        with pytest.raises(ValueError):
            groups(url)


def test_transitive_cross_dataset_components_and_conflicts(tmp_path):
    builder = CorpusBuilder(tmp_path / "corpus.sqlite", [])
    text = " ".join(f"word{i}" for i in range(120))
    builder.add(record("a", text, "https://a.example.com", "train"))
    builder.add(record("b", text + " extra", "https://other.org", "", "finemath"))
    builder.add(record("c", "different document", "https://sub.other.org", "test"))
    result = builder.finish(100)
    assert result["conflicting_components"] == 1
    assert result["counts"]["near_duplicate_edges"] >= 1
    assert list(builder.documents()) == []
    builder.close()


def test_exact_canonical_dedup_before_role_packing(tmp_path):
    builder = CorpusBuilder(tmp_path / "corpus.sqlite", [])
    builder.add(record("a", "alpha   beta\n gamma", "https://example.com", "train"))
    builder.add(record("b", "alpha beta gamma", "https://example.org"))
    result = builder.finish(100)
    documents = list(builder.documents())
    assert len(documents) == 1
    assert result["counts"]["exact_duplicates"] == 1
    assert (
        documents[0]["content_sha256"]
        == hashlib.sha256(canonicalize("alpha beta gamma").encode()).hexdigest()
    )
    assert audit(documents) == {"train": 1}
    builder.close()


def test_answer_bearing_copy_inside_long_document_quarantines_source(tmp_path):
    reference = (
        "Mira bought twelve books then gave four books to her sister eight remain"
    )
    builder = CorpusBuilder(
        tmp_path / "corpus.sqlite", [{"doc_id": "benchmark-1", "text": reference}]
    )
    builder.add(
        record("a", "unrelated prefix " * 50 + reference, "https://example.com")
    )
    builder.add(record("b", "other page", "https://a.example.com"))
    result = builder.finish(100)
    assert result["counts"]["benchmark_documents"] == 1
    assert result["benchmark_excluded_components"] == 1
    assert list(builder.documents()) == []
    builder.close()


def test_independent_pair_audit_and_lsh_probability():
    config = DedupConfig()
    text = " ".join(f"word{i}" for i in range(200))
    result = false_negative_audit([text, text + " extra", "different document"], config)
    assert result["exact_positive_pairs"] == 1
    assert not result["missed_pairs"]
    assert config.theoretical_miss_probability() < 0.001


def test_invalid_source_quarantine_is_recorded(tmp_path):
    builder = CorpusBuilder(tmp_path / "corpus.sqlite", [])
    builder.add(record("a", "some text", "missing"))
    result = builder.finish(100)
    assert result["counts"]["invalid_source"] == 1
    assert result["invalid_source_records"][0]["doc_id"] == "a"
    builder.close()
