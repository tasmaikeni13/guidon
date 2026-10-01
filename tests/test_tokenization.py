from tokenizers import Tokenizer, models, pre_tokenizers

from guidon.artifacts import object_sha256
from guidon.tokenization import eligible_tapes, parallel_tokenize


def test_parallel_encoding_preserves_order_priority_partial_budget_and_partition():
    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, "one": 1, "two": 2}))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    settings = {
        "pilot": dict(role="train", dataset="fixture", tokenizer="word"),
        "confirmation": dict(role="train", dataset="fixture", tokenizer="word"),
        "guide": dict(role="guidance", dataset="fixture", tokenizer="word"),
    }
    rows = [
        dict(text="one two one", split="train", dataset="fixture", cluster_id="a"),
        dict(text="two one", split="train", dataset="fixture", cluster_id="b"),
        dict(text="one", split="guidance", dataset="fixture", cluster_id="c"),
        dict(text="two two", split="test", dataset="fixture", cluster_id="d"),
    ]
    remaining = dict(pilot=2, confirmation=3, guide=2)
    allocations = []
    for row, encoded in parallel_tokenize(
        rows, {"word": tokenizer}, settings, remaining, batch_size=4
    ):
        eligible = eligible_tapes(row, settings, remaining)
        if not eligible:
            assert row["split"] == "test"
            assert not encoded
            continue
        name = eligible["word"][0]
        ids = encoded["word"]
        assert ids == tokenizer.encode(row["text"], add_special_tokens=False).ids
        take = min(len(ids) + 1, remaining[name])
        allocations.append((row["cluster_id"], name, take))
        remaining[name] -= take
    assert allocations == [
        ("a", "pilot", 2),
        ("b", "confirmation", 3),
        ("c", "guide", 2),
    ]
    assert remaining == dict(pilot=0, confirmation=0, guide=0)
    settings["confirmation"]["source_partition"] = dict(modulus=2, remainders=[0])
    partition = int(object_sha256({"cluster": "b", "seed": 20261003})[:16], 16)
    expected = {"word": ["confirmation"]} if partition % 2 == 0 else {}
    assert (
        eligible_tapes(rows[1], settings, dict(pilot=0, confirmation=1, guide=0))
        == expected
    )
