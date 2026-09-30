"""Fail closed on manifest overlaps before constructing training iterators.

This audits declared identities. Near-duplicate clustering, benchmark exclusion,
and correct source-group extraction must be performed upstream (Phase 03).
"""

import argparse
import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path

ROLES = frozenset({"train", "guidance", "development", "validation", "test"})
TRAINING_ROLES = frozenset({"train", "guidance"})


def audit(records: Iterable[Mapping[str, str]]) -> dict[str, int]:
    """Require unique document IDs and no cross-role content/cluster/source group."""
    identities: dict[str, dict[str, str]] = {
        key: {} for key in ("doc_id", "content_sha256", "cluster_id", "source_group")
    }
    counts: Counter = Counter()
    for record in records:
        role = record.get("split")
        if role not in ROLES:
            raise ValueError(f"Unknown split role: {role!r}")
        for field, seen in identities.items():
            value = record.get(field)
            if not isinstance(value, str) or not value:
                raise ValueError(f"Missing nonempty identity: {field}")
            if field == "content_sha256" and not re.fullmatch(r"[0-9a-f]{64}", value):
                raise ValueError("content_sha256 must be a lowercase SHA-256 digest")
            if value in seen and (seen[value] != role or field == "doc_id"):
                raise ValueError(
                    f"Split overlap or duplicate {field}: {seen[value]} / {role}"
                )
            seen[value] = role
        counts[role] += 1
    if not counts:
        raise ValueError("Empty manifest")
    return dict(sorted(counts.items()))


def training_paths(paths: Mapping[str, Path]) -> dict[str, Path]:
    """The optimizer's data reader may only receive train and guidance paths."""
    if set(paths) != TRAINING_ROLES:
        raise ValueError("Training inputs must contain exactly train and guidance")
    return {role: Path(path) for role, path in paths.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="Document-level JSONL identities")
    args = parser.parse_args()
    with args.manifest.open() as handle:
        records = (json.loads(line) for line in handle if line.strip())
        print(json.dumps(audit(records), indent=2))


if __name__ == "__main__":
    main()
