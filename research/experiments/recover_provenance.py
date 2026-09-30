"""Recover one missing recording-fault manifest without rerunning any trajectory."""

from __future__ import annotations

import gzip
import hashlib
import json
import platform
import subprocess

import numpy as np

from research.experiments.simulate import ROOT, draw_seeds


def recover() -> None:
    protocol_path = ROOT / "research/phase02/drift-followup-protocol.json"
    raw_path = ROOT / "artifacts/phase02/drift-discovery-v1.jsonl.gz"
    destination = ROOT / "research/phase02/drift-discovery-v1-provenance.json"
    if destination.exists():
        raise FileExistsError("Manifest already exists")
    protocol = json.loads(protocol_path.read_text())
    root = protocol["discovery_seed"]
    expected = {
        (c["id"], int(seed))
        for c in protocol["conditions"]
        for seed in draw_seeds(root, c["id"], 1000)
    }
    found = set()
    with gzip.open(raw_path, "rt") as handle:
        for line in handle:
            row = json.loads(line)
            identity = (row["condition"], row["seed"])
            assert identity not in found
            found.add(identity)
    assert found == expected
    names = [
        "research/experiments/simulate.py",
        "research/experiments/synthetic.py",
        "research/experiments/register_simulation.py",
        "src/guidon/reference.py",
    ]
    snapshot = {
        name: subprocess.check_output(["git", "show", f"0902877:{name}"], cwd=ROOT)
        for name in names
    }
    provenance = {
        "protocol_version": protocol["version"],
        "stream": "discovery",
        "root_seed": root,
        "code_commit": "0902877a8c79b1108c9bfbb8bcfd09c96e0d2233",
        "rows": len(found),
        "condition_order": [
            protocol["conditions"][i]["id"]
            for i in np.random.default_rng(root).permutation(
                len(protocol["conditions"])
            )
        ],
        "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        "analysis_spec_sha256": hashlib.sha256(
            (ROOT / "research/phase02/analysis_spec.md").read_bytes()
        ).hexdigest(),
        "raw_path": str(raw_path.relative_to(ROOT)),
        "raw_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        "source_sha256": {
            name: hashlib.sha256(data).hexdigest() for name, data in snapshot.items()
        },
        "elapsed_seconds": None,
        "elapsed_reason": (
            "Manifest write failed after raw completion; exact original "
            "duration unavailable, not invented"
        ),
        "recording_fault": (
            "relative output path, raw complete; provenance recovered after "
            "exact identity/count check; must independently verify before use"
        ),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "machine": platform.machine(),
            "backend": "NumPy float64 CPU",
        },
        "exclusions": [],
        "restarts": [],
        "llm_training_executed": False,
    }
    destination.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    print(
        f"Recovered immutable provenance for {len(found)} existing "
        f"trajectories; no restart"
    )


if __name__ == "__main__":
    recover()
