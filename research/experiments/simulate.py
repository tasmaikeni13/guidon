"""Execute the registered cheap discovery/confirmation synthetic conditions."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np

from research.experiments.synthetic import Condition, simulate

ROOT = Path(__file__).resolve().parents[2]


def draw_seeds(root: int, condition_id: str, draws: int) -> np.ndarray:
    prefix = int.from_bytes(
        hashlib.sha256(condition_id.encode()).digest()[:8], "little"
    )
    return np.random.SeedSequence([root, prefix]).generate_state(draws, dtype=np.uint64)


def run(
    stream: str, protocol_path: Path, raw_path: Path, provenance_path: Path
) -> None:
    protocol_path = protocol_path.resolve()
    raw_path = raw_path.resolve()
    provenance_path = provenance_path.resolve()
    if raw_path.exists() or provenance_path.exists():
        raise FileExistsError(
            "Raw evidence and provenance are immutable; use a new run ID"
        )
    protocol = json.loads(protocol_path.read_text())
    root = protocol[f"{stream}_seed"]
    cases = protocol["conditions"]
    order = np.random.default_rng(root).permutation(len(cases))
    count = protocol["draws_per_primary_condition_per_stream"]
    started = time.monotonic()
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with (
        raw_path.open("wb") as handle,
        gzip.GzipFile(fileobj=handle, mode="wb", mtime=0, filename="") as raw,
    ):
        for position, index in enumerate(order):
            condition = Condition(**cases[index])
            rows, _ = simulate(condition, draw_seeds(root, condition.id, count))
            for row in rows:
                row["stream"] = stream
                raw.write(
                    (json.dumps(row, sort_keys=True, allow_nan=False) + "\n").encode()
                )
            raw.flush()
            print(
                f"{stream} {position + 1}/{len(cases)} {condition.id}: {count} draws",
                flush=True,
            )
    source_paths = [
        Path(__file__),
        ROOT / "research/experiments/synthetic.py",
        ROOT / "research/experiments/register_simulation.py",
        ROOT / "src/guidon/reference.py",
    ]
    provenance = {
        "code_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "protocol_version": protocol["version"],
        "stream": stream,
        "root_seed": root,
        "rows": len(cases) * count,
        "condition_order": [cases[i]["id"] for i in order],
        "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        "analysis_spec_sha256": hashlib.sha256(
            (ROOT / "research/phase02/analysis_spec.md").read_bytes()
        ).hexdigest(),
        "raw_path": str(raw_path.relative_to(ROOT)),
        "raw_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        "source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in source_paths
        },
        "elapsed_seconds": time.monotonic() - started,
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
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    print(f"Complete: {provenance['rows']} rows, {provenance['elapsed_seconds']:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stream", choices=["discovery", "confirmation"], required=True
    )
    parser.add_argument(
        "--protocol", type=Path, default=ROOT / "research/phase02/protocol.json"
    )
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--provenance", type=Path)
    args = parser.parse_args()
    run(
        args.stream,
        args.protocol,
        args.raw or ROOT / f"artifacts/phase02/{args.stream}-v1.jsonl.gz",
        args.provenance or ROOT / f"research/phase02/{args.stream}-v1-provenance.json",
    )
