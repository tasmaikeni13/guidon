"""Read-only verification of phase evidence and its transitive dependency hashes."""

from __future__ import annotations

import json
from importlib import metadata
from pathlib import Path

from guidon.artifacts import sha256_file

ROOT = Path(__file__).resolve().parents[2]


def verify(identities: list[str], root: Path = ROOT) -> dict[str, str]:
    verified: dict[str, str] = {}
    active: set[str] = set()

    def visit(identity: str) -> None:
        if identity in verified:
            return
        if identity in active:
            raise ValueError("Cyclic phase dependency")
        active.add(identity)
        path = root / "research" / "gates" / f"{identity}.json"
        gate = json.loads(path.read_text())
        if gate["status"] != "passed" or gate["theory_version"] != "0.2":
            raise ValueError(f"Phase {identity} has no valid current gate")
        for package, expected in gate.get("runtime_package_versions", {}).items():
            if metadata.version(package) != expected:
                raise ValueError(f"Phase {identity} runtime pin differs: {package}")
        for name, expected in gate["source_and_evidence_sha256"].items():
            if sha256_file(root / name) != expected:
                raise ValueError(f"Stale phase {identity} source/evidence: {name}")
        for record in gate["raw_artifacts"]:
            if sha256_file(root / record["path"]) != record["sha256"]:
                raise ValueError(
                    f"Stale phase {identity} raw artifact: {record['path']}"
                )
        for dependency in gate["depends_on"]:
            dependency_path = root / dependency["path"]
            if sha256_file(dependency_path) != dependency["sha256"]:
                raise ValueError(f"Stale phase {identity} prerequisite")
            visit(dependency_path.stem)
        verified[identity] = sha256_file(path)
        active.remove(identity)

    for identity in identities:
        visit(identity)
    state = json.loads((root / "phases/state.json").read_text())
    for phase in state["phases"]:
        if phase["id"] in verified:
            if (
                phase["status"] != "passed"
                or phase["evidence"]["path"] != f"research/gates/{phase['id']}.json"
                or phase["evidence"]["sha256"] != verified[phase["id"]]
            ):
                raise ValueError(
                    "Phase state does not match its directly verified gate"
                )
    return verified
