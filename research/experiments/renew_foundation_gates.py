"""Renew unchanged mathematical/statistical evidence after audited scope migration.

Historical gates are archived intact. No raw run, decision, threshold or protected
method source is replaced. Runtime pins replace broad dependency-lock hashing;
phase-local claims/implementations retain hashes. Global progress documentation
belongs to its downstream gate and cannot retroactively change a theorem result.
"""

from __future__ import annotations

import argparse
import ast
import importlib.metadata
import json
import re
from pathlib import Path

from guidon.artifacts import atomic_json, sha256_file

ROOT = Path(__file__).resolve().parents[2]
ADMINISTRATIVE = {
    "AGENTS.md",
    "README.md",
    "pyproject.toml",
    "uv.lock",
    "phases/README.md",
    "research/state.md",
    "research/completion_audit.md",
    "research/verification.md",
}
VERIFIER = "research/experiments/verify_gates.py"


def function_ast(path: Path, name: str) -> str:
    node = next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == name
    )
    return ast.dump(node, include_attributes=False)


def run(original_root: Path, *, apply: bool) -> dict:
    historical_log = original_root / "verification-v2.log"
    if not historical_log.read_text().startswith("Phases 01/02 passed:"):
        raise ValueError("Original signed snapshot must pass its unchanged verifier")
    original = {
        n: json.loads((original_root / f"research/gates/{n}.json").read_text())
        for n in ("01", "02")
    }
    if function_ast(ROOT / VERIFIER, "independently_check_summaries") != function_ast(
        original_root / VERIFIER, "independently_check_summaries"
    ):
        raise ValueError("Independent statistical summary checker changed")
    theorem_names = {
        name
        for p in (ROOT / "proofs/Guidon").glob("*.lean")
        for name in re.findall(r"^theorem (\w+)", p.read_text(), re.M)
    }
    lean_log = ROOT / "artifacts/phase04/lean-audit-v1.log"
    assert len(theorem_names) == 32 and "sorryAx" not in lean_log.read_text()
    for match in re.finditer(r"depends on axioms: \[(.*?)\]", lean_log.read_text()):
        assert set(match[1].split(", ")) <= {
            "propext",
            "Classical.choice",
            "Quot.sound",
        }
    changes, protected = {}, {}
    for gate in original.values():
        for name, digest in gate["source_and_evidence_sha256"].items():
            current = sha256_file(ROOT / name)
            if name in ADMINISTRATIVE or name == VERIFIER:
                changes[name] = {
                    "historical_sha256": digest,
                    "current_sha256": current,
                    "scope": (
                        "global progress/dependency declarations"
                        if name in ADMINISTRATIVE
                        else (
                            "phase status assertions generalized; "
                            "independent summary checker unchanged"
                        )
                    ),
                }
            else:
                if current != digest:
                    raise ValueError(f"Protected scientific source changed: {name}")
                protected[name] = digest
        for raw in gate["raw_artifacts"]:
            if sha256_file(ROOT / raw["path"]) != raw["sha256"]:
                raise ValueError("Original raw archive is missing or changed")
    pins = {
        p: importlib.metadata.version(p) for p in ("jax", "jaxlib", "numpy", "scipy")
    }
    if pins != {"jax": "0.6.2", "jaxlib": "0.6.2", "numpy": "2.2.6", "scipy": "1.15.3"}:
        raise ValueError("Numerical runtime differs from the signed foundation")
    value = dict(
        passed=True,
        scope=(
            "unchanged phase 01/02 mathematics, protocols, seeds, "
            "raw observations and decisions"
        ),
        historical_commit="108cbd5",
        historical_gates={
            n: sha256_file(original_root / f"research/gates/{n}.json") for n in original
        },
        historical_verifier_log=str(historical_log.relative_to(ROOT)),
        historical_verifier_log_sha256=sha256_file(historical_log),
        actual_lean_axiom_log_sha256=sha256_file(lean_log),
        protected_source_sha256=protected,
        administrative_scope_migration=changes,
        runtime_package_versions=pins,
        scientific_draws_or_thresholds_changed=False,
        raw_recovery_record_sha256=sha256_file(
            ROOT / "research/phase03/upstream-raw-recovery-v1.json"
        ),
    )
    if not apply:
        return value
    record_path = ROOT / "research/phase03/foundation-equivalence-v1.json"
    atomic_json(record_path, value)
    for n, gate in original.items():
        archive = ROOT / f"research/gates/archive/{n}-108cbd5.json"
        archive.parent.mkdir(parents=True, exist_ok=True)
        with archive.open("xb") as handle:
            handle.write((original_root / f"research/gates/{n}.json").read_bytes())
        assert sha256_file(archive) == value["historical_gates"][n]
        gate["source_and_evidence_sha256"] = {
            name: digest
            for name, digest in gate["source_and_evidence_sha256"].items()
            if name not in ADMINISTRATIVE and name != VERIFIER
        }
        for path in (
            record_path,
            ROOT / VERIFIER,
            Path(__file__),
            ROOT / "src/guidon/gates.py",
            ROOT / "research/phase03/upstream-raw-recovery-v1.json",
        ):
            gate["source_and_evidence_sha256"][str(path.relative_to(ROOT))] = (
                sha256_file(path)
            )
        gate["runtime_package_versions"] = {
            key: pins[key]
            for key in (("jax", "jaxlib", "numpy") if n == "01" else ("numpy", "scipy"))
        }
        gate["renewal"] = {
            "historical_gate_path": f"research/gates/archive/{n}-108cbd5.json",
            "historical_gate_sha256": value["historical_gates"][n],
            "equivalence_record": str(record_path.relative_to(ROOT)),
            "equivalence_record_sha256": sha256_file(record_path),
            "reason": (
                "Administrative scope migration after exact raw recovery; "
                "all scientific evidence unchanged"
            ),
        }
        for dependency in gate["depends_on"]:
            dependency["sha256"] = sha256_file(ROOT / dependency["path"])
        gate["next_action"] = (
            "Phases 03/04 authorized and in progress; "
            "no downstream gate is promoted by this renewal."
        )
        atomic_json(ROOT / f"research/gates/{n}.json", gate, replace=True)
    state_path = ROOT / "phases/state.json"
    state = json.loads(state_path.read_text())
    for phase in state["phases"][:2]:
        phase["evidence"]["sha256"] = sha256_file(ROOT / phase["evidence"]["path"])
    state["phases"][2].update(
        status="in_progress",
        reason=(
            "Authorized joint corpus indexing is live; actual capacities, "
            "sealing and production reader audits pending."
        ),
    )
    state["phases"][3].update(
        status="in_progress",
        reason=(
            "Trainer/evaluator/checkpoints implemented; source parity and "
            "compiled TPU kernel checks pass; full model and real-data "
            "profiling/resume evidence pending."
        ),
    )
    state["phases"][4].update(
        status="not_started",
        reason=(
            "Rewritten as a 125M/600M-token/2500-update hyperparameter sweep "
            "with 3 independent pilot seeds; not authorized to execute yet."
        ),
    )
    state["llm_confirmation_executed"] = False
    state["next_action"] = (
        "Complete actual Phase 03 data/sealing audits and Phase 04 full trainer "
        "verification/profiling on the existing TPU; preserve the original "
        "2.5B/42,43,44 confirmation contract."
    )
    atomic_json(state_path, state, replace=True)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-root", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    value = run(args.original_root.resolve(), apply=args.apply)
    print(
        json.dumps(
            {
                "passed": value["passed"],
                "protected_sources": len(value["protected_source_sha256"]),
                "scientific_draws_or_thresholds_changed": False,
                "applied": args.apply,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
