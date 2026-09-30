"""Verify current phase evidence and dependency hashes; never promote a stale gate."""

from __future__ import annotations

import gzip
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from guidon.protocol import audit
from research.experiments.simulate import ROOT, draw_seeds


def checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_gate(identity: str) -> dict:
    gate_path = ROOT / f"research/gates/{identity}.json"
    gate = json.loads(gate_path.read_text())
    assert gate["status"] == "passed", identity
    assert gate["theory_version"] == "0.2"
    for name, expected in gate["source_and_evidence_sha256"].items():
        assert checksum(ROOT / name) == expected, f"Stale {identity}: {name}"
    for raw in gate["raw_artifacts"]:
        assert checksum(ROOT / raw["path"]) == raw["sha256"], raw["path"]
    for prerequisite in gate["depends_on"]:
        assert checksum(ROOT / prerequisite["path"]) == prerequisite["sha256"]
        assert (
            json.loads((ROOT / prerequisite["path"]).read_text())["status"] == "passed"
        )
    return gate


def independently_check_summaries() -> None:
    protocol = json.loads((ROOT / "research/phase02/protocol.json").read_text())
    keys = (
        "current_guide_linear_gain",
        "evaluation_linear_gain",
        "actual_guide_advantage",
        "actual_train_advantage",
        "fallback_fraction",
        "coefficient_time_variance",
        "update_norm_mean",
    )
    seed_sets = []
    for stream in ("discovery", "confirmation"):
        observations = defaultdict(list)
        with gzip.open(
            ROOT / f"artifacts/phase02/{stream}-v1.jsonl.gz", "rt"
        ) as handle:
            for line in handle:
                row = json.loads(line)
                observations[row["condition"]].append(row)
        summary = json.loads(
            (ROOT / f"research/phase02/{stream}-v1-summary.json").read_text()
        )
        found_seeds = set()
        for case in protocol["conditions"]:
            identity = case["id"]
            rows = observations[identity]
            seeds = {r["seed"] for r in rows}
            expected = set(
                map(int, draw_seeds(protocol[f"{stream}_seed"], identity, 1000))
            )
            assert seeds == expected
            found_seeds |= seeds
            evaluation = np.asarray(
                [r["evaluation_loss"] for r in rows], dtype=np.longdouble
            )
            contrasts = {
                "adamw_minus_guidon": evaluation[:, 0] - evaluation[:, 1],
                "control_minus_guidon": evaluation[:, 2] - evaluation[:, 1],
            }
            for key in keys:
                contrasts[key] = np.asarray([r[key] for r in rows], dtype=np.longdouble)
            for key, x in contrasts.items():
                mean = np.sum(x) / len(x)
                sd = np.sqrt(np.sum((x - mean) ** 2) / (len(x) - 1))
                se = sd / np.sqrt(np.longdouble(len(x)))
                observed = summary[identity][key]
                np.testing.assert_allclose(
                    [mean, se, mean - 1.96 * se, mean + 1.96 * se],
                    [observed[k] for k in ("mean", "mcse", "lower95", "upper95")],
                    rtol=1e-11,
                    atol=1e-12,
                )
        seed_sets.append(found_seeds)
    assert seed_sets[0].isdisjoint(seed_sets[1])


def verify() -> None:
    state = json.loads((ROOT / "phases/state.json").read_text())
    assert state["protocol_version"] == "0.2"
    for phase in state["phases"][:2]:
        assert phase["status"] == "passed"
        assert checksum(ROOT / phase["evidence"]["path"]) == phase["evidence"]["sha256"]
        verify_gate(phase["id"])
    assert all(
        p["status"] == "invalidated" and p["evidence"] is None
        for p in state["phases"][2:]
    )
    assert not state["llm_training_executed"]
    theorem_names = {
        name
        for source in (ROOT / "proofs/Guidon").glob("*.lean")
        for name in re.findall(r"^theorem (\w+)", source.read_text(), re.M)
    }
    audit_text = (ROOT / "proofs/Audit.lean").read_text()
    for name in theorem_names:
        assert f"#print axioms Guidon.{name}" in audit_text
    axioms = (ROOT / "research/verification/phase01/axioms.txt").read_text()
    assert "sorryAx" not in axioms
    for line in axioms.splitlines():
        match = re.search(r"depends on axioms: \[(.*)\]", line)
        if match:
            assert set(match.group(1).split(", ")) <= {
                "propext",
                "Classical.choice",
                "Quot.sound",
            }
    for name, tokens, updates, probes in (
        ("pretrain_125m", 2500000000, 4769, 73),
        ("continued_pythia160m", 1000000000, 1908, 28),
    ):
        config = json.loads((ROOT / f"configs/{name}.json").read_text())
        result = audit(config)
        assert result["training_loss_tokens_per_run"] == tokens
        assert result["updates"] == updates and result["guidance_probes"] == probes
        assert not result["ready_for_confirmation"]
        assert config["optimizer_defaults"]["signal_floor"] == 0.01
    for stream in ("discovery", "confirmation"):
        assert json.loads(
            (ROOT / f"research/phase02/{stream}-v1-decisions.json").read_text()
        )["local_mechanism_passed"]
        assert json.loads(
            (ROOT / f"research/phase02/{stream}-v1-verification.json").read_text()
        )["passed"]
        assert json.loads(
            (ROOT / f"research/phase02/drift-{stream}-v1-verification.json").read_text()
        )["passed"]
        assert json.loads(
            (ROOT / f"research/phase02/drift-{stream}-age-v1.json").read_text()
        )["registered_stale_reversal_passed"]
    assert json.loads(
        (ROOT / "research/phase02/power-v1-verification.json").read_text()
    )["passed"]
    assert json.loads(
        (ROOT / "research/phase02/opposite-evaluation-v1.json").read_text()
    )["all_arms_worsen_independent_evaluation"]
    independently_check_summaries()
    print(
        "Phases 01/02 passed: all gate/source/raw/dependency hashes verified; "
        f"{len(theorem_names)} theorem audits; independent summaries and "
        "stream identities; unchanged token/seed contracts"
    )


if __name__ == "__main__":
    verify()
