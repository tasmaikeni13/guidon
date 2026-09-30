"""Recheck actual emitted coefficients and weights from phase-01 immutable raw data."""

from __future__ import annotations

import gzip
import hashlib
import json
from collections import Counter

from research.experiments.check_math import check_weights, self_check
from research.experiments.simulate import ROOT


def verify() -> None:
    summary = json.loads((ROOT / "research/phase01/math-v3.json").read_text())
    protocol_path = ROOT / "research/phase01/protocol.json"
    protocol = json.loads(protocol_path.read_text())
    raw = ROOT / summary["raw_path"]
    assert hashlib.sha256(raw.read_bytes()).hexdigest() == summary["raw_sha256"]
    assert (
        hashlib.sha256(protocol_path.read_bytes()).hexdigest()
        == summary["protocol_sha256"]
    )
    counts = Counter()
    resolved_failures = 0
    overflows = 0
    with gzip.open(raw, "rt") as handle:
        for line in handle:
            row = json.loads(line)
            counts[
                (row["groups"], row["dtype"], row["scale"], row["perturbation"])
            ] += 1
            checked = check_weights(
                row["a"],
                row["c"],
                row["weights"],
                protocol["radius"],
                protocol["relative_neutrality_limit"],
            )
            assert checked["passed"], row
            assert checked == row["checked"]
            assert row["schedule_ok"]
            assert row["parameters_finite"]
            if not row["moments_finite"]:
                overflows += 1
                assert not row["numerics_ok"]
            if (
                1e-6 <= row["scale"] <= 1e15
                and row["projected_signal"] > protocol["resolved_projected_signal"]
            ):
                assert row["numerics_ok"]
                assert (
                    row["weight_difference"]
                    <= protocol["weight_parity_atol_when_resolved"]
                )
                assert row["parameter_difference"] <= protocol["parameter_parity_atol"]
                resolved_failures += row["fallback_jax"]
    assert len(counts) == len(protocol["groups"]) * 2 * len(
        protocol["gradient_scales"]
    ) * len(protocol["collinearity_perturbations"])
    assert set(counts.values()) == {protocol["draws_per_cell"]}
    assert resolved_failures == 0
    report = {
        "passed": True,
        "raw_sha256": summary["raw_sha256"],
        "verified_rows": sum(counts.values()),
        "cells": len(counts),
        "resolved_fallbacks": resolved_failures,
        "visible_moment_overflows": overflows,
        "independent_checker_self_test": self_check(),
    }
    (ROOT / "research/phase01/verification.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(
        f"Verified {report['verified_rows']} emitted coefficient/weight records; "
        f"all invariants pass; {overflows} overflows flagged"
    )


if __name__ == "__main__":
    verify()
