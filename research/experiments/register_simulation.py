"""Construct the phase-02 design before sampling; no outcome-dependent choices."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from research.experiments.synthetic import Condition

ROOT = Path(__file__).resolve().parents[2]


def conditions() -> list[Condition]:
    baseline = Condition("intended")
    result = []
    # Walsh columns yield balanced orthogonal binary main effects. This is a
    # fractional screen; two-factor effects are aliased and never called causal.
    factors = [
        ("noise", (0, 0.5)),
        ("batch_size", (8, 128)),
        ("covariance", (0, 0.8)),
        ("alignment", (0.1, 1)),
        ("condition_number", (1, 10000)),
        ("mismatch", (0, 1)),
        ("groups", (1, 64)),
        ("interval", (1, 256)),
        ("radius", (0, 0.3)),
        ("guide_stream_size", (512, 4096)),
        ("delay", (0, 64)),
        ("beta1", (0, 0.99)),
    ]
    for i in range(32):
        settings = {
            name: levels[(i & column).bit_count() % 2]
            for column, (name, levels) in enumerate(factors, start=1)
        }
        settings["learning_rate"] = 0.01 / np.sqrt(settings["condition_number"])
        result.append(replace(baseline, id=f"screen-{i:02}", **settings))
    result.append(baseline)
    for factor, values in {
        "groups": [1, 2, 64],
        "interval": [1, 64, 256],
        "radius": [0, 0.05, 0.3],
        "noise": [0, 0.5],
        "batch_size": [8, 128],
        "guide_stream_size": [512],
        "delay": [16, 64],
        "beta1": [0, 0.99],
        "mismatch": [1],
        "condition_number": [1, 10000],
    }.items():
        for value in values:
            result.append(
                replace(baseline, id=f"target-{factor}-{value}", **{factor: value})
            )
    for intervention in ("reverse", "shuffle", "collinear", "remove"):
        result.append(
            replace(baseline, id=f"control-{intervention}", intervention=intervention)
        )
    for rotation, interval in (
        (0.001, 1),
        (0.001, 64),
        (0.2, 16),
        (0.2, 64),
        (0.2, 256),
    ):
        result.append(
            replace(
                baseline,
                id=f"rotate-{rotation}-{interval}",
                family="rotating",
                rotation=rotation,
                interval=interval,
                condition_number=1,
            )
        )
    for family in ("regression", "classification"):
        for mismatch in (0, 1):
            result.append(
                replace(
                    baseline,
                    id=f"spurious-{family}-{mismatch}",
                    family=family,
                    groups=2,
                    condition_number=1,
                    interval=1,
                    mismatch=mismatch,
                    learning_rate=0.03,
                    steps=48,
                )
            )
    result.append(
        replace(
            baseline,
            id="curvature-large-step",
            groups=2,
            condition_number=10000,
            learning_rate=2,
            noise=0,
            interval=1,
            steps=8,
        )
    )
    return result


if __name__ == "__main__":
    destination = ROOT / "research/phase02/protocol.json"
    if destination.exists():
        raise FileExistsError("Do not overwrite a registered simulation design")
    design = {
        "version": "simulation-v1-guidon-0.2",
        "theory_version": "0.2",
        "discovery_seed": 271801,
        "confirmation_seed": 271802,
        "draws_per_primary_condition_per_stream": 1000,
        "arms": ["adamw", "guidon", "adamw_plus_guidance"],
        "order": (
            "SeedSequence-root randomized condition order; condition-specific "
            "SHA256-derived per-draw seeds"
        ),
        "stopping": (
            "run every condition and retain all draws; no exclusions based on losses"
        ),
        "screening": (
            "32-cell orthogonal two-level fractional factorial; main effects "
            "descriptive, interactions aliased; all targeted followups "
            "predeclared"
        ),
        "intended_regime_ids": [
            "intended",
            "target-groups-2",
            "target-groups-64",
            "target-interval-1",
            "target-interval-64",
            "target-radius-0.05",
            "target-radius-0.3",
            "target-noise-0",
            "target-batch_size-128",
        ],
        "primary_invariants": {
            "relative_neutrality": 1e-5,
            "min_proxy_gain": -1e-12,
            "min_fresh_gain": -1e-12,
            "finite_parameters": True,
            "radius_zero_and_removed_guide_bitwise": True,
        },
        "mechanism_decisions": {
            "representative_transfer": (
                "mean current-guide and evaluation linear gain positive, lower 95% "
                "Monte Carlo interval >0 in every intended condition"
            ),
            "signal_corruption": (
                "intended mean true guide gain exceeds both reverse and shuffle "
                "controls; reverse mean <0"
            ),
            "collinearity": (
                "weights differ from unit by at most 1e-12 for exact c parallel a"
            ),
            "controlled_drift": (
                "rotate-0.2-64 or rotate-0.2-256 mean current guide gain below "
                "fresh rotate-0.001-1; no stale-current guarantee"
            ),
            "actual_loss": (
                "record reversals; first-order identities do not impose actual "
                "descent or superiority to information control"
            ),
            "evaluation_noninterference": (
                "replay each family with changed evaluation data; all parameter "
                "paths bitwise equal and evaluation hashes different"
            ),
        },
        "analysis": (
            "all three arm losses, paired differences and Monte Carlo SE; no "
            "condition selected as LLM validation; 95% MC intervals descriptive"
            " except registered local mechanisms"
        ),
        "conditions": [asdict(c) for c in conditions()],
    }
    destination.write_text(json.dumps(design, indent=2) + "\n")
    print(f"Registered {len(design['conditions'])} conditions")
