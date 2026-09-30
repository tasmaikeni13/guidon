"""Independent raw-data verifier and scalar public-reference replay apparatus."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np

from guidon.reference import Config, init, step
from research.experiments.check_math import self_check
from research.experiments.simulate import ROOT
from research.experiments.synthetic import Condition, problem_gradient, simulate


def manufactured_checks() -> dict:
    theta = np.array([[0.3, -0.4]])
    c = Condition("manufactured", groups=2, condition_number=100)
    target = np.array([0.75, -0.75])
    h = np.diag([1.0, 100.0])
    observed = problem_gradient(theta, c, 0, np.zeros_like(theta), guide=True)[0]
    expected = h @ (theta[0] - target)
    np.testing.assert_allclose(observed, expected, rtol=0, atol=1e-13)
    for j in range(2):
        e = np.eye(2)[j] * 1e-5
        plus = 0.5 * (theta[0] + e - target) @ h @ (theta[0] + e - target)
        minus = 0.5 * (theta[0] - e - target) @ h @ (theta[0] - e - target)
        assert abs((plus - minus) / (2e-5) - observed[j]) < 1e-8
    p = (np.zeros(1), np.zeros(1))
    cfg = Config(guide_warmup=0)
    _, _, stats = step(
        p,
        (np.ones(1), np.ones(1)),
        init(p),
        cfg,
        guide_gradients=(2 * np.ones(1), np.zeros(1)),
    )
    np.testing.assert_allclose(stats["weights"], [1.15, 0.85], atol=1e-14)
    return {
        "known_hessian_gradient": True,
        "finite_difference_gradient": True,
        "hand_weights": stats["weights"].tolist(),
        "fault_rejection": self_check()["faults_rejected"],
    }


def split_blocks(x, groups):
    return (x.copy(),) if groups == 1 else tuple(np.array([v]) for v in x)


def reference_replay(condition: Condition, seeds: np.ndarray) -> dict:
    rows, data = simulate(condition, seeds, capture=True)
    changed, altered = simulate(condition, seeds, changed_evaluation=True, capture=True)
    assert np.array_equal(data["paths"], altered["paths"])
    assert all(
        x["evaluation_sha256"] != y["evaluation_sha256"]
        for x, y in zip(rows, changed, strict=True)
    )
    maximum = 0.0
    weight_maximum = 0.0
    for i in range(len(seeds)):
        parameters = [
            split_blocks(data["initial"][i], condition.groups) for _ in range(3)
        ]
        states = [init(p) for p in parameters]
        config = Config(
            learning_rate=condition.learning_rate,
            beta1=condition.beta1,
            beta2=0.95,
            weight_decay=0,
            radius=condition.radius,
            guide_interval=condition.interval,
            guide_warmup=0,
            signal_floor=condition.signal_floor,
        )
        for t, tape in enumerate(data["tapes"]):
            for arm in range(3):
                g = tape["g"][arm, i].copy()
                h = None if tape["h"] is None else tape["h"][arm, i].copy()
                cfg = config if arm == 1 else replace(config, radius=0)
                if arm == 2 and h is not None:
                    g = (condition.batch_size * g + condition.guide_batch_size * h) / (
                        condition.batch_size + condition.guide_batch_size
                    )
                probe = None
                if arm == 1 and cfg.probe_due(t):
                    if condition.intervention == "reverse":
                        h = -h
                    elif condition.intervention == "remove":
                        h = np.zeros_like(h)
                    elif condition.intervention in ("shuffle", "collinear"):
                        old_m = np.concatenate(states[1].first)
                        old_v = np.concatenate(states[1].second)
                        m = cfg.beta1 * old_m + (1 - cfg.beta1) * g
                        v = cfg.beta2 * old_v + (1 - cfg.beta2) * g**2
                        u = (m / (1 - cfg.beta1 ** (t + 1))) / (
                            np.sqrt(v / (1 - cfg.beta2 ** (t + 1))) + cfg.epsilon
                        )
                        h = (
                            g.copy()
                            if condition.intervention == "collinear"
                            else np.divide(
                                np.roll(h * u, 1), u, out=np.zeros_like(h), where=u != 0
                            )
                        )
                    probe = split_blocks(h, condition.groups)
                parameters[arm], states[arm], metrics = step(
                    parameters[arm],
                    split_blocks(g, condition.groups),
                    states[arm],
                    cfg,
                    guide_gradients=probe,
                )
                observed = np.concatenate(parameters[arm])
                expected = data["paths"][t, arm, i]
                discrepancy = float(np.max(np.abs(observed - expected)))
                maximum = max(maximum, discrepancy)
                np.testing.assert_allclose(observed, expected, rtol=2e-11, atol=2e-12)
                if arm == 1:
                    weight_maximum = max(
                        weight_maximum,
                        float(np.max(np.abs(metrics["weights"] - tape["weights"][i]))),
                    )
                    np.testing.assert_allclose(
                        metrics["weights"], tape["weights"][i], rtol=2e-11, atol=2e-12
                    )
                    if condition.intervention == "collinear" and cfg.probe_due(t):
                        np.testing.assert_allclose(metrics["weights"], 1, atol=1e-12)
    return {
        "max_parameter_discrepancy": maximum,
        "max_weight_discrepancy": weight_maximum,
        "evaluation_noninterference_bitwise": True,
        "replayed_draws": len(seeds),
    }


def verify(
    raw_path: Path,
    provenance_path: Path,
    output_path: Path,
    protocol_path: Path = ROOT / "research/phase02/protocol.json",
) -> None:
    protocol = json.loads(protocol_path.read_text())
    provenance = json.loads(provenance_path.read_text())
    assert hashlib.sha256(raw_path.read_bytes()).hexdigest() == provenance["raw_sha256"]
    assert (
        hashlib.sha256(protocol_path.read_bytes()).hexdigest()
        == provenance["protocol_sha256"]
    )
    assert (
        hashlib.sha256(
            (ROOT / "research/phase02/analysis_spec.md").read_bytes()
        ).hexdigest()
        == provenance["analysis_spec_sha256"]
    )
    for name, checksum in provenance["source_sha256"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != checksum:
            # Immutable older records retain their exact executed source snapshot.
            # The sole driver change is CLI path resolution, audited separately.
            assert name == "research/experiments/simulate.py", name
            snapshot = subprocess.check_output(
                ["git", "show", f"0902877:{name}"], cwd=ROOT
            )
            assert hashlib.sha256(snapshot).hexdigest() == checksum, name
    cases = {c["id"]: Condition(**c) for c in protocol["conditions"]}
    counters = {key: 0 for key in cases}
    seen = set()
    selected_rows = {key: {} for key in cases}
    with gzip.open(raw_path, "rt") as handle:
        for line in handle:
            row = json.loads(line)
            case = cases[row["condition"]]
            identity = (case.id, row["seed"])
            assert identity not in seen
            seen.add(identity)
            counters[case.id] += 1
            assert np.isfinite(np.asarray(row["final_parameters"])).all()
            assert row["max_relative_neutrality"] <= 1e-5
            assert row["min_proxy_gain"] >= -1e-12
            assert row["min_fresh_gain"] >= -1e-12
            assert 0 <= row["fallback_fraction"] <= 1
            assert row["guide_examples"] <= case.guide_stream_size
            assert row["fresh_probes"] == len(range(0, case.steps, case.interval))
            if case.radius == 0 or case.intervention == "remove":
                assert row["adamw_limit_bitwise"]
            if case.family in ("quadratic", "rotating"):
                # Independent final-loss calculation from declared Hessian/center.
                dim = case.dim
                target = np.resize([0.75, -0.75], dim)
                angle = case.rotation * (case.steps - 1)
                target[:2] = (
                    np.array(
                        [
                            [np.cos(angle), -np.sin(angle)],
                            [np.sin(angle), np.cos(angle)],
                        ]
                    )
                    @ target[:2]
                )
                eigen = np.exp(np.linspace(0, np.log(case.condition_number), dim))
                loss = 0.5 * np.mean(
                    eigen * (np.asarray(row["final_parameters"]) - target) ** 2, axis=-1
                )
                np.testing.assert_allclose(
                    loss, row["evaluation_loss"], rtol=1e-12, atol=1e-12
                )
            if len(selected_rows[case.id]) < 3:
                selected_rows[case.id][row["seed"]] = row
    assert set(counters.values()) == {
        protocol["draws_per_primary_condition_per_stream"]
    }
    replays = {}
    for case in cases.values():
        seeds = np.array(list(selected_rows[case.id]), dtype=np.uint64)
        generated, _ = simulate(case, seeds)
        for row in generated:
            original = selected_rows[case.id][row["seed"]]
            assert row["trajectory_sha256"] == original["trajectory_sha256"]
            np.testing.assert_array_equal(
                row["evaluation_loss"], original["evaluation_loss"]
            )
        replays[case.id] = reference_replay(case, seeds)
    report = {
        "passed": True,
        "stream": provenance["stream"],
        "rows_verified": len(seen),
        "condition_counts": counters,
        "manufactured": manufactured_checks(),
        "independent_reference_replays": replays,
        "raw_sha256": provenance["raw_sha256"],
    }
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        f"Verified {len(seen)} raw draws, {len(cases) * 3} reference replays;"
        " all families bitwise evaluation-isolated"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stream", choices=["discovery", "confirmation"], required=True
    )
    args = parser.parse_args()
    verify(
        ROOT / f"artifacts/phase02/{args.stream}-v1.jsonl.gz",
        ROOT / f"research/phase02/{args.stream}-v1-provenance.json",
        ROOT / f"research/phase02/{args.stream}-v1-verification.json",
    )
