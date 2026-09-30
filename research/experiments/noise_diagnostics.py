"""Secondary coefficient-noise measurements using existing confirmation identities.

Measure the first probe of each registered quadratic/rotating condition. Shortening
the measurement to that first update leaves its initialization and RNG tape intact.
No new primary decision or favorable-condition selection is made.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import replace

import numpy as np

from guidon.reference import controller
from research.experiments.analyze_simulation import mean_se
from research.experiments.simulate import ROOT, draw_seeds
from research.experiments.synthetic import Condition, center, simulate


def run() -> None:
    protocol = json.loads((ROOT / "research/phase02/protocol.json").read_text())
    cases = [
        Condition(**c)
        for c in protocol["conditions"]
        if c["family"] in ("quadratic", "rotating") and c["intervention"] == "ordinary"
    ]
    raw_path = ROOT / "artifacts/phase02/coefficient-noise-v1.jsonl.gz"
    if raw_path.exists():
        raise FileExistsError("Coefficient-noise observations are immutable")
    results = {}
    with (
        raw_path.open("wb") as handle,
        gzip.GzipFile(fileobj=handle, mode="wb", mtime=0, filename="") as raw,
    ):
        for case in cases:
            seeds = draw_seeds(protocol["confirmation_seed"], case.id, 1000)
            _, data = simulate(replace(case, steps=1), seeds, capture=True)
            tape = data["tapes"][0]
            theta = data["initial"]
            eigen = np.geomspace(1, case.condition_number, case.dim)
            train_center = (
                0.25 * center(case.dim, 0, case.rotation)
                if case.family == "rotating"
                else np.zeros(case.dim)
            )
            guide_center = (
                case.alignment
                * (1 - 2 * case.mismatch)
                * center(case.dim, -case.delay, case.rotation)
            )
            population_g = (theta - train_center) * eigen
            population_h = (theta - guide_center) * eigen
            g = tape["g"][1]
            h = tape["h"][1]
            u = g / (np.abs(g) + 1e-8)
            observed_a = g * u
            observed_c = h * u
            population_a = population_g * u
            population_c = population_h * u
            if case.groups == 1:
                observed_a = observed_a.sum(axis=-1, keepdims=True)
                observed_c = observed_c.sum(axis=-1, keepdims=True)
                population_a = population_a.sum(axis=-1, keepdims=True)
                population_c = population_c.sum(axis=-1, keepdims=True)
            a_errors = observed_a - population_a
            c_errors = observed_c - population_c
            correction_errors = []
            true_gain = []
            for i, seed in enumerate(seeds):
                normalized_c = (
                    population_c[i] / np.max(np.abs(population_c[i]))
                    if np.max(np.abs(population_c[i]))
                    else population_c[i]
                )
                oracle_weights, _ = controller(
                    population_a[i], normalized_c, case.radius, 1e-5, case.signal_floor
                )
                difference = tape["weights"][i] - oracle_weights
                correction_errors.append(difference)
                true_gain.append(float(population_c[i] @ (tape["weights"][i] - 1)))
                row = {
                    "condition": case.id,
                    "seed": int(seed),
                    "sample_a": observed_a[i].tolist(),
                    "population_a_at_sample_u": population_a[i].tolist(),
                    "sample_c": observed_c[i].tolist(),
                    "population_c_at_sample_u": population_c[i].tolist(),
                    "delta_minus_noiseless_delta": difference.tolist(),
                }
                raw.write((json.dumps(row, sort_keys=True) + "\n").encode())
            errors = np.asarray(correction_errors)
            results[case.id] = {
                "n": 1000,
                "a_error_coordinate_variance_mean": float(
                    np.var(a_errors, axis=0, ddof=1).mean()
                ),
                "c_error_coordinate_variance_mean": float(
                    np.var(c_errors, axis=0, ddof=1).mean()
                ),
                "a_error_max_coordinate_mean_abs": float(
                    np.max(np.abs(a_errors.mean(axis=0)))
                ),
                "c_error_max_coordinate_mean_abs": float(
                    np.max(np.abs(c_errors.mean(axis=0)))
                ),
                "normalization_correction_bias_norm": float(
                    np.linalg.norm(errors.mean(axis=0))
                ),
                "correction_error_norm": mean_se(np.linalg.norm(errors, axis=-1)),
                "true_population_fresh_gain": mean_se(true_gain),
                "scope": (
                    "u depends on sampled g; variance includes that dependence. "
                    "Correction bias is diagnostic, not an unbiasedness claim."
                ),
            }
    output = {
        "scope": "secondary first-probe measurements on existing confirmation seeds",
        "raw_path": str(raw_path.relative_to(ROOT)),
        "raw_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        "rows": len(cases) * 1000,
        "conditions": results,
    }
    (ROOT / "research/phase02/coefficient-noise-v1.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n"
    )
    print(
        f"Measured coefficient errors and normalization sensitivity for "
        f"{len(cases)} registered conditions; {output['rows']} existing draw identities"
    )


if __name__ == "__main__":
    run()
