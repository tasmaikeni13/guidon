"""Independent three-pair raw interval checker, with analytic normal calibration."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
from collections import defaultdict

import numpy as np
from scipy.stats import nct

from research.experiments.simulate import ROOT


def verify() -> None:
    summary = json.loads((ROOT / "research/phase02/power-v1.json").read_text())
    protocol_path = ROOT / "research/phase02/power-protocol.json"
    protocol = json.loads(protocol_path.read_text())
    raw_path = ROOT / summary["raw_path"]
    assert hashlib.sha256(raw_path.read_bytes()).hexdigest() == summary["raw_sha256"]
    assert (
        hashlib.sha256(protocol_path.read_bytes()).hexdigest()
        == summary["protocol_sha256"]
    )
    # The CDF of t with 2 df is 1/2 + x/(2*sqrt(x^2+2)); invert it.
    critical = np.sqrt(2 * 0.95**2 / (1 - 0.95**2))
    # Test the quantile's probability residual: SciPy's numerical ppf differs
    # from the df=2 closed form by about 5e-11 in x, but only 6e-13 in the CDF.
    reported = summary["critical_t_df2"]
    assert abs(0.5 + reported / (2 * np.sqrt(reported**2 + 2)) - 0.975) < 2e-11
    records = defaultdict(list)
    seen = set()
    with gzip.open(raw_path, "rt") as handle:
        for row in csv.DictReader(handle):
            identity = (row["distribution"], float(row["sigma"]), float(row["effect"]))
            draw_id = (*identity, int(row["draw"]))
            assert draw_id not in seen
            seen.add(draw_id)
            differences = np.array(
                [float(row[key]) for key in ("d42", "d43", "d44")], dtype=np.longdouble
            )
            mean = differences.sum() / 3
            sd = np.sqrt(np.sum((differences - mean) ** 2) / 2)
            width = critical * sd / np.sqrt(np.longdouble(3))
            np.testing.assert_allclose(
                [mean, mean - width, mean + width],
                [float(row[key]) for key in ("mean", "lower", "upper")],
                rtol=1e-10,
                atol=1e-12,
            )
            records[identity].append(
                (float(mean), float(mean - width), float(mean + width))
            )
    assert len(records) == len(protocol["distributions"]) * len(
        protocol["paired_difference_sd"]
    ) * len(protocol["effects"])
    results = []
    for cell in summary["cells"]:
        identity = (cell["distribution"], cell["sigma"], cell["effect"])
        values = np.asarray(records[identity])
        assert len(values) == protocol["draws_per_cell"]
        mean, lower, upper = values.T
        covered = (lower <= cell["effect"]) & (cell["effect"] <= upper)
        rejected = (lower > 0) | (upper < 0)
        assert abs(float(covered.mean()) - cell["coverage"]) < 1e-12
        assert (
            abs(float(rejected.mean()) - cell["two_sided_rejection_probability"])
            < 1e-12
        )
        np.testing.assert_allclose(
            (upper - lower).mean(), cell["mean_interval_width"], rtol=1e-10
        )
        if identity[0] == "normal":
            coverage_error = abs(cell["coverage"] - 0.95) / np.sqrt(
                0.95 * 0.05 / len(values)
            )
            noncentrality = np.sqrt(3) * cell["effect"] / cell["sigma"]
            target_power = nct.sf(critical, 2, noncentrality) + nct.cdf(
                -critical, 2, noncentrality
            )
            power_error = abs(float(rejected.mean()) - target_power) / np.sqrt(
                target_power * (1 - target_power) / len(values)
            )
            assert (
                coverage_error
                <= protocol["coverage_or_power_calibration_error_max_mcse"]
            )
            assert (
                power_error <= protocol["coverage_or_power_calibration_error_max_mcse"]
            )
            results.append(
                {
                    "cell": identity,
                    "coverage_error_mcse": coverage_error,
                    "power_error_mcse": power_error,
                }
            )
    output = {
        "passed": True,
        "raw_rows_verified": len(seen),
        "raw_sha256": summary["raw_sha256"],
        "critical_t_from_independent_df2_cdf": float(critical),
        "normal_calibration": results,
        "nonnormal_coverage": "reported as sensitivity, not required to equal 0.95",
    }
    (ROOT / "research/phase02/power-v1-verification.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n"
    )
    print(
        f"Verified all {len(seen)} three-pair intervals; "
        "Gaussian coverage/power agree with independent analytic targets"
    )


if __name__ == "__main__":
    verify()
