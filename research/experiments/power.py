"""Three-pair interval calibration and design sensitivity; no LLM seed extension."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json

import numpy as np
from scipy.stats import nct, t

from research.experiments.simulate import ROOT


def run() -> None:
    raw_path = ROOT / "artifacts/phase02/power-v1.csv.gz"
    protocol_path = ROOT / "research/phase02/power-protocol.json"
    protocol = json.loads(protocol_path.read_text())
    if raw_path.exists():
        raise FileExistsError("Power raw data is immutable")
    critical = float(t.ppf(0.975, 2))
    rng = np.random.default_rng(protocol["seed"])
    cells = [
        (distribution, sigma, effect)
        for distribution in protocol["distributions"]
        for sigma in protocol["paired_difference_sd"]
        for effect in protocol["effects"]
    ]
    order = rng.permutation(len(cells))
    summary = []
    n = protocol["draws_per_cell"]
    with (
        raw_path.open("wb") as handle,
        gzip.GzipFile(fileobj=handle, mode="wb", mtime=0, filename="") as compressed,
    ):
        import io

        with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as raw:
            writer = csv.writer(raw)
            writer.writerow(
                [
                    "distribution",
                    "sigma",
                    "effect",
                    "draw",
                    "d42",
                    "d43",
                    "d44",
                    "mean",
                    "lower",
                    "upper",
                ]
            )
            for index in order:
                distribution, sigma, effect = cells[index]
                if distribution == "normal":
                    errors = rng.normal(size=(n, 3))
                elif distribution == "t3":
                    errors = rng.standard_t(3, size=(n, 3)) / np.sqrt(3)
                else:
                    errors = (np.exp(rng.normal(size=(n, 3))) - np.exp(0.5)) / np.sqrt(
                        (np.e - 1) * np.e
                    )
                differences = effect + sigma * errors
                mean = differences.mean(axis=-1)
                width = critical * differences.std(axis=-1, ddof=1) / np.sqrt(3)
                lower, upper = mean - width, mean + width
                covered = (lower <= effect) & (effect <= upper)
                significant = (lower > 0) | (upper < 0)
                positive = lower > 0
                for i in range(n):
                    writer.writerow(
                        [
                            distribution,
                            sigma,
                            effect,
                            i,
                            *differences[i],
                            mean[i],
                            lower[i],
                            upper[i],
                        ]
                    )
                coverage = float(covered.mean())
                power = float(significant.mean())
                expected = None
                if distribution == "normal":
                    noncentrality = np.sqrt(3) * effect / sigma
                    expected = float(
                        nct.sf(critical, 2, noncentrality)
                        + nct.cdf(-critical, 2, noncentrality)
                    )
                summary.append(
                    {
                        "distribution": distribution,
                        "sigma": sigma,
                        "effect": effect,
                        "n": n,
                        "coverage": coverage,
                        "coverage_mcse": float(np.sqrt(coverage * (1 - coverage) / n)),
                        "two_sided_rejection_probability": power,
                        "power_mcse": float(np.sqrt(power * (1 - power) / n)),
                        "beneficial_interval_probability": float(positive.mean()),
                        "point_target_probability": float((mean >= 0.005).mean()),
                        "mean_interval_width": float((2 * width).mean()),
                        "width_mcse": float((2 * width).std(ddof=1) / np.sqrt(n)),
                        "normal_theory_rejection_probability": expected,
                    }
                )
    result = {
        "critical_t_df2": critical,
        "n_pairs": 3,
        "raw_path": str(raw_path.relative_to(ROOT)),
        "raw_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        "rows": n * len(cells),
        "cells": summary,
        "minimum_two_sided_exact_signflip_p": 0.25,
    }
    (ROOT / "research/phase02/power-v1.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(f"Power calibration: {result['rows']} synthetic three-pair experiments")


if __name__ == "__main__":
    run()
