"""Secondary age/sensitivity diagnostics on the existing registered raw seeds.

No new conditions, seeds or primary decisions are selected here. All five
registered rotating conditions are measured, including the null/reversal regimes.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
from scipy.stats import norm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from research.experiments.simulate import ROOT, draw_seeds  # noqa: E402
from research.experiments.synthetic import Condition, center, simulate  # noqa: E402


def run(protocol_path, stream, raw_path, summary_path, figure_path) -> None:
    protocol = json.loads(protocol_path.read_text())
    count = protocol["draws_per_primary_condition_per_stream"]
    cases = [
        Condition(**c) for c in protocol["conditions"] if c["family"] == "rotating"
    ]
    if raw_path.exists():
        raise FileExistsError("Secondary diagnostics raw output is immutable")
    result = {}
    group_sizes = sorted({c.groups for c in cases})
    fig, axes = plt.subplots(
        1,
        len(group_sizes),
        figsize=(5 * len(group_sizes), 5),
        layout="constrained",
        squeeze=False,
    )
    with (
        raw_path.open("wb") as handle,
        gzip.GzipFile(fileobj=handle, mode="wb", mtime=0, filename="") as raw,
    ):
        for case in cases:
            seeds = draw_seeds(protocol[f"{stream}_seed"], case.id, count)
            rows, captured = simulate(case, seeds, capture=True)
            first = np.zeros((count, case.dim))
            second = np.zeros_like(first)
            gains = []
            for index, tape in enumerate(captured["tapes"]):
                gradient = tape["g"][1]
                first = case.beta1 * first + (1 - case.beta1) * gradient
                second = 0.95 * second + 0.05 * gradient**2
                direction = (first / (1 - case.beta1 ** (index + 1))) / (
                    np.sqrt(second / (1 - 0.95 ** (index + 1))) + 1e-8
                )
                theta_before = (
                    captured["initial"]
                    if index == 0
                    else captured["paths"][index - 1, 1]
                )
                target = (
                    case.alignment
                    * (1 - 2 * case.mismatch)
                    * center(case.dim, index, case.rotation)
                )
                true_gradient = (theta_before - target) * np.geomspace(
                    1, case.condition_number, case.dim
                )
                gain = np.sum(
                    true_gradient * (tape["weights"] - 1) * direction, axis=-1
                )
                gains.append(gain)
            gains = np.stack(gains)
            for index, row in enumerate(rows):
                np.testing.assert_allclose(
                    gains[:, index].mean(),
                    row["current_guide_linear_gain"],
                    rtol=1e-11,
                    atol=1e-11,
                )
                output = {
                    "condition": case.id,
                    "seed": row["seed"],
                    "per_step_current_gain": gains[:, index].tolist(),
                    "trajectory_sha256": row["trajectory_sha256"],
                }
                raw.write((json.dumps(output, sort_keys=True) + "\n").encode())
            mean = gains.mean(axis=-1)
            se = gains.std(axis=-1, ddof=1) / np.sqrt(count)
            result[case.id] = {
                "n": count,
                "mean_current_gain_per_step": mean.tolist(),
                "mcse_per_step": se.tolist(),
                "steps_with_negative_mean_upper95": np.flatnonzero(
                    mean + 1.96 * se < 0
                ).tolist(),
                "draws_with_any_negative_gain": int(np.sum(np.any(gains < 0, axis=0))),
            }
            ax = axes[0, group_sizes.index(case.groups)]
            ax.plot(np.arange(case.steps), mean, label=case.id)
            ax.fill_between(
                np.arange(case.steps), mean - 1.96 * se, mean + 1.96 * se, alpha=0.15
            )
    for group_size, ax in zip(group_sizes, axes[0], strict=True):
        ax.axhline(0, color="gray", linewidth=1)
        ax.set_xlabel("Update / age (intervals 64/256)")
        ax.set_ylabel("Current population-guide linear gain")
        ax.set_title(f"{group_size} groups, first two coordinates rotate")
        ax.legend(fontsize=7)
    folder = ROOT / "research/phase02/figures"
    folder.mkdir(exist_ok=True)
    fig.savefig(figure_path)
    plt.close(fig)
    summary = {
        "scope": (
            "secondary diagnostics on existing confirmation draws; not new confirmation"
        ),
        "raw_path": str(raw_path.relative_to(ROOT)),
        "raw_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        "rows": len(cases) * count,
        "conditions": result,
    }
    if "primary_followup_prediction" in protocol:
        cutoff = norm.isf(0.05 / 34)
        reversing = []
        for identity in ("drift-g2-w0.2-k64", "drift-g2-w0.2-k256"):
            row = result[identity]
            upper = np.array(row["mean_current_gain_per_step"]) + cutoff * np.array(
                row["mcse_per_step"]
            )
            reversing.append(bool(np.any(upper[8:25] < 0)))
        fresh = result["drift-g2-w0.2-k1"]
        summary["registered_stale_reversal_passed"] = bool(
            any(reversing) and np.mean(fresh["mean_current_gain_per_step"]) > 0
        )
        summary["bonferroni_z_34_checks"] = float(cutoff)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print({k: v["steps_with_negative_mean_upper95"] for k, v in result.items()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol", type=Path, default=ROOT / "research/phase02/protocol.json"
    )
    parser.add_argument(
        "--stream", choices=["discovery", "confirmation"], default="confirmation"
    )
    parser.add_argument(
        "--raw",
        type=Path,
        default=ROOT / "artifacts/phase02/age-diagnostics-v1.jsonl.gz",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=ROOT / "research/phase02/age-diagnostics-v1.json",
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=ROOT / "research/phase02/figures/stale-current-gain.svg",
    )
    args = parser.parse_args()
    run(
        args.protocol.resolve(),
        args.stream,
        args.raw.resolve(),
        args.summary.resolve(),
        args.figure.resolve(),
    )
