"""Regenerate statistical summaries and standalone figures from immutable raw data."""

from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from research.experiments.simulate import ROOT  # noqa: E402


def mean_se(x) -> dict:
    x = np.asarray(x, dtype=float)
    mean = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(len(x)))
    return {
        "mean": mean,
        "mcse": se,
        "lower95": mean - 1.96 * se,
        "upper95": mean + 1.96 * se,
    }


def analyze(stream: str) -> dict:
    groups = defaultdict(list)
    raw_path = ROOT / f"artifacts/phase02/{stream}-v1.jsonl.gz"
    with gzip.open(raw_path, "rt") as handle:
        for line in handle:
            row = json.loads(line)
            groups[row["condition"]].append(row)
    summary = {}
    for identity, rows in sorted(groups.items()):
        evaluation = np.asarray([row["evaluation_loss"] for row in rows])
        losses = {
            arm: mean_se(evaluation[:, i])
            for i, arm in enumerate(("adamw", "guidon", "adamw_plus_guidance"))
        }
        summary[identity] = {
            "n": len(rows),
            "evaluation_loss": losses,
            "adamw_minus_guidon": mean_se(evaluation[:, 0] - evaluation[:, 1]),
            "control_minus_guidon": mean_se(evaluation[:, 2] - evaluation[:, 1]),
            **{
                key: mean_se([r[key] for r in rows])
                for key in (
                    "current_guide_linear_gain",
                    "evaluation_linear_gain",
                    "actual_guide_advantage",
                    "actual_train_advantage",
                    "fallback_fraction",
                    "coefficient_time_variance",
                    "update_norm_mean",
                )
            },
            "max_relative_neutrality": max(r["max_relative_neutrality"] for r in rows),
            "min_proxy_gain": min(r["min_proxy_gain"] for r in rows),
            "min_fresh_gain": min(r["min_fresh_gain"] for r in rows),
            "all_adamw_limit_bitwise": all(r["adamw_limit_bitwise"] for r in rows),
        }
    destination = ROOT / f"research/phase02/{stream}-v1-summary.json"
    destination.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def decisions(summary: dict, protocol: dict) -> dict:
    intended = {}
    for identity in protocol["intended_regime_ids"]:
        x = summary[identity]
        intended[identity] = bool(
            x["current_guide_linear_gain"]["lower95"] > 0
            and x["evaluation_linear_gain"]["lower95"] > 0
        )
    signal = summary["intended"]["current_guide_linear_gain"]
    reverse = summary["control-reverse"]["current_guide_linear_gain"]
    shuffle = summary["control-shuffle"]["current_guide_linear_gain"]
    corruption = bool(
        signal["lower95"] > reverse["upper95"]
        and signal["lower95"] > shuffle["upper95"]
        and reverse["upper95"] < 0
    )
    fresh = summary["rotate-0.001-1"]["current_guide_linear_gain"]
    drift = {
        key: summary[key]["current_guide_linear_gain"]
        for key in ("rotate-0.2-64", "rotate-0.2-256")
    }
    drift_detected = any(x["upper95"] < fresh["lower95"] for x in drift.values())
    result = {
        "intended_transfer": intended,
        "corruption_prediction": corruption,
        "controlled_drift_detected": drift_detected,
        "collinearity_and_noninterference": "checked by separate scalar verifier",
        "llm_superiority": "unmeasured",
    }
    result["local_mechanism_passed"] = (
        all(intended.values()) and corruption and drift_detected
    )
    return result


def plots(summary: dict) -> None:
    folder = ROOT / "research/phase02/figures"
    folder.mkdir(exist_ok=True)
    identities = [
        "intended",
        "control-reverse",
        "control-shuffle",
        "target-mismatch-1",
        "rotate-0.001-1",
        "rotate-0.2-64",
        "spurious-regression-0",
        "spurious-classification-0",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7), layout="constrained")
    for ax, metric, label in zip(
        axes,
        ("adamw_minus_guidon", "control_minus_guidon"),
        (
            "AdamW − GUIDON evaluation loss",
            "Information control − GUIDON evaluation loss",
        ),
        strict=True,
    ):
        means = [summary[k][metric]["mean"] for k in identities]
        errors = [1.96 * summary[k][metric]["mcse"] for k in identities]
        ax.errorbar(means, np.arange(len(identities)), xerr=errors, fmt="o", capsize=3)
        ax.axvline(0, color="gray", linewidth=1)
        ax.set_yticks(np.arange(len(identities)), identities)
        ax.set_xlabel(label + "\npointwise 95% Monte Carlo intervals")
        ax.invert_yaxis()
    fig.suptitle("Synthetic confirmation; loss units depend on problem, not LLM NLL")
    fig.savefig(folder / "synthetic-comparisons.svg")
    fig.savefig(folder / "synthetic-comparisons.png", dpi=160)
    plt.close(fig)
    identities = [
        "intended",
        "control-reverse",
        "control-shuffle",
        "control-collinear",
        "rotate-0.001-1",
        "rotate-0.2-64",
        "rotate-0.2-256",
    ]
    fig, ax = plt.subplots(figsize=(8, 4.7), layout="constrained")
    ax.bar(
        np.arange(len(identities)),
        [summary[k]["current_guide_linear_gain"]["mean"] for k in identities],
    )
    ax.set_xticks(np.arange(len(identities)), identities, rotation=25, ha="right")
    ax.axhline(0, color="gray", linewidth=1)
    ax.set_ylabel("Mean current population-guide linear gain per step")
    ax.set_title("Signal corruption and drift distinguish local mechanism explanations")
    fig.savefig(folder / "mechanisms.svg")
    plt.close(fig)
    power = json.loads((ROOT / "research/phase02/power-v1.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for distribution in ("normal", "t3", "skew_lognormal"):
        cells = sorted(
            [
                r
                for r in power["cells"]
                if r["distribution"] == distribution and r["effect"] == 0.005
            ],
            key=lambda r: r["sigma"],
        )
        sigma = [r["sigma"] for r in cells]
        axes[0].plot(
            sigma,
            [r["two_sided_rejection_probability"] for r in cells],
            "o-",
            label=distribution,
        )
        axes[1].plot(
            sigma, [r["mean_interval_width"] for r in cells], "o-", label=distribution
        )
    axes[0].set_ylabel("Two-sided rejection probability")
    axes[1].set_ylabel("Mean 95% paired t interval width (nats/token)")
    for ax in axes:
        ax.set_xlabel("Paired seed-difference SD (nats/token)")
        ax.legend()
    fig.suptitle("Three independent seed pairs; true difference 0.005 nats/token")
    fig.savefig(folder / "three-pair-power.svg")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stream", choices=["discovery", "confirmation"], required=True
    )
    args = parser.parse_args()
    summary = analyze(args.stream)
    protocol = json.loads((ROOT / "research/phase02/protocol.json").read_text())
    result = decisions(summary, protocol)
    (ROOT / f"research/phase02/{args.stream}-v1-decisions.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    if args.stream == "confirmation":
        plots(summary)
    print(json.dumps(result, indent=2))
