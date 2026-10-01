"""Register the 125M / 600M-token / 2,500-update three-seed Phase 05 sweep.

This command writes a reviewable run matrix and configs. It never launches jobs
or opens evaluation. Every arm has twelve configurations and all three seeds.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np

from guidon.artifacts import atomic_json, object_sha256, sha256_file
from guidon.protocol import audit as audit_confirmation

ARMS = ("adamw", "guidon", "adamw_plus_guidance")
PILOT_SEEDS = [101, 102, 103]


def audit(plan: dict[str, Any], base: dict[str, Any]) -> dict[str, Any]:
    audit_confirmation(base)
    if plan["phase"] != "05" or plan["cohort"] != "pilot":
        raise ValueError("Phase 05 sweep must use the pilot cohort")
    if plan["seeds"] != PILOT_SEEDS:
        raise ValueError("The three preregistered pilot seeds are 101, 102, 103")
    if plan["confirmation_seeds"] != [42, 43, 44]:
        raise ValueError("Confirmation retains seeds 42, 43, 44")
    if plan["arms"] != list(ARMS):
        raise ValueError("Both optimizers and the equal-information arm are required")
    if plan["training_loss_tokens"] != 600_000_000 or plan["updates"] != 2500:
        raise ValueError("Every sweep run requires 600M loss tokens and 2,500 updates")
    labels = plan["loss_tokens_per_update"]
    if labels != 240_000 or labels * plan["updates"] != plan["training_loss_tokens"]:
        raise ValueError("Sweep arithmetic must register 240,000 labels per update")
    capacity = plan["global_batch_sequences"] * plan["sequence_length"]
    if labels > capacity or plan["sequence_length"] > base["sequence_length"]:
        raise ValueError("The registered model/batch cannot accommodate the sweep")
    if (
        plan["dataset"] != base["data"]["dataset"]
        or plan["dataset_revision"] != base["data"]["revision"]
    ):
        raise ValueError("Sweep must use the pinned FineWeb-Edu corpus")
    if plan["model_parameters"] != 125_226_240:
        raise ValueError("Sweep must retain the actual 125M decoder")
    if plan["configurations_per_arm"] != 12:
        raise ValueError("Equal twelve-configuration tuning budgets are registered")
    if not plan["final_evaluation_locked"] or plan["selection_role"] != "development":
        raise ValueError("Sweep selection may use only pilot development evidence")
    if plan["signal_floor"] != 0.01:
        raise ValueError("The checked v0.2 signal floor is fixed")
    interval, warmup = plan["guide_interval"], plan["guide_warmup_updates"]
    if interval != 64 or warmup != 128:
        raise ValueError("Main search uses the registered shared information cadence")
    probes = len(range(warmup, plan["updates"], interval))
    guide_tokens = probes * plan["guide_batch_sequences"] * plan["sequence_length"]
    count = len(ARMS) * len(PILOT_SEEDS) * plan["configurations_per_arm"]
    return {
        "phase": "05",
        "model_parameters": 125_226_240,
        "seeds": PILOT_SEEDS,
        "required_runs": count,
        "training_loss_tokens_per_run": 600_000_000,
        "updates_per_run": 2500,
        "positive_labels_per_update": labels,
        "physical_batch_positions": capacity,
        "masked_positions_per_update": capacity - labels,
        "guide_probes_per_guidon_or_control_run": probes,
        "guide_tokens_per_guidon_or_control_run": guide_tokens,
        "total_main_search_loss_tokens": count * 600_000_000,
        "total_additional_search_guidance_tokens": (
            2 * len(PILOT_SEEDS) * plan["configurations_per_arm"] * guide_tokens
        ),
        "executed": False,
        "confirmation_training_loss_tokens": base["training_loss_tokens"],
        "confirmation_seeds": base["seeds"],
    }


def trial_settings(plan: dict[str, Any], base: dict[str, Any]) -> list[dict[str, Any]]:
    """Balanced fixed draws, with the documented default retained as trial zero."""
    rng = np.random.default_rng(plan["trial_generation_seed"])
    count = plan["configurations_per_arm"]
    defaults = base["optimizer_defaults"]
    columns = {
        key: rng.permutation(np.resize(np.asarray(values), count - 1)).tolist()
        for key, values in plan["search_space"].items()
        if key != "guidon_radius"
    }
    radii = rng.permutation(
        np.resize(np.asarray(plan["search_space"]["guidon_radius"]), count - 1)
    ).tolist()
    trials = []
    for index in range(count):
        values = copy.deepcopy(defaults)
        minimum_lr_fraction = base["schedule"]["minimum_learning_rate_fraction"]
        if index:
            for key, entries in columns.items():
                if key == "minimum_learning_rate_fraction":
                    minimum_lr_fraction = entries[index - 1]
                else:
                    values[key] = entries[index - 1]
            values["guidon_radius"] = radii[index - 1]
        values.update(
            guide_interval=plan["guide_interval"],
            guide_warmup_updates=plan["guide_warmup_updates"],
            guide_batch_sequences=plan["guide_batch_sequences"],
        )
        trials.append(
            {
                "trial_id": f"trial-{index:02d}",
                "optimizer_defaults": values,
                "minimum_learning_rate_fraction": minimum_lr_fraction,
            }
        )
    return trials


def register(plan_path: Path, destination: Path) -> dict[str, Any]:
    plan = json.loads(plan_path.read_text())
    base_path = (plan_path.parent / plan["base_confirmation_config"]).resolve()
    base = json.loads(base_path.read_text())
    arithmetic = audit(plan, base)
    trials = trial_settings(plan, base)
    destination.mkdir(parents=True, exist_ok=False)
    rows = []
    for trial in trials:
        for arm in ARMS:
            config = copy.deepcopy(base)
            config.update(
                phase="05",
                regime="hyperparameter_sweep",
                status="registered_not_executed",
                cohort="pilot",
                seeds=PILOT_SEEDS,
                optimizer=arm,
                training_loss_tokens=600_000_000,
                loss_tokens_per_update=240_000,
                expected_updates=2500,
                global_batch_sequences=plan["global_batch_sequences"],
                sequence_length=plan["sequence_length"],
                trial_id=trial["trial_id"],
                sweep_plan_sha256=sha256_file(plan_path),
            )
            config["optimizer_defaults"] = trial["optimizer_defaults"]
            config["schedule"]["minimum_learning_rate_fraction"] = trial[
                "minimum_learning_rate_fraction"
            ]
            config["gates"].update(frozen=False, final_evaluation_locked=True)
            name = f"{trial['trial_id']}-{arm}.json"
            atomic_json(destination / name, config)
            for seed in PILOT_SEEDS:
                rows.append(
                    {
                        "run_id": f"phase05-{trial['trial_id']}-{arm}-seed{seed}",
                        "trial_id": trial["trial_id"],
                        "optimizer": arm,
                        "seed": seed,
                        "config": name,
                        "config_sha256": sha256_file(destination / name),
                        "state": "registered_not_executed",
                    }
                )
    order = np.random.default_rng(plan["run_order_seed"]).permutation(len(rows))
    rows = [rows[i] | {"order": position} for position, i in enumerate(order)]
    bundle = {
        "schema_version": 1,
        "plan_sha256": sha256_file(plan_path),
        "base_confirmation_sha256": sha256_file(base_path),
        "arithmetic": arithmetic,
        "trials": trials,
        "runs": rows,
        "selection": plan["selection"],
        "bundle_identity": object_sha256(rows),
    }
    atomic_json(destination / "matrix.json", bundle)
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--register", type=Path)
    args = parser.parse_args()
    if args.register:
        value = register(args.plan, args.register)["arithmetic"]
    else:
        plan = json.loads(args.plan.read_text())
        base = json.loads(
            (args.plan.parent / plan["base_confirmation_config"]).read_text()
        )
        value = audit(plan, base)
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
