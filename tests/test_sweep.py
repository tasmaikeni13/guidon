import copy
import json
from pathlib import Path

import pytest

from guidon.sweep import audit, register

ROOT = Path(__file__).resolve().parents[1]


def plan_and_base():
    plan = json.loads((ROOT / "configs/sweep_125m_600m.json").read_text())
    base = json.loads((ROOT / "configs/pretrain_125m.json").read_text())
    return plan, base


def test_sweep_exact_user_budget_and_unchanged_confirmation():
    plan, base = plan_and_base()
    result = audit(plan, base)
    assert result["model_parameters"] == 125_226_240
    assert result["training_loss_tokens_per_run"] == 600_000_000
    assert result["updates_per_run"] == 2500
    assert result["positive_labels_per_update"] * 2500 == 600_000_000
    assert len(result["seeds"]) == 3
    assert result["required_runs"] == 108
    assert result["guide_probes_per_guidon_or_control_run"] == 38
    assert result["confirmation_training_loss_tokens"] == 2_500_000_000
    assert result["confirmation_seeds"] == [42, 43, 44]


@pytest.mark.parametrize(
    "key,value",
    [
        ("training_loss_tokens", 32_000_000),
        ("updates", 2499),
        ("final_evaluation_locked", False),
    ],
)
def test_sweep_rejects_budget_and_boundary_changes(key, value):
    plan, base = plan_and_base()
    plan[key] = value
    with pytest.raises(ValueError):
        audit(plan, base)


def test_registration_includes_every_arm_trial_and_seed(tmp_path):
    plan_path = ROOT / "configs/sweep_125m_600m.json"
    bundle = register(plan_path, tmp_path / "registered")
    assert len({row["run_id"] for row in bundle["runs"]}) == 108
    for arm in ("adamw", "guidon", "adamw_plus_guidance"):
        rows = [r for r in bundle["runs"] if r["optimizer"] == arm]
        assert len(rows) == 36
        assert {r["seed"] for r in rows} == {101, 102, 103}
    for trial in bundle["trials"]:
        configs = [
            json.loads(
                (
                    tmp_path / "registered" / f"{trial['trial_id']}-{arm}.json"
                ).read_text()
            )
            for arm in ("adamw", "guidon", "adamw_plus_guidance")
        ]
        settings = [copy.deepcopy(c["optimizer_defaults"]) for c in configs]
        assert settings[0] == settings[1] == settings[2]
        assert all(c["loss_tokens_per_update"] == 240_000 for c in configs)
    with pytest.raises(FileExistsError):
        register(plan_path, tmp_path / "registered")
