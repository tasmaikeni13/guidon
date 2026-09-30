import json
from pathlib import Path

import pytest

from guidon.protocol import audit

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "filename,budget,probes,partial",
    [
        ("pretrain_125m.json", 2_500_000_000, 73, 194_816),
        ("continued_pythia160m.json", 1_000_000_000, 28, 182_784),
    ],
)
def test_planned_protocol_matches_required_budgets(filename, budget, probes, partial):
    config = json.loads((ROOT / "configs" / filename).read_text())
    result = audit(config)
    assert result["training_loss_tokens_per_run"] == budget
    assert result["guidance_probes"] == probes
    assert result["final_partial_loss_tokens"] == partial
    assert result["full_updates"] * result["batch_loss_tokens"] + partial == budget
    assert result["required_primary_runs"] == 6
    assert not result["ready_for_confirmation"]


def test_reusing_guidance_invalidates_the_protocol():
    config = json.loads((ROOT / "configs" / "pretrain_125m.json").read_text())
    config["data"]["guidance_reuse"] = True
    with pytest.raises(ValueError, match="recycled"):
        audit(config)


def test_shared_sources_keep_one_global_role_assignment_seed():
    scratch = json.loads((ROOT / "configs" / "pretrain_125m.json").read_text())
    continued = json.loads((ROOT / "configs" / "continued_pythia160m.json").read_text())
    assert scratch["data"]["split_seed"] == continued["data"]["split_seed"]
