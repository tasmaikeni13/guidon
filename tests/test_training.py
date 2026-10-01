import json
from pathlib import Path

import jax
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from guidon import checkpoint
from guidon.model import ModelConfig, init, parameter_count
from guidon.packing import TapeWriter, write_capability
from guidon.train import run


def inputs(tmp_path):
    model = dict(
        family="gpt2_decoder",
        layers=1,
        hidden_size=8,
        attention_heads=2,
        mlp_size=32,
        vocab_size=16,
        max_position_embeddings=16,
        tie_embeddings=True,
        dropout=0,
        bias=True,
    )
    model["expected_parameters"] = parameter_count(
        init(jax.random.PRNGKey(0), ModelConfig.from_scratch_protocol(model))
    )
    config = dict(
        model=model,
        seeds=[101],
        cohort="pilot",
        training_loss_tokens=128,
        loss_tokens_per_update=32,
        global_batch_sequences=4,
        sequence_length=8,
        hardware=dict(model_compute_dtype="float32"),
        schedule=dict(warmup_fraction=0.02, minimum_learning_rate_fraction=0.1),
        optimizer_defaults=dict(
            learning_rate=0.01,
            beta1=0.9,
            beta2=0.95,
            epsilon=1e-8,
            weight_decay=0.01,
            guidon_radius=0.15,
            guide_interval=2,
            guide_warmup_updates=1,
            guide_batch_sequences=2,
            neutrality_tolerance=1e-5,
            signal_floor=0.01,
            global_gradient_clip_norm=1.0,
        ),
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    for role in ("train", "guidance"):
        writer = TapeWriter(
            tmp_path / role, role, "pilot", {"revision": "fixture"}, 0, shard_tokens=11
        )
        writer.add(dict(doc_id=role + "-a", split=role), [1, 2, 3, 4] * 100)
        writer.finish()
    capability = tmp_path / "training.json"
    write_capability(
        capability,
        {r: tmp_path / r / "manifest.json" for r in ("train", "guidance")},
        process_role="training",
        cohort="pilot",
        preparation_sha256="fixture",
    )
    return config_path, capability


@pytest.mark.parametrize("optimizer", ["adamw", "guidon", "adamw_plus_guidance"])
def test_interrupt_resume_exact_global_state_rng_and_loss_cursors(
    tmp_path, optimizer, monkeypatch
):
    config, capability = inputs(tmp_path)
    whole = run(
        config,
        optimizer,
        101,
        capability,
        tmp_path / "whole",
        purpose="verification",
        max_updates=4,
    )
    first = run(
        config,
        optimizer,
        101,
        capability,
        tmp_path / "resumed",
        purpose="verification",
        max_updates=4,
        interrupt_after_updates=2,
    )
    resumed = run(
        config,
        optimizer,
        101,
        capability,
        tmp_path / "resumed",
        purpose="verification",
        max_updates=4,
        resume=Path(first["checkpoint"]),
    )
    a = json.loads((Path(whole["checkpoint"]) / "commit.json").read_text())
    b = json.loads((Path(resumed["checkpoint"]) / "commit.json").read_text())
    assert a["logical_state_sha256"] == b["logical_state_sha256"]
    assert whole["cursors"] == resumed["cursors"]
    assert whole["cursors"]["completed_loss_tokens"] == 128
    expected_guide = 0 if optimizer == "adamw" else 32
    assert whole["cursors"]["guidance"] == expected_guide
    assert resumed["complete_training_budget"]
    assert not first["complete_training_budget"]
    assert first["interrupted"]
    if optimizer == "adamw":
        monkeypatch.setattr(
            "guidon.train.implementation_identity", lambda: {"train.py": "changed"}
        )
        with pytest.raises(ValueError, match="identity differs"):
            run(
                config,
                optimizer,
                101,
                capability,
                tmp_path / "resumed",
                purpose="verification",
                max_updates=4,
                resume=Path(first["checkpoint"]),
            )


def test_checkpoint_rejects_corrupt_or_changed_identity(tmp_path):
    from guidon import jax_optimizer as jo

    placement = NamedSharding(
        Mesh(np.asarray(jax.devices()), ("data",)), PartitionSpec()
    )
    params = (jax.device_put(np.ones(3, np.float32), placement),)
    state = jo.init(params, (0,), (True,))
    identity = {"config": "original", "seed": 101}
    saved = checkpoint.save(
        tmp_path,
        params,
        state,
        jax.random.PRNGKey(101),
        dict(train=0, guidance=0, completed_loss_tokens=0),
        identity,
    )
    with pytest.raises(ValueError, match="identity differs"):
        checkpoint.load(saved, {"config": "different", "seed": 101}, placement)
    (saved / "state.npz").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="corrupt"):
        checkpoint.load(saved, identity, placement)


def test_direct_training_cannot_skip_registered_gates(tmp_path):
    config, capability = inputs(tmp_path)
    with pytest.raises(ValueError, match="registered run bundle"):
        run(config, "adamw", 101, capability, tmp_path / "unauthorized", max_updates=4)


def test_final_payload_cannot_influence_full_training_trajectory(tmp_path, monkeypatch):
    config, capability = inputs(tmp_path)
    final = tmp_path / "sealed-evaluation"
    final.mkdir()
    payload = final / "unavailable-labels.json"
    payload.write_text(json.dumps({"labels": [1, 2, 3]}))
    original_open = Path.open

    def guarded_open(path, *args, **kwargs):
        if path.resolve().is_relative_to(final.resolve()):
            raise AssertionError("Training attempted to open final evaluation")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    first = run(
        config,
        "guidon",
        101,
        capability,
        tmp_path / "first",
        purpose="verification",
        max_updates=4,
    )
    with original_open(payload, "w") as handle:
        json.dump({"labels": [15] * 1000}, handle)
    second = run(
        config,
        "guidon",
        101,
        capability,
        tmp_path / "second",
        purpose="verification",
        max_updates=4,
    )
    commits = [
        json.loads((Path(result["checkpoint"]) / "commit.json").read_text())
        for result in (first, second)
    ]
    assert commits[0]["logical_state_sha256"] == commits[1]["logical_state_sha256"]


def test_finite_guidance_loss_with_nonfinite_gradient_aborts(tmp_path, monkeypatch):
    import jax.numpy as jnp

    from guidon import jax_optimizer as jo
    from guidon.model import optimizer_layout
    from guidon.train import make_graph

    config_path, _ = inputs(tmp_path)
    config = json.loads(config_path.read_text())
    model = ModelConfig.from_scratch_protocol(config["model"])
    params = init(jax.random.PRNGKey(101), model)

    @jax.custom_jvp
    def finite_primal(value, bad):
        return value

    @finite_primal.defjvp
    def derivative(primals, tangents):
        value, bad = primals
        value_dot, _ = tangents
        return value, value_dot * jnp.where(bad, jnp.nan, 1.0)

    def objective(p, batch, _config):
        value = sum(jnp.sum(x) for x in jax.tree.leaves(p))
        return finite_primal(value, batch["bad"]), jnp.asarray(1.0)

    monkeypatch.setattr("guidon.model.loss_sums", objective)
    groups, masks = optimizer_layout(params)
    state = jo.init(params, groups, masks)._replace(count=jnp.asarray(1))
    graph = make_graph(model, params, config, "guidon", has_guide=True)
    _, _, metrics = graph(
        params,
        state,
        {"bad": jnp.asarray(False)},
        {"bad": jnp.asarray(True)},
        jnp.asarray(0.01),
    )
    assert np.isfinite(metrics["guide_nll"])
    assert not bool(metrics["guide_gradients_finite"])
    assert not bool(metrics["numerics_ok"])
