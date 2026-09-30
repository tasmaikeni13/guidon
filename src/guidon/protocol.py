"""Audit the planned run matrix and exact token/probe arithmetic, without training."""

import argparse
import json
from pathlib import Path


def audit(config: dict) -> dict:
    if config["seeds"] != [42, 43, 44]:
        raise ValueError("The registered confirmation seeds are 42, 43, 44")
    if config["optimizers"] != ["adamw", "guidon"]:
        raise ValueError("AdamW and GUIDON are the registered optimizers")
    if config["required_information_control"] != "adamw_plus_guidance":
        raise ValueError("The extra-information AdamW control is required")
    if config["data"]["guidance_reuse"]:
        raise ValueError("Fresh guidance may not be recycled")
    if not config["data"]["group_before_pack"]:
        raise ValueError("Documents must be grouped before sequence packing")
    if not config["gates"]["final_evaluation_locked"]:
        raise ValueError("Final evaluation must be sealed")
    tokens = config["training_loss_tokens"]
    batch_tokens = config["global_batch_sequences"] * config["sequence_length"]
    if tokens <= 0 or batch_tokens <= 0:
        raise ValueError("Token budgets must be positive")
    full, remainder = divmod(tokens, batch_tokens)
    updates = full + int(remainder > 0)
    settings = config["optimizer_defaults"]
    interval, warmup = settings["guide_interval"], settings["guide_warmup_updates"]
    if interval < 1 or warmup < 0:
        raise ValueError("Invalid probe schedule")
    probes = len(range(warmup, updates, interval))
    guide_tokens = (
        probes * settings["guide_batch_sequences"] * config["sequence_length"]
    )
    if guide_tokens > config["data"]["minimum_guidance_tokens"]:
        raise ValueError("The guidance reservation cannot support a single fresh pass")
    if config["regime"] == "from_scratch":
        if tokens != 2_500_000_000:
            raise ValueError("Scratch confirmation requires exactly 2.5B loss tokens")
        model = config["model"]
        h = model["hidden_size"]
        if model["mlp_size"] != 4 * h or model["attention_heads"] != 12:
            raise ValueError(
                "The parameter formula requires the registered GPT-2 layout"
            )
        expected = (
            model["vocab_size"] * h
            + model["max_position_embeddings"] * h
            + model["layers"] * (12 * h * h + 13 * h)
            + 2 * h
        )
        if expected != model["expected_parameters"] or expected != 125_226_240:
            raise ValueError("The registered tied GPT-2 architecture must be 125M")
    return {
        "regime": config["regime"],
        "status": config["status"],
        "required_primary_runs": 6,
        "required_information_control_runs": 3,
        "training_loss_tokens_per_run": tokens,
        "batch_loss_tokens": batch_tokens,
        "full_updates": full,
        "final_partial_loss_tokens": remainder,
        "updates": updates,
        "guidance_probes": probes,
        "additional_guidance_tokens_per_guidon_or_control_run": guide_tokens,
        "guide_token_fraction": guide_tokens / tokens,
        "ready_for_confirmation": bool(
            config["data"]["manifest_ready"] and config["gates"]["frozen"]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(json.loads(args.config.read_text())), indent=2))


if __name__ == "__main__":
    main()
