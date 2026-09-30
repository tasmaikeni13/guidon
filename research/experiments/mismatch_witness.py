"""All-arm opposite-evaluation witness, executed through the public reference."""

from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import replace

import numpy as np

from guidon.reference import Config, init, step
from research.experiments.simulate import ROOT


def run() -> None:
    protocol_path = ROOT / "research/phase02/mismatch-protocol.json"
    protocol = json.loads(protocol_path.read_text())
    raw_path = ROOT / "artifacts/phase02/opposite-evaluation-v1.jsonl.gz"
    if raw_path.exists():
        raise FileExistsError("Mismatch witness raw is immutable")
    counts = {}
    with (
        raw_path.open("wb") as handle,
        gzip.GzipFile(fileobj=handle, mode="wb", mtime=0, filename="") as raw,
    ):
        for stream in ("discovery", "confirmation"):
            counts[stream] = 0
            for draw in range(protocol["draws_per_stream"]):
                rng = np.random.default_rng(
                    np.random.SeedSequence([protocol[f"{stream}_seed"], draw, 0])
                )
                eval_rng = np.random.default_rng(
                    np.random.SeedSequence([protocol[f"{stream}_seed"], draw, 3])
                )
                initial = float(rng.uniform(0.5, 1.5))
                target = float(eval_rng.uniform(3, 4))
                parameters = [(np.array([initial]),) for _ in range(3)]
                states = [init(p) for p in parameters]
                config = Config(
                    learning_rate=0.01,
                    weight_decay=0,
                    radius=0.15,
                    guide_warmup=0,
                    guide_interval=1,
                )
                residual = 0.0
                for _index in range(protocol["steps"]):
                    for arm in range(3):
                        gradient = (parameters[arm][0].copy(),)
                        guide = gradient if arm == 1 else None
                        cfg = config if arm == 1 else replace(config, radius=0)
                        parameters[arm], states[arm], stats = step(
                            parameters[arm],
                            gradient,
                            states[arm],
                            cfg,
                            guide_gradients=guide,
                        )
                        residual = max(residual, abs(stats["training_residual"]))
                final = np.array([p[0][0] for p in parameters])
                train_before = 0.5 * initial**2
                train_after = 0.5 * final**2
                eval_before = 0.5 * (initial - target) ** 2
                eval_after = 0.5 * (final - target) ** 2
                assert np.all(train_after < train_before)
                assert np.all(eval_after > eval_before)
                # Evaluation target is used only after all updates. Parameter paths
                # are a pure function of initialization/train/guide here; P9 applies.
                row = {
                    "stream": stream,
                    "draw": draw,
                    "initial": initial,
                    "evaluation_target": target,
                    "final_parameters": final.tolist(),
                    "train_and_guide_before": train_before,
                    "train_and_guide_after": train_after.tolist(),
                    "evaluation_before": eval_before,
                    "evaluation_after": eval_after.tolist(),
                    "max_training_residual": residual,
                }
                raw.write((json.dumps(row, sort_keys=True) + "\n").encode())
                counts[stream] += 1
    # Separate raw-file pass checks objective formulas and every inequality.
    with gzip.open(raw_path, "rt") as handle:
        for line in handle:
            row = json.loads(line)
            final = np.array(row["final_parameters"])
            np.testing.assert_allclose(0.5 * final**2, row["train_and_guide_after"])
            np.testing.assert_allclose(
                0.5 * (final - row["evaluation_target"]) ** 2, row["evaluation_after"]
            )
            assert all(x > row["evaluation_before"] for x in row["evaluation_after"])
    result = {
        "passed": True,
        "all_arms_worsen_independent_evaluation": True,
        "counts": counts,
        "raw_path": str(raw_path.relative_to(ROOT)),
        "raw_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        "scope": (
            "counterexample to universal transfer; not evidence about "
            "representative guidance or LLM superiority"
        ),
    }
    (ROOT / "research/phase02/opposite-evaluation-v1.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(
        f"Opposite-evaluation witness verified for all three arms on "
        f"{sum(counts.values())} independent draws"
    )


if __name__ == "__main__":
    run()
