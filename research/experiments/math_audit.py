"""Phase 01 adversarial audit of actual NumPy and compiled JAX steps."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from guidon import jax_optimizer as jo
from guidon.reference import Config, controller
from research.experiments.check_math import check_weights, oracle, self_check

ROOT = Path(__file__).resolve().parents[2]


def run(output: Path, summary_path: Path) -> None:
    protocol_path = ROOT / "research/phase01/protocol.json"
    protocol = json.loads(protocol_path.read_text())
    rng = np.random.default_rng(protocol["seed"])
    rows = []
    for number in protocol["groups"]:
        n = protocol["draws_per_cell"]
        gradients, guides, metadata = [], [], []
        for scale in protocol["gradient_scales"]:
            for perturb in protocol["collinearity_perturbations"]:
                for draw in range(n):
                    g = rng.uniform(0.25, 1, number) * scale
                    h = g + perturb * scale * rng.normal(size=number)
                    gradients.append(g)
                    guides.append(h)
                    metadata.append((scale, perturb, draw))
        gs, hs = np.asarray(gradients), np.asarray(guides)
        for dtype in ("float32", "bfloat16"):
            cfg = Config(learning_rate=1e-3, weight_decay=0, guide_warmup=0)
            ids, mask = tuple(range(number)), (False,) * number
            update = jo.make_step(ids, mask, cfg)

            def single(g, h, number=number, update=update, ids=ids, mask=mask):
                p = tuple(jnp.zeros(1, jnp.float32) for _ in range(number))
                gt = tuple(g[i : i + 1] for i in range(number))
                ht = tuple(h[i : i + 1] for i in range(number))
                result, state, metrics = update(p, gt, jo.init(p, ids, mask), ht)
                return jnp.concatenate(result), jnp.concatenate(state.second), metrics

            interface = jnp.float32 if dtype == "float32" else jnp.bfloat16
            result, second, stats = jax.jit(jax.vmap(single))(
                jnp.asarray(gs, interface), jnp.asarray(hs, interface)
            )
            result, second = np.asarray(result), np.asarray(second)
            stats = jax.tree.map(np.asarray, stats)
            gq = np.asarray(jnp.asarray(gs, interface), dtype=np.float64)
            hq = np.asarray(jnp.asarray(hs, interface), dtype=np.float64)
            for i, (scale, perturb, draw) in enumerate(metadata):
                u64 = gq[i] / (np.abs(gq[i]) + cfg.epsilon)
                a64, c64 = gq[i] * u64, hq[i] * u64
                cunit = c64 / np.max(np.abs(c64)) if np.max(np.abs(c64)) else c64
                w64, fallback64 = controller(
                    a64, cunit, cfg.radius, cfg.neutrality_tolerance
                )
                w32 = stats["weights"][i]
                # Check against actual FP32 moment arithmetic, not ideal float64 a.
                gf = gq[i].astype(np.float32)
                with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                    mf = (1 - cfg.beta1) * gf
                    vf = (1 - cfg.beta2) * gf**2
                    uf = (mf / np.float32(1 - cfg.beta1)) / (
                        np.sqrt(vf / np.float32(1 - cfg.beta2)) + cfg.epsilon
                    )
                af = gf.astype(np.float64) * uf.astype(np.float64)
                cf = hq[i] * uf.astype(np.float64)
                cf = cf / np.max(np.abs(cf)) if np.max(np.abs(cf)) else cf
                af = stats["train_coefficients"][i].astype(np.float64)
                cf = stats["guide_coefficients"][i].astype(np.float64)
                checked = check_weights(
                    af, cf, w32, cfg.radius, cfg.neutrality_tolerance
                )
                expected = oracle(a64, cunit, cfg.radius)
                an = a64 / np.max(np.abs(a64)) if np.max(np.abs(a64)) else a64
                q = cunit - an * (an @ cunit) / (an @ an) if an @ an else cunit
                rows.append(
                    {
                        "groups": number,
                        "dtype": dtype,
                        "scale": scale,
                        "perturbation": perturb,
                        "draw": draw,
                        "fallback_fp64": bool(fallback64),
                        "fallback_jax": bool(stats["fallback"][i]),
                        "numerics_ok": bool(stats["numerics_ok"][i]),
                        "a": af.tolist(),
                        "c": cf.tolist(),
                        "weights": w32.tolist(),
                        "moments_finite": bool(np.isfinite(second[i]).all()),
                        "parameters_finite": bool(np.isfinite(result[i]).all()),
                        "schedule_ok": bool(stats["schedule_ok"][i]),
                        "weight_difference": float(np.max(np.abs(w32 - w64))),
                        "oracle_difference": float(np.max(np.abs(expected - w64))),
                        "projected_signal": float(np.max(np.abs(q))),
                        "parameter_difference": float(
                            np.max(np.abs(result[i] + cfg.learning_rate * w64 * u64))
                        ),
                        "checked": checked,
                    }
                )
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Immutable raw output exists: {output}")
    with (
        output.open("wb") as handle,
        gzip.GzipFile(fileobj=handle, mode="wb", mtime=0, filename="") as raw,
    ):
        for row in rows:
            raw.write(
                (json.dumps(row, sort_keys=True, allow_nan=False) + "\n").encode()
            )
    resolved = [
        r
        for r in rows
        if r["scale"] <= 1e15
        and r["scale"] >= 1e-6
        and r["projected_signal"] > protocol["resolved_projected_signal"]
    ]
    groups = {}
    for dtype in ("float32", "bfloat16"):
        selected = [r for r in resolved if r["dtype"] == dtype]
        groups[dtype] = {
            "n": len(selected),
            "fallback_fraction": float(np.mean([r["fallback_jax"] for r in selected])),
            "max_relative_neutrality": max(
                r["checked"]["relative_neutrality"] for r in selected
            ),
            "max_weight_difference": max(r["weight_difference"] for r in selected),
            "max_parameter_difference": max(
                r["parameter_difference"] for r in selected
            ),
            "invariant_failures": sum(not r["checked"]["passed"] for r in selected),
        }
    summary = {
        "version": protocol["version"],
        "rows": len(rows),
        "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        "raw_path": str(output.relative_to(ROOT)),
        "raw_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "environment": jo.environment(),
        "checker_self_test": self_check(),
        "resolved": groups,
        "all_invariant_failures": sum(not r["checked"]["passed"] for r in rows),
        "overflow_moment_cases": sum(not r["moments_finite"] for r in rows),
        "near_collinear_fallbacks": sum(
            r["fallback_jax"]
            for r in rows
            if r["projected_signal"] <= protocol["resolved_projected_signal"]
        ),
        "near_collinear_large_discrepancies": sum(
            r["weight_difference"] > 2e-4
            for r in rows
            if r["projected_signal"] <= protocol["resolved_projected_signal"]
        ),
        "nullspace_dimensions": {str(b): b - 1 for b in protocol["groups"]},
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw", type=Path, default=ROOT / "artifacts/phase01/math-v3.jsonl.gz"
    )
    parser.add_argument(
        "--summary", type=Path, default=ROOT / "research/phase01/math-v3.json"
    )
    args = parser.parse_args()
    run(args.raw.resolve(), args.summary.resolve())
