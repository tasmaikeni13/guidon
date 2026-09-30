# Phase 04 — Training scripts and kernels for v4-32

**Method dependency:** GUIDON v0.2 uses $\tau=\rho_t/\max(\|q\|_\infty,0.01)$ and exact-idempotent numerical reprojection; the old v0.1 plans were invalidated by Phase 01. Require the current hash-verified prerequisite gates.

**Entry:** 01/03 valid; user has started this phase and made the existing pod available.
Do not provision another pod. A v4-32 slice has 16 physical chips with topology
2x2x4 ([Google specification](https://docs.cloud.google.com/tpu/docs/v4)). Discover
JAX's actual logical devices after distributed initialization; do not equate the
name's 32 TensorCores with 32 independent chips or assume one host.

Implement `guidon.train`, `guidon.checkpoint`, `guidon.evaluate`, a dataset loader,
and a checked TPU launcher. These are future deliverables, not existing commands.
Use the proposed CLI contract:

```text
python -m guidon.train --config CONFIG --optimizer adamw|guidon|adamw_plus_guidance
    --seed SEED --manifest MANIFEST --run-dir RUN_DIR [--resume CHECKPOINT]
```

Provide explicit syntax and dry-run modes before declaring that command available.
Implement the registered 125M GPT-2 decoder; count its actual parameters. Import
Pythia's GPT-NeoX weights/tokenizer with exact architecture and test logits against
the source implementation before continued training. Use identical model/loss
code, gradient clipping, schedule, masks, batches, and checkpoint policy across arms.
FP32 parameters/moments, bfloat16 matrix compute, numerically stable FP32 NLL.

Install a matched TPU stack on every worker: the current CPU prototype pins JAX
0.6.2; its TPU extra resolves libtpu 0.0.17.*. Consult
[official installation](https://docs.jax.dev/en/latest/installation.html) and record
the actually working wheel versions. If the pod image needs a newer stack, update
the lock, rerun numerical/sharding/logit parity, and invalidate affected efficiency
evidence. Initialize every process before devices, use a global `Mesh` with
`NamedSharding`, and select a measured data/model mesh from discovered hardware.

Current `jax_optimizer.py` is the readable XLA path for both optimizers. Implement
and benchmark an optimized XLA/Pallas path for **both**, with float32 fused moment
update, bias correction, coefficient partial sums, and scaled parameter update.
Preserve the optimization barrier on realized weights and the conservative guide-sign certificate. Respect the two-pass dependency: current $u$ is needed for $a,c$, and coefficients
are needed before applying $w$. Prove/test tile padding, masks, group boundaries,
ties, and reduction semantics. A global parameter element is counted once; do not
sum already replicated, globally averaged gradients a second time across replicas.
Use stable global reduction and fuse operations only while preserving the method.
Keep an unfused correctness oracle. Custom kernels earn adoption only by measured
end-to-end improvement; optimized XLA is an acceptable final kernel path if Pallas
does not help. Document the tried implementation and measured rejection.

Compile ordinary and probe step graphs separately. The host/data scheduler supplies
guidance only when `probe_due`; assert `schedule_ok` and `numerics_ok` and log certificate/fallback
statistics. The guide backward pass belongs in total cost and peak memory. Do not
benchmark only an optimizer microkernel and call that training speed.

Run shape/precision/sharding parity, tiny real-data overfit, multi-host collective
and loss accounting tests, and an interrupt/resume identity test. Checkpoints include
parameters, moments, guide coefficients/age, all RNGs, token/probe cursors, schedule,
config/data hashes, and completed loss-token count; write atomically across workers.

**Outputs:** executable scripts/launchers, optimized baseline and candidate kernels,
tests, environment lock, HLO/profiler traces, per-step/probe timing, peak memory,
communication/FLOP estimates, and gate 04. Profile 200+ synchronized steady-state
updates with I/O/checkpoint overhead included in a separate end-to-end measure.
Target at most 1.10x wall time and compute; if missed, diagnose traffic, graph
recompilation, collectives, or probe cadence, revise theory/config as necessary,
propagate invalidation, and remeasure. CPU parity alone cannot pass this gate.
