# Foundation verification

Verified locally on 2026-09-30, Linux x86_64, Python 3.10.12, NumPy 2.2.6,
JAX/jaxlib 0.6.2, Optax 0.2.5, pytest 8.4.2, Ruff 0.11.13. Exact resolved dependencies
are in `uv.lock`. This environment has CPU devices; no TPU or model-training
performance is claimed.

## Python

```bash
uv sync --extra dev --extra jax
uv run ruff check src tests
uv run ruff format --check src tests
uv run pytest -q
XLA_FLAGS=--xla_force_host_platform_device_count=4 uv run pytest -q
```

Final pass counts and exact command output are retained in
`research/verification/python.txt` and `research/verification/sharded.txt`.
The ordinary run skips the multi-device test when only one device is present;
the four-device run exercises it. These checks cover the independent Optax baseline,
float64 reference/JAX parity, current reprojection of stale coefficients, guidance
moment isolation, schedule violations, degenerate/extreme coefficient cases,
declared split overlaps, input-path roles, and required protocol/token arithmetic.

They do not train an LLM, validate a real data manifest, test multi-host TPU
communication, estimate generalization, or pass an experimental phase gate.

## Lean

```bash
cd proofs
lake build
lake env lean Audit.lean
```

Lean 4.24.0/mathlib v4.24.0 build successfully, including raw-normalization bridge
lemmas. The axiom audit reports only standard `propext`, `Classical.choice`,
`Quot.sound`; `evaluation_noninterference` has no axioms. There are no project
`axiom` declarations, `sorry`, or `admit`. Final outputs are in
`research/verification/lean.txt` and `research/verification/axioms.txt`.
The [theory map](../theory.md) describes the exact mathematical and conditional scope.

## Protocol and scope

```bash
uv run python -m guidon.protocol configs/pretrain_125m.json
uv run python -m guidon.protocol configs/continued_pythia160m.json
```

Scratch: 4,769 updates; 194,816 loss tokens on the final partial update; 73 probes
and 4,784,128 extra guide tokens at the proposed cadence. Continued: 1,908 updates;
182,784 final partial tokens; 28 probes and 1,835,008 extra guide tokens. Each has
six primary and three required information-control runs in its future matrix.
Both report `ready_for_confirmation=false`; real manifests and frozen gates are absent.

Local Markdown link targets and the nine-phase dependency/state consistency are
checked during release. Original prompt and supplied skills remain preserved.
The supplied skills are preserved as an initialized, pinned submodule with a
tracked `.gitmodules` URL and a publicly available commit. Large dependency caches,
environments, data, runs, and checkpoints are ignored.
