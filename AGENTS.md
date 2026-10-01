# Working on GUIDON

This is an active research task. When the user says “read AGENTS.md and continue
where left,” first read [research/HANDOFF.md](research/HANDOFF.md), then continue
the listed next action without asking the user to repeat context. The handoff is
the current source of truth for progress; keep it updated as work advances.
Complete the currently authorized Phase 03/04 work and the Phase 05 protocol
rewrite. Do not start final confirmation, open sealed evaluation, or provision a
new TPU. Do not run large sweeps unless the user explicitly directs execution.

## Where to look

- `theory.md`: equations, assumptions, theorem map, and explicit hypotheses.
- `src/guidon/reference.py`: readable NumPy algorithm specification.
- `src/guidon/jax_optimizer.py`: JAX implementation and shared AdamW baseline.
- `proofs/`: pinned Lean/mathlib statements and dependency audit.
- `configs/`: proposed protocols; never silently alter the 125M/2.5B/42,43,44 contract.
- `research/`: literature, boundaries, live evidence, and failed branches.
- `phases/state.json`: gate status; use evidence hashes when changing status.
- `research/HANDOFF.md`: exact continuation point and current run state.
- `skills/`: user-supplied research methods and references; read relevant ones.

## Scientific rules

Treat guidance as optimization data. Training readers get only `train` and
`guidance`; development is for pilot selection; final validation/test remain sealed
until the registered confirmation. Never tune on a guidance score and label it
independent evaluation. The user has explicitly chosen the pinned FineWeb-Edu
`sample-10BT` stream and waived corpus-wide deduplication for this task. Use a
deterministic, documented token slice from that stream; do not resume the stopped
corpus-wide dedup/index job. Keep benchmark/evaluation boundaries and role-restricted
readers intact, and state plainly that the resulting run does not establish
deduplication-based leakage isolation.

Maintain strong AdamW, equal tuning budgets, and the AdamW-plus-guidance information
control. Retain every required seed and all null, failed, and restarted runs. No
selective restarts, benchmark cherry-picking, invented results, or changes to success
thresholds after seeing sealed evidence. A failed confirmation makes the inspected
evaluation development evidence for subsequent redesign; obtain a new sealed set.

Search primary literature for method changes and novelty claims. Record equations,
sources, counterexamples, and failure mechanisms. Keep theory, implementations,
proofs, configs, phase dependencies, and claims consistent. Propagate invalidation
to all dependent gates. Empirical superiority is a hypothesis until actual runs
support it. Never add proof placeholders or custom axioms to make Lean pass.

## Local verification

```bash
git submodule update --init --recursive
uv sync --extra dev --extra jax
uv run ruff check src tests
uv run ruff format --check src tests
uv run pytest -q
XLA_FLAGS=--xla_force_host_platform_device_count=4 uv run pytest -q tests/test_sharding.py
uv run python -m guidon.protocol configs/pretrain_125m.json
uv run python -m guidon.protocol configs/continued_pythia160m.json
cd proofs
lake build
lake env lean Audit.lean
```

Use focused tests for code changes; run all listed checks before a foundation release.
Python follows PEP 8 through Ruff. Prefer explicit types, small pure functions, and
comments explaining mathematical choices. A compiled CPU test is not TPU profiling.
Check [research/HANDOFF.md](research/HANDOFF.md) for the exact in-progress
deliverables and remaining verification. Do not describe partial data preparation,
compiled kernels, or code presence as a completed training/evaluation result.

Do not commit datasets, checkpoints, credentials, or local dependency caches. Keep
small auditable records and links/hashes for large artifacts. Commit and push work
when requested; do not change unrelated repositories or account settings.

This concise, project-specific guidance follows
[official AGENTS.md guidance](https://learn.chatgpt.com/docs/agent-configuration/agents-md)
and [OpenAI's advice on maintaining instructions](https://developers.openai.com/blog/rethinking-skills-and-prompts-for-gpt-6-astra).
