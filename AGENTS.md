# Working on GUIDON

This is an experimental optimizer foundation. The user will start research phases
separately. Creating or reviewing the repository does not authorize large training
runs, TPU provisioning, or opening sealed evaluation. When a phase is requested,
follow [phases/README.md](phases/README.md) and that phase's document. Complete its
authorized repairs without requesting repeated permission for routine fixes.

## Where to look

- `theory.md`: equations, assumptions, theorem map, and explicit hypotheses.
- `src/guidon/reference.py`: readable NumPy algorithm specification.
- `src/guidon/jax_optimizer.py`: JAX implementation and shared AdamW baseline.
- `proofs/`: pinned Lean/mathlib statements and dependency audit.
- `configs/`: proposed protocols; never silently alter the 125M/2.5B/42,43,44 contract.
- `research/`: literature, boundaries, live evidence, and failed branches.
- `phases/state.json`: gate status; use evidence hashes when changing status.
- `skills/`: user-supplied research methods and references; read relevant ones.

## Scientific rules

Treat guidance as optimization data. Training readers get only `train` and
`guidance`; development is for pilot selection; final validation/test remain sealed
until the registered confirmation. Never tune on a guidance score and label it
independent evaluation. Deduplicate before packing; verify source groups, benchmark
exclusions, token cursors, and dataset/model revisions.

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
The training harness, dataset builder, Pallas kernels, and full paper are future
phase deliverables, not existing completed capabilities.

Do not commit datasets, checkpoints, credentials, or local dependency caches. Keep
small auditable records and links/hashes for large artifacts. Commit and push work
when requested; do not change unrelated repositories or account settings.

This concise, project-specific guidance follows
[official AGENTS.md guidance](https://learn.chatgpt.com/docs/agent-configuration/agents-md)
and [OpenAI's advice on maintaining instructions](https://developers.openai.com/blog/rethinking-skills-and-prompts-for-gpt-6-astra).
