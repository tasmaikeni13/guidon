# GUIDON

**Guidance Using Independent Development Objectives and Neutrality**

GUIDON is an experimental AdamW variant for LLM pretraining and continued
pretraining. It periodically uses a separate guidance batch to reweight existing
AdamW updates by model block. A small projection preserves the current training
gradient's first-order progress while improving the fresh guidance linear model.

The repository contains a working optimizer prototype, machine-checked coefficient
theory, and a nine-phase plan for testing whether that mechanism improves independent
generalization within 10% of AdamW's compute and wall time. **LLM performance and TPU
speed are unmeasured.** The large research runs and full paper are future phases.

## What is here

| Artifact | Status |
|---|---|
| [Theory and equations](theory.md) | Written, assumptions and unproved empirical hypotheses explicit |
| [Lean proofs](proofs/README.md) | Checked with pinned Lean 4.24.0/mathlib; no project axioms or placeholders |
| [NumPy reference](src/guidon/reference.py) | Implemented and tested |
| [JAX optimizer and AdamW baseline](src/guidon/jax_optimizer.py) | JIT and global-array reduction correctness checked on CPU |
| [Data-boundary checks](src/guidon/data_boundary.py) | Declared identity and reader-role checks; full corpus preparation is Phase 03 |
| [Experiment configs](configs/) | Proposed, pinned sources; deliberately not frozen or launch ready |
| [Nine research phases](phases/README.md) | Phases 01/02 completed locally with evidence; 03–09 remain unexecuted |
| [Math fidelity and simulations](research/phase02/README.md) | Numerical repair, 174k paired synthetic draws, 240k three-pair design experiments; all failures retained |
| [Agent instructions](AGENTS.md) | Setup, scope, scientific boundaries, verification, and repair rules |

## Guidance and leakage

Any examples that influence an optimizer are optimization data, even when they
are called validation. GUIDON therefore separates ordinary training, guidance,
pilot development, sealed validation, and sealed test. Guidance cannot serve as
evidence of independent generalization. Only the last two splits supply final
confirmation, after tuning and checkpoint rules are frozen.

The [data protocol](research/data_protocol.md) requires document/near-duplicate/source
group isolation, benchmark exclusions, restricted readers, fresh guidance within
each run, and an audit of continued-pretraining exposure. The Lean isolation result
is conditional on that boundary; it cannot certify external datasets. The method
does not promise a universal improvement on arbitrary future distributions.

## Install and verify locally

```bash
git submodule update --init --recursive
uv sync --extra dev --extra jax
uv run ruff check src tests
uv run ruff format --check src tests
uv run pytest -q
XLA_FLAGS=--xla_force_host_platform_device_count=4 uv run pytest -q tests/test_sharding.py
uv run python -m guidon.protocol configs/pretrain_125m.json
uv run python -m guidon.protocol configs/continued_pythia160m.json
```

For the proofs, install the pinned Lean toolchain through Elan, then:

```bash
cd proofs
lake update
lake exe cache get
lake build
lake env lean Audit.lean
```

`uv.lock` pins the Python environment; `proofs/lake-manifest.json` pins proof
dependencies. For an existing TPU VM, `uv sync --extra dev --extra tpu` selects the
matched TPU extra; compatibility and multi-host profiling must still pass Phase 04.
The checked local results are recorded in [verification](research/verification.md).

## Use the optimizer

The caller computes training and, at scheduled probes, guidance gradients. Here is
a hand-checkable two-block illustration of the reference API:

```python
import numpy as np

from guidon.reference import Config, init, step

params = (np.zeros(4), np.zeros(4))
train_grad = (np.ones(4), np.ones(4))
guide_grad = (2 * np.ones(4), np.zeros(4))
config = Config(guide_warmup=0)
params, state, metrics = step(
    params, train_grad, init(params), config, guide_gradients=guide_grad
)
print(metrics["weights"])  # [1.15, 0.85]
```

For JAX use `jax_optimizer.init(params, group_ids, decay_mask)` and
`jax.jit(jax_optimizer.make_step(group_ids, decay_mask, config))`. The step accepts
`(params, training_gradients, state, guide_gradients=None, learning_rate=None)`.
Group/mask pytrees match parameter leaves; multiple leaves can share a contiguous
group ID. Set `Config(radius=0)` for the shared AdamW baseline. A training harness
must abort on `metrics["schedule_ok"] == False` or `metrics["numerics_ok"] == False`, handle nonfinite training gradients,
and log numerical fallbacks. Current code returns updated parameters directly;
it is not an Optax transformation. See the tests for concrete pytree/sharding use.

## Research progress

[Phase 01](research/phase01/proof_fidelity_v0.2.md) verified mathematical and numerical fidelity and repaired weak-signal normalization and XLA rounding certification. [Phase 02](research/phase02/mechanism_report_v0.2.md) confirmed the registered local mechanism on synthetic problems and documented failure regimes and weak three-seed statistical resolution. These results do not establish LLM superiority or TPU efficiency. Reproduction and raw-artifact retention are documented in [the experiment guide](research/phase02/README.md).

The next phase is [Phase 03](phases/03_data_and_evaluation_boundary.md), on a separate user request. Advance through the gates in [phases/README.md](phases/README.md). That contract specifies how to
research failures, revise math/proofs, invalidate and rewrite dependent phases,
preserve all results, and obtain new sealed evaluation after a redesign.

The scratch confirmation is 125,226,240 parameters and exactly **2.5B FineWeb-Edu
training loss tokens per optimizer per seed**, using **42, 43, 44**. A required
additional AdamW arm receives the identical guide examples to control extra
information. Continued pretraining uses pinned Pythia-160M and a 1B-token
FineMath/retention mixture. Both include source-held evaluation, fixed secondary
benchmarks, complete cost accounting, and uncertainty analysis.

The full trainer, data builder, custom TPU kernels, benchmark harness, and paper
are deliverables of those later phases. Nothing in these instructions launches
them during repository preparation.

The [literature ledger](research/literature.md) records close precedents and coverage
limits. The proposed composition's novelty is provisional; the projection itself
is standard mathematics. User-supplied research skills are preserved in `skills/`. The original prompt was removed by the owner in commit `f85bfcb` and remains in Git history. The skills are a pinned submodule of
the supplied [skills repository](https://github.com/tasmaikeni13/skills), at commit
`0f4d8e0f9d6269094e585caded2cc8d0a9d2b0be`; initialize it with the setup command above.
