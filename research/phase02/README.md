# Reproducing phases 01/02

Run from the repository root with `uv sync --extra dev --extra jax`. Scripts
refuse to overwrite raw evidence. Use a clean artifact directory or a new explicit
output path for a repeat; keep the old file and its hash. Raw JSONL/CSV is gzip
compressed with deterministic headers under ignored `artifacts/phase01` and
`artifacts/phase02`; datasets/checkpoints/caches are not committed. Small summaries,
protocols, manifests, verifier results and standalone figures are tracked.

```bash
uv run python -m research.experiments.math_audit
uv run python -m research.experiments.verify_math
uv run python -m research.experiments.simulate --stream discovery
uv run python -m research.experiments.verify_simulation --stream discovery
uv run python -m research.experiments.analyze_simulation --stream discovery
uv run python -m research.experiments.simulate --stream confirmation
uv run python -m research.experiments.verify_simulation --stream confirmation
uv run python -m research.experiments.power
uv run python -m research.experiments.verify_power
uv run python -m research.experiments.analyze_simulation --stream confirmation
uv run python -m research.experiments.mechanism_diagnostics
uv run python -m research.experiments.noise_diagnostics
uv run python -m research.experiments.mismatch_witness
```

The original v1/v2 coefficient records are failed/intermediate evidence. Their
summaries, protocol snapshots and original code snapshot/reference commit remain;
`math_audit` defaults to the repaired v3. Exact reproduction across a different
BLAS/XLA/hardware environment may change floating reductions; verify invariants
and compare the recorded environment rather than asserting cross-platform byte
identity. On the recorded CPU stack, deterministic streams and raw headers make
trajectory files reproducible.

The drift interaction followup is registered separately. Run each stream with
`simulate --protocol research/phase02/drift-followup-protocol.json`, and explicit
`--raw artifacts/phase02/drift-STREAM-v1.jsonl.gz --provenance
research/phase02/drift-STREAM-v1-provenance.json`. Its validator accepts that protocol
as the fourth argument to `verify_simulation.verify`; the default CLI remains the
main study. For per-step analysis use `mechanism_diagnostics --protocol` with the
same followup, `--stream`, and explicit raw/summary/figure paths. Exact invocations
and the original recording-fault traceback remain in `research/verification/phase02`.
The original drift-discovery raw file was complete and its provenance recovered
without rerunning trajectories; a fresh reproduction can write its manifest
normally with the repaired CLI. Do not run the recovery helper on healthy records.

The [mechanism report](mechanism_report_v0.2.md) interprets results and all competing
models. The [analysis specification](analysis_spec.md) and JSON protocols precede
draws; the [uncertainty budget](uncertainty_budget.md), [failure ledger](failures.md)
and [driver equivalence](driver_equivalence.md) explain boundaries and faults.
[Standalone figures](figures/) show synthetic comparisons, mechanism controls,
three-pair power and all drift groups. Complete per-condition summaries retain
null/mismatch/negative outcomes and are the source for every plotted number.

`verify_gates` rechecks source/evidence hashes, all required raw records and gate
prerequisites. Large raw files are retained locally and identified by hash; they
are deliberately absent from Git. After a clean clone, reproduce them with the
commands above before requesting a full raw-evidence gate check. Running code and
retaining synthetic evidence does not authorize Phase 03 data or Phase 04 training.
