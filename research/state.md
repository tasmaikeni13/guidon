# Research state — 2026-09-30 — GUIDON 0.2

Current authorized scope is Phase 01 and Phase 02 plus GitHub publication. Both
phases have passed gates backed by directly verified source, protocol and evidence
hashes. Phase 02 includes registered discovery, confirmation, followup, verifier
and statistical-design evidence. No LLM training, TPU profiling, corpus
preparation, pilots or sealed real evaluation have been performed. The authoritative
status and evidence hashes are in `phases/state.json`.

Established evidence: all 32 public Lean theorem statements compile with standard
foundations only; exact noise/drift bounds and signal-floor properties are added.
The final 7,168-case actual FP32/bfloat16 coefficient audit passes its independent
invariants. Near-collinear full-radius amplification and XLA rounded-correction
certification were repaired; original failed/intermediate records are retained.
Overflow is now a visible run failure. See `research/phase01/claims.md` and the
versioned proof-fidelity report.

Phase 02 comprises 174,000 independent synthetic trajectory draws paired across
three arms (136k main, 36k registered drift interaction followup, 2k opposite-
evaluation witness), plus 240,000 synthetic three-pair design experiments.
Every condition retains at least 1,000 draws in each stream. The independent
verifier checks every raw record and replays sampled complete trajectories through
the public reference. Changing evaluation gives bitwise identical training paths.
Secondary coefficient/age diagnostics reuse those identities, not new confirmations.

Live H1 survives the registered representative local-mechanism tests: corruption
reverses/weakens transfer, null controls reach AdamW, and stale-current benefit
can reverse at controlled drift. Generalization and cost remain hypotheses.
Extra-information AdamW wins the representative regression comparison; mismatch
and universal-transfer failures remain. Original partial rotations did not
reverse because stationary coordinates dominated; fresh group-interaction tests
preserve that absence and locate the two-group boundary. The detailed competing-
model, failures and uncertainty records are in `research/phase02/`.

Three LLM seed pairs remain required (42/43/44). At paired SD 0.005 nats, detecting
a 0.005-nat effect has only about 18% two-sided power under the simulated normal
model; skewed differences under-cover nominal t intervals. Never substitute
synthetic draws/documents/checkpoints for trained-model replication.

The 0.1 plans are archived and phases 02–09 were invalidated by the controller
change. Phase 02 is renewed with 0.2 evidence; 03–09 remain unexecuted and require
their current prerequisites and a separate user request. The original
125,226,240-parameter / 2.5B / 42,43,44 contract and success thresholds are unchanged.

Next action after publication: the user may request Phase 03. Its missing
deliverables are an executable corpus preparation/deduplication/source-grouping
pipeline, benchmark exclusion audit, pinned tokenizer/model/data evidence, and
sealed readers/process boundaries. No existing full trainer or corpus builder is
claimed. Before resuming, run `uv run python -m research.experiments.verify_gates`
to verify the current evidence and transitive prerequisites.
