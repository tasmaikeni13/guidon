# Autonomous research, one phase at a time

This is the execution contract for the active research task. **Read
`../research/HANDOFF.md` after `../AGENTS.md` whenever continuing from a new
session.** Phases 01 and 02 have local audit/simulation evidence. Phase 03/04 code
exists, but production data preparation and remaining hardware evidence are
unfinished; no LLM training run has been performed. Current signoff is in
`state.json` and `research/gates/`. Follow the user's active authorization and the
handoff; do not open sealed evaluation or provision a new TPU.

The objective remains better independent LLM generalization than well-tuned AdamW,
near its compute and speed, with evaluation isolated from learning. Preserve the
required scratch experiment: approximately 125M parameters, exactly 2.5B FineWeb-Edu
training loss tokens, optimizers AdamW/GUIDON, seeds **42, 43, 44**. The added
AdamW-plus-guidance arm controls extra information; it is the same optimizer.
The active data plan uses pinned FineWeb-Edu `sample-10BT` without corpus-wide
deduplication, as directed by the user on 2026-10-01; see Phase 03 and the handoff
for limitations and exact resumption steps.

## Order and evidence

| Phase | Work | Depends on | Required gate record |
|---|---|---|---|
| [01](01_math_and_proofs.md) | Derivation, attack, formal and numerical fidelity | — | `research/gates/01.json` |
| [02](02_simulation_and_statistics.md) | Monte Carlo, mechanism tests, statistical design | 01 | `research/gates/02.json` |
| [03](03_data_and_evaluation_boundary.md) | Dataset preparation and sealed evaluation | 01, 02 | `research/gates/03.json` |
| [04](04_tpu_implementation_and_kernels.md) | Training scripts, TPU kernels, profiling, resume | 01, 03 | `research/gates/04.json` |
| [05](05_pilots_tuning_and_freeze.md) | Fair pilots, ablations, frozen protocols | 02, 03, 04 | `research/gates/05.json` |
| [06](06_scratch_confirmation.md) | 125M/2.5B scratch confirmation | 05 | `research/gates/06.json` |
| [07](07_continued_pretraining.md) | Pythia/FineMath continued-pretraining confirmation | 05, 06 | `research/gates/07.json` |
| [08](08_generalization_and_analysis.md) | Independent benchmarks, real-world shift, claims | 06, 07 | `research/gates/08.json` |
| [09](09_paper_and_release.md) | Paper, comments, formatting, repository release | 01–08 | `research/gates/09.json` |

Gate records contain status, protocol/theory version, code commit, commands,
environment/hardware, dataset/config/checkpoint hashes, raw artifact paths and
checksums, observed decision statistics, outstanding assumptions, and the precise
pass/fail reason. Existence of a file is not a passed gate. Update
`phases/state.json` only after verifying its evidence. Large artifacts live outside
Git; small provenance and summaries remain tracked. There are nine phases.

## Self-correcting loop

1. **Specify:** state the claim, mathematical domain, measurable signature, comparator,
   resource envelope, falsifier, and next decisive test before running it.
2. **Check:** reproduce the relevant baseline and verifier. Separate an implementation
   failure from a failed scientific prediction. Preserve the traceback and raw output.
3. **Study:** search current primary papers and official implementations, read the
   closest methods, inspect counterexamples and negative results, and update
   `research/literature.md`. Use the relevant supplied skills and their references.
4. **Revise:** write a hypothesis card, derive equations and edge cases, implement the
   smallest causal change, update `theory.md`, and rebuild Lean. Never insert a
   placeholder proof or assume the desired conclusion.
5. **Test:** run invariants and independent checkers, then simulations, pilot data,
   and mechanism controls. Choose the next experiment for information gained per
   cost, including conditions that could reject every live explanation.
6. **Decide:** promote, revise, or reject with evidence. Retain negative and failed
   branches, their mechanism, and conditions for reopening. Continue repair/research
   toward the original target; a merely passing toy or convenient substitute is not
   success. If resources or access prevent the next necessary test, record a concrete
   blocker and the exact resumption action rather than inventing success.

Use `research/state.md` and the supplied research-loop templates for durable
hypotheses, run records, measurement uncertainty, and failure taxonomy. A visible
progress log is useful; it is not evidence for a scientific claim.

## Adaptive dependencies

Every gate must record the equations, theorem names, code, and data/protocol decisions
it relies on. An upstream change invalidates that gate **and its transitive dependents**.
For example, replacing the projection in 01 invalidates 02–09; changing source groups
in 03 invalidates 04–09; fixing a measured kernel semantic bug in 04 invalidates 05–09.
A profiling-only improvement requires semantic parity and renewed efficiency
evidence; prior loss evidence may be retained only with a written equivalence audit.

Archive the old theory/config/gate records by version, mark dependent entries
`invalidated`, rewrite all affected phase instructions, update equations, theorem
maps, implementations, tests, hypotheses, analysis plans, and later paper text.
Rerun from the earliest invalidated prerequisite within the user's authorized scope.
Do not leave future phases assuming abandoned mathematics. At every handoff write
the next exact command or the missing deliverable to implement.

## Failure, restarts, and evaluation

Infrastructure failures resume the **same** run from a consistent checkpoint and
data/probe/RNG cursors. A scientific redesign gets a new protocol and run ID.
Never restart a healthy seed because its score is unfavorable; all three required
seeds remain in the report. Before restarting, classify the fault, explain evidence,
test the fix cheaply, propagate invalidation, and keep the failed run and its cost.

If a full competitive run misses the registered target, diagnose it and return to
the appropriate mathematical/data/implementation/pilot phase. Continue adaptive
research rather than quietly accepting an inferior replacement. Once final losses
or benchmark answers have influenced a redesign, that evaluation is spent: reclassify
it as development and obtain new sealed groups before a new confirmation. Infinite
retries against the same test cannot produce publishable confirmation. Report every
attempt and total search cost; a negative result remains valid evidence.

No formula here guarantees eventual AdamW superiority. Three seeds provide limited
uncertainty resolution. Preserve the requested seed count and report inconclusive
evidence honestly; any additional confirmation must be a new, explicitly registered
extension, with budgets identified before seeing its results.

## Commands and future artifacts

Current verification commands are in [AGENTS.md](../AGENTS.md). Phase 03/04 trainer,
data preparation, packing, sealing, Pallas, and profiling code is present; the direct
sample-10BT token build and required end-to-end evidence remain incomplete. Do not
treat code presence, compilation, or synthetic tests as a passed data gate, completed
hardware profiling, or an LLM result. Follow the handoff for the next action.

Each requested phase is complete only when its outputs exist, its gate is directly
verified, dependencies remain valid, and the state/next action is refreshed. The
final phase writes the paper and presents supported results, not aspirational wins.
