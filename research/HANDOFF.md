# Active research handoff

Read this file after `AGENTS.md` whenever a new session is asked to continue.
The user explicitly said to continue from this point without asking them to repeat
the task. Update this handoff and `phases/state.json` as work advances.

## Current instruction from the user

Use the pinned FineWeb-Edu `sample-10BT` data directly. The user waived
corpus-wide deduplication and asked to stop the expensive all-candidate scan. Make
a deterministic 2.5B positive-next-token training tape from that source, then
register a deterministic 600M-token pilot subset from the same stream. Preserve the
original 2.5B confirmation contract, model size, and seeds 42/43/44. Phase 05 is
currently specified as a three-seed 125M / 600M / 2,500-update hyperparameter sweep
with pilot seeds 101/102/103; the sweep has not run. Keep final validation/test
sealed. Do not provision a new TPU or execute confirmation. The user explicitly
authorized pushing repository work to their GitHub.

The no-dedup choice is a documented limitation: do not claim source isolation,
deduplication-based leakage prevention, or clean benchmark exposure. Keep reader
roles and sealed-evaluation controls. The choice supersedes the earlier Phase 03
instruction to deduplicate every corpus document before packing.

## Exact stop point

On 2026-10-01, the user told us to stop the corpus scan. All scan workers and the
controller were terminated. The last log record reports **6,640,000 documents
indexed** in the FineWeb-Edu candidate; this was only a partial index. It did not
produce token tapes or a valid Phase 03 manifest and did not pass a gate. Its log
and source/protocol registration are retained under ignored `artifacts/phase03/`
for local audit. The temporary SQLite database is under
`/dev/shm/guidon-phase03-v3/`; it is disposable partial indexing scratch, not a
training corpus. Remove it if it still exists before rebuilding data.

No training job or sweep is running. Phase 03 and Phase 04 remain `in_progress`;
Phase 05 is `not_started`. Never mark them passed based on code presence, compile
checks, or this handoff.

## Resume actions

1. Check `git status`, read `phases/state.json`, and inspect the retained
   `artifacts/phase03/index-v3.log` only if it still exists. Confirm no old process
   is running before starting preparation.
2. Use the pinned FineWeb-Edu repo/revision/config already recorded in
   `configs/data_phase03.json`: `HuggingFaceFW/fineweb-edu`, revision
   `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`, config `sample-10BT`. The file list
   and hashes are in that config. Ignore its old multi-source/dedup processing
   settings for the active training tape; retain the file only as source provenance.
3. Implement or adapt preparation to stream those pinned shards, tokenize with the
   pinned GPT-2 tokenizer, and pack exactly 2.5B positive next-token labels without
   crossing document boundaries. Stop at the registered token count; do not scan
   all 10B tokens once enough valid labels exist. Produce a deterministic 600M
   pilot slice with hashes from the same source stream. Avoid keeping two copies of
   the 600M slice in RAM or duplicating the whole source corpus locally.
4. Preserve evaluation boundaries and readers. Check the actual manifests, masks,
   token cursors, tape hashes, source revision, access roles, and sealed payload
   behavior. Clearly record that no global dedup or source-overlap exclusion was
   run. Update Phase 03 docs/config/gate evidence to match the implemented design;
   keep its status in progress until all checks pass.
5. Finish Phase 04's remaining evidence on the already-existing TPU only: full
   model real-data step/profile, interrupted multi-host checkpoint/resume, and the
   outstanding end-to-end checks identified in `phases/04_tpu_implementation_and_kernels.md`.
   Do not launch large experimental training as a substitute for these bounded
   implementation checks.
6. Keep the Phase 05 sweep protocol, run matrix, three seeds, and controls
   preregistered. Do not launch the 108 sweep jobs unless the user explicitly asks
   to execute Phase 05 after gates 03/04 pass.
7. Before calling the requested foundation work complete, refresh `README.md`,
   `AGENTS.md`, phase docs, and state; run the repository's required release checks;
   verify evidence hashes and that no data, checkpoints, private keys, credentials,
   or caches are staged.

## Work already present

The working tree contains substantial uncommitted Phase 03/04 implementation and
audit work from this session: corpus/source/benchmark tooling, token packing,
sealing, restricted readers, a JAX trainer/checkpoint path, model loading, Pallas
kernels, TPU compile/runtime/profile tooling, and a rewritten Phase 05 sweep config.
There are focused tests and retained failure records. Actual-data capacity and
production reader/sealing audits remain pending. CPU parity, unit tests, mathematical
proofs, or TPU compilation do not constitute real-data training, successful resume
on hardware, Phase 03/04 exit, or empirical GUIDON performance.

Phase 01/02 gates are locally hash-verified. A public evaluator key and encrypted
artifact-custody records were created for audit retention; private key bytes remain
outside the repository. Do not grant the training/research process access to the
private evaluator key and do not open sealed evaluation in this continuation.

## GitHub and artifact rules

The remote is `origin` (`tasmaikeni13/guidon`). The user authorized a push. Push only
the reviewed source/docs/small audit records and confirm the final remote commit.
Never add ignored `artifacts/`, dataset shards, tape binaries, model weights,
credentials, local caches, or private evaluator key material. Preserve public
provenance and hashes where needed to locate large encrypted artifacts.
