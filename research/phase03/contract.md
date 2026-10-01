# Phase 03/04 execution contract — registered 2026-10-01

The user authorized completing Phase 03 data preparation and Phase 04 trainer,
kernels, TPU parity/profiling/resume, rewriting Phase 05 as a three-seed 125M
600M-token/2,500-update hyperparameter sweep, and updating/publicizing supporting
files. This does not launch the Phase 05 sweep, Phase 06/07 confirmation, provision
another TPU, or open final evaluation.

The existing `my-tpu-v4` in `projectalgebraicai/us-central2-b` is HEALTHY/READY,
v4-32, topology 2×2×4, four workers. Metadata and read-only TPU API inspection
agree. Discover logical JAX devices only after initializing every process.
Matched stack: JAX/JAXlib 0.6.2, libtpu 0.0.17, FP32 parameters/moments and
bfloat16 matrix compute. No TPU efficiency evidence has yet been collected.

Primary data deliverables: joint canonicalization/source/near-duplicate components
across selected pinned FineWeb-Edu and FineMath candidates plus all registered
benchmark references; whole-component hash roles; quarantined conflicts; at least
10M actual loss tokens per guidance/development/validation/test reservation in
each regime; separate pilot and confirmation tapes; independent shifted groups;
immutable shard/manifest/cursor hashes and dataset license/provenance records.
Scratch confirmation needs 2.5B, scratch pilot 600M, continued training 1B with an
exact 90/10 loss-token mixture. Published nominal token counts are not evidence
of capacity. No `manifest_ready` promotion until actual capacity and exclusion,
source-isolation, reader and sealing audits pass.

Validation/test payloads must be encrypted to an evaluator public key whose
private key is unavailable to this research session. The user authorized choosing
the custody arrangement without further questions. A one-shot external Cloud Run
job generated a 3072-bit RSA key and deposited it in a write-only Secret Manager
version. This research identity has an explicit HTTP 403 denial for private-key
access and cannot change that secret's IAM. The public key and the retained
post-deposit metadata error are audited in `evaluator-custody-v1.json`. A separate
evaluator may receive access from the project owner after registered confirmation;
no final evaluation is authorized by the present phase request.
Preparation may inspect evaluation text internally for contamination but must
not emit payloads, answers or scores to the optimizer or research logs. Training
receives only role-limited train/guidance capabilities; development uses a
different process. A future frozen evaluator verifies the run/protocol gates.

Phase 04 requires actual registered-architecture count, source GPT-NeoX logit
parity, globally counted loss/gradients, tiny real-data overfit, multihost parity,
atomic interruption/resume identity, optimized paths for both arms, and at least
200 synchronized steady-state full-model updates with total guide compute,
memory, collectives and separate end-to-end I/O/checkpoint cost included. An
optimizer microbenchmark or CPU correctness test cannot pass the TPU gate.

Falsifiers: a cross-role source/duplicate/exclusion leak; borrowed/repeated guide
labels; incorrect token masks/cursors; source logit mismatch; nondeterministic
resume; missed scheduled probes/nonfinite state; or wall-time/compute ratio >1.10.
An empirical failure is retained and diagnosed; no threshold is relaxed.

Prior Phase 01/02 raw archives are absent from this clean checkout. Restore them
from deterministic scripts and verify recorded checksums. Restored discovery
(68,000 rows) and final coefficient audit (7,168 rows) currently match their
recorded hashes. Recovery outputs/provenance live under ignored
`artifacts/recovery/`; original signed summaries are not overwritten. Broad
snapshot hashes in existing gates need an explicit equivalence audit when docs,
new implementation dependencies and verification tooling are updated. Mathematical
optimizer/proof/theory sources and confirmation configs stay unchanged.
