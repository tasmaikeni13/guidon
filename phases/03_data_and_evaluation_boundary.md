# Phase 03 — Data preparation and evaluation isolation

**Entry:** 01/02 valid. Implement the full data tooling before TPU training. Current
`guidon.data_boundary` checks declared identities; it does not construct trustworthy
clusters or certify a corpus.

Use the pinned revisions in `configs/`. Scratch: FineWeb-Edu `sample-10BT`, GPT-2
tokenizer, exactly 2,500,000,000 loss tokens in the main training tape. Continued:
Pythia-160M-deduped with its own tokenizer and FineMath `finemath-4plus`, 1B tokens
in a 90% math / 10% independent FineWeb-Edu retention mixture. Keep a single global
role assignment so scratch evaluation sources cannot later become continued-training
or guidance sources. Token budgets are measured after actual tokenization, not
inferred from a dataset's published nominal count.

Implement `guidon.prepare_data` and immutable manifest generation. Canonicalize
documents, extract registrable-domain/source groups using a pinned public-suffix
list, build exact and near-duplicate clusters across all candidate splits and
benchmarks, then assign whole clusters/source groups to roles. Use audited MinHash
or another declared similarity method, record thresholds and false-negative checks.
Quarantine a connected component that conflicts with role/source constraints.
Exclude all registered benchmark documents and answer-bearing near copies from
both train and guidance before packing. Record exclusion counts and sampled audits.

Reserve at least 10M tokens each for guidance, development, validation, and test,
plus independent shifted-domain evaluation. Split by immutable hash and the fixed
split seed before inspecting model outcomes. Guidance is fresh **within each run**;
paired arms/seeds use the same frozen tapes for fairness. Pilot and confirmation
guidance streams are distinct. Confirm capacity after filtering; insufficient data
requires a documented deterministic extension, never borrowing sealed evaluation.

Pack sequences only within a role. Register EOS, boundary attention/loss masks,
tokenizer revision, stride, padding, and accounting conventions. Count exactly the
next-token labels with positive loss mask; partial final batches consume only their
registered labels. Hash manifests, token shards, probe tapes, and sample cursors.
Create metadata fields `doc_id`, `content_sha256`, `cluster_id`, `source_group`, `split`;
run `uv run python -m guidon.data_boundary PATH_TO_MANIFEST.jsonl`.

Build a training process whose readers receive only train/guidance paths and
enforce this with an access audit. Separate pilot-development and frozen evaluator
processes. Keep final payloads unavailable to the research agent until the prescribed
confirmation. Membership/contamination audit tooling may inspect evaluation text
for exclusions without passing labels, examples, or scores to optimization/tuning;
record and constrain that information flow. A common data preparation hash must
be independent of which optimizer wins.

For Pythia, audit incrementally added data against its known Pile training lineage
and benchmarks. Shared initialization controls existing exposure but does not erase
it. Mark unverifiable base exposure and exclude contaminated public tasks from any
clean-generalization claim. Prefer newly reserved source groups for primary losses.
Do not claim that a post-2023 crawl proves a page was absent from 2023 training.

**Outputs:** data builder/loader tests (including adversarial leaks), manifest hashes,
dataset cards/license provenance, quarantine/contamination reports, sealed evaluator
access design, enough token shards, and `research/gates/03.json`. Set `manifest_ready`
only for a real audited manifest and link its hash. **Exit:** no known cross-role
document/cluster/source overlap, no evaluator input channel, exact replayable budgets.
On failure repair grouping/preprocessing, revise the data protocol, invalidate 04–09,
and rerun the audit. Do not waive an unresolved evaluation leak.
