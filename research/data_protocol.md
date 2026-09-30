# Data and evaluation protocol

The optimizer consumes only training and **guidance validation**. Guidance is
optimization data. Ordinary training gradients update moments; guide gradients
produce block coefficients that influence parameters. Hiding that influence behind
scalars would not make it independent evaluation.

Five roles have distinct uses: `train`, `guidance`, `development` for pilot decisions,
sealed `validation` for the frozen primary contrast, and sealed `test` for final
transfer/shift. No evaluator payload enters gradient computation, clipping,
preprocessing fit, source assignment, schedule, sampling, seed choice, model import,
checkpoint selection, or manual/autonomous method search.

## Upstream preparation

Pin dataset/model/tokenizer revisions and preprocessing code. Before packing, group
canonical documents by exact content, declared near-duplicate cluster, and source
(whole registrable domain where metadata permits). Resolve their connected components
and assign every component to one global role across scratch and continued regimes.
Quarantine contradictions. Register exclusion rules and audit sampled cluster errors.
Known benchmark documents, solutions, and near copies are excluded from train and
guidance. Public web duplication remains an empirical audit problem, not a proof.

The declared-identity checker expects document-level JSONL records like:

```json
{"doc_id":"opaque-id","content_sha256":"64-lowercase-hex-characters","cluster_id":"dedup-component","source_group":"registered-source","split":"train"}
```

The digest above is schematic, not an accepted real digest. `guidon.data_boundary`
rejects malformed/missing identities, duplicate document IDs, and cross-role content,
cluster, or source identities. It cannot infer whether a supplied cluster is honest,
detect an unrecorded duplicate, or establish that a publisher grouping is correct.
Phase 03 implements and validates those upstream parts before setting a gate.

Reserve pilot and confirmation cohorts before outcomes. Guidance samples are never
reused **inside a trajectory**; the same frozen guide tape is shared across paired
arms/seeds to isolate optimizer effects. Do not recycle a small guide set adaptively.
No packing crosses roles. Count actual masked next-token labels, freeze special-token
and boundary conventions, and preserve deterministic stream/probe cursors on resume.

## Runtime boundary

`training_paths` accepts exactly train/guidance paths. A future trainer must wire its
readers through this boundary, verify manifest and shard hashes, and have no final
evaluation capability. Pilot selection uses a separate development evaluator. Final
evaluation runs in a distinct sealed process on frozen checkpoints and a fixed
protocol; logs must establish that training did not open final payloads.

The Lean result proves that changing evaluation payloads cannot alter a pure
trajectory **if its authorized inputs and initial state are fixed independently**.
It does not prove that an arbitrary caller chose clean inputs. Test the entire future
data/training/checkpoint pipeline by changing inaccessible evaluation fixtures and
checking the same resulting checkpoint hash. Inspect preprocessing and orchestration
as well as the optimizer function.

## Repeated research and pretrained exposure

Adaptive reuse of evaluation can bias results; see the primary
[holdout-reuse analysis](https://arxiv.org/abs/1506.02629). This project does not
implement a reusable-holdout or privacy mechanism. If a confirmation or test result
influences redesign, move that set to development and reserve new evaluation groups.
Log every attempt and its cost; never erase unfavorable seeds or replace them.

For continued pretraining, audit Pythia's known corpus and incremental data/benchmark
overlap. A shared base checkpoint controls historical exposure in a comparison but
cannot make unknown base contamination disappear. Only audited clean evaluation
supports a clean claim; contaminated/uncertain public tasks are separately reported.
An evaluation boundary concerns this workflow's data use, not a promise of universal
deployment accuracy or source-distribution independence.
