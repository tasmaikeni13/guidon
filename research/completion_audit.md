# Historical foundation deliverable audit

Historical scope: the initial foundation preparation at `5967ec0`. The owner subsequently removed `prompt.md` in `f85bfcb`; it remains in history. The current Phase 01/02 execution has a separate requirement audit below. This audit does not mark future experiments or the
scientific superiority hypothesis complete.

| Requirement from `prompt.md` | Current evidence |
|---|---|
| Named optimizer and full form | README/theory: GUIDON, Guidance Using Independent Development Objectives and Neutrality |
| Training guided by a separate landscape without final evaluation leakage | Explicit five-role protocol; guidance disclosed as optimization data; reader/identity tests; conditional Lean noninterference; full data/trainer audit required in Phase 03/04 |
| LLM scratch and continued pretraining | Callable optimizer prototypes; two proposed configurations; Phases 06/07 preserve both regimes |
| AdamW baseline and benchmarking | Shared JAX moment/decay implementation, zero-radius limit, independent Optax parity tests; fair tuning/information controls and end-to-end benchmarking specified |
| Near-parity speed/compute and better generalization | Prospective 1.10x resource and 0.005-nat effect targets; full TPU/LLM evidence intentionally deferred, no achieved-performance claim |
| Research/study and proposed novelty | Primary-source ledger, closest-work matrix, mechanism reconstruction, alternative hypotheses, falsifiers, and explicit novelty/access limitations |
| `theory.md`, important ideas and equations, no full paper yet | Theory notes with update equations, P1–P11, assumptions, counterexample, costs, and empirical targets; no full paper written |
| Lean proofs, corrected until they compile | Pinned sources build successfully; normalization bridge and quadratic optimum checked; axiom audit clean of project axioms/placeholders |
| `phases/README.md` with autonomous repair/adaptation | Nine-phase dependency DAG, evidence gates, research loop, failure records, transitive invalidation and rewrite rules, honest confirmation/restart policy |
| Starting mathematical/statistical phases | 01 derivation/proof/adversarial analysis; 02 Monte Carlo, manufactured problems, uncertainty/power and mechanism interventions |
| Script/kernel plans for v4-32 and AdamW | 04 specifies training/data/evaluation/checkpoint CLI deliverables, multi-host discovery, both XLA/Pallas paths, profiling and semantic/resume gates |
| Full 3-seed 125M/2.5B FineWeb-Edu competition | 06 and checked config specify 42/43/44, 125,226,240 parameters, exactly 2.5B main loss tokens, required six primary runs and additional AdamW information control |
| Choose continued model/high-quality data and advantage benchmarks | 07 chooses pinned Pythia-160M, FineMath-4+ and retention mixture; 08 specifies source/temporal shifts, knowledge/context/commonsense/math tasks with contamination and frozen-selection rules |
| Diagnose/research/restart and remake phases on failure | Phase contract and every gate specify repairs, versioned redesign, dependent rewrites, preserved healthy/unfavorable seeds, and fresh evaluation after adaptive redesign |
| Final cleanup, comments, PEP 8, human README/paper, organization | 09 requires these deliverables and clean-clone reproduction; initial Python is Ruff formatted/linted and README describes actual status |
| `AGENTS.md`, researched best practices | Concise repo-specific file with relevant-source pointers and actual checks, citing fetched official guidance |
| Preserve user control of execution | At initial foundation: all nine phases `not_started`; configurations not frozen/not launch ready; no LLM or TPU training performed |
| Use user's GitHub and push | Final publication verifies the configured user's repository, branch, and exact local/remote commit equality |

Direct local verification: [record and commands](verification.md), retained pytest
and Lean output in `research/verification/`, source inspection of the theorem map,
configuration arithmetic, nine-phase state/DAG, and local Markdown link checks.

Limits are requirements of honest research: empirical superiority and general
deployment behavior cannot be proved by the coefficient lemmas; a guide-influenced
set cannot be an untouched holdout; absolute novelty cannot be established by a
finite search. The planned research tests those claims and keeps negative evidence.

## Current request: complete phases 01/02 and push

This audit preserves the requested scope. CPU simulation/math evidence completes
these phases; the future LLM objective is not substituted with a favorable toy.
Gate hashes and `verify_gates` validate current artifacts directly.

| Explicit requirement | Authoritative evidence and its scope |
|---|---|
| Read AGENTS, phase README, requested phases/dependencies and relevant supplied methods | Implemented phase contracts, theory/experimental/literature skill references, registered protocols and literature update; later phases read to propagate v0.2 dependencies |
| 01.1 Re-derive and match both implementations; clipping, accumulation, ties, zero/empty layouts and timing | `theory.md`, phase01 claim table and fidelity report; source hashes; scalar/JIT/Optax tests and regression witnesses |
| 01.2 Rebuild/audit Lean, exact domains, conditional P7/P8/P9 | 32 public theorem axiom printouts, successful pinned build, P1–P14 assumption table; no custom axioms/placeholders |
| 01.3 Attack every named failure boundary and P10 | Exact/rational hand, momentum, opposite gradient, curvature, decay, stale/conflicting and skew-normalization witnesses; zero/one/collinear tests; compiled P10 |
| 01.4 Noise, projection sensitivity, normalization bias, stale error and freedom | Checked deterministic L1 error/drift bounds; explicit conjectural adaptive noise assumptions; measured coefficient errors/bias, covariance/batch/source/moment and group/rotation factors |
| 01.5 FP32/bfloat16 interface versus exact/float64; scale/collinearity/cancellation; repair failures | 7168 final actual-emitted coefficient/weight records; independent SVD/longdouble checker; old v1/v2 retained; floored-scale/reprojection/barrier proofs and overflow flags |
| 01 required script, claim/dependency table, counterexamples, versioned report and frozen tolerances/fault rejection | `math_audit.py`, `check_math.py`, `verify_math.py`; phase01 protocol, claims and fidelity v0.2; tested mutations; gate01 |
| 02 scripts and independent verifier; ≥1000 cheap draws/condition; streams/order/raws | 68 main + 18 fresh followup conditions, both streams at 1000 each; 2000 opposing-evaluation draws; source/protocol/raw hashes and per-draw outputs; complete scalar replay verification |
| 02 all generators and requested ranges, fractional screen/targeting and 3 arms | Explicit known Hessians/gradients, spurious regression/classification, rotating objectives, hand case; protocol fields cover every named level; fixed shared tuning apparatus and information-matched moments/data control |
| 02 intervention table and expected/absent/reversed behavior | Corrupt/remove/zero/collinear/fresh/stale/curvature/evaluation controls; original nonreversal retained; fresh group interaction shows reversal; report includes alternate explanation and all-arm failure |
| 02 invariant residuals, losses, oracle regret where defined, norms, fallback, variance, transfer and MC errors | Immutable per-draw fields; every condition in both summaries; separate first-probe and age diagnostics; independent summary recalculation |
| 02 three-pair power/width, 0.005-nat target, df2/assumptions, no pseudoreplication | 240000 raw design experiments; independent exact-df2/normal calibration; heavy-tail/skew coverage; frozen paired-seed analysis, secondary conditional bootstrap and multiplicity policies |
| 02 standalone figures, provenance, competing models, uncertainty and null/failure/multiplicity specification | phase02 figures, manifests, mechanism report, uncertainty budget, frozen analysis spec and complete failure ledger |
| Exit dependencies valid and statuses refreshed; preserve original LLM contract | Gates 01/02 with evidence hashes; transitive invalidation history and updated 03–09 plans; unchanged 125M/2.5B/42,43,44 and continued protocol audits; no sealed evaluation/training |
| Commit and push using owner's GitHub | Committed source snapshots and final remote HEAD equality; existing owner commit removing prompt preserved; no datasets/checkpoints/credentials/caches committed |

All relevant Python/Lean/sharding/config checks pass. One default-device sharding
skip is covered by the separate four-device test. Raw evidence is local outside
Git and hash-addressed; clone reproduction commands regenerate it. That storage
boundary does not convert absent artifacts into a passed gate: local gate validation
requires every raw file to exist and match its checksum.
