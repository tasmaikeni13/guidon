# Prompt deliverable audit

Scope: prepare and push the optimizer foundation and phases; the user explicitly
starts phase execution later. This audit does not mark future experiments or the
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
| Preserve user control of execution | All nine phases `not_started`; configurations not frozen/not launch ready; no LLM or TPU training performed |
| Use user's GitHub and push | Final publication verifies the configured user's repository, branch, and exact local/remote commit equality |

Direct local verification: [record and commands](verification.md), retained pytest
and Lean output in `research/verification/`, source inspection of the theorem map,
configuration arithmetic, nine-phase state/DAG, and local Markdown link checks.

Limits are requirements of honest research: empirical superiority and general
deployment behavior cannot be proved by the coefficient lemmas; a guide-influenced
set cannot be an untouched holdout; absolute novelty cannot be established by a
finite search. The planned research tests those claims and keeps negative evidence.
