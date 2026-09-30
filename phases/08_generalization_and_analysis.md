# Phase 08 — Independent generalization and complete analysis

**Method dependency:** GUIDON v0.2 uses $\tau=\rho_t/\max(\|q\|_\infty,0.01)$ and exact-idempotent numerical reprojection; the old v0.1 plans were invalidated by Phase 01. Require the current hash-verified prerequisite gates.

**Entry:** valid 06/07 and the evaluator/tasks/analysis frozen in 05. Use exact final
checkpoints; no prompt, checkpoint, task, decoding, or threshold selection based on
the scores below. Any dataset unable to pass the registered contamination/access
checks is marked excluded/inconclusive with its reason, not silently replaced.

The primary scratch outcome is independently reserved FineWeb-Edu validation NLL.
Primary continued outcomes are independent math NLL and retention NLL. Test broader
transfer through prespecified source-held publisher/domain slices, a temporal
post-cutoff slice with verified document provenance, and realistic punctuation,
case, and boilerplate perturbations. A new crawl date alone is not a publication
date. Keep guidance/development publishers out of these evaluation groups.
Measure per-document continuation loss, per-domain averages, worst registered domain
regression, and uncertainty at the source/document group level.

Secondary frozen tasks: LAMBADA OpenAI (context-dependent prediction), ARC-Easy
(knowledge), HellaSwag (completion), PIQA (physical commonsense), and WinoGrande
(reference reasoning); for continued pretraining add GSM8K and MATH with fixed
zero-shot prompts and deterministic decoding. Pin the official
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) and task
revisions in Phase 03/05. Small 125–160M models may score poorly on generation tasks;
report this directly and do not change prompts until they appear to win. These
benchmarks are finite proxies for real-world uses, not deployment guarantees.

Regenerate analysis entirely from immutable raw outputs. For each primary contrast
report all three paired differences, mean, effect size in nats/token and relative
perplexity, the registered interval and assumptions, and matched resource ratios.
Use per-source/document paired bootstrap for conditional evaluation uncertainty;
seed-level intervals describe training variability. Do not pool tokens/checkpoints
as independent runs. Report multiplicity policy for secondary tasks and an aggregate
fixed before scores. Negative or inconclusive tasks stay in the main report.

Compare with AdamW-plus-guidance and the registered mechanism ablations. Inspect
learning curves, coefficient/projection diagnostics, actual-versus-linear guide
change, alignment noise, fallback rates, and cost traces. A lower NLL alone does
not identify the causal mechanism. Confirm the claimed effect remains after equal
information, tuning, and resource accounting. Include inference settings/cost;
GUIDON changes training only, so exported architectures must be identical within
each regime.

**Exit:** frozen evaluation and reproducible tables/standalone plots exist, clean
scope is explicit, alternate explanations have been tested, and the registered
generalization/resource targets hold. A broad advantage claim requires both regimes;
a narrower empirical observation remains labeled narrower, not replacement success.
If targets fail, preserve all evidence and revise upstream research/phases; inspected
test sets are spent for future confirmation. Write `research/gates/08.json` and
a claim–evidence matrix separating formal, simulated, exploratory, and confirmatory
evidence. Every permitted paper sentence must map to a supported claim.
