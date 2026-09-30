# Predeclared analysis — simulation v1, GUIDON 0.2

This file and `protocol.json` are frozen before any simulation draw. The primary
question is whether the local block controller's risky population-transfer and
signal-corruption predictions survive a noisy, representative synthetic regime.
The intended IDs, noise/drift controls, falsifiers and tolerances are explicit in
that JSON. They were chosen from the equations and known failure boundaries,
not from observed losses. No synthetic outcome is LLM confirmation.

A draw is an independently initialized synthetic trajectory, paired across AdamW,
GUIDON and information-matched AdamW. Discovery and confirmation use separate
fixed root streams; all predeclared targeted conditions are run in both. Conditions
are randomized in order. The 32-cell screen balances 12 binary factors but aliases
interactions; it cannot identify main-effect causality without the targeted checks.
Every main-effect direction, null, mismatch, curvature failure and fallback stays
in the report. There is no optional stopping, selective exclusion or seed restart.

Quadratics have explicit diagonal Hessians, analytic gradients and zero-loss
instantaneous oracles; regret sums loss relative to the time-varying evaluation
minimizer only for these problems. Rotating conditions change train and guide
centers with controlled angular drift. Spurious regression/classification use
fresh Gaussian feature batches with train correlation 0.95 and guide/evaluation
correlation 0 when representative. Labels use the causal first feature. Guide
mismatch changes guide correlation or reverses quadratic centers. Independent
evaluation samples use a separate RNG and never enter updates. Logistic/regression
finite evaluation losses have no claimed exact oracle. A finite guidance source
population creates a common coefficient offset, plus new noise at each probe;
stream size varies this uncertainty. Every gradient batch is fresh. This stylized
noise model is not a web-data/model validation experiment.

Loss reductions, current population-guide/evaluation linear gains, update norms,
coefficient time variance, fallback fractions and neutrality residuals are retained
per draw. Report means and paired standard deviations with SE = s/sqrt(N).
Pointwise 95% Monte Carlo intervals use 1.96 SE and describe simulation precision,
not physical validity. The intended local comparisons form an intersection rule:
all named intended conditions must pass. Corruption comparisons are prespecified.
The rest of the screen is descriptive, with no “significant optimizer” labels.
Across-condition and distribution comparisons must remain visible, including
conditions where GUIDON and information-matched AdamW both fail the target.

For future LLM confirmation, define d_s = NLL_AdamW,s - NLL_GUIDON,s for seeds
42, 43, 44. The experimental unit is one paired training seed. Report all three
d_s and their mean, sample SD, and the 95% paired t interval
mean(d) ± t_(0.975,2) s/sqrt(3), t_(0.975,2)=4.302652729911275.
Independent normal paired differences justify exact t coverage; assess that
assumption cautiously with only three pairs. The registered meaningful point
difference remains 0.005 nats/token. Distinguish a mean ≥0.005, an interval excluding
zero, and an interval whose lower bound exceeds 0.005; these are different findings.
Simulation studies cover variance 0.002, 0.005, 0.01, 0.02 and Gaussian, t3 and skewed
errors, effect 0 or 0.005, 10,000 simulated three-pair experiments per cell. Report
coverage, power against zero, probability of attaining the point threshold, and
interval widths with Monte Carlo SE. No extra LLM seeds are silently introduced.

Document/source-group paired bootstrap is secondary and conditional on the trained
models; resampling documents cannot produce independent training seeds. Preserve
source clusters, then aggregate within each seed. Training variability, evaluation
sampling uncertainty, tuning/model-selection uncertainty and numerical/contamination
bias remain separate. Shared initialization in continued pretraining is disclosed.
The primary scratch and math/retention hypotheses use the future frozen protocol;
all registered secondary tasks get all per-task differences and Holm-adjusted
p-values only if a defensible seed-level test is possible. With three pairs, an
exact two-sided sign-flip test cannot attain p<0.05 (minimum 2/8). Otherwise report
effect sizes and intervals without declaring significance. Fixed task aggregates
and directions must be registered in Phase 05 before scores; failed/excluded tasks
remain in the denominator or are reported as inconclusive per its frozen rule.

Infrastructure faults retain the traceback, raw output, artifact hash and run ID.
Repair with a new version if mathematics or design changes, invalidate dependent
gates, and use fresh confirmation draws for the changed prediction. Scientific
nulls are outcomes, not infrastructure failures. Both compute cost and every
failed/discovery/confirmation draw count are retained in provenance.
