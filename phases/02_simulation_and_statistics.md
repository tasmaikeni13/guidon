# Phase 02 — Monte Carlo, mechanisms, and statistical design

**Entry:** Phase 01 valid. **Purpose:** falsify behavioral predictions cheaply and
design honest uncertainty analysis before expensive LLM trials.

Build `research/experiments/simulate.py` and a separate verifier. Use at least 1,000
Monte Carlo draws per primary synthetic condition, fixed discovery/confirmation
random streams, and raw per-draw output. This sample count applies to cheap draws,
not to independent LLM training seeds. Implement manufactured quadratic problems
with known Hessians and gradients, a spurious-feature regression/classification
problem with independently generated evaluation data, and nonstationary/rotating
gradient problems. A deterministic two-block example must reproduce the hand result.

Vary noise covariance and batch size, alignment, condition number $1$–$10^4$,
guidance distribution mismatch, group count 1/2/14/64, interval 1/16/64/256, radius
0/0.05/0.15/0.3, guide-stream size, delay, and moment lag. Use factorial screening
and targeted followups instead of an unjustified exhaustive costly grid. Randomize
condition order. Compare AdamW, GUIDON, and AdamW receiving the same extra examples.

Register competing predictions:

| Intervention | Mechanism prediction | Alternative explanation tested |
|---|---|---|
| Remove guide or set radius zero | AdamW limiting behavior | Hidden implementation changes |
| Shuffle/sign-reverse guide coefficients before projection | Fresh guide advantage weakens/reverses | Extra compute/data alone |
| Make guide coefficients collinear with training | Correction vanishes | Generic layerwise schedule |
| Increase age under controlled drift | Current-guide benefit can disappear | Claim of stale-current-guide guarantee |
| Hold training progress fixed and vary curvature/step size | First-order promise need not give actual descent | Mistaking linear algebra for a loss theorem |
| Change evaluation while fixing training/guidance | Bitwise identical reference training trajectory | Evaluation information entering learning |

Check invariant residuals, actual train/guide/test losses, regret versus the relevant
oracle only where defined, update norms, fallback fraction, coefficient variance,
and guide-to-evaluation transfer. Quantify Monte Carlo standard errors, confidence
coverage, and sensitivity to assumed distributions. Include conditions that reject
every live candidate; a realistic mismatch failure is useful evidence.

For confirmation, the experimental unit is a paired training seed, not a token,
checkpoint, or document. Simulate the power and interval width of **three pairs**
for plausible variance and a 0.005 nats/token difference. Predefine the paired mean
and all individual differences; a 95% paired t interval uses two degrees of freedom
and requires its distributional assumption. Document/group bootstrap is secondary,
conditional on trained models, and cannot create extra independent seeds. Avoid a
claim of significance based on pseudoreplication. Report weak power explicitly.

**Outputs:** scripts, immutable raw CSV/JSONL, standalone figures, simulation
provenance, competing-model/mechanism report, uncertainty budget, and a predeclared
analysis specification covering null/failure results and multiplicity across tasks.

**Exit:** invariant coverage passes, risky predictions survive in the *registered
intended regime*, and failure regimes are documented; simulation bias and statistical
resolution are understood. If the intended mechanism fails, diagnose, study, revise
Phase 01/theory/proofs, invalidate dependents, and repeat on fresh simulation draws.
Do not select a favorable toy as proof of LLM superiority. Write gate 02.
