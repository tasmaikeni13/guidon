# Phase 02 mechanism and model-validation report — GUIDON 0.2

The registered local mechanism survived independent synthetic confirmation.
Representative, low-noise guidance produced positive population-guide/evaluation
linear gain in every named intended condition, and corrupting that signal degraded
or reversed it. This passes the cheap mechanism gate. It does not establish LLM
superiority, a hardware budget, or a generalization guarantee. Negative and null
outcomes remain part of the evidence.

The main design has 68 conditions, with 1,000 independent trajectory draws per
condition in each fixed discovery/confirmation stream: 136,000 paired draws across
AdamW, GUIDON and information-matched AdamW. A 32-cell orthogonal fractional screen
covers noise, covariance, batch size, alignment, condition number, guide mismatch,
group count, cadence, radius, stream size, delay and momentum. Targeted conditions
cover intermediate/default levels, manufactured quadratics, spurious regression
and classification, signal corruption/removal/collinearity, controlled rotation
and a large-step curvature boundary. All 68 conditions and all three arms remain
in the summaries. Main effects are descriptive because interactions are aliased.

An independent verifier inspected every raw draw's identities, finite parameters,
neutrality, proxy/fresh gains, guidance count and limits. It recomputed all known
quadratic final evaluation losses and reproduced three draws per condition in each
stream through the public scalar NumPy optimizer, matching all arm parameters and
controller weights within 2e−12 absolute / 2e−11 relative. Every replay also changed
evaluation data and retained bitwise identical parameter paths, with different
evaluation hashes. The manufactured Hessian/gradient finite-difference check and
hand two-block weights [1.15,0.85] pass. Faulty sign/projection/clipping/radius and
nonfinite-weight implementations are rejected by the separate checker.

Confirmation maximum relative neutrality was 2.20e−14, below 1e−5. Mean fallback
fraction was zero in every main condition. The exact-real and finite FP32 audits
remain separate: these trajectory simulations use float64 CPU arithmetic.

| Competing explanation | Intervention / result | What remains live |
|---|---|---|
| M1 representative feedback within the current training-neutral plane | Intended population-guide gain 11.0849±0.0383 MC SE per step; reverse −10.9353±0.0420 and shuffle −3.0931±0.0348 | Supports this local causal signature, not a universal loss theorem |
| M2 extra examples / altered moments explain a win | Information-matched AdamW beats GUIDON in representative spurious regression | Extra data remains an explanation for future learning wins; keep the control |
| M3 generic layerwise schedule or hidden Adam changes | Radius zero / removed guide are bitwise AdamW; fresh collinear coefficients give unit weights; corruption changes transfer | A useful generic schedule remains a future pilot alternative; not eliminated by these tests |
| M4 stored-guide gain guarantees current-guide benefit | Rotation/cadence weakens gain; fresh interaction followup finds stale current reversal | Unconditional stale guarantee rejected; a supplied drift bound remains necessary |
| M5 evaluation information enters learning | Changing evaluation leaves every reference-replayed trajectory bitwise unchanged | Actual corpus/process boundary still needs Phase 03/04 |
| M6 first-order progress ensures actual loss/transfer | Taylor curvature witnesses, mismatch outcomes and all-arm opposite-evaluation failures | Actual loss budgets and representative data cannot be omitted |

The following are confirmation paired mean evaluation-loss differences; positive
means GUIDON has lower loss. Each condition uses its own loss scale, so do not
aggregate these rows as an LLM NLL or cross-task performance ranking.

| Condition | AdamW−GUIDON (MC SE) | Information control−GUIDON (MC SE) |
|---|---:|---:|
| Representative quadratic `intended` | 0.242024 (0.000627) | 0.263355 (0.000646) |
| Reverse coefficients | −0.240305 (0.000680) | −0.219007 (0.000658) |
| Shuffle coefficients | −0.069884 (0.000791) | −0.048554 (0.000797) |
| Opposite quadratic guidance center | −0.239352 (0.000684) | −0.247201 (0.000631) |
| Representative spurious regression | 0.062901 (0.000212) | **−0.081201 (0.000367)** |
| Representative spurious classification | 0.056667 (0.000195) | 0.024472 (0.000193) |
| Mismatched spurious classification | −0.001102 (0.000286) | −0.000890 (0.000282) |
| Large-step curvature boundary | −0.036266 (0.035663) | 257.356822 (9.096172) |

Across all 68 conditions, 36 mean AdamW−GUIDON contrasts and 19 information-
control contrasts were zero or negative, including registered null limits.
They are visible in the complete JSON summaries, not dropped. The large-step
example is uncertain in its AdamW comparison; do not call that mean a confirmed
regression. Small exact high-curvature witnesses independently disprove the
unconditional descent claim. Raw train/guide/evaluation losses, oracle regret where
it is defined, norms, coefficients, fallbacks and all paired contrasts are retained.

The original rotation changed only two coordinates among 14. Its aggregate gain
weakened but never reversed: stationary dimensions masked the intended stale
failure. We retained that outcome and registered a fresh 18-condition group×drift×
cadence interaction followup, 1,000 draws per condition in both new streams (36,000
additional paired draws). Groups 2/14/64, drift 0.001/0.2 and cadence 1/64/256 all
remain. Discovery and confirmation both show negative current gain at updates
8–23 for two-group fast rotation with stale 64/256 cadence. The prespecified
8–24 window passes even with a Bonferroni normal MC bound over 34 checks.
Fresh fast-rotation feedback retains positive mean gain. The 14/64-group diluted
rotations do not reverse, and those outcomes remain in the report and figure.
This followup diagnoses the degrees-of-freedom/active-rotation interaction; it
changes no optimizer equation or intended-regime success threshold.

A separately registered opposite-evaluation quadratic uses one nonzero training
group, train/guide optimum zero, initial parameters uniform in [0.5,1.5] and an
independently generated evaluation optimum in [3,4]. All three arms reduce
training/guidance loss and increase evaluation loss in all 2,000 discovery and
confirmation draws. This rejects universal transfer for every live implemented
candidate, including the extra-information explanation. It is a failure witness,
not a substitute success task. Total independent trajectory draw count is 174,000;
secondary diagnostics reuse existing identities and are not extra confirmation.

Secondary first-probe diagnostics separate coefficient error from trajectory
change. In the intended condition, mean per-coordinate training coefficient-error
variance is 0.000312; batch 8 gives 0.001248 and batch 128 gives 0.0000785. The
zero-noise condition gives exactly zero coefficient error. Guide-error variance
and correction bias grow with noise. These values include u's dependence on g;
they do not certify unbiased Adam-dependent coefficients. The exact skew-noise
normalization-bias witness and the P13/P14 deterministic bounds remain the honest
mathematical scope. All diagnostic conditions, not just the examples above, remain
in `coefficient-noise-v1.json` and immutable per-draw raw records.

The statistical design uses 240,000 synthetic three-pair experiments. With a true
0.005-nat effect and Gaussian paired-difference SD 0.002/0.005/0.01/0.02, two-sided
rejection probabilities are 61.66%/18.12%/8.26%/5.91%, with mean 95% interval widths
0.00879/0.02203/0.04387/0.08830 nats/token. All Gaussian coverage and power checks
agree with independently computed analytic targets within the registered four
MC-SE tolerance. Skewed errors give about 84–85% coverage rather than nominal 95%;
t3 errors give about 96%. Three pairs cannot diagnose these assumptions reliably.
The seed count remains 42/43/44; document bootstrap cannot fix training-seed power.
See the frozen analysis specification and uncertainty budget.

A relative-output-path provenance failure occurred after the complete drift
discovery raw file had been written. The raw file and traceback were retained,
its missing manifest recovered from exact registered identities and the committed
source snapshot, and every record independently verified. No run was restarted.
The CLI path repair has a written driver-equivalence audit. A power-verifier check
also initially demanded tighter quantile-coordinate agreement than SciPy's ppf
accuracy; the final checker verifies the exact df=2 CDF probability residual and
all raw interval formulas. This affects no scientific threshold, effect, raw draw
or inference decision. Both failed checker logs remain in the failure ledger.

Model validation is limited to these declared synthetic generators. Diagonal
quadratics, Gaussian spurious features, known drift and float64 arithmetic omit
Transformer geometry, real source reuse, non-Gaussian gradients, clipping/packing,
architecture/tuning interactions and hardware communication. Fixed common
synthetic hyperparameters are an apparatus comparison, not a claim to have tuned
an LLM baseline. Phase 05 retains equal-budget strong AdamW tuning. Next requested
work is Phase 03's executable corpus preparation and sealed-reader/source audit;
Phase 04 must still build the trainer and prove resume/numerical parity before
any expensive confirmation. No LLM training or sealed real evaluation occurred.
