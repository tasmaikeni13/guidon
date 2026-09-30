# Claim, assumption and dependency audit — GUIDON 0.2

All vectors below are finite real coefficient vectors. The executed floating-point
checks form a separate evidence layer. The Lean build and complete public-theorem
axiom audit are in `research/verification/phase01/`.

| Claim | Checked theorem(s) | Exact domain and assumptions | Dependencies / implemented equation | What it cannot establish |
|---|---|---|---|---|
| P1 projection neutrality | `projection_neutral` | Finite real vectors; division by zero is zero in Lean | dot algebra, nonnegative sum of squares; `reference.project`, JAX coefficient subtraction | Population neutrality, floating-point equality |
| P2 projection gain identity | `projection_gain` | Same domain, zero a allowed | P1, dot commutativity | Actual guide loss decrease |
| P3 raw training linear progress | `training_progress`, `normalized_training_progress` | Any real scale, positive-normalization reconstruction | P1, `normalized_reconstruction`; block dots and fixed shared decay | Momentum descent or population loss |
| P4 fresh/stored linear gain | `guide_progress`, `guide_nonworsening`, `normalized_guide_progress` | Nonnegative scale; raw fresh gain adds maxAbs(c) | P2, `floorScale_nonneg`; fresh pre-update guide, stored C | Current-guide gain on stale steps |
| P5 0.2 radius and positivity | `floor_correction_bounded`, `floor_weight_interval`, `floor_weights_positive` | Nonempty finite groups, rho≥0, kappa>0; positive weights additionally rho<1 | `abs_le_maxAbs`, max denominator; controller scale | Relative perturbation versus a decay-cancelled full baseline |
| P6 parameter perturbation energy | `floor_perturbation_energy` | Above; nonnegative block energies; application needs disjoint coordinates | P5 squared, summed; D−D0=delta*u | Overlapping/tied coordinate double counting |
| P7 conditional actual descent | `conditional_descent` | eta≥0, supplied Taylor upper bound and curvature-step budget | Scalar inequality; no code enforces smoothness/budget | Universal AdamW/Transformer convergence |
| P8 actual guide comparison | `guide_loss_comparison` | Supplied upper/lower remainders, sum dominated by eta*gain | Scalar inequality and P4 | Guaranteed loss decrease without curvature bounds |
| P9 information noninterference | `evaluation_noninterference`, `transcript_extensionality` | Same initial state, transition and authorized transcript; no evaluation-derived choice upstream | Pure trajectory recursion; simulation independent RNG/reader inputs | Dataset cleanliness, pretraining contamination certificate |
| P10 impossibility witness | `generalization_counterexample` | Explicit real quadratics F=(x−1)^2, E=(x+1)^2 | `norm_num`; rational witness independently checked | Any unconditional external generalization claim |
| P11 quadratic optimum | `quadratic_optimality`, `quadratic_optimum_attained` | Feasible a·z=0; displayed algebra for any tau; dividing into penalized objective requires tau>0 | Completion of square, P1/P2; scalar tau chosen after C | Infinity-ball LP optimum or fixed data-independent penalty |
| P12 numerical second projection fidelity | `projection_idempotent` | Exact finite real arithmetic | P1; numerical reproject-q pass | Bitwise idempotence in FP32 |
| P13 sampled-to-population coefficient error | `dot_error_l1`, `population_neutrality_error` | Coordinatewise abs(delta)≤rho and sampled neutrality; no independence assumed | Absolute sum bound, dot subtraction | Small population error unless coefficient error is small |
| P14 stale-current coefficient drift | `stale_gain_lower` | Coordinatewise radius; both raw coefficient vectors expressed with the current adaptive direction | Same absolute sum bound | Drift control or guide representativeness supplied by the algorithm |

Existing `stepScale` theorems remain valid for the archived 0.1 normalization.
They are audited too, but P5/P6 for the current algorithm use the floor theorems.
Standard dependencies `propext`, `Classical.choice`, `Quot.sound` are expected;
no project axioms or proof placeholders are allowed. `evaluation_noninterference`
uses no axioms; `transcript_extensionality` uses `propext`.

| Further analysis | Status and exact assumptions | Phase 02 measurement / falsifier |
|---|---|---|
| Degrees of freedom B−1 for a≠0; B at zero | Standard rank-nullity applied to one nonzero linear functional; SVD oracle checks all requested group sizes | One-group correction absent; vary 1/2/14/64 |
| Projection sensitivity near a=0 | No unconditional Lipschitz claim: normalization changes direction arbitrarily at zero. A bounded-noise/separated-norm result would need a norm lower bound | Vary noise covariance, batch size and moment lag; retain opposite-gradient cases |
| Normalization bias | Exact skew zero-mean coefficient noise has nonzero expected correction; rational witness. No unbiasedness theorem | Compare IID/correlated noise and finite guide source sizes; coefficient variance/transfer |
| Fixed-u variance scale ∝1/n | Measurement-model prediction with IID fresh gradients, fixed u and finite covariance; does not apply directly to adaptive u(g) | Batch 8/32/128, guide stream 512/4096, dependence-sensitive summaries |
| Joint adaptive coefficient noise bounds | Conjectural without a distribution for dependent g,u. No distribution-free population-descent claim | Test fresh true population/evaluation gain in the registered low-noise regime and high-noise failures |
| Deterministic drift budget | Checked P14; imposing L1 drift≤L_c*age yields a supplied budget, not a proved property of training | Controlled rotation, delay 0/16/64, interval 1/16/64/256 |
| Floored q→delta continuous at q=0 | Direct positive-denominator formula; max-norm normalization of raw c and a still has zero-norm discontinuities | Extreme scales, near-collinearity and zero-coefficient witnesses |

The statistical noise scaling and sensitivity proposals are not promoted to new
formal theorem claims. Their unresolved adaptive-dependence assumptions are carried
forward explicitly; the deterministic bounds are the checked results.
