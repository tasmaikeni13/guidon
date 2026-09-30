# GUIDON: Guidance Using Independent Development Objectives and Neutrality

Version 0.2, 2026-09-30. Experimental theory notes, not a paper.

## Question and information boundary

Can occasional feedback from a separate development landscape improve an AdamW
trajectory without sacrificing its instantaneous training progress or adding much
compute? Better generalization and TPU throughput are **hypotheses**, not established
results. A scalar derived from examples is still information from those examples.

Use five disjoint document/source groups:

| Split | Permitted use |
|---|---|
| `train` | Ordinary gradients, Adam moments |
| `guidance` | Optimizer feedback; counts as optimization data |
| `development` | Pilot decisions and hyperparameter selection for every arm |
| `validation` | Sealed confirmation losses after the protocol is frozen |
| `test` | Sealed final and distribution-shift evaluation |

The requested validation landscape is implemented as **guidance validation**: it
is separate from ordinary training, but cannot be described as an untouched
evaluation set. If “without leakage” means that *no information whatsoever from a
validation set can influence learning*, validation guidance is incompatible with
that meaning. Here it means no final validation/test information enters optimization,
preprocessing, sampling, tuning, initialization, or checkpoint selection. See
[the data protocol](research/data_protocol.md) for the operational checks.

This boundary prevents evaluation contamination caused by this research workflow.
It does not establish IID sampling, remove all undetected web duplicates, certify
a pretrained model's entire history, or guarantee performance in every deployment.

## Baseline and parameterization

Partition parameters into $B$ **disjoint coordinate blocks**: tied embeddings occur
once, each Transformer block gets one group, and the remaining final normalization
gets one group. Let $g_t$ be the current accumulated training gradient, after the
same global clipping used by both optimizers. AdamW computes

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\qquad
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^{\odot2},
$$
$$
\hat m_t=m_t/(1-\beta_1^{t+1}),\qquad
\hat v_t=v_t/(1-\beta_2^{t+1}),\qquad
u_{t,j}=\hat m_{t,j}/(\sqrt{\hat v_{t,j}}+\epsilon).
$$

With a fixed decay mask $M$, its positive step direction is
$D^0_t=u_t+\lambda M\theta_t$, and $\theta_{t+1}=\theta_t-\eta_tD^0_t$.
GUIDON uses identical moments and decay. Only the adaptive block directions are
reweighted. Decay remains decoupled and is never multiplied by guidance weights.
[AdamW](https://arxiv.org/abs/1711.05101) supplies this baseline.

## Controller

At a probe step $s$, compute $h_s=\nabla G_s(\theta_s)$ from a **fresh, never reused**
guidance batch, at the same pre-update parameters as $g_s$. Define

$$a_{t,j}=\langle g_{t,j},u_{t,j}\rangle,\qquad
c_{s,j}=\langle h_{s,j},u_{s,j}\rangle.$$

Store only $C_s=c_s/\|c_s\|_\infty$ (zero if $c_s=0$); discard $h_s$.
At every step, form $A_t=a_t/\|a_t\|_\infty$ (zero if $a_t=0$) and

$$
q_t=P_{A_t}^{\perp}C_s
=C_s-\frac{A_t^\top C_s}{A_t^\top A_t}A_t,
\quad\text{with }0/0=0\text{ in this formula}.
$$

Before the first probe set the correction to zero. For age $d=t-s$, use

$$
\rho_t=\rho\max(0,1-d/K),\qquad
\tau_t=\rho_t/\max(\|q_t\|_\infty,\kappa),\qquad\kappa>0,
$$
$$
\delta_t=\tau_tq_t,\quad w_{t,j}=1+\delta_{t,j},\quad
D_t=(w_{t,j}u_{t,j})_{j=1}^B+\lambda M\theta_t,\quad
\theta_{t+1}=\theta_t-\eta_tD_t.
$$

Positive $c_j$ means that increasing the block's descent step improves the guide
linear model; hence the **plus** sign in $w=1+\delta$. Normalizing $a$ does not change
its orthogonal complement. Normalizing $c$ only rescales the positive step scale.
Defaults are $\rho=0.15$, $K=64$, $\kappa=0.01$, first probe at zero-indexed update 128. The signal floor acts only on the positive scale, so $a^\top\delta=0$ still holds. It removes the full-radius discontinuity at $q=0$; below the floor, $\|\delta\|_\infty=\rho_t\|q\|_\infty/\kappa$. Both implementations reproject $q$ once for roundoff control; `projection_idempotent` proves this is the same exact-real equation. They are
provisional, subject to equally budgeted pilot tuning. Radius zero is AdamW.

In exact arithmetic, for any $\tau\ge0$, $\delta=\tau P_A^\perp C$ solves

$$\max_{A^\top z=0}\left(C^\top z-\frac{\|z\|_2^2}{2\tau}\right),\qquad\tau>0.$$

Indeed, on the feasible plane,

$$2\tau C^\top z-\|z\|_2^2
=\tau^2\|q\|_2^2-\|z-\tau q\|_2^2\le\tau^2\|q\|_2^2.$$

The infinity normalization chooses a feasible radius. It does **not** solve the
separate linear program with an infinity-ball constraint, nor does it make the
quadratic penalty independent of the observed guide vector.

## Proved properties and their scope

The finite real coefficient statements are machine checked in
[Core.lean](proofs/Guidon/Core.lean). Names below are authoritative.

| ID | Lean theorem | Statement |
|---|---|---|
| P1 | `projection_neutral` | $A^\top q=0$, including $A=0$ |
| P2 | `projection_gain` | $C^\top q=\|q\|_2^2$ |
| P3 | `training_progress`, `normalized_training_progress` | $a_t^\top w_t=a_t^\top\mathbf1$, including raw-coefficient normalization |
| P4 | `guide_progress`, `guide_nonworsening`, `normalized_guide_progress` | $C_s^\top(w_t-1)=\tau_t\|q_t\|_2^2\ge0$; raw fresh gain includes $\|c_s\|_\infty$ |
| P5 | `floor_correction_bounded`, `floor_weight_interval`, `floor_weights_positive` | $|\delta_j|\le\rho_t$; $1-\rho_t\le w_j\le1+\rho_t$, positive for $\rho_t<1$ |
| P6 | `floor_perturbation_energy` | $\sum_j\delta_j^2e_j\le\rho_t^2\sum_je_j$ for all $e_j\ge0$ |
| P7 | `conditional_descent` | Taylor upper bound plus step-size budget implies actual training nonincrease |
| P8 | `guide_loss_comparison` | Explicit upper/lower Taylor bounds plus a gain budget imply guided loss no worse than the baseline step |
| P9 | `evaluation_noninterference`, `transcript_extensionality` | Fixed authorized input transcript implies the same parameter trajectory for arbitrary evaluation payloads |
| P10 | `generalization_counterexample` | Training/guidance improvement can worsen independent evaluation |
| P11 | `quadratic_optimality`, `quadratic_optimum_attained` | The feasible quadratic bound and its attained optimum |

The scaling in P3 preserves zero under positive normalization of $a$; multiplying
back gives $a_t^\top\delta_t=0$. Because decay is identical,

$$\langle g_t,D_t\rangle=\langle g_t,D_t^0\rangle.$$

At a **fresh** probe with $m_c=\|c_t\|_\infty$,

$$\langle h_t,D_t-D_t^0\rangle=m_c\tau_t\|q_t\|_2^2\ge0.$$

Between probes P4 concerns the stored coefficient surrogate $C_s$, not the current
guidance gradient. Reprojecting a previously stored *weight vector* instead of $C_s$
would be a different method. No stale-current-guide guarantee is claimed.

For disjoint blocks set $e_j=\|u_{t,j}\|_2^2$ in P6 to obtain
$\|D_t-D_t^0\|_2\le\rho_t\|u_t\|_2$.
This bound is relative to the adaptive direction, not to $D^0$, which could be small
because of cancellation with decay.

If $F$ has an $L_F$ smoothness upper bound at this step, then

$$F(\theta-\eta D)\le F(\theta)-\eta b+
\tfrac12 L_F\eta^2\|D\|_2^2,\qquad b=\langle\nabla F(\theta),D^0\rangle.$$

P7 applies if $\eta\ge0$ and $L_F\eta\|D\|^2\le2b$.
Adam momentum need not make $b$ positive; minibatch $g_t$ need not equal a population
gradient. P7 is a conditional local statement, not an AdamW convergence proof.

For a fresh guide gradient, two-sided Taylor remainder bounds give

$$G(\theta-\eta D)-G(\theta-\eta D^0)
\le-\eta m_c\tau\|q\|^2+
\tfrac12 L_G\eta^2(\|D\|^2+\|D^0\|^2).$$

P8 requires the positive first-order gain to dominate that remainder. The code does
not estimate $L_G$ or enforce this condition; actual guide or training loss can rise.
P7/P8 assume supplied Taylor bounds rather than proving smoothness of a Transformer.

P9 proves an information-flow property of a pure state-transition specification.
It assumes inputs, seeds, preprocessing, and initial state are chosen without the
sealed evaluation payload. The Python manifest tests and future data/process audit
must justify that assumption. It is not a proof that external datasets are clean.
P10 uses $F(x)=(x-1)^2$ and independent $E(x)=(x+1)^2$: moving $0\to1$ reduces $F$
from 1 to 0 and raises $E$ from 1 to 4. Better real-world learning cannot be proved
from these coefficient identities.

## Numerical and resource model

Float32 kernels are not exact-real proofs. Normalize coefficient vectors before
projection; use zero-denominator branches; test

$$|A^\top\delta|\le \varepsilon_{\rm cert}
\max(\rho_t\|A\|_1,10^{-30}),\qquad\varepsilon_{\rm cert}=10^{-5}.$$

The certificate uses the **realized** correction $w-1$, after floating-point addition,
and also checks that its stored-guide dot product is nonnegative. The guide sign certificate exceeds a conservative $4B\epsilon_{32}\sum_j|C_j(w_j-1)|$ summation-error budget. JAX places an optimization barrier before forming the realized correction, preventing XLA from cancelling the rounded addition/subtraction. Nonfinite
coefficients or a failed certificate return unit weights and log a fallback.
Radius bounds in floating point carry rounding error; P5/P6 are exact-real statements.
Nonfinite training gradients invalidate a run and must be diagnosed.
Missing/unexpected probe calls are protocol errors; JAX reports `schedule_ok=False`
and the trainer must abort. It must also assert `numerics_ok`; finite input gradients can still overflow moments or parameter updates. No independent coordinate clipping follows projection.

Adam stores $2P$ moment elements. GUIDON adds $B$ coefficients and counters, one
$O(P)$ train dot reduction per step, an $O(B)$ projection, and a guide
forward/backward plus $O(P)$ guide reduction every $K$ updates. The guide gradient
is temporary $O(P)$ memory. XLA fusion, memory traffic, and communication still
determine wall time. With 524,288 training and 65,536 guidance loss tokens per probe,
the amortized extra model-token fraction is approximately $1/(8K)=0.1953\%$;
this is **not** a measured speed or FLOP-overhead claim.

## Empirical targets and failure boundaries

Target: improve sealed mean NLL by at least 0.005 nats/token on the registered
125M/2.5B task, remain within 10% wall time and 10% measured training compute, and
avoid material regression on registered shifted domains. Repeat for continued
pretraining. Compare train-only AdamW and AdamW that receives the identical guidance
examples as ordinary gradient data. All actual values remain unmeasured.

Expected failures: guide mismatch, noisy or collinear coefficients, one effective
group, too few independent guide sources, stale feedback, moment/train misalignment,
and reduction/communication overhead. Use removal, permutation, delay, radius, and
information-matched controls to distinguish the proposed mechanism from more data
or a better learning-rate schedule. See [phases](phases/README.md).

## Relation to prior work

Validation-based example weighting precedes this idea
([Ren et al.](https://proceedings.mlr.press/v80/ren18a.html)); so do alignment-based
domain weighting ([DoGE](https://arxiv.org/abs/2310.15393)), validation-driven
layerwise learning rates ([MetaLR](https://arxiv.org/abs/2206.01408)), and constrained
gradient combinations ([CAGrad](https://arxiv.org/abs/2110.14048)). Layerwise LLM
schedules also have recent precedents ([LLR](https://arxiv.org/abs/2605.22297)).
The candidate contribution is their specifically constrained composition: training-
neutral reweighting of **existing disjoint AdamW directions**, fresh sparse guidance,
and reprojected stale coefficient feedback. The projection mathematics is standard.
This composition is apparently distinct from the read closest formulations; novelty
is provisional, including an inaccessible 2026 feasible-set projection comparison,
and must be checked again before a paper. Evidence and search limits are
recorded in [the literature ledger](research/literature.md).

## Coefficient uncertainty and drift (Phase 01)

`dot_error_l1`, `population_neutrality_error`, and `stale_gain_lower` in
[Robustness.lean](proofs/Guidon/Robustness.lean) prove deterministic bounds for
$|\delta_j|\le\rho_t$:

$$|a_*^\top\delta|\le\rho_t\|a_*-a_t\|_1,\qquad
c_t^\top\delta\ge c_s^\top\delta-\rho_t\|c_t-c_s\|_1.$$

The first inequality assumes sampled neutrality; the second uses raw coefficients
expressed at the current adaptive direction, so coefficient drift includes changes
in both guide gradients and Adam directions. A drift assumption
$\|c_t-c_s\|_1\le L_c d$ gives an error budget $\rho_t L_c d$;
the code does not estimate or enforce it. Normalizing stored coefficients hides
raw magnitude changes, so its certificate alone cannot substitute for that bound.

For fixed adaptive directions and unbiased coefficient estimates with finite
coordinate variances, Cauchy–Schwarz suggests a noise scale proportional to
$\rho_t\sum_j\sigma_j/\sqrt{n}$. This is a **measurement model**,
not a formal theorem about adaptive Adam moments: the same sampled gradient enters
$a$ and $u$, and they are dependent. Unbiased raw coefficient noise does not imply
an unbiased normalized correction (the exact skew-noise witness in the Phase 01
report disproves that claim). Projection sensitivity grows when the training
coefficient norm approaches zero, and direction sensitivity grows near collinearity;
the signal floor controls the latter amplification but does not certify population
progress. Phase 02 measures these effects rather than assuming independence.

The valid nullspace has $B-1$ dimensions for nonzero $a$ and $B$ for $a=0$
by checked `effective_degrees_of_freedom` / `zero_constraint_degrees_of_freedom`
(rank-nullity of the coefficient functional). One nonzero training group therefore
has no correction. Canonical
tied parameters occur once; accumulation precedes the common global clipping.
A positive global clip rescales $a$ for a fixed $u$, preserving its nullspace, but
clipping changes Adam moments and the protected gradient is the supplied clipped
gradient. Future trainers must preserve these interface assumptions.
