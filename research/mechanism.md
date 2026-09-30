# Mechanism reconstruction and hypothesis portfolio

Target socket: change the distribution of an existing AdamW step across disjoint
model blocks using occasional external development feedback, within a tight
compute/state budget and without feeding evaluation into learning.

## Source mechanisms and transfer

**Geometric constrained motion:** [CAGrad](https://arxiv.org/abs/2110.14048) constrains
an update around an average task gradient while optimizing local improvement. Its
state/observations are task gradients and a small coefficient problem. The useful
relation is a secondary objective subject to an explicit local protection constraint;
its convergence assumptions and multi-task objective are not transported here.

**Online feedback through layer rates:** [MetaLR](https://arxiv.org/abs/2206.01408)
uses online meta-feedback to adapt individual layer learning rates during transfer.
The useful relation is a separate performance measurement controlling a small set
of step scales. GUIDON replaces unconstrained rate adaptation with a current-training
neutral correction and sparse probes. Sparse-delay stability is a new hypothesis,
not a guarantee inherited from MetaLR.

| Source role/relation | Target reconstruction | Mismatch / assumption |
|---|---|---|
| Candidate direction acts on parameters | Fixed AdamW block directions $u_j$ | Moment directions need not be descent directions |
| Local protection constraint | $a^\top\delta=0$ | Protects sampled first-order training progress only |
| Secondary objective from separate feedback | $c^\top\delta$ | Guidance is optimization data, not sealed evaluation |
| Low-dimensional constrained actuation | Block scales $1+\delta_j$ with radius bound | At most $B-1$ degrees of freedom; partition sensitivity |
| Online feedback at a measurement timescale | Fresh probe coefficients, then current reprojection | Staleness can destroy current-guide benefit |

Domain-neutral bridge: for finite real vectors $a,c$, maximize
$c^\top\delta-\|\delta\|^2/(2\tau)$ subject to $a^\top\delta=0$. Its solution is
$\delta=\tau(c-aa^\top c/\|a\|^2)$, with the zero-$a$ convention in theory.
Choose positive scaling for a radius bound. Exact dimensions are coefficient-space
scalars; normalized vectors remove arbitrary positive loss scaling. The bridge is
reconstructed in `reference.controller` and the JAX step; P1–P6/P11 certify algebra.

Compatibility: inputs/actuators are available, state is O(block count), train and
guide gradients are observable at a probe, and disjoint groups make the parameter
perturbation energy additive. Unverified: guide representativeness, coefficient SNR,
drift, and hardware cost. Removal/corruption predictions and matched controls are
specified in Phase 02. No analogy to control stability is asserted without a proof.

## Structurally distinct branches

| ID | Causal change | Observable prediction | Simplest confound / cheapest kill | Status |
|---|---|---|---|---|
| H1 | Training-neutral block scale projection | Fresh linear guide gain with identical training linear progress | Extra data; compare information-matched AdamW and corrupt guide | Chosen prototype; empirical benefit untested |
| H2 | Guide selects among momentum timescales | Better response when old momentum opposes representative guide | Ordinary momentum tuning; rotating-gradient control | Proposed alternative; state/compute cost must be budgeted |
| H3 | Sparse guide-forward candidate step selection | Candidate ranking predicts independent loss at probe | Selection noise; fresh holdout and equal-search compute | Proposed alternative; added candidate forwards may miss speed target |
| H4 | Alignment-guided domain sampling | Source allocation improves target loss | Known DoGE-style data allocation; equal-source data controls | Established family, not chosen as a novelty claim |
| H5 | Full guide-gradient conflict projection | Parameter-space rotation helps in conflicts unavailable to block scales | Extra gradient injection/information; compare same guide exposure | Known multiobjective family; larger action/state path |
| H6 | Training-only sharpness correction | Curvature-sensitive robustness without guide examples | Extra backward/perturbation passes; matched-compute test | Alternative, likely conflicts with near-parity budget |

These are different causal branches, not radius variants. They have not been run.
Any replacement that survives an actual kill test must replace the math/proof/code
and downstream instructions together; retain the original 125M/2.5B comparison.

Risky prediction for H1: its independent benefit should weaken when guide coefficients
are permuted, vanish when $c\parallel a$, and deteriorate with guide mismatch/drift,
even though training neutrality remains. Falsifier: equal-information AdamW or a
frozen block schedule explains the apparent gain. Present status: **implemented**,
exact coefficient properties checked; no empirical ML confirmation.
