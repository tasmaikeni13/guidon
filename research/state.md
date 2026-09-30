# Research state — 2026-09-30

The requested foundation is being prepared; the user starts research phases later.
Authoritative run/gate status is `phases/state.json`: all nine phases not started.
There are no LLM, TPU, pilot, Monte Carlo study, or benchmark results to report.

Established evidence: the coefficient proofs compile; the axiom audit uses only
Lean's standard foundations (and evaluation noninterference uses no axioms); current
NumPy/JAX tests exercise projection, moment isolation, probe schedules, Optax parity,
declared split boundaries, token/probe arithmetic, and global array reduction.
See `research/verification.md` for exact final commands and retained outputs.

Live hypothesis H1 is the training-neutral block controller. The full portfolio,
mechanism mapping, risky predictions, and alternative explanations are in
`research/mechanism.md`. Its generalization and cost predictions remain unknown.
Novelty is provisional; `research/literature.md` records closest precedents and a
blocked primary-source comparison, not a promise of exhaustive novelty.

## Failure/limitation ledger

| Candidate assertion / branch | Evidence or mechanism | Resolution / reopening |
|---|---|---|
| Guidance can be untouched validation | Its coefficients influence parameters | Label it optimization data; seal distinct evaluation |
| Local identities prove universal generalization | P10 counterexample: train 1→0, eval 1→4 | Replace with prospective empirical hypothesis |
| Cache weights without current reprojection | $(1,1)\cdot(0.1,-0.1)=0$ but $(1,2)\cdot(0.1,-0.1)=-0.1$ | Store coefficients; reproject using current $a$ |
| Preserve first-order progress ⇒ actual descent | Taylor curvature and negative momentum progress | Conditional P7, measured true losses |
| Small probe-token fraction ⇒ near-parity speed | Every-step reductions/memory and pod communication | End-to-end Phase 04 measurement |
| Check certificate before forming float32 weights | Adding 1 can alter the realized correction | Check $w-1$ and proxy sign after weight construction |
| Existing proofs permit placeholders to pass | Initial Lean elaboration errors were implementation errors | Corrected definitions/proofs; final build and axiom audit clean |
| Broad “validation-guided layerwise rates” novelty | MetaLR and other precedents | Narrow contribution to constrained composition; resolve remaining FSP access |

Next action for the user: request Phase 01. Its wider assumption/numerical/statistical
audit uses the checked foundation and must produce its own gate evidence. A passing
unit test does not mark that research phase complete.
