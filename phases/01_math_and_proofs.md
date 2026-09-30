# Phase 01 — Mathematical behavior and proof fidelity

**Entry:** requested by the user; optimizer foundation present. Existing proofs are
starting evidence, not signoff for this wider analysis.

**Question:** does the exact implemented controller preserve current training
first-order progress, improve the fresh guide linear model, and remain bounded
under the intended numerical and stochastic conditions?

1. Re-derive moments, coefficient normalization, projection, radius, age decay,
   and fixed decoupled decay from `theory.md`. Match each equation to both Python
   implementations. Audit clipped versus unclipped gradients, accumulation, tied
   parameters, zero coefficients, empty layouts, and probe timing.
2. Rebuild Lean and audit theorem dependencies. Prove any added claimed mathematical
   properties with their exact domains and assumptions. P7/P8 are conditional on
   Taylor bounds; P9 is conditional on independent authorized inputs. Never present
   them as universal Transformer convergence or an external-data certificate.
3. Attack the design: one group; $c\parallel a$; $a=0$; adverse moment alignment;
   opposite population/training gradients; large curvature; decay cancellation;
   conflicting guidance; and discontinuity of direction normalization near $q=0$.
   Preserve exact/rational counterexamples where possible. Verify that P10 rejects
   any proposed unconditional generalization guarantee.
4. Analyze sample noise in $a,c$, projection sensitivity, the normalization's bias,
   stale-surrogate error, and the effective degrees of freedom ($B-1$ for $a\ne0$).
   Seek bounds under explicitly stated variance/drift assumptions; if they cannot
   be proved, mark them conjectures and design measurements for Phase 02.
5. Audit float32 and bfloat16 interfaces against float64 and exact witnesses. Sweep
   extreme scales, near-collinearity, group cardinality, and coefficient cancellation.
   If fallback is frequent, derive a signal threshold or regularized replacement,
   update the exact theory and proofs, and invalidate downstream plans. Adding a
   denominator epsilon without re-deriving neutrality is a failed fix.

**Existing commands:** `uv run pytest -q tests/test_reference.py tests/test_jax.py`;
inside `proofs`, `lake build` and `lake env lean Audit.lean`.

**Build:** an independent numerical/adversarial analysis script in
`research/experiments/math_audit.py`; a claim/assumption/dependency table; minimal
counterexamples; a versioned proof-fidelity report. Record quantitative tolerances
before the randomized check, including relative neutrality, finite weights, and
nonnegative *fresh linear* gain. Test rejection of known faulty implementations.

**Exit:** every theorem claim has a checked proof and correct interpretation;
numerical discrepancies and fallbacks are explained; expected/absent/reversed behavior
is explicit. A proof that says less than the proposed mechanism invalidates the
mechanism claim. Research, revise equations, and rerun until this gate is met. Write
`research/gates/01.json`; revise 02–09 if any load-bearing equation changes.
