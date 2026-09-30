# Machine-checked coefficient theory

```bash
cd proofs
lake update
lake exe cache get
lake build
lake env lean Audit.lean
```

Lean is pinned to 4.24.0, with mathlib tag `v4.24.0` resolved in
`lake-manifest.json` to commit `f897ebcf72cd16f89ab4577d0c826cd14afaafc7`.
No project axioms or proof placeholders are permitted. The audit prints dependencies;
Lean's standard `propext`, `Classical.choice`, and `Quot.sound` are expected.

`Core.lean` proves finite real coefficient identities, normalized radius bounds,
the quadratic optimum, and conditional Taylor inequalities. `Isolation.lean`
proves evaluation noninterference for the specified pure transition and gives a
counterexample to unconditional generalization. [theory.md](../theory.md) maps P1–P11
to actual theorem names and states assumptions.

These theorems do not certify floating-point code, data deduplication, the pretrained
corpus, Transformer smoothness, statistical significance, TPU efficiency, or improved
LLM generalization. Tests establish a separate implementation evidence layer.
An equation change requires updating the statements, rebuilding, auditing assumptions,
and revalidating every dependent phase. Do not weaken a statement solely to silence
Lean while leaving a stronger claim in the theory notes.
