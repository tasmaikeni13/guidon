# Research contract — GUIDON v0.1

Question: can sparse independent development feedback improve LLM generalization
by moving within AdamW's blockwise training-progress-neutral coefficient plane?
Target failure: a training trajectory that lowers sampled training loss while
allocating updates poorly for independent source distributions.

Regimes: approximately 125M decoder trained from scratch on exactly 2.5B FineWeb-Edu
tokens; Pythia-160M continued for 1B domain/retention tokens. Hardware target: the
user's existing 16-chip v4-32 slice. Seeds 42/43/44 are required full confirmation.
The foundation was prepared earlier. The current task executes Phases 01/02 locally and pushes their verified artifacts to GitHub. Later phases require a separate user request.

Comparison class: strong tuned AdamW, plus AdamW with identical additional guidance
examples. Only AdamW is a competitor optimizer. Match architectures, initial states,
main token streams, loss masks, schedules and tuning budgets; expose guide information
and added compute separately. Compare exact-final-token checkpoints.

Primary metrics: independent validation NLL in nats per supervised token (lower);
mean paired difference across the three seeds; full wall time, measured compute,
peak memory, and retention/registered shifted-domain regression. Proposed meaningful
effect: at least 0.005 nats/token lower NLL; resource ceilings 1.10x AdamW; retention
and shifted-domain ceiling 0.01 nats/token regression. These are prospective decisions,
not observed values. Pilot outcome thresholds must be frozen before final evaluation.

Success evidence: full recorded confirmation, all seeds, information and mechanism
controls, effect sizes with honest uncertainty, clean evaluation boundary, and
end-to-end TPU measurements in both regimes. A proof of an instantaneous identity,
CPU timing, guidance loss, or one favorable seed cannot establish that claim.

Kill/revision triggers: violated numerical neutrality, persistent fallback/schedule
failures, guide mismatch/no causal signal, ineffective coefficient dimensions,
equal-information AdamW explaining the gain, missed resource ceiling, data leakage,
or inconclusive/failing confirmation. Preserve evidence; research and revise the
causal mechanism while retaining the original experiment contract. Final evidence
used to redesign is spent and replaced by newly sealed groups.

Assumptions: guidance distribution represents a useful target mixture; enough fresh
sources remain after filtering; independent evaluation can be protected operationally;
simple block reweighting has useful freedom; extra reductions can meet the pod
budget. None are proved by the current coefficient theorems. Exact-real statements
and explicitly conditional Taylor consequences are separately proved in Lean.

Literature search boundary: primary papers and official artifacts available through
2026-09-30. Novelty is scoped to the recorded searches, with unresolved coverage
identified. No universal novelty or generalization claim is made.
